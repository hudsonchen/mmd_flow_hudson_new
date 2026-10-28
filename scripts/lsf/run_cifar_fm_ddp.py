"""Four-GPU CIFAR flow matching, exact DDPM U-Net, resumable optimizer updates."""
import argparse
from contextlib import nullcontext
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from cifar_flow_matching import DDPMFlow, FeatureMoments, flow_target, sample_heun, skewed_times, update_ema
from data import download_cifar10, load_cifar10_split
from fid import load_inception, frechet_distance
from run_storage import new_run_path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rank_indices(count, global_batch, world, rank, update, seed):
    batches = count // global_batch
    epoch, batch = divmod(update - 1, batches)
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))
    local = global_batch // world
    start = batch * global_batch + rank * local
    return permutation[start:start + local]


def reduce_moments(moments, device):
    for key in ['sum', 'outer']:
        value = torch.from_numpy(getattr(moments, key)).to(device)
        dist.all_reduce(value)
        setattr(moments, key, value.cpu().numpy())
    count = torch.tensor(moments.count, device=device, dtype=torch.long)
    dist.all_reduce(count)
    moments.count = int(count)
    return moments.statistics()


def rng_state(device):
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'], device)


def checkpoint(output, model, ema, optimizer, scaler, update, attempts, best, config, device):
    local = rng_state(device)
    states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, states, dst=0)
    if dist.get_rank() == 0:
        save_torch(output/'latest.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), update=update,
            attempts=attempts, best_fid=best, rank_rng=states, config=config,
            lineage_has_flow_matching=True, objective='flow-matching'))
    dist.barrier()


@torch.inference_mode()
def preview(ema, output, update, args, device):
    if dist.get_rank() == 0:
        from torchvision.utils import save_image
        rng = torch.Generator(device=device).manual_seed(8765432)
        noise = torch.randn(32, 3, 32, 32, generator=rng, device=device)
        raw = torch.cat([sample_heun(ema, z, args.heun_steps, args.amp)
                         for z in noise.split(args.sample_batch)])
        save_image(((raw+1)/2).clamp(0, 1), output/f'preview_{update:07d}.png', nrow=8)
        write_json(output/f'preview_{update:07d}.json', dict(update=update, count=len(raw),
            solver='heun', grid='edm_rho7_exact_endpoints', steps=args.heun_steps,
            nfe=2*args.heun_steps, seed=8765432,
            raw_pixel_variance=float(raw.var(0).mean()),
            raw_pixel_outside_fraction=float((raw.abs()>1).float().mean())))
    dist.barrier()


@torch.inference_mode()
def reference_stats(raw, inception, args, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    moments = FeatureMoments()
    shard = raw[rank::world]
    for images in shard.split(args.fid_batch):
        features = inception(images.to(device).float()/255)[0].flatten(1)
        moments.add(features.cpu().numpy())
    return reduce_moments(moments, device)


@torch.inference_mode()
def evaluate(ema, inception, reference, output, update, args, device, intervals=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    intervals = intervals or args.heun_steps
    count = args.fid_samples // world + (rank < args.fid_samples % world)
    # CPU RNG creates reproducible noise independent of CUDA kernels and batch size.
    noise = torch.randn(count, 3, 32, 32,
                        generator=torch.Generator().manual_seed(100000 + args.seed + rank))
    moments = FeatureMoments()
    outside = 0
    preview_images = []
    started = time.monotonic()
    for i, z in enumerate(noise.split(args.sample_batch)):
        raw = sample_heun(ema, z.to(device), intervals, args.amp)
        outside += int((raw.abs() > 1).sum())
        pixels = ((raw+1)/2).clamp(0, 1)
        for batch in pixels.split(args.fid_batch):
            moments.add(inception(batch)[0].flatten(1).cpu().numpy())
        if rank == 0 and len(preview_images)*args.sample_batch < 100:
            preview_images.append(pixels.cpu())
        if rank == 0 and i % 10 == 0:
            write_json(output/'evaluation_status.json', dict(update=update, solver='heun',
                steps=intervals, rank0_generated=min((i+1)*args.sample_batch, count),
                rank0_target=count, elapsed_seconds=time.monotonic()-started))
    stats = reduce_moments(moments, device)
    tally = torch.tensor(outside, device=device, dtype=torch.long)
    dist.all_reduce(tally)
    row = None
    if rank == 0:
        from torchvision.utils import save_image
        row = dict(update=update, fid=frechet_distance(*reference[:2], *stats[:2]),
            generated_count=stats[2], reference_count=reference[2], reference_split='train',
            solver='heun', grid='edm_rho7_exact_endpoints', steps=intervals, nfe=intervals*2,
            raw_pixel_outside_fraction=int(tally)/(stats[2]*3072),
            evaluation_seconds=time.monotonic()-started,
            preprocessing='unquantized float [0,1], same Inception as Riesz and DDPM')
        suffix = '' if intervals == args.heun_steps else f'_heun{intervals}'
        write_json(output/f'metrics_{update:07d}{suffix}.json', row)
        save_image(torch.cat(preview_images)[:100], output/f'samples_{update:07d}{suffix}.png', nrow=10)
        print(json.dumps(row), flush=True)
    payload = [row]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=250000)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--heun-steps', type=int, default=50)
    p.add_argument('--sampler-audit-steps', type=int, nargs='+', default=[100, 250])
    p.add_argument('--sample-batch', type=int, default=32)
    p.add_argument('--fid-batch', type=int, default=32)
    p.add_argument('--fid-samples', type=int, default=10000)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--preview-every', type=int, default=1000)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if min(args.steps,args.microbatch,args.accumulation,args.threads,args.sample_batch,
           args.fid_batch,args.eval_every,args.preview_every) < 1 or args.fid_samples < 4:
        p.error('Counts must be positive and FID requires at least four samples')
    if args.heun_steps < 3 or any(s < 3 for s in args.sampler_audit_steps):
        p.error('Heun requires at least three intervals')
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('This experiment requires exactly four GPUs')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', device_id=device, timeout=datetime.timedelta(hours=2))
    output = new_run_path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output/'config.json').exists() and not args.resume:
            raise ValueError('Existing training output requires --resume')
    dist.barrier()
    torch.manual_seed(args.seed)
    model = DDPMFlow().to(device)
    parameters = sum(v.numel() for v in model.parameters())
    if parameters != 61804419:
        raise RuntimeError(f'Wrong DDPM backbone: {parameters}')
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              gradient_as_bucket_view=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9,.95),
                                 weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp, init_scale=1024)
    torch.manual_seed(args.seed + 1009 * rank + 1)
    torch.cuda.manual_seed(args.seed + 1009 * rank + 1)
    global_batch = args.microbatch * args.accumulation * world
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(parameters=parameters, world_size=world, global_batch=global_batch,
        objective='flow-matching', architecture='ddpm-unet', initialization='random',
        time_sampling='t=sigmoid(1.2-1.2*N(0,1)), clipped to [0.0001,1]',
        path='x_t=(1-t)*noise+t*image; target=image-noise',
        time_embedding='999*(1-t)', optimizer='AdamW beta=(0.9,0.95), constant LR',
        sampling='Heun, EDM rho7 grid anchored at 0 and 1; 2 network calls per interval',
        recipe='Meta flow_matching CIFAR recipe adapted to the unchanged DDPM backbone',
        fid='FID-10K versus all 50000 training images; unquantized pixels',
        lineage_has_flow_matching=True)
    update, attempts, best = 0, 0, float('inf')
    resume_rng = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Resume configuration differs from the saved run')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        update, attempts, best = state['update'], state['attempts'], state['best_fid']
        resume_rng = state['rank_rng'][rank]
        del state
        if (output/'best.json').exists():
            best = min(best, json.loads((output/'best.json').read_text())['fid'])
    if rank == 0:
        write_json(output/'config.json', config)
        write_json(output/'status.json', dict(status='initializing', update=update))
    hardware = dict(rank=rank, gpu=torch.cuda.get_device_name(device),
        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        capability=torch.cuda.get_device_capability(device), torch=torch.__version__,
        cuda=torch.version.cuda, parameters=parameters)
    write_json(output/f'hardware_rank{rank}.json', hardware)
    # Host-resident uint8 data avoids four full floating-point GPU copies.
    archive = download_cifar10(args.data_root)
    raw = load_cifar10_split(archive, True)[0].reshape(-1,3,32,32).pin_memory()
    inception = load_inception(device).requires_grad_(False)
    cache = output/'fid_reference.npz'
    reference_count = 128 if args.smoke else len(raw)
    if cache.exists():
        with np.load(cache) as saved:
            reference = (saved['mean'], saved['cov'], int(saved['count']))
            if reference[2] != reference_count:
                raise ValueError('Wrong cached FID reference count')
    else:
        reference = reference_stats(raw[:reference_count], inception, args, device)
        if rank == 0:
            np.savez(cache, mean=reference[0], cov=reference[1], count=reference[2],
                dataset_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    dist.barrier()
    if resume_rng is not None:
        restore_rng(resume_rng, device)
    elif not args.smoke:
        preview(ema, output, 0, args, device)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    overflows = 0
    consecutive_overflows = 0
    target_updates = 3 if args.smoke else args.steps
    while update < target_updates:
        attempts += 1
        ids = rank_indices(len(raw),global_batch,world,rank,update+1,args.seed+42)
        clean = raw[ids].to(device, non_blocking=True).float()/127.5-1
        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.zeros((), device=device)
        for micro, images in enumerate(clean.split(args.microbatch)):
            context = ddp.no_sync() if micro+1 < args.accumulation else nullcontext()
            with context:
                flip = torch.rand(len(images),device=device) < .5
                images = torch.where(flip[:,None,None,None],images.flip(-1),images)
                noise = torch.randn_like(images)
                t = skewed_times(len(images),device)
                x, velocity = flow_target(images,noise,t)
                with torch.autocast('cuda',dtype=torch.float16,enabled=args.amp):
                    prediction = ddp(x,t)
                loss = (prediction.float()-velocity).square().mean()/args.accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite FM loss')
                batch_loss += loss.detach()
                scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            overflows += 1
            consecutive_overflows += 1
            if consecutive_overflows > 20:
                raise RuntimeError('Repeated FP16 overflow')
            continue
        consecutive_overflows = 0
        if not torch.isfinite(grad):
            raise RuntimeError('Nonfinite unscaled gradient')
        update += 1
        update_ema(model,ema,update,args.ema)
        dist.all_reduce(batch_loss); batch_loss /= world
        if rank == 0 and (update%100 == 0 or args.smoke):
            row = dict(status='running',update=update,loss=float(batch_loss),grad=float(grad),
                lr=args.lr,seconds=time.monotonic()-start,world_size=world,
                peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9,
                amp_scale=scaler.get_scale(),overflow_attempts=overflows)
            write_json(output/'status.json',row)
            with (output/'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if args.smoke:
            continue
        if update%1000 == 0 or update==args.steps:
            checkpoint(output,model,ema,optimizer,scaler,update,attempts,best,config,device)
        if update%args.preview_every == 0:
            preview(ema,output,update,args,device)
        if update%args.eval_every == 0 or update==args.steps:
            if rank==0:write_json(output/'status.json',dict(status='evaluating',update=update))
            row = evaluate(ema,inception,reference,output,update,args,device)
            if row['fid'] < best:
                best = row['fid']
                if rank==0:
                    # Atomic latest checkpoint already contains this update's full recovery state.
                    destination=output/'best.pt.tmp'
                    destination.unlink(missing_ok=True)
                    os.link(output/'latest.pt',destination)
                    destination.replace(output/'best.pt')
                    write_json(output/'best.json',row)
            if update==5000 or update==args.steps:
                audit_steps = args.sampler_audit_steps if update==args.steps else args.sampler_audit_steps[:1]
                for intervals in audit_steps:
                    evaluate(ema,inception,reference,output,update,args,device,intervals)
            dist.barrier()
    if args.smoke:
        # Full parameter equality verifies actual gradient synchronization across all ranks.
        flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        master = flat.clone(); dist.broadcast(master,src=0)
        max_difference = (flat-master).abs().max(); dist.all_reduce(max_difference,op=dist.ReduceOp.MAX)
        if float(max_difference)>1e-6:raise RuntimeError('DDP parameters diverged')
        del flat,master
        preview(ema,output,update,args,device)
        smoke_args=copy.copy(args)
        smoke_args.fid_samples=128
        smoke_fid=evaluate(ema,inception,reference,output,update,smoke_args,device)
        checkpoint(output,model,ema,optimizer,scaler,update,attempts,best,config,device)
        # Validate serialized training, EMA, optimizer, and rank RNG state on every rank.
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['update']==update and len(saved['rank_rng'])==world
        assert len(saved['optimizer']['state'])==len(list(model.parameters()))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(),saved['model'][key],rtol=0,atol=0)
        del saved
        write_json(output/f'smoke_rank{rank}.json',dict(**hardware,passed=True,
            optimizer_updates=update,global_batch=global_batch,microbatch=args.microbatch,
            accumulation=args.accumulation,parameter_max_difference=float(max_difference),
            peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9,
            seconds=time.monotonic()-start,finite_sampler_and_inception=True,
            checkpoint_roundtrip=True,amp_overflows=overflows,
            full_fid_pipeline_passed=True,smoke_fid_not_quality_metric=smoke_fid['fid']))
        dist.barrier()
        if rank==0:
            rows=[json.loads((output/f'smoke_rank{i}.json').read_text()) for i in range(world)]
            write_json(output/'smoke.json',dict(passed=True,ranks=rows))
    if rank==0:write_json(output/'status.json',dict(status='complete',update=update))
    dist.barrier(); dist.destroy_process_group()


if __name__=='__main__':
    main()
