"""Four-GPU direct-velocity CIFAR Riesz, one global interacting particle batch."""
import argparse
from contextlib import nullcontext
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from cifar_flow_matching import DDPMFlow, FeatureMoments
from cifar_riesz_velocity import global_particle_loss, training_time, sample_velocity
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
            lineage_has_flow_matching=False, objective='coupled'))
    dist.barrier()


@torch.inference_mode()
def preview(ema, output, update, args, device):
    if dist.get_rank() == 0:
        from torchvision.utils import save_image
        rng = torch.Generator(device=device).manual_seed(8765432)
        noise = torch.randn(32, 3, 32, 32, generator=rng, device=device)
        raw = torch.cat([sample_velocity(ema, z, args.sample_steps, args.amp)
                         for z in noise.split(args.sample_batch)])
        save_image(((raw+1)/2).clamp(0, 1), output/f'preview_{update:07d}.png', nrow=8)
        write_json(output/f'preview_{update:07d}.json', dict(update=update, count=len(raw),
            solver='euler', grid='uniform', steps=args.sample_steps,
            nfe=args.sample_steps, seed=8765432,
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
    intervals = intervals or args.sample_steps
    count = args.fid_samples // world + (rank < args.fid_samples % world)
    # CPU RNG creates reproducible noise independent of CUDA kernels and batch size.
    noise = torch.randn(count, 3, 32, 32,
                        generator=torch.Generator().manual_seed(100000 + args.seed + rank))
    moments = FeatureMoments()
    outside = 0
    preview_images = []
    started = time.monotonic()
    for i, z in enumerate(noise.split(args.sample_batch)):
        raw = sample_velocity(ema, z.to(device), intervals, args.amp)
        outside += int((raw.abs() > 1).sum())
        pixels = ((raw+1)/2).clamp(0, 1)
        for batch in pixels.split(args.fid_batch):
            moments.add(inception(batch)[0].flatten(1).cpu().numpy())
        if rank == 0 and len(preview_images)*args.sample_batch < 100:
            preview_images.append(pixels.cpu())
        if rank == 0 and i % 10 == 0:
            write_json(output/'evaluation_status.json', dict(update=update, solver='euler',
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
            solver='euler', grid='uniform', steps=intervals, nfe=intervals,
            raw_pixel_outside_fraction=int(tally)/(stats[2]*3072),
            evaluation_seconds=time.monotonic()-started,
            preprocessing='unquantized float [0,1], same Inception as Riesz and DDPM')
        suffix = '' if intervals == args.sample_steps else f'_euler{intervals}'
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
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch-per-rank', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--time-sampling', choices=['uniform','skewed'], default='skewed')
    p.add_argument('--horizon', type=float, default=.01)
    p.add_argument('--reference-time-power', type=float, default=1.)
    p.add_argument('--distance-powers', type=float, nargs='+', default=[.8,1.,1.5])
    p.add_argument('--tau', type=float, nargs='+', default=[.2,.05,.02])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--sample-steps', type=int, default=100)
    p.add_argument('--sample-batch', type=int, default=32)
    p.add_argument('--fid-batch', type=int, default=32)
    p.add_argument('--fid-samples', type=int, default=10000)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--preview-every', type=int, default=1000)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if min(args.steps,args.batch_per_rank,args.sample_steps,args.threads,args.sample_batch,
           args.fid_batch,args.eval_every,args.preview_every) < 1 or args.fid_samples < 4:
        p.error('Counts must be positive and FID requires at least four samples')
    if not 0 < args.horizon <= 1 or not math.isfinite(args.reference_time_power) or args.reference_time_power < 0:
        p.error('Invalid horizon or time weighting')
    if any(not math.isfinite(v) or not 0 < v <= 2 for v in args.distance_powers) or len(set(args.distance_powers)) != len(args.distance_powers):
        p.error('Kernel powers must be distinct and in (0,2]')
    if any(not math.isfinite(v) or v <= 0 for v in args.tau):p.error('Invalid temperatures')
    if args.eval_every % 1000:p.error('FID cadence must align with recovery checkpoints')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp, init_scale=1024)
    torch.manual_seed(args.seed + 1009 * rank + 1)
    torch.cuda.manual_seed(args.seed + 1009 * rank + 1)
    global_batch = args.batch_per_rank * world
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(parameters=parameters, world_size=world, global_batch=global_batch,
        objective='coupled', parameterization='velocity', architecture='ddpm-unet', dataset='cifar10', initialization='random',
        time_sampling=args.time_sampling, shared_time_across_all_ranks=True,
        time_distribution='sigmoid(1.2-1.2*N(0,1)), clipped to [0.0001,0.999]' if args.time_sampling=='skewed' else 'uniform [0,0.999)',
        path='x_t=(1-t)*noise+t*image; prediction=x_t+h*v_theta; target=x_(t+h)',
        time_embedding='999*(1-t)', optimizer='Adam beta=(0.9,0.999); 500-update warmup, cosine to 0.1*lr',
        ema='decay .99 through update 1000, then .999',
        sampling='100 uniform Euler steps by default; one network call per interval',
        normalization='none', affinity='row-column', reference_coupling='paired', include_self_negative=True,
        feature_space='pixels', prime_steps=0, fork=None,
        training_rule='global paired pixel Riesz; sum power-kernel fields; no velocity MSE',
        interaction_batch=global_batch, ddp_backward_scale=world,
        fid='FID-10K versus all 50000 training images; unquantized pixels',
        lineage_has_flow_matching=False)
    update, attempts, best = 0, 0, float('inf')
    resume_rng = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state.get('lineage_has_flow_matching', True):
            raise ValueError('Riesz resume must not load FM weights')
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
    if '2080 Ti' not in hardware['gpu']:
        raise RuntimeError(f"Expected RTX 2080 Ti, got {hardware['gpu']}")
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
        # Same random-with-replacement data recipe as the existing Riesz runs.
        ids = torch.randint(len(raw), (args.batch_per_rank,))
        clean = raw[ids].to(device, non_blocking=True).float()/127.5-1
        flip = torch.rand(len(clean), device=device) < .5
        clean = torch.where(flip[:,None,None,None], clean.flip(-1), clean)
        noise = torch.randn_like(clean)
        shared_time = training_time(device, args.time_sampling) if rank == 0 else torch.zeros((), device=device)
        if args.smoke and rank == 0:
            shared_time.fill_([.001, .77, .999][update])
        dist.broadcast(shared_time, src=0)
        t = float(shared_time)
        h = min(args.horizon, 1-t)
        source = (1-t)*noise + t*clean
        target = (1-t-h)*noise + (t+h)*clean
        lr = args.lr*min((update+1)/500,1)*(.1+.9*.5*(1+math.cos(math.pi*update/args.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            velocity = ddp(source, shared_time)
        prediction = source + h*velocity.float()
        predictions = [torch.empty_like(prediction) for _ in range(world)]
        targets = [torch.empty_like(target) for _ in range(world)]
        dist.all_gather(predictions, prediction.detach())
        dist.all_gather(targets, target)
        loss, force_info = global_particle_loss(prediction, predictions, targets, rank, t, args.horizon,
            args.distance_powers, args.tau, args.reference_time_power)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite global Riesz loss')
        batch_loss = loss.detach()
        # Each global loss only differentiates this rank's queries. Undo DDP averaging.
        scaler.scale(loss*world).backward()
        del predictions, targets, prediction, velocity, source, target, force_info, loss
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
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(current, .001 if update>1000 else .01)
        dist.all_reduce(batch_loss); batch_loss /= world
        if rank == 0 and (update%100 == 0 or args.smoke):
            row = dict(status='running',update=update,loss=float(batch_loss),grad=float(grad),
                lr=lr,time=t,seconds=time.monotonic()-start,world_size=world,
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
            optimizer_updates=update,global_batch=global_batch,batch_per_rank=args.batch_per_rank,
            parameter_max_difference=float(max_difference),
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
