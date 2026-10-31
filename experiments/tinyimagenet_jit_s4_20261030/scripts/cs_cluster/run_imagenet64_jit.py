"""Matched Tiny ImageNet JiT-S/4 Riesz and conditional-flow-matching training."""
import argparse
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
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from imagenet64_jit import JiTVelocity, sample_time, learning_rate, sample_velocity, sampling_nfe
from imagenet64_data import build_index, ImageNet64, RandomImageBatches
from torch.utils.data import DataLoader, Subset
from training_objectives import training_loss

from fid import FeatureMoments, load_inception, frechet_distance
from imagenet64_fid_reference import load_benchmark_reference
from run_storage import new_run_path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


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


def validate_resume_config(saved, current, allow_evaluation_change=False):
    """Permit only explicitly requested sampler/output changes, never loss changes."""
    old, new = dict(saved), dict(current)
    old.setdefault('solver', 'euler')
    old.setdefault('required_gpu', 'RTX 2080 Ti')
    for key in ('fid_reference', 'fid_reference_sha256'):
        old.setdefault(key, None)
        new.setdefault(key, None)
    evaluation_keys = {'solver', 'sample_steps', 'sampling', 'fid_reference', 'fid_reference_sha256', 'fid'}
    permitted = {'output'} | evaluation_keys if allow_evaluation_change else set()
    differences = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if differences - permitted:
        raise ValueError(f'Resume configuration differs: {sorted(differences - permitted)}')
    return bool(differences & evaluation_keys)


def checkpoint(output, model, ema, optimizer, scaler, update, attempts, best, config, device):
    local = rng_state(device)
    states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, states, dst=0)
    if dist.get_rank() == 0:
        save_torch(output/'latest.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), update=update,
            attempts=attempts, best_fid=best, rank_rng=states, config=config,
            lineage_has_flow_matching=config['objective']=='flow_matching', objective=config['objective']))
    dist.barrier()


@torch.inference_mode()
def preview(ema, output, update, args, device, kind="ema"):
    if dist.get_rank() == 0:
        from torchvision.utils import save_image
        rng = torch.Generator(device=device).manual_seed(8765432)
        noise = torch.randn(32, 3, 64, 64, generator=rng, device=device)
        labels = torch.arange(len(noise),device=device)%200
        raw = torch.cat([sample_velocity(ema,z,args.sample_steps,args.amp,args.solver,
            labels[i:i+len(z)],args.cfg,(args.cfg_min_time,args.cfg_max_time))
            for i,z in zip(range(0,len(noise),args.sample_batch),noise.split(args.sample_batch))])
        save_image(((raw+1)/2).clamp(0, 1), output/f'preview_{kind}_{update:07d}.png', nrow=8)
        write_json(output/f'preview_{kind}_{update:07d}.json', dict(update=update, weights=kind, count=len(raw),
            solver=args.solver, grid='uniform', steps=args.sample_steps,cfg=args.cfg,
            cfg_interval=[args.cfg_min_time,args.cfg_max_time],
            nfe=sampling_nfe(args.sample_steps,args.solver,args.prediction), seed=8765432,
            raw_pixel_variance=float(raw.var(0).mean()),
            raw_pixel_outside_fraction=float((raw.abs()>1).float().mean())))
    dist.barrier()


@torch.inference_mode()
def reference_stats(raw, inception, args, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    moments = FeatureMoments()
    shard = Subset(raw,range(rank,len(raw),world))
    loader = DataLoader(shard,batch_size=args.fid_batch,num_workers=args.workers,pin_memory=True)
    for index, images in enumerate(loader):
        features = inception(images.to(device).float()/255)[0].flatten(1)
        moments.add(features.cpu().numpy())
        if rank == 0 and index % 100 == 0:
            write_json(args.output/'reference_status.json',dict(rank0_processed=moments.count,rank0_total=len(shard)))
    return reduce_moments(moments, device)


@torch.inference_mode()
def evaluate(ema, inception, reference, output, update, args, device, intervals=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    intervals = intervals or args.sample_steps
    count = args.fid_samples // world + (rank < args.fid_samples % world)
    # CPU RNG creates reproducible noise independent of CUDA kernels and batch size.
    noise = torch.randn(count, 3, 64, 64,
                        generator=torch.Generator().manual_seed(100000 + args.seed + rank))
    moments = FeatureMoments()
    outside = 0
    preview_images = []
    started = time.monotonic()
    for i, z in enumerate(noise.split(args.sample_batch)):
        global_ids = torch.arange(rank,args.fid_samples,world,device=device)
        labels = global_ids[i*args.sample_batch:i*args.sample_batch+len(z)]%200
        raw = sample_velocity(ema,z.to(device),intervals,args.amp,args.solver,labels,
            args.cfg,(args.cfg_min_time,args.cfg_max_time))
        outside += int((raw.abs() > 1).sum())
        pixels = ((raw+1)/2).clamp(0, 1)
        for batch in pixels.split(args.fid_batch):
            moments.add(inception(batch)[0].flatten(1).cpu().numpy())
        if rank == 0 and len(preview_images)*args.sample_batch < 100:
            preview_images.append(pixels.cpu())
        if rank == 0 and i % 10 == 0:
            write_json(output/'evaluation_status.json', dict(update=update, solver=args.solver,
                steps=intervals, rank0_generated=min((i+1)*args.sample_batch, count),
                rank0_target=count, elapsed_seconds=time.monotonic()-started))
    stats = reduce_moments(moments, device)
    tally = torch.tensor(outside, device=device, dtype=torch.long)
    dist.all_reduce(tally)
    row = None
    if rank == 0:
        from torchvision.utils import save_image
        row = dict(update=update, fid=frechet_distance(*reference[:2], *stats[:2]),
            generated_count=stats[2], reference_count=reference[2],
            reference_split='validation',
            reference_path=str(args.fid_reference) if args.fid_reference else str(output/'fid_reference.npz'),
            reference_sha256=args.fid_reference_sha256,
            cfg=args.cfg,cfg_interval=[args.cfg_min_time,args.cfg_max_time],class_labels='balanced modulo 200',
            solver=args.solver, grid='uniform', steps=intervals, nfe=sampling_nfe(intervals,args.solver,args.prediction),
            raw_pixel_outside_fraction=int(tally)/(stats[2]*12288),
            evaluation_seconds=time.monotonic()-started,
            preprocessing='unquantized float [0,1], same Inception as Riesz and DDPM')
        suffix = '' if intervals == args.sample_steps else f'_{args.solver}{intervals}'
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
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--objective', choices=['riesz','flow_matching'], required=True)
    p.add_argument('--patch-size', type=int, choices=[4], default=4)
    p.add_argument('--prediction', choices=['x'], default='x')
    p.add_argument('--denominator-floor', type=float, default=.05)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--expected-world-size', type=int, choices=[1], default=1)
    p.add_argument('--global-batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup-epochs', type=float, default=5.)
    p.add_argument('--cfg-dropout', type=float, default=.1)
    p.add_argument('--cfg', type=float, default=2.9)
    p.add_argument('--cfg-min-time', type=float, default=.1)
    p.add_argument('--cfg-max-time', type=float, default=1.)
    p.add_argument('--ema-decay', type=float, default=.9999)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--time-mean', type=float, default=-.8)
    p.add_argument('--time-std', type=float, default=.8)
    p.add_argument('--horizon', type=float, default=.01)
    p.add_argument('--reference-time-power', type=float, default=1.)
    p.add_argument('--distance-powers', type=float, nargs='+', default=[.8,1.,1.5])
    p.add_argument('--tau', type=float, nargs='+', default=[.2,.05,.02])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--solver', choices=['euler','heun'], default='heun')
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--required-gpu', default='H100')
    p.add_argument('--sample-batch', type=int, default=32)
    p.add_argument('--fid-batch', type=int, default=32)
    p.add_argument('--fid-samples', type=int, default=10000)
    p.add_argument('--fid-reference', type=Path)
    p.add_argument('--fid-reference-sha256')
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--preview-every', type=int, default=1000)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--allow-evaluation-change', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if bool(args.fid_reference) != bool(args.fid_reference_sha256):
        p.error('--fid-reference and --fid-reference-sha256 must be supplied together')
    benchmark_reference = None
    benchmark_metadata = None
    if args.fid_reference:
        benchmark_reference, benchmark_metadata = load_benchmark_reference(
            args.fid_reference, args.fid_reference_sha256)
    args.steps = math.ceil(args.epochs*100000/args.global_batch)
    args.warmup_steps = args.warmup_epochs*100000/args.global_batch
    if min(args.steps,args.global_batch,args.sample_steps,args.threads,args.sample_batch,
           args.fid_batch,args.eval_every,args.preview_every) < 1 or args.fid_samples < 4:
        p.error('Counts must be positive and FID requires at least four samples')
    if not 0 <= args.cfg_dropout <= 1 or not 0 <= args.cfg_min_time <= args.cfg_max_time <= 1 or args.cfg < 1:
        p.error('Invalid CFG settings')
    if not 0 < args.horizon <= 1 or not math.isfinite(args.reference_time_power) or args.reference_time_power < 0:
        p.error('Invalid horizon or time weighting')
    if args.warmup_steps < 0 or args.workers < 0 or args.time_std <= 0 or not 0 < args.ema_decay < 1:
        p.error('Invalid JiT optimizer or time sampling configuration')
    if any(not math.isfinite(v) or not 0 < v <= 2 for v in args.distance_powers) or len(set(args.distance_powers)) != len(args.distance_powers):
        p.error('Kernel powers must be distinct and in (0,2]')
    if any(not math.isfinite(v) or v <= 0 for v in args.tau):p.error('Invalid temperatures')
    if args.eval_every % 1000:p.error('FID cadence must align with recovery checkpoints')
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != args.expected_world_size or world < 1 or args.global_batch < world:
        raise ValueError('Worker count must match the requested GPU allocation and batch')
    batch_sizes = [args.global_batch//world+int(i<args.global_batch%world) for i in range(world)]
    local_batch = batch_sizes[rank]
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
    model = JiTVelocity(args.patch_size, args.prediction, args.denominator_floor,
                       args.gradient_checkpointing).to(device)
    parameters = sum(v.numel() for v in model.parameters())
    if rank == 0:
        initial_hash = hashlib.sha256()
        for key,value in model.state_dict().items():
            initial_hash.update(key.encode())
            initial_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        write_json(output/'initialization.json', dict(seed=args.seed, parameters=parameters,
            state_sha256=initial_hash.hexdigest(), random_initialization=True))
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              gradient_as_bucket_view=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    peak_lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, betas=(.9,.95), weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    torch.manual_seed(args.seed + 1009 * rank + 1)
    torch.cuda.manual_seed(args.seed + 1009 * rank + 1)
    global_batch = args.global_batch
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('resume','allow_evaluation_change')}
    config.update(parameters=parameters, world_size=world, global_batch=global_batch, rank_batch_sizes=batch_sizes,
        epoch_definition='100000 sampled training images per dataset-equivalent epoch, class batches with replacement',
        objective=args.objective, parameterization=args.prediction, architecture=f'JiT-S/{args.patch_size}', dataset='tinyimagenet200', initialization='random',
        velocity_conversion='(x_hat-x_t)/max(1-t,denominator_floor)' if args.prediction=='x' else 'direct network output',
        time_sampling='jit_logit_normal', shared_time_across_all_ranks=True,
        time_distribution=f'sigmoid({args.time_mean}+{args.time_std}*N(0,1)), clipped to [0.0001,0.999]',
        path='x_t=(1-t)*noise+t*image; Riesz: prediction=x_t+h*v_theta, target=x_(t+h); FM: target velocity=image-noise',
        time_embedding='raw t as official JiT', optimizer='Adam beta=(0.9,0.95), eps=1e-8, weight_decay=0; linear warmup then constant',
        peak_learning_rate=peak_lr, ema=args.ema_decay, precision='bfloat16 network; float32 Riesz field',
        image_size=64, image_tokens=(64//args.patch_size)**2, in_context_tokens=32,
        backbone_source='LTH14/JiT@cbc743a2ada5e9762697da2c83f8c4f8379e8c17',
        conditioning='class-matched batches for both objectives; CFG dropout per batch; dropped batches sample all classes',
        sampling=f'{args.sample_steps} uniform {args.solver} intervals; {sampling_nfe(args.sample_steps,args.solver,args.prediction)} velocity evaluations; final Euler for x-prediction',
        normalization='none', affinity='row-column', reference_coupling='paired', include_self_negative=True,
        feature_space='pixels', prime_steps=0, fork=None,
        training_rule=('global paired pixel Riesz; unchanged power-kernel fields; no velocity MSE' if args.objective=='riesz' else 'conditional velocity MSE; no Riesz term'),
        interaction_batch=global_batch, ddp_backward_scale=world,
        fid='FID-10K versus 10000 held-out Tiny ImageNet validation images; unquantized generated pixels',
        lineage_has_flow_matching=args.objective=='flow_matching', hidden_size=384, depth=12, heads=6, classes=200)
    if args.fid_reference:
        config['fid'] = 'FID-10K versus saved ImageNet64 benchmark statistics; unquantized generated pixels'
    update, attempts, best = 0, 0, float('inf')
    resume_rng = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state.get('objective') != args.objective:
            raise ValueError('Cannot resume across training objectives')
        if args.objective=='riesz' and state.get('lineage_has_flow_matching', True):
            raise ValueError('Riesz resume must not load FM weights')
        evaluation_changed = validate_resume_config(state['config'], config, args.allow_evaluation_change)
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        update, attempts, best = state['update'], state['attempts'], state['best_fid']
        resume_rng = state['rank_rng'][rank]
        if evaluation_changed:
            best = float('inf')
        if rank == 0:
            write_json(output/'continuation.json', dict(checkpoint=str(args.resume), update=update,
                prior_config=state['config'], current_config=config, evaluation_changed=evaluation_changed,
                best_fid_reset=evaluation_changed, optimizer_and_rng_restored=True))
        del state
        if not evaluation_changed and (output/'best.json').exists():
            best = min(best, json.loads((output/'best.json').read_text())['fid'])
    if rank == 0:
        write_json(output/'config.json', config)
        if benchmark_metadata is not None:
            write_json(output/'fid_reference.json', benchmark_metadata)
        write_json(output/'status.json', dict(status='initializing', update=update))
    hardware = dict(rank=rank, gpu=torch.cuda.get_device_name(device),
        allocated_gpu=os.environ.get('MMD_ALLOCATED_GPU'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        capability=torch.cuda.get_device_capability(device), torch=torch.__version__,
        cuda=torch.version.cuda, parameters=parameters)
    if args.required_gpu not in hardware['gpu']:
        raise RuntimeError(f"Expected {args.required_gpu}, got {hardware['gpu']}")
    write_json(output/f'hardware_rank{rank}.json', hardware)
    # Index once; workers decode original images directly into uint8 64px crops.
    if rank == 0:
        if not (output/'dataset_index.json').exists():
            build_index(args.data_root, output)
        print('Tiny ImageNet training index ready', flush=True)
    dist.barrier()
    dataset = ImageNet64(args.data_root, output/'dataset_index.json')
    identity = json.loads((output/'dataset_identity.json').read_text())
    if hashlib.sha256((output/'dataset_index.json').read_bytes()).hexdigest() != identity['index_sha256']:
        raise RuntimeError('ImageNet index hash mismatch')
    if Path(identity['root']) != args.data_root.resolve():
        raise RuntimeError('ImageNet data root changed')
    validation = ImageNet64(args.data_root, split='val')
    validation_identity = json.loads((args.data_root/'val_identity.json').read_text())
    if rank == 0:
        write_json(output/'validation_identity.json', validation_identity)
    inception = None
    reference = benchmark_reference
    # FID initialization is deferred to the first evaluation, so training starts promptly.
    def ensure_fid():
        nonlocal inception, reference
        if inception is None:
            inception = load_inception(device).requires_grad_(False)
        if reference is not None:
            return
        cache = output/'fid_reference.npz'
        reference_count = 128 if args.smoke else len(validation)
        if cache.exists():
            with np.load(cache) as saved:
                if int(saved['count']) != reference_count or str(saved['index_sha256']) != validation_identity['index_sha256']:
                    raise ValueError('Wrong cached FID reference identity')
                reference = (saved['mean'],saved['cov'],int(saved['count']))
        else:
            reference = reference_stats(Subset(validation, range(reference_count)), inception, args, device)
            if rank == 0:
                np.savez(cache,mean=reference[0],cov=reference[1],count=reference[2],
                         index_sha256=validation_identity['index_sha256'],preprocessing=validation_identity['preprocessing'])
        dist.barrier()
    batches = RandomImageBatches(dataset,local_batch,args.seed+7,rank,args.cfg_dropout)
    loader = DataLoader(batches,batch_size=None,sampler=range(attempts,args.steps*22),
        num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,
        generator=torch.Generator().manual_seed(args.seed+rank+311))
    batch_iterator = iter(loader)
    if resume_rng is not None:
        restore_rng(resume_rng, device)
    elif not args.smoke:
        preview(ema, output, 0, args, device)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    training_seconds = 0.
    overflows = 0
    consecutive_overflows = 0
    time_bins = [dict(count=0, grad_sum=0., grad_max=0., clipped=0, overflow_attempts=0)
                 for _ in range(5)]
    target_updates = 3 if args.smoke else args.steps
    while update < target_updates:
        iteration_started = time.monotonic()
        attempts += 1
        # Same random-with-replacement data recipe as the existing Riesz runs.
        batch = next(batch_iterator)
        clean = batch['images'].to(device, non_blocking=True).float()/127.5-1
        labels = torch.full((len(clean),),int(batch['label']),device=device,dtype=torch.long)
        flip = torch.rand(len(clean), device=device) < .5
        clean = torch.where(flip[:,None,None,None], clean.flip(-1), clean)
        noise = torch.randn_like(clean)
        shared_time = sample_time(device, args.time_mean, args.time_std) if rank == 0 else torch.zeros((), device=device)
        if args.smoke and rank == 0:
            shared_time.fill_([.001, .77, .999][update])
        dist.broadcast(shared_time, src=0)
        t = float(shared_time)
        time_bin = time_bins[min(int(t*5),4)]
        h = min(args.horizon, 1-t)
        source = (1-t)*noise + t*clean
        target = (1-t-h)*noise + (t+h)*clean
        lr = learning_rate(update, peak_lr, args.warmup_steps)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
            velocity = ddp(source, shared_time, labels)
        loss = training_loss(velocity, source, clean, noise, t, args.horizon,
            args.objective, args.distance_powers, args.tau, args.reference_time_power)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        batch_loss = loss.detach()
        scaler.scale(loss).backward()
        del velocity, source, target, loss
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            overflows += 1
            time_bin['overflow_attempts'] += 1
            consecutive_overflows += 1
            if consecutive_overflows > 20:
                raise RuntimeError('Repeated FP16 overflow')
            continue
        consecutive_overflows = 0
        if not torch.isfinite(grad):
            raise RuntimeError('Nonfinite unscaled gradient')
        torch.cuda.synchronize(device)
        training_seconds += time.monotonic()-iteration_started
        update += 1
        grad_value = float(grad)
        time_bin['count'] += 1
        time_bin['grad_sum'] += grad_value
        time_bin['grad_max'] = max(time_bin['grad_max'], grad_value)
        time_bin['clipped'] += int(grad_value > 1.)
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(current, 1-args.ema_decay)
        dist.all_reduce(batch_loss); batch_loss /= world
        if rank == 0 and (update%100 == 0 or update<=3 or args.smoke):
            row = dict(status='running',update=update,objective=args.objective,
                training_seconds=training_seconds,loss=float(batch_loss),grad=float(grad),
                lr=lr,time=t,label=int(batch['label']),epoch=update*global_batch/len(dataset),seconds=time.monotonic()-start,world_size=world,
                peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9,
                amp_scale=scaler.get_scale(),overflow_attempts=overflows)
            write_json(output/'status.json',row)
            with (output/'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if args.smoke:
            continue
        if update==3 or update%1000 == 0 or update==args.steps:
            if rank == 0:
                write_json(output/'time_gradient_diagnostics.json', dict(update=update,
                    since_resume=True, edges=[0.,.2,.4,.6,.8,1.], bins=time_bins,
                    definition='Pre-clip gradient norms after DDP reduction; overflow attempts separate'))
            checkpoint(output,model,ema,optimizer,scaler,update,attempts,best,config,device)
        if update%args.preview_every == 0:
            preview(ema,output,update,args,device)
            preview(model,output,update,args,device,kind='current')
        if update%args.eval_every == 0 or update==args.steps:
            if rank==0:write_json(output/'status.json',dict(status='evaluating',update=update))
            ensure_fid()
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
        ensure_fid()
        smoke_fid=evaluate(ema,inception,reference,output,update,smoke_args,device)
        checkpoint(output,model,ema,optimizer,scaler,update,attempts,best,config,device)
        # Validate serialized training, EMA, optimizer, and rank RNG state on every rank.
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['update']==update and len(saved['rank_rng'])==world
        assert len(saved['optimizer']['state'])==len([p for p in model.parameters() if p.requires_grad])
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(),saved['model'][key],rtol=0,atol=0)
        del saved
        write_json(output/f'smoke_rank{rank}.json',dict(**hardware,passed=True,
            optimizer_updates=update,global_batch=global_batch,batch_per_rank=local_batch,
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
