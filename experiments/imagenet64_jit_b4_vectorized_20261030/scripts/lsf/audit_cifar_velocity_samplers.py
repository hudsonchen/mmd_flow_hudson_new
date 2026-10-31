"""Compare samplers with fixed EMA weights, shared latents, and training-reference FID."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

VARIANTS = [('euler100', 'euler', 'uniform', 100),
            ('heun50', 'heun', 'uniform', 50),
            ('heun50_edm', 'heun', 'edm', 50),
            ('heun100', 'heun', 'uniform', 100)]


def write_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def edm_time_grid(intervals, device='cpu'):
    """EDM rho=7 spacing mapped to flow time; exact t=0 and t=1 endpoints.

    Unlike the upstream example's initial t=1/81, include the Gaussian endpoint.
    `intervals` always means actual Heun steps; NFE is exactly 2*intervals.
    """
    import torch
    if intervals < 3:
        raise ValueError('At least three integration intervals are required')
    sigma = torch.linspace(80. ** (1/7), .002 ** (1/7), intervals - 1,
                           dtype=torch.float64, device=device).pow(7)
    return torch.cat([sigma.new_zeros(1), 1 / (1 + sigma), sigma.new_ones(1)])


def sample(model, noise, solver, grid_kind, steps, amp=True):
    import torch
    with torch.inference_mode():
        x = noise.float().clone()
        grid = (edm_time_grid(steps, x.device) if grid_kind == 'edm'
                else torch.arange(steps + 1, device=x.device, dtype=torch.float64)/steps)
        for left, right in zip(grid[:-1], grid[1:]):
            dt = (right-left).float()
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                first = model(x, left.float()).float()
                if solver == 'heun':
                    second = model(x + dt*first, right.float()).float()
            x = x + dt*(first if solver == 'euler' else .5*(first+second))
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite sampler output')
        return x


def self_test():
    import math
    import torch
    from cifar_riesz_velocity import sample_velocity
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.calls = 0; self.times = []
        def forward(self, x, t):
            self.calls += 1; self.times.append(float(t)); return x
    z = torch.ones(2, 3, 2, 2)
    errors = {}
    for name, solver, grid, steps in VARIANTS:
        model = Field().eval()
        value = sample(model, z, solver, grid, steps, amp=False)
        assert model.calls == steps*(1 if solver == 'euler' else 2)
        assert min(model.times) == 0
        assert max(model.times) == (1 if solver == 'heun' else float(torch.tensor(.99)))
        errors[name] = float((value-math.e).abs().max())
        if name == 'euler100':
            torch.testing.assert_close(value, sample_velocity(Field(), z, 100), atol=2e-6, rtol=2e-6)
    assert errors['heun50'] < errors['euler100']/10
    assert errors['heun100'] < errors['heun50']/3
    print(json.dumps(dict(self_test='passed', exponential_ode_errors=errors)))


def submit(args):
    from run_storage import new_run_path
    output = new_run_path(Path('cifar')/args.name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Atomic replacement by the trainer leaves this hard-linked inode unchanged.
    os.link(args.run/'training/latest.pt', output/'checkpoint.pt')
    shutil.copy2(args.run/'training/fid_reference.npz', output/'fid_reference.npz')
    shutil.copy2(Path(__file__), output/'audit.py')
    command = [sys.executable, '-u', str(output/'audit.py'), '--run', str(args.run),
               '--output', str(output), '--count', str(args.count), '--batch', str(args.batch)]
    write_json(output/'launch.json', dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=str(args.run), checkpoint_origin=str(args.run/'training/latest.pt'),
        checkpoint_preservation='hard link; trainer atomically replaces latest.pt',
        source_directory=str(args.run/'source'), command=command, variants=VARIANTS,
        generated_count=args.count, reference_count=50000, seed=100000,
        note='Exploratory small-sample FID; do not compare absolute scores to FID-10K'))
    cmd = ['bsub', '-q', 'dbeigpu', '-m', 'dbei-ai1', '-J', 'cifar_velocity_sampler_audit',
           '-n', '4', '-R', 'span[hosts=1] rusage[mem=8GB]',
           '-gpu', 'num=1:mode=shared:gmem=16G:mig=7/7', '-W', '00:45',
           '-env', 'all,OMP_NUM_THREADS=4,MKL_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4',
           '-oo', str(output/'scheduler.log'), *command]
    result = subprocess.run(cmd, capture_output=True, text=True)
    record = dict(command=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        record['job_id'] = re.search(r'<(\d+)>', result.stdout).group(1)
    write_json(output/'submission.json', record)
    print(json.dumps(dict(output=str(output), **record)))
    result.check_returncode()


def evaluate(args):
    import numpy as np
    import torch
    from torchvision.utils import save_image
    from velocity_model import DDPMFlow
    from fid import FeatureMoments
    from fid import load_inception, frechet_distance
    from PIL import Image, ImageDraw
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    output = args.output
    cp_path = output/'checkpoint.pt'
    cp = torch.load(cp_path, map_location='cpu', mmap=True, weights_only=False)
    assert cp['config']['parameterization'] == 'velocity'
    assert not cp['lineage_has_flow_matching']
    model = DDPMFlow().cuda().eval().requires_grad_(False)
    model.load_state_dict(cp['ema'])
    update = cp['update']
    with cp_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(output/'checkpoint_info.json', dict(update=update, sha256=digest,
        weights='ema', config=cp['config'], gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda))
    del cp
    inception = load_inception('cuda').requires_grad_(False)
    with np.load(output/'fid_reference.npz') as reference:
        ref_mean, ref_cov = reference['mean'], reference['cov']
        assert int(reference['count']) == 50000
    noise = torch.randn(args.count, 3, 32, 32, generator=torch.Generator().manual_seed(100000))
    rows, previews = [], []
    for name, solver, grid, steps in VARIANTS:
        start = time.monotonic()
        moments = FeatureMoments()
        outside, pixels_all = 0, []
        generation_seconds = 0
        for offset in range(0, args.count, args.batch):
            torch.cuda.synchronize(); batch_start = time.monotonic()
            raw = sample(model, noise[offset:offset+args.batch].cuda(), solver, grid, steps)
            torch.cuda.synchronize(); generation_seconds += time.monotonic()-batch_start
            outside += int((raw.abs()>1).sum())
            pixels = ((raw+1)/2).clamp(0,1)
            pixels_all.append(pixels.cpu())
            with torch.inference_mode():
                features = inception(pixels)[0].flatten(1).cpu().numpy()
            moments.add(features)
            if offset == 0:
                save_image(pixels[:32], output/f'{name}_preview.png', nrow=8)
            write_json(output/'status.json', dict(status='running', sampler=name,
                generated=min(offset+args.batch,args.count), target=args.count,
                checkpoint_update=update, elapsed_seconds=time.monotonic()-start))
        stats = moments.statistics()
        fid = frechet_distance(ref_mean, ref_cov, stats[0], stats[1])
        images = torch.cat(pixels_all)
        np.save(output/f'{name}_samples.npy', images.numpy())
        save_image(images[:64], output/f'{name}_samples.png', nrow=8)
        # First eight identical latent indices per row, enlarged without interpolation.
        preview = (images[:8].permute(0,2,3,1).numpy()*255).round().astype('uint8')
        previews.append((name, preview))
        canvas = Image.new('RGB',(8*96, len(previews)*120),'white')
        draw = ImageDraw.Draw(canvas)
        for index,(label,tiles) in enumerate(previews):
            draw.text((4,index*120+3),label,fill='black')
            for j,tile in enumerate(tiles):
                canvas.paste(Image.fromarray(tile).resize((96,96),Image.Resampling.NEAREST),(j*96,index*120+22))
        canvas.save(output/'comparison.png')
        row = dict(sampler=name, solver=solver, grid=grid, intervals=steps,
            nfe=steps*(1 if solver=='euler' else 2), fid=fid, generated_count=args.count,
            reference_count=50000, reference_split='train', checkpoint_update=update,
            checkpoint_sha256=digest, noise_seed=100000, amp=True,
            preprocessing='unquantized float [0,1]', raw_pixel_outside_fraction=outside/(args.count*3072),
            generation_seconds=generation_seconds, total_seconds=time.monotonic()-start)
        rows.append(row)
        write_json(output/f'{name}_metrics.json',row)
        write_json(output/'results.json',rows)
        print(json.dumps(row),flush=True)
    write_json(output/'status.json',dict(status='complete', checkpoint_update=update, results=rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--count', type=int, default=2048)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--name', default='velocity_riesz_sampler_audit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--submit', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.count < 64 or args.batch < 8:
        p.error('Need at least 64 samples and batch size 8')
    sys.path.insert(0, str(args.run/'source'))
    # Archived runs predate the module split; use their original frozen helpers.
    if not (args.run/'source/velocity_model.py').exists():
        import cifar_flow_matching
        sys.modules['velocity_model'] = cifar_flow_matching
        import fid
        fid.FeatureMoments = cifar_flow_matching.FeatureMoments
    if args.self_test:
        self_test()
    elif args.submit:
        submit(args)
    else:
        if not args.output: p.error('--output required for evaluation')
        evaluate(args)


if __name__ == '__main__':
    main()
