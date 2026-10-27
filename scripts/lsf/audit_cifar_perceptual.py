"""Independent held-out FID for saved one-pass VGG Riesz and time-DiT FM models."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
os.environ['DRIFT_COMPILE'] = '0'
from run_storage import new_run_path


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def launch(args):
    import torch
    output = new_run_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = new_run_path('cifar')
    studies = [
        ('multi128', 'priming_dit_vgg_multiscale_b128_r256_fresh_seed42_20261026', 'posttraining/checkpoint_step_025000.pt'),
        ('multi256', 'priming_dit_vgg_multiscale_b256_r1024_fresh_seed42_20261026', 'posttraining/checkpoint_step_025000.pt'),
        ('fm', 'time_dit_fm10_pixels_ema999_fresh_seed42_20261026', 'training/checkpoint_latest.pt'),
    ]
    # The FM source supplies its exact time-dependent architecture and sampler.
    # Both one-pass studies must have the identical backbone and FID implementation.
    archive = root / studies[-1][1] / 'source'
    for _, name, _ in studies:
        for file in ['backbone.py', 'fid.py', '_fid_inception.py', 'data.py']:
            if sha256(root/name/'source'/file) != sha256(archive/file):
                raise ValueError(f'Archived implementations differ: {name}/{file}')
    hashes = {}
    for name in ['backbone.py', 'flow_path.py', 'loss.py', 'fid.py', '_fid_inception.py', 'data.py',
                 'run_storage.py', 'assets/pt_inception-2015-12-05-6726825d.pth', 'assets/inception.sha256']:
        source = archive/name
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = sha256(target)
    for source in list(ROOT.glob('*LICENSE*')) + [ROOT/'assets/SCORE_SDE_LICENSE', Path(__file__)]:
        relative = source.relative_to(ROOT)
        target = output/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256(target)
    models = []
    for key, name, file in studies:
        source = root/name/file
        target = output/f'{key}_checkpoint.pt'
        # Copy from a single open inode; the running trainer atomically replaces latest.
        with source.open('rb') as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
        checkpoint = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        c = checkpoint['config']
        if key != 'fm' and (c.get('training_stage') != 'distributional' or c.get('objective') != 'softmax-riesz'):
            raise ValueError('Expected a Riesz distributional checkpoint')
        models.append(dict(key=key, original=str(source), checkpoint=str(target), sha256=sha256(target),
            update=checkpoint['update_count'], config=c))
        del checkpoint
    command = [str(ROOT/'.venv/bin/python'), '-u', str(output/'source/scripts/lsf/audit_cifar_perceptual.py'),
               '--output', str(output), '--worker', '--seed', str(args.seed), '--batch-size', str(args.batch_size)]
    manifest = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), models=models,
        hashes=hashes, noise_seed=args.seed, generated_count=10000, reference_split='test', reference_count=10000,
        data_root=str(new_run_path('datasets')), queue='longgpu', host='longgpu01', command=command,
        note='Independent validation of archived selection; no model updates or changes to original runs. FM snapshot update differs from VGG selection updates; no equal-compute claim.')
    write_json(output/'manifest.json', manifest)
    result = subprocess.run(['bsub', '-q', 'longgpu', '-m', 'longgpu01', '-J', 'cifar_perceptual_audit',
        '-n', '2', '-R', 'span[hosts=1] rusage[mem=6GB]', '-gpu', 'num=1:mode=shared', '-W', '02:00',
        '-oo', str(output/'scheduler.log'), *command], text=True, capture_output=True, check=True)
    write_json(output/'submission.json', dict(stdout=result.stdout, stderr=result.stderr,
        job_id=re.search(r'<(\d+)>', result.stdout).group(1)))
    print(output, result.stdout, sep='\n')


def worker(args):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from backbone import CIFAR10DiTParticleGenerator
    from flow_path import TimeConditionedDiT, FlowSampler
    from data import download_cifar10, load_cifar10_split
    from fid import CIFAR10FID
    from torchvision.utils import save_image
    output = args.output
    manifest = json.loads((output/'manifest.json').read_text())
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = True
    for name, expected in manifest['hashes'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Frozen audit source hash mismatch: {name}')
    archive = download_cifar10(Path(manifest['data_root']))
    real = load_cifar10_split(archive, False)[0]
    evaluator = CIFAR10FID(real, device='cuda', batch_size=64)
    result = dict(noise_seed=args.seed, generated_count=10000, reference_split='test', reference_count=len(real),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])
    for entry in manifest['models']:
        path = Path(entry['checkpoint'])
        if sha256(path) != entry['sha256']:
            raise ValueError('Audit checkpoint changed')
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        c = checkpoint['config']
        options = {k: c['dit_'+k] for k in ['width', 'depth', 'heads', 'patch_size']}
        model = (TimeConditionedDiT if entry['key']=='fm' else CIFAR10DiTParticleGenerator)(**options).cuda().eval()
        model.load_state_dict(checkpoint['ema_model'])
        sampler = FlowSampler(model, c['flow_steps']) if entry['key']=='fm' else model
        del checkpoint
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
        samples = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, 10000, args.batch_size):
                noise = torch.randn(min(args.batch_size, 10000-start), 3072, generator=generator, device='cuda')
                pixels = sampler(noise)
                if not torch.isfinite(pixels).all() or (pixels<0).any() or (pixels>1).any():
                    raise ValueError('Invalid audit sample pixels')
                samples.append(pixels.cpu())
        samples = torch.cat(samples).reshape(-1, 3, 32, 32)
        stats = evaluator.statistics(samples)
        row = dict(key=entry['key'], update=entry['update'], checkpoint_sha256=entry['sha256'],
            fid=evaluator.score(stats), sampling_nfe=c['flow_steps'] if entry['key']=='fm' else 1,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        np.save(output/f"{entry['key']}_samples.npy", (samples.numpy()*255).round().astype(np.uint8))
        save_image(samples[:100], str(output/f"{entry['key']}_samples.png"), nrow=10)
        result['rows'].append(row)
        write_json(output/'audit.json', result)
        print(json.dumps(row), flush=True)
        del model, sampler, samples
        torch.cuda.empty_cache()
    write_json(output/'status.json', dict(status='complete', **result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int, default=7754321)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.seed < 2**63:
        parser.error('Invalid batch size or seed')
    (worker if args.worker else launch)(args)


if __name__ == '__main__':
    main()
