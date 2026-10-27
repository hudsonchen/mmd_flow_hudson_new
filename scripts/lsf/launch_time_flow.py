"""Freeze, validate, and launch time-conditioned Riesz transport or flow matching."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_storage import new_run_path
from run_cifar_pair import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--objective', choices=['riesz', 'flow-matching'], default='riesz')
    parser.add_argument('--riesz-time-mode', choices=['discrete', 'continuous'], default='discrete')
    parser.add_argument('--transport-horizon', type=float, default=.1)
    parser.add_argument('--flow-steps', type=int, default=10,
                        help='Sampling steps; does not set the continuous Riesz training interval')
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--softmax-feature-space', choices=['pixels', 'vgg'], default='vgg')
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--rollout-fraction', type=float, default=0.)
    parser.add_argument('--rollout-warmup-steps', type=int, default=5000)
    parser.add_argument('--endpoint-weight', type=float, default=0.)
    parser.add_argument('--fork-checkpoint', type=Path)
    parser.add_argument('--queue', default='dbeigpu')
    parser.add_argument('--gpu-memory', type=int, default=80)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.objective == 'flow-matching' and (args.softmax_feature_space != 'pixels' or args.rollout_fraction):
        parser.error('Flow matching requires pixels and no rollout mixing')
    if args.flow_steps < 1 or not math.isfinite(args.transport_horizon) or not 0 < args.transport_horizon <= 1:
        parser.error('Invalid sampling step count or transport horizon')
    if args.riesz_time_mode == 'continuous' and (
            args.objective != 'riesz' or args.rollout_fraction or args.endpoint_weight):
        parser.error('Continuous-time Riesz requires no rollout mixing or endpoint loss')
    if not math.isfinite(args.endpoint_weight) or args.endpoint_weight < 0:
        parser.error('Endpoint weight must be finite and nonnegative')
    if args.endpoint_weight and (args.objective != 'riesz' or args.softmax_feature_space != 'pixels'):
        parser.error('Endpoint Riesz requires the pixel-space Riesz objective')
    if args.softmax_feature_space == 'vgg' and (args.source_cache is None or not args.source_cache.is_dir()):
        parser.error('Requires an existing VGG cache')
    if (not 0 <= args.ema_decay < 1 or not 0 <= args.rollout_fraction <= 1
            or args.rollout_warmup_steps < 0 or args.gpu_memory < 1):
        parser.error('Invalid EMA, rollout schedule or GPU memory')
    if args.fork_checkpoint is not None and not args.fork_checkpoint.is_file():
        parser.error('Fork checkpoint does not exist')
    study = new_run_path(Path('cifar') / args.name).resolve()
    study.mkdir(parents=True, exist_ok=False)
    source, output, cache = study / 'source', study / 'training', study / 'perceptual_cache'
    source.mkdir()
    output.mkdir()
    if args.softmax_feature_space == 'vgg':
        shutil.copytree(args.source_cache, cache)
    parent = None
    parent_sha256 = None
    if args.fork_checkpoint is not None:
        parent = study / 'parent_checkpoint.pt'
        # Copy an opened atomic checkpoint; never train against a moving latest path.
        shutil.copyfile(args.fork_checkpoint, parent)
        with parent.open('rb') as stream:
            parent_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    (source / 'assets').symlink_to(ROOT / 'assets', target_is_directory=True)
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*LICENSE*'))
    files += list((ROOT / 'scripts/lsf').glob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
    files += list((ROOT / 'tests/fixtures').glob('*.json'))
    hashes = {}
    for original in files:
        relative = original.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write_json(source / 'sha256.json', hashes)
    python = str(ROOT / '.venv/bin/python')
    command = [python, '-u', str(source / 'train_time_flow.py'), '--output', str(output),
               '--data-root', str(ROOT / 'data'), '--flow-steps', str(args.flow_steps), '--steps', '250000',
               '--save-every', '5000',
               '--objective', args.objective,
               '--riesz-time-mode', args.riesz_time_mode, '--transport-horizon', str(args.transport_horizon),
               '--softmax-feature-space', args.softmax_feature_space,
               '--ema-decay', str(args.ema_decay), '--rollout-fraction', str(args.rollout_fraction),
               '--rollout-warmup-steps', str(args.rollout_warmup_steps),
               '--endpoint-weight', str(args.endpoint_weight),
               '--repulsion-weight', '1.0', '--seed', '42']
    smoke = [python, '-u', str(source / 'scripts/lsf/smoke_time_flow.py'),
             '--data-root', str(ROOT / 'data'), '--output', str(study / 'smoke.json'),
             '--softmax-feature-space', args.softmax_feature_space,
             '--objective', args.objective,
             '--riesz-time-mode', args.riesz_time_mode, '--transport-horizon', str(args.transport_horizon),
             '--flow-steps', str(args.flow_steps),
             '--endpoint-weight', str(args.endpoint_weight),
             '--ema-decay', str(args.ema_decay), '--rollout-fraction', str(args.rollout_fraction)]
    if args.softmax_feature_space == 'vgg':
        command += ['--lpips-cache', str(cache)]
        smoke += ['--lpips-cache', str(cache)]
    if parent is not None:
        command += ['--fork-checkpoint', str(parent)]
        smoke += ['--fork-checkpoint', str(parent)]
    manifest = dict(study=str(study), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        comparison=(f'Conditional affine flow matching with velocity MSE and {args.flow_steps}-step Euler sampling'
                    if args.objective == 'flow-matching' else
                    'Continuous-time Riesz marginal matching with one model call per training update'
                    if args.riesz_time_mode == 'continuous' else
                    'Local Riesz transport plus a separate full-rollout endpoint Riesz loss'
                    if args.endpoint_weight else
                    f'{args.flow_steps}-step time-conditioned Riesz-softmax transport along fixed affine marginals'),
        objective=args.objective,
        initialization='fork_live_weights_reset_optimizer_and_ema' if parent else 'fresh_time_conditioned_DiT',
        parent_checkpoint_original=str(args.fork_checkpoint.resolve()) if parent else None,
        parent_checkpoint_sha256=parent_sha256,
        feature_space=args.softmax_feature_space, ema_decay=args.ema_decay,
        rollout_fraction=args.rollout_fraction, rollout_warmup_steps=args.rollout_warmup_steps,
        riesz_time_mode=args.riesz_time_mode, transport_horizon=args.transport_horizon,
        sampling_steps=args.flow_steps, training_time_grid=None if args.riesz_time_mode == 'continuous'
        or args.objective == 'flow-matching' else [k/args.flow_steps for k in range(args.flow_steps+1)],
        endpoint_weight=args.endpoint_weight, endpoint_backprop_steps=args.flow_steps if args.endpoint_weight else 0,
        paired_velocity_supervision=args.objective == 'flow-matching', smoke_command=smoke,
        stages=[dict(stage='training', output=str(output), command=command)])
    write_json(study / 'manifest.json', manifest)
    test_code = (
        "import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite(); "
        "[suite.addTests(loader.discover('tests', pattern=p)) for p in "
        "['test_time_flow.py', 'test_dit.py', 'test_perceptual.py']]; "
        "result=unittest.TextTestRunner(verbosity=2).run(suite); "
        "raise SystemExit(not result.wasSuccessful())")
    with (study / 'validation_tests.log').open('w') as stream:
        subprocess.run([python, '-c', test_code], cwd=source, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    if args.submit:
        job_name = (f'cifar_dit_fm{args.flow_steps}' if args.objective == 'flow-matching'
                    else 'cifar_dit_riesz_continuous' if args.riesz_time_mode == 'continuous'
                    else f'cifar_dit_riesz{args.flow_steps}_endpoint' if args.endpoint_weight
                    else f'cifar_dit_riesz{args.flow_steps}_{args.softmax_feature_space}')
        submission = ['bsub', '-q', args.queue, '-J', job_name, '-n', '4',
                      '-R', 'span[hosts=1] rusage[mem=32GB]',
                      '-gpu', f'num=1:mode=exclusive_process:gmem={args.gpu_memory}G', '-W', '48:00',
                      '-oo', str(study / 'scheduler.log'), python, '-u',
                      str(source / 'scripts/lsf/run_compact_dit.py'),
                      '--manifest', str(study / 'manifest.json')]
        result = subprocess.run(submission, text=True, capture_output=True)
        match = re.search(r'Job <(\d+)>', result.stdout)
        deployment = dict(status='submitted' if result.returncode == 0 else 'failed',
            command=submission, manifest=str(study / 'manifest.json'), stdout=result.stdout,
            stderr=result.stderr, returncode=result.returncode, job_id=match.group(1) if match else None)
        write_json(study / 'deployment.json', deployment)
        print(json.dumps(deployment), flush=True)
        result.check_returncode()
    print(study, flush=True)


if __name__ == '__main__':
    main()
