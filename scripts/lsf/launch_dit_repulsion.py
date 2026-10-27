"""Launch a fresh posttraining ablation from a completed DiT priming study."""
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
from run_cifar_pair import set_option, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-study', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--repulsion-weight', type=float, default=.5)
    parser.add_argument('--softmax-feature-space', choices=['pixels', 'vgg'], default='pixels')
    parser.add_argument('--perceptual-pool-size', type=int, default=4)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.repulsion_weight) or args.repulsion_weight < 0:
        parser.error('Repulsion weight must be finite and nonnegative')
    if args.perceptual_pool_size < 1:
        parser.error('Invalid perceptual pool size')
    source_study = args.source_study.resolve()
    source_manifest = json.loads((source_study / 'manifest.json').read_text())
    checkpoint = source_study / 'reconstruction/checkpoint_latest.pt'
    status = json.loads((checkpoint.parent / 'job_status.json').read_text())
    if status['status'] != 'complete' or not checkpoint.is_file():
        raise RuntimeError('Requires a completed reconstruction checkpoint')
    study = new_run_path(Path('cifar') / args.name).resolve()
    study.mkdir(parents=True, exist_ok=False)
    output = study / 'posttraining'
    output.mkdir()
    source = study / 'source'
    source.mkdir()
    (source / 'assets').symlink_to(ROOT / 'assets', target_is_directory=True)
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*LICENSE*'))
    files += list((ROOT / 'scripts/lsf').glob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
    files += list((ROOT / 'tests/fixtures').glob('*.json'))
    checksums = {}
    for original in files:
        relative = original.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        checksums[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write_json(source / 'sha256.json', checksums)
    command = list(next(s for s in source_manifest['stages'] if s['stage'] == 'posttraining')['command'])
    command[2] = str(source / 'main.py')
    for flag in ('--resume', '--fork-checkpoint'):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    set_option(command, '--output', output)
    set_option(command, '--init-checkpoint', checkpoint)
    set_option(command, '--repulsion-weight', args.repulsion_weight)
    set_option(command, '--save-every', 5000)
    set_option(command, '--softmax-feature-space', args.softmax_feature_space)
    set_option(command, '--perceptual-pool-size', args.perceptual_pool_size)
    if args.softmax_feature_space == "vgg":
        if '--multiscale-loss' in command:
            command.remove('--multiscale-loss')
        # Seed the new run's cache from the already verified priming assets.
        previous_cache = source_study / 'perceptual_cache'
        cache = study / 'perceptual_cache'
        shutil.copytree(previous_cache, cache)
        set_option(command, '--lpips-cache', cache)
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = dict(
        study=str(study), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reference_manifest=str(source_study / 'manifest.json'),
        comparison='LPIPS-pretrained DiT; posttraining force and VGG-feature ablation',
        repulsion_weight=args.repulsion_weight, attraction_weight=1.,
        softmax_feature_space=args.softmax_feature_space,
        perceptual_pool_size=args.perceptual_pool_size,
        initialization=dict(checkpoint=str(checkpoint), sha256=checkpoint_sha,
                            optimizer='fresh', inherited_online_and_ema=True),
        smoke_command=[command[0], '-u', str(source / 'scripts/lsf/smoke_dit.py'),
                       str(ROOT / 'data'), str(study / 'smoke.json'),
                       '--repulsion-weight', str(args.repulsion_weight)],
        stages=[dict(stage='posttraining', output=str(output), command=command)])
    if args.softmax_feature_space == "vgg":
        manifest['smoke_command'] += ['--softmax-feature-space', args.softmax_feature_space,
                                     '--perceptual-pool-size', str(args.perceptual_pool_size),
                                     '--lpips-cache', str(cache)]
    write_json(study / 'manifest.json', manifest)
    # Validate the exact frozen code that will run on the worker.
    test_code = (
        "import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite(); "
        "[suite.addTests(loader.discover('tests', pattern=p)) for p in "
        "['test_unnormalized_softmax.py', 'test_losses.py', 'test_priming.py', 'test_dit.py', 'test_perceptual.py']]; "
        "result=unittest.TextTestRunner(verbosity=2).run(suite); "
        "raise SystemExit(not result.wasSuccessful())")
    with (study / 'validation_tests.log').open('w') as stream:
        subprocess.run([command[0], '-c', test_code], cwd=source,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    if args.submit:
        job_name = 'cifar_dit_vgg_flow' if args.softmax_feature_space == 'vgg' else f'cifar_dit_rep{args.repulsion_weight:g}'
        submission = ['bsub', '-q', 'dbeigpu', '-J', job_name, '-n', '4',
                      '-R', 'span[hosts=1] rusage[mem=32GB]',
                      '-gpu', 'num=1:mode=exclusive_process:gmem=80G', '-W', '48:00',
                      '-oo', str(study / 'scheduler.log'), command[0], '-u',
                      str(source / 'scripts/lsf/run_compact_dit.py'),
                      '--manifest', str(study / 'manifest.json')]
        result = subprocess.run(submission, text=True, capture_output=True)
        match = re.search(r'Job <(\d+)>', result.stdout)
        deployment = dict(status='submitted' if result.returncode == 0 else 'failed',
                          command=submission, manifest=str(study / 'manifest.json'),
                          stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                          job_id=match.group(1) if match else None)
        write_json(study / 'deployment.json', deployment)
        print(json.dumps(deployment), flush=True)
        result.check_returncode()
    print(study, flush=True)


if __name__ == '__main__':
    main()
