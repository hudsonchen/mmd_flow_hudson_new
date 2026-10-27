"""Train multiscale VGG DiT from a preserved checkpoint with explicit batch sizes."""
import argparse
import copy
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
from run_storage import new_run_path
from run_cifar_pair import set_option, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-study', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--restart-from-priming', action='store_true',
                        help='Start post-training at zero with fresh Adam/RNG from completed reconstruction')
    parser.add_argument('--name', required=True)
    parser.add_argument('--queue', default='longgpu02')
    parser.add_argument('--gpu-memory', default='32G')
    parser.add_argument('--train-particles', type=int, default=128)
    parser.add_argument('--target-batch-size', type=int, default=256)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.train_particles < 2 or args.target_batch_size < 1:
        parser.error('Require at least two generated particles and one real image')
    previous = args.source_study.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error('A preserved checkpoint is required')
    if checkpoint.name == 'checkpoint_latest.pt' and not args.restart_from_priming:
        parser.error('Use an immutable numbered checkpoint, not a live recovery checkpoint')
    if args.restart_from_priming:
        import torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if saved['arguments']['training_stage'] != 'reconstruction':
            parser.error('Fresh primed post-training requires a reconstruction checkpoint')
        duration = saved['arguments']['epochs'] if saved['arguments']['epochs'] is not None else saved['arguments']['steps']
        if saved['iteration'] != duration:
            parser.error('Reconstruction must be complete')
        del saved
    old_manifest = json.loads((previous / 'manifest.json').read_text())
    study = new_run_path(Path('cifar') / args.name).resolve()
    study.mkdir(parents=True, exist_ok=False)
    source, output, cache = study / 'source', study / 'posttraining', study / 'perceptual_cache'
    source.mkdir()
    output.mkdir()
    # Copy the initialization so the continuation remains independently recoverable.
    initial = study / 'initialization'
    initial.mkdir()
    shutil.copy2(checkpoint, initial / checkpoint.name)
    checkpoint = initial / checkpoint.name
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    shutil.copytree(previous / 'perceptual_cache', cache)
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
    command = list(next(s for s in old_manifest['stages'] if s['stage'] == 'posttraining')['command'])
    command[2] = str(source / 'main.py')
    for flag in ('--resume', '--fork-checkpoint'):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    init_flag = '--init-checkpoint' if args.restart_from_priming else '--fork-checkpoint'
    for flag, value in [('--output', output), (init_flag, checkpoint),
                        ('--train-particles', args.train_particles), ('--target-batch-size', args.target_batch_size),
                        ('--perceptual-pool-size', 8), ('--lpips-cache', cache),
                        ('--save-every', 5000), ('--checkpoint-every', 1000),
                        ('--keep-checkpoint-every', 5000)]:
        set_option(command, flag, value)
    if args.restart_from_priming:
        set_option(command, '--schedule-start-update', 0)
        set_option(command, '--warmup-steps', 1000)
    if '--vgg-multiscale' not in command:
        command.append('--vgg-multiscale')
    python = command[0]
    manifest = dict(study=str(study), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reference_manifest=str(previous / 'manifest.json'),
        comparison=(f'Multiscale VGG feature fields with {args.train_particles}-generated/'
                    f'{args.target_batch_size}-real interaction batch'),
        initialization=dict(checkpoint=str(checkpoint), original_checkpoint=str(args.checkpoint.resolve()),
                            sha256=checkpoint_sha, inherited_online=True, inherited_ema=True,
                            inherited_optimizer=not args.restart_from_priming,
                            inherited_rng=not args.restart_from_priming,
                            posttraining_start_update=0 if args.restart_from_priming else None,
                            learning_rate_schedule='restart_at_zero' if args.restart_from_priming else 'unchanged'),
        feature_recipe=dict(layers=5, branches_per_layer=['spatial', 'patch_stats', 'global_stats'],
                            maximum_spatial_side=8, weighting='equal_15_branches',
                            scales='independent_batch_distance', temperatures=[.2, .05, .02],
                            repulsion_weight=1., field_rms_normalization=False),
        generated_batch=args.train_particles, real_batch=args.target_batch_size, fid_every=5000,
        smoke_command=[python, '-u', str(source / 'scripts/lsf/smoke_dit.py'), str(ROOT / 'data'),
                       str(study / 'smoke.json'), '--checkpoint', str(checkpoint),
                       '--softmax-feature-space', 'vgg', '--vgg-multiscale',
                       '--perceptual-pool-size', '8', '--lpips-cache', str(cache),
                       '--train-particles', str(args.train_particles),
                       '--target-batch-size', str(args.target_batch_size)],
        stages=[dict(stage='posttraining', output=str(output), command=command)])
    if args.restart_from_priming:
        manifest['smoke_command'].append('--fresh-optimizer')
    write_json(study / 'manifest.json', manifest)
    recovery = copy.deepcopy(manifest)
    recovery_command = recovery['stages'][0]['command']
    if '--fork-checkpoint' in recovery_command:
        index = recovery_command.index('--fork-checkpoint')
        del recovery_command[index:index + 2]
    set_option(recovery_command, '--resume', output / 'checkpoint_latest.pt')
    recovery['purpose'] = 'Resume this continuation after a durable checkpoint exists'
    write_json(study / 'manifest_resume.json', recovery)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', DRIFT_COMPILE='0')
    with (study / 'validation_tests.log').open('w') as stream:
        subprocess.run([python, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=source,
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if args.submit:
        submission = ['bsub', '-q', args.queue, '-J', f'cifar_dit_vgg_multi_b{args.train_particles}', '-n', '4',
                      '-R', 'span[hosts=1] rusage[mem=32GB]',
                      '-gpu', f'num=1:mode=exclusive_process:gmem={args.gpu_memory}', '-W', '48:00',
                      '-oo', str(study / 'scheduler.log'), python, '-u',
                      str(source / 'scripts/lsf/run_compact_dit.py'), '--manifest', str(study / 'manifest.json')]
        result = subprocess.run(submission, text=True, capture_output=True)
        match = re.search(r'Job <(\d+)>', result.stdout)
        deployment = dict(status='submitted' if result.returncode == 0 else 'failed',
                          command=submission, job_id=match.group(1) if match else None,
                          stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        write_json(study / 'deployment.json', deployment)
        print(json.dumps(deployment), flush=True)
        result.check_returncode()
    print(study, flush=True)


if __name__ == '__main__':
    main()
