"""Run the Penn reconstruction and tuned no-RMS recipe on four local H100s."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def commands(repo, study):
    common = [sys.executable, '-u', str(repo / 'main.py'), '--dataset', 'cifar10',
              '--data-root', os.environ['DATA_ROOT'], '--target-size', '0',
              '--particles', '64', '--learning-rate', '0.0002', '--optimizer', 'Adam',
              '--adam-beta2', '0.999', '--seed', '42', '--architecture', 'unet',
              '--base-channels', '128', '--num-res-blocks', '4', '--latent-dim', '3072',
              '--objective', 'softmax-riesz', '--threads', '8', '--device', 'cuda',
              '--data-parallel-gpus', '4', '--fid-samples', '10000',
              '--fid-batch-size', '64', '--trajectory-particles', '8']
    return {
        'reconstruction': common + ['--training-stage', 'reconstruction',
            '--corruption-level', '0.8', '--epochs', '80', '--target-batch-size', '64',
            '--ema-decay', '0.9999', '--save-every', '20', '--checkpoint-every', '1',
            '--output', str(study / 'reconstruction')],
        'posttraining': common + ['--steps', '1000000', '--target-batch-size', '256',
            '--train-particles', '128', '--ema-decay', '0.9999', '--random-flip',
            '--max-grad-norm', '1', '--warmup-steps', '1000', '--lr-schedule', 'cosine',
            '--min-learning-rate', '0.0001', '--multiscale-loss', '--coarse-sizes', '8', '16',
            '--patch-sizes', '7', '14', '--crops-per-size', '4',
            '--global-loss-weight', '2', '--coarse-loss-weight', '1', '--patch-loss-weight', '1',
            '--no-field-rms-normalization', '--no-normalize-per-temperature',
            '--save-every', '2500', '--checkpoint-every', '1000',
            '--keep-checkpoint-every', '25000', '--init-checkpoint',
            str(study / 'reconstruction/checkpoint_latest.pt'),
            '--output', str(study / 'posttraining')],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    study = args.study.resolve()
    stages = commands(repo, study)
    if args.dry_run:
        print(json.dumps(stages, indent=2))
        return
    study.mkdir(parents=True, exist_ok=True)
    # An advisory lock prevents two allocations from resuming the same run.
    import fcntl
    lock = (study / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*repo.glob('*.py'), *Path(__file__).parent.glob('*priming*'),
                        Path(__file__).parent / 'smoke_cifar_4gpu.py']
              if p.is_file()}
    manifest = dict(source=str(repo), commit=commit, source_sha256=hashes,
                    reference_commit='7a93ac4467cad2ad2b65e8018462c5c1585046d8',
                    job_id=os.environ.get('JOB_ID'), host=os.uname().nodename,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    commands=stages, gpu_count=4, reconstruction_epochs=80,
                    posttraining_updates=1000000, field_rms_normalization='none')
    write_json(study / 'deployment.json', manifest)

    def status(state, stage=None, **extra):
        value = dict(status=state, stage=stage, job_id=os.environ.get('JOB_ID'),
                     updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        write_json(study / 'job_status.json', value)
        if stage:
            write_json(study / stage / 'job_status.json', value)

    stage = None
    try:
        status('preflight')
        subprocess.run([sys.executable, '-u', str(repo / 'scripts/cs_cluster/smoke_cifar_4gpu.py'),
                        str(study / 'preflight.json')], check=True)
        import torch
        for stage, command in stages.items():
            output = study / stage
            output.mkdir(exist_ok=True)
            checkpoint = output / 'checkpoint_latest.pt'
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
                complete = saved['iteration'] == (80 if stage == 'reconstruction' else 1000000)
                del saved
                if complete:
                    status('complete', stage)
                    continue
                command = command + ['--resume', str(checkpoint)]
            status('running', stage)
            print(shlex.join(command), flush=True)
            with (output / 'run.log').open('a', buffering=1) as log:
                log.write(shlex.join(command) + '\n')
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            expected = 62560 if stage == 'reconstruction' else 1000000
            assert saved['update_count'] == expected
            assert saved['config']['data_parallel_gpus'] == 4
            if stage == 'posttraining':
                assert saved['config']['softmax_field_normalization'] == 'none'
                assert saved['config']['priming_initialization']['source_updates'] == 62560
            del saved
            fid = json.loads((output / 'fid_metrics.json').read_text())
            assert fid[-1]['iteration'] == (80 if stage == 'reconstruction' else 1000000)
            status('complete', stage, updates=expected)
        status('complete', posttraining_updates=1000000)
    except BaseException as error:
        status('failed', stage, error=repr(error))
        raise


if __name__ == '__main__':
    main()
