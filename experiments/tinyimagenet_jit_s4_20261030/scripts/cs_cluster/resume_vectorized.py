"""Resume only the matched Tiny ImageNet Riesz run after the H100 validation gate."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    study = args.study.resolve()
    assert str(study).endswith('/tinyimagenet/jit_s4_pair_20261030/riesz')
    lock = (study/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    manifest = json.loads((source/'source_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(source/name) != expected:
            raise RuntimeError('Source hash mismatch: '+name)
    benchmark = json.loads(args.benchmark.read_text())
    assert benchmark['passed'] and benchmark['stage'] == 'complete'
    assert benchmark['speedup'] > 1.2 and 'H100' in benchmark['gpu']
    for name, expected in benchmark['source_sha256'].items():
        assert digest(source/name) == expected
    original = json.loads((study/'deployment.json').read_text())
    assert original['objective'] == 'riesz'
    # Dataset and sampler are unchanged; retain their original passed preflight.
    assert json.loads((study/'preflight/smoke.json').read_text())['passed']
    config = json.loads((study/'training/config.json').read_text())
    assert config['objective'] == 'riesz' and config['steps'] == 58594
    assert config['global_batch'] == 1024 and not config['lineage_has_flow_matching']
    for split in ['train', 'val']:
        identity = original['dataset'][split]
        root = Path(config['data_root'])
        for suffix, key in [('index.json','index_sha256'),('rgb64.npy','cache_sha256')]:
            assert digest(root/f'{split}_{suffix}') == identity[key]
    command = list(original['commands']['training'])
    old_entry = str(Path(original['source'])/'scripts/cs_cluster/jit_rank_entrypoint.py')
    command[command.index(old_entry)] = str(source/'scripts/cs_cluster/jit_rank_entrypoint.py')
    command += ['--resume', str(args.checkpoint.resolve())]
    # This is an arithmetic implementation change, not a loss/configuration change.
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    deployment = dict(job_id=os.environ['JOB_ID'],source=str(source),source_manifest=manifest,
        prior_deployment_sha256=digest(study/'deployment.json'),command=command,
        checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),
        benchmark=str(args.benchmark),benchmark_sha256=digest(args.benchmark),
        objective='riesz',dataset_and_evaluation_unchanged=True,
        original_preflight=str(study/'preflight/smoke.json'),
        change='Algebraically equivalent vectorized FP32 Riesz field',
        optimizer_ema_rng_resume_required=True)
    write_json(study/'vectorized_deployment.json',deployment)
    def status(state, **extra):
        value = dict(status=state,stage='training',job_id=os.environ['JOB_ID'],
                     updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        write_json(study/'job_status.json',value)
        write_json(study/'vectorized_job_status.json',value)
    try:
        status('running')
        with (study/'training/run.log').open('a', buffering=1) as log:
            log.write('\nVectorized Riesz continuation: '+json.dumps(deployment)+'\n')
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete')
    except BaseException as error:
        status('failed',error=repr(error))
        raise


if __name__ == '__main__':
    main()
