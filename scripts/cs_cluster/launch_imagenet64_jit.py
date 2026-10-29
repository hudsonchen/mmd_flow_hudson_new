"""Validate three allocated H100s, then launch the requested ImageNet64 recipe."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    study = args.study.resolve()
    if not str(study).startswith('/SAN/intelsys/imagenet_mmd_flow/'):
        raise ValueError('This cs-cluster launch requires explicit SAN output storage')
    study.mkdir(parents=True,exist_ok=True)
    lock = (study/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Keep the orchestrator free of CUDA contexts on exclusive-process GPUs.
    devices = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=3 or len(set(devices))!=3:
        raise RuntimeError(f'Expected three distinct allocated devices, got {devices}')
    tracked = subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    hashes = {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in tracked if (repo/p).is_file()}
    common = [sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
        str(repo/'scripts/cs_cluster/jit_rank_entrypoint.py'),
        '--data-root','/SAN/intelsys/imagenet_mmd_flow/ILSVRC/Data/CLS-LOC/train',
        '--epochs','600','--global-batch','1024','--lr','0.0002','--warmup-epochs','5',
        '--time-mean','-0.8','--time-std','0.8','--cfg-dropout','0.1',
        '--cfg','2.9','--cfg-min-time','0.1','--cfg-max-time','1',
        '--solver','heun','--sample-steps','50','--required-gpu','H100']
    commands = dict(preflight=common+['--output',str(study/'preflight'),'--smoke','--sample-steps','2'],
                    training=common+['--output',str(study/'training')])
    manifest = dict(source=str(repo),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),source_sha256=hashes,
        job_id=os.environ.get('JOB_ID'),host=os.uname().nodename,allocated_device_ids=devices,
        nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),commands=commands,
        source_transfer='GitHub branch deployment',dataset='ImageNet training RGB center-cropped to 64',
        pure_pixel_training=True,pretrained_encoder=False,initialization='random',
        optimizer='Adam',betas=[.9,.95],weight_decay=0,global_batch=1024,
        rank_batch_sizes=[342,341,341],epochs=600,warmup_epochs=5,
        epoch_definition='dataset-equivalent image count; class batches sampled with replacement')
    if (study/'deployment.json').exists():
        old=json.loads((study/'deployment.json').read_text())
        if old['source_commit']!=manifest['source_commit']:
            raise RuntimeError('Existing run source differs; use a fresh run')
    write_json(study/'deployment.json',manifest)
    stage=None
    def status(state,**extra):
        write_json(study/'job_status.json',dict(status=state,stage=stage,
            job_id=os.environ.get('JOB_ID'),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    try:
        for stage, command in commands.items():
            output=study/stage
            output.mkdir(exist_ok=True)
            if stage=='preflight' and (output/'smoke.json').exists():
                if not json.loads((output/'smoke.json').read_text())['passed']:
                    raise RuntimeError('Existing preflight did not pass')
                continue
            if stage=='training':
                for name in ['dataset_index.json','dataset_identity.json']:
                    if not (output/name).exists():
                        os.link(study/'preflight'/name,output/name)
                if (output/'latest.pt').exists():
                    command=command+['--resume',str(output/'latest.pt')]
            status('running')
            print(json.dumps(dict(stage=stage,command=command)),flush=True)
            with (output/'run.log').open('a',buffering=1) as log:
                subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
            if stage=='preflight':
                smoke=json.loads((output/'smoke.json').read_text())
                if not smoke['passed'] or len(smoke['ranks'])!=3:
                    raise RuntimeError('Three-H100 validation failed')
                status('preflight_passed')
        status('complete')
    except BaseException as error:
        status('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
