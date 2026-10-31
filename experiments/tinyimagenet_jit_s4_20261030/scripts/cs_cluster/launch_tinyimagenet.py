"""One allocated H100: verify frozen source/data, preflight, then train."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--objective',choices=['riesz','flow_matching'],required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo))
    from run_storage import new_run_path
    study = new_run_path(args.study.resolve())
    if not str(study).startswith('/SAN/intelsys/imagenet_mmd_flow/'):
        raise ValueError('This cluster experiment requires explicit SAN storage')
    study.mkdir(parents=True,exist_ok=True)
    lock = (study/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:
        raise RuntimeError(f'Expected one allocated GPU, got {devices}')
    source = json.loads((repo/'source_manifest.json').read_text())
    for name,digest in source['sha256'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen source changed: {name}')
    data_identity = {}
    for split,count in [('train',100000),('val',10000)]:
        identity = json.loads((args.data_root/f'{split}_identity.json').read_text())
        if identity['count']!=count or len(identity['classes'])!=200:
            raise ValueError('Wrong Tiny ImageNet split')
        for suffix,key in [('index.json','index_sha256'),('rgb64.npy','cache_sha256')]:
            with (args.data_root/f'{split}_{suffix}').open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=identity[key]:
                    raise ValueError(f'Dataset hash mismatch: {split}/{suffix}')
        data_identity[split] = identity
    if data_identity['train']['classes'] != data_identity['val']['classes']:
        raise ValueError('Train/validation class mappings differ')
    common = [sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=1',
        str(repo/'scripts/cs_cluster/jit_rank_entrypoint.py'),
        '--data-root',str(args.data_root),'--objective',args.objective,
        '--epochs','600','--global-batch','1024','--lr','0.0002','--warmup-epochs','5',
        '--time-mean','-0.8','--time-std','0.8','--cfg-dropout','0.1',
        '--cfg','2.9','--cfg-min-time','0.1','--cfg-max-time','1',
        '--solver','heun','--sample-steps','50','--required-gpu','H100',
        '--expected-world-size','1','--patch-size','4','--prediction','x',
        '--denominator-floor','0.05','--gradient-checkpointing','--seed','0',
        '--horizon','0.01','--ema-decay','0.9999','--eval-every','5000',
        '--preview-every','1000','--fid-samples','10000','--workers','2']
    commands = dict(preflight=common+['--output',str(study/'preflight'),'--smoke','--sample-steps','2'],
                    training=common+['--output',str(study/'training')])
    manifest = dict(source=str(repo),source_manifest=source,job_id=os.environ.get('JOB_ID'),
        host=os.uname().nodename,allocated_device_ids=devices,commands=commands,
        dataset=data_identity,objective=args.objective,initialization='random; seed 0; no pretrained weights',
        architecture='custom JiT-S/4: depth 12, width 384, 6 heads, patch 4, RGB64, 200 classes',
        loss_source_unchanged=args.objective=='riesz',
        training_budget='600 dataset-equivalent epochs = 58594 updates at batch 1024',
        evaluation='FID-10K versus held-out 10000-image validation split every 5000 updates',
        sampling='50 Heun intervals, final Euler; CFG 2.9 on [0.1,1]; EMA 0.9999')
    write_json(study/'deployment.json',manifest)
    stage = None
    def status(state,**extra):
        write_json(study/'job_status.json',dict(status=state,stage=stage,
            job_id=os.environ.get('JOB_ID'),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    try:
        for stage,command in commands.items():
            output = study/stage
            output.mkdir(exist_ok=True)
            if stage=='preflight' and (output/'smoke.json').exists():
                if not json.loads((output/'smoke.json').read_text())['passed']:
                    raise RuntimeError('Existing preflight failed')
                continue
            if stage=='training' and (output/'latest.pt').exists():
                command = command+['--resume',str(output/'latest.pt')]
            status('running')
            print(json.dumps(dict(stage=stage,command=command)),flush=True)
            with (output/'run.log').open('a',buffering=1) as log:
                subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
            if stage=='preflight':
                smoke = json.loads((output/'smoke.json').read_text())
                if not smoke['passed'] or len(smoke['ranks'])!=1:
                    raise RuntimeError('H100 preflight failed')
                status('preflight_passed')
        status('complete')
    except BaseException as error:
        status('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
