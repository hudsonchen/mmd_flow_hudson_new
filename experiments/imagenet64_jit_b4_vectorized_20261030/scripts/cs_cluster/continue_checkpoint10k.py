"""Resume the same ImageNet64 run with only its checkpoint interval changed."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    study=Path(os.environ['STUDY'])
    source=Path(__file__).resolve().parents[2]
    lock=(study/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=4:
        raise RuntimeError('Expected four allocated H100 devices')
    manifest=json.loads((source/'checkpoint_change_source.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Source mismatch: {name}')
    old=json.loads((study/'deployment.json').read_text())
    command=old['commands']['training'].copy()
    command[0]=sys.executable
    old_entry=str(Path(old['source'])/'scripts/cs_cluster/jit_rank_entrypoint.py')
    command[command.index(old_entry)]=str(source/'scripts/cs_cluster/jit_rank_entrypoint.py')
    command+=['--resume',str(study/'training/latest.pt'),'--checkpoint-every','10000','--allow-checkpoint-change']
    record=dict(job_id=os.environ['JOB_ID'],source=str(source),source_manifest=manifest,
        prior_job_id=old['job_id'],command=command,checkpoint_every=10000,
        exceptions='initial, final, and improving FID checkpoints',
        preserves='model, EMA, optimizer, scaler, per-rank RNG, update and data-attempt count',
        host=os.uname().nodename,allocated_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    (study/'checkpoint_change_deployment.json').write_text(json.dumps(record,indent=2)+'\n')
    def status(state,**extra):
        p=study/'job_status.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(status=state,stage='training',job_id=os.environ['JOB_ID'],
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
        tmp.replace(p)
    status('running')
    try:
        with (study/'training/run.log').open('a',buffering=1) as log:
            log.write('\nCheckpoint interval continuation: '+json.dumps(record)+'\n')
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete')
    except BaseException as error:
        status('failed',error=repr(error))
        raise


if __name__=='__main__':main()
