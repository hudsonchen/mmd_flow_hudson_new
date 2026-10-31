"""Hold the replacement job until a complete current checkpoint is verified."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

STUDY=Path('/SAN/intelsys/imagenet_mmd_flow/mmd_flow_hudson_new/results/imagenet64/jit_b4_xpred_4h100_20261029')
SOURCE=Path(__file__).resolve().parents[2]
OLD_JOB='7659554'
TARGET=4000


def record(**values):
    path=STUDY/'checkpoint_change_control.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    old.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(old,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(values),flush=True)


def main():
    sys.path.insert(0,str(SOURCE))
    spec=importlib.util.spec_from_file_location('resume_runner',SOURCE/'scripts/cs_cluster/run_imagenet64_jit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=STUDY/'checkpoint_change_control.json'
    if path.exists():raise RuntimeError('Switch already recorded; inspect before retrying')
    config=json.loads((STUDY/'training/config.json').read_text())
    assert not module.validate_resume_config(config,{**config,'checkpoint_every':10000},allow_checkpoint_change=True)
    job=subprocess.check_output(['qsub','-terse','-h','-o',str(STUDY/'checkpoint10k_scheduler.log'),
        '-v',f'STUDY={STUDY},SOURCE_DIR={SOURCE}',
        str(SOURCE/'scripts/cs_cluster/continue_checkpoint10k.sh')],cwd=SOURCE,text=True).strip()
    if not job.isdigit():raise RuntimeError(f'Unexpected job id: {job}')
    record(status='waiting_for_checkpoint',prior_job_id=OLD_JOB,replacement_job_id=job,
           target_checkpoint=TARGET,checkpoint_every=10000)
    last_identity=None
    saved_update=0
    latest=STUDY/'training/latest.pt'
    while True:
        stat=latest.stat();identity=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
        if identity!=last_identity:
            saved=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
            saved_update=saved['update']
            assert len(saved['rank_rng'])==4 and saved['optimizer']['state']
            assert not saved['lineage_has_flow_matching']
            module.validate_resume_config(saved['config'],{**config,'checkpoint_every':10000},allow_checkpoint_change=True)
            del saved
            last_identity=identity
            record(latest_verified_checkpoint=saved_update,checkpoint_bytes=stat.st_size)
        if saved_update>=TARGET and (STUDY/f'training/preview_{saved_update:07d}.json').exists():break
        if subprocess.run(['qstat','-j',OLD_JOB],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError('Original job ended before the planned checkpoint boundary')
        time.sleep(5)
    record(status='stopping_original',resume_update=saved_update)
    subprocess.run(['qdel',OLD_JOB],check=True)
    for _ in range(30):
        if subprocess.run(['qstat','-j',OLD_JOB],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
        time.sleep(2)
    else:raise RuntimeError('Original job still present; replacement remains held')
    subprocess.run(['qrls',job],check=True)
    record(status='replacement_released',resume_update=saved_update)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        record(status='failed',error=repr(error))
        raise
