"""Four-GPU validation gate, then resume ImageNet64 with automatic original-code fallback."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    study=Path(os.environ['STUDY']).resolve()
    source=Path(__file__).resolve().parents[2]
    work=source.parent
    assert str(study).endswith('/imagenet64/jit_b4_xpred_4h100_20261029')
    lock=(study/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==4
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    manifest=json.loads((source/'vectorized_source.json').read_text())
    for name,expected in manifest['sha256'].items():
        if digest(source/name)!=expected:raise RuntimeError('Source mismatch: '+name)
    prior=json.loads((study/'checkpoint_change_deployment.json').read_text())
    for name,expected in prior['source_manifest']['sha256'].items():
        if digest(Path(prior['source'])/name)!=expected:raise RuntimeError('Fallback source mismatch: '+name)
    audit=json.loads((work/'relaunch_audit.json').read_text())
    checkpoint=work/'resume_checkpoint.pt'
    assert digest(checkpoint)==audit['checkpoint_sha256']
    config=json.loads((study/'training/config.json').read_text())
    assert config['world_size']==4 and config['global_batch']==1024
    assert config['checkpoint_every']==10000 and config['eval_every']==5000
    assert not config['lineage_has_flow_matching']
    assert digest(Path(config['fid_reference']))==config['fid_reference_sha256']
    def status(state,stage,**extra):
        row=dict(status=state,stage=stage,job_id=os.environ['JOB_ID'],
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        write_json(work/'job_status.json',row)
        write_json(study/'job_status.json',row)
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=4',str(source/'scripts/benchmark_vectorized_ddp.py'),
        '--study',str(study),'--checkpoint',str(checkpoint),'--output',str(work/'benchmark.json')]
    status('running','four_gpu_benchmark')
    with (work/'benchmark.log').open('a',buffering=1) as log:
        process=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=-1
    benchmark=json.loads((work/'benchmark.json').read_text()) if (work/'benchmark.json').exists() else {}
    optimized=code==0 and benchmark.get('passed',False) and benchmark.get('speedup',0)>1.2
    if optimized:
        assert benchmark['world_size']==4 and benchmark['checkpoint_sha256']==audit['checkpoint_sha256']
        for name,expected in benchmark['source_sha256'].items():assert digest(source/name)==expected
    selected=source if optimized else Path(prior['source'])
    command=prior['command'].copy()
    command[0]=sys.executable
    old_entry=str(Path(prior['source'])/'scripts/cs_cluster/jit_rank_entrypoint.py')
    command[command.index(old_entry)]=str(selected/'scripts/cs_cluster/jit_rank_entrypoint.py')
    command[command.index('--resume')+1]=str(checkpoint)
    deployment=dict(job_id=os.environ['JOB_ID'],source=str(selected),optimized=optimized,
        source_manifest=manifest if optimized else prior['source_manifest'],command=command,
        checkpoint=str(checkpoint),checkpoint_sha256=audit['checkpoint_sha256'],
        checkpoint_update=audit['checkpoint_update'],benchmark_exit_code=code,
        benchmark_sha256=digest(work/'benchmark.json') if benchmark else None,
        checkpoint_every=10000,eval_every=5000,world_size=4,global_batch=1024,
        original_preflight_retained=True,optimizer_ema_scaler_all_rank_rng_resume_required=True)
    write_json(study/'vectorized_deployment.json',deployment)
    stage='optimized_training' if optimized else 'original_fallback_training'
    status('running',stage)
    try:
        with (study/'training/run.log').open('a',buffering=1) as log:
            log.write('\nRiesz implementation continuation: '+json.dumps(deployment)+'\n')
            subprocess.run(command,cwd=selected,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete',stage)
    except BaseException as error:
        status('failed',stage,error=repr(error));raise


if __name__=='__main__':main()
