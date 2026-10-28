"""Freeze and submit a four-2080-Ti direct-velocity Riesz preflight followed by fresh training."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_storage import new_run_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--batch-per-rank',type=int,default=64)
    p.add_argument('--walltime',default='120:00')
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    study=new_run_path(Path('cifar')/args.name).resolve()
    study.mkdir(parents=True,exist_ok=False)
    source=study/'source'
    files=['cifar_riesz_velocity.py','cifar_flow_matching.py','small_flow.py','loss.py','backbone.py','data.py','fid.py','_fid_inception.py',
           'run_storage.py','scripts/lsf/run_cifar_riesz_ddp.py','scripts/lsf/launch_cifar_riesz_ddp.py',
           'scripts/lsf/run_cifar_fm_ddp.py','scripts/lsf/monitor_pixel_flow.py',
           'tests/test_cifar_riesz_velocity.py','tests/test_riesz_power.py','tests/test_cifar_flow_matching.py',
           'assets/pt_inception-2015-12-05-6726825d.pth','assets/inception.sha256',
           'assets/SCORE_SDE_LICENSE','SOFTMAX_RIESZ_LICENSE.txt','AGENTS.md']
    hashes={}
    for name in files:
        dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst)
        with dst.open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    logs=[]
    for pattern in ['test_cifar_riesz_velocity.py','test_riesz_power.py','test_cifar_flow_matching.py']:
        test=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','unittest','discover',
            '-s',str(source/'tests'),'-p',pattern,'-v'],cwd=source,text=True,capture_output=True,
            env={**__import__('os').environ,'DRIFT_COMPILE':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        logs.append(test.stdout+test.stderr)
        (study/'tests.log').write_text('\n'.join(logs))
        if test.returncode:raise RuntimeError(f'Tests failed: {study}/tests.log')
    command=[str(ROOT/'.venv/bin/python'),'-m','torch.distributed.run','--standalone',
        '--nnodes=1','--nproc-per-node=4',str(source/'scripts/lsf/run_cifar_riesz_ddp.py'),
        '--output',str(study/'training'),'--data-root',str(new_run_path(Path('datasets'))),
        '--steps',str(args.steps),'--batch-per-rank',str(args.batch_per_rank)]
    smoke=command.copy();smoke[smoke.index('--output')+1]=str(study/'preflight');smoke+=['--smoke']
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes=hashes, command=command, smoke_command=smoke,
        queue='longgpu',host='longgpu01',gpus=4,required_gpu='RTX 2080 Ti',parameters=61804419,
        global_batch=4*args.batch_per_rank,initialization='random seed 0; no FM, diffusion or reconstruction weights',
        recipe='Direct velocity; paired pixel Riesz; powers 0.8/1.0/1.5 summed without RMS normalization',
        time_sampling='Meta sigmoid(1.2-1.2*N(0,1)), capped at 0.999; one shared time for all global particles',
        reference_time_weight='Retain (1-t)^1 from previous Riesz recipe; no denoiser division in the model',
        meta_time_reference='https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/examples/image/training/train_loop.py',
        optimizer='Existing Riesz Adam schedule and EMA; AMP added for 2080 Ti capacity',
        sampling='100 uniform Euler steps; 100 network calls',
        distributed_loss='Gather detached predictions/targets across four ranks; full 256-particle field; attach local queries; multiply loss by 4 before DDP backward',
        validation=dict(cpu_tests_passed=True,gpu_preflight='required before training'),
        checkpoint='Atomic latest every 1000 updates, best at completed FID; optimizer/scaler/all-rank RNG included',
        fid='FID-10K against all 50000 training images every 5000 updates; preview every 1000',
        storage=str(study))
    (study/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.submit:
        resources=['-q','longgpu','-m','longgpu01','-n','8','-R',
            'span[hosts=1] rusage[mem=16GB]','-gpu','num=4:mode=shared:gmem=9G:j_exclusive=yes',
            '-env','all,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2']
        def submit(cmd,name,wall,log,dependency=None):
            invocation=['bsub',*resources,'-J',name,'-W',wall,'-oo',str(log)]
            if dependency:invocation+=['-w',f'done({dependency})']
            result=subprocess.run([*invocation,*cmd],text=True,capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr+result.stdout)
            return dict(job_id=re.search(r'<(\d+)>',result.stdout).group(1),
                        stdout=result.stdout,command=[*invocation,*cmd])
        preflight=submit(smoke,'cifar_riesz_velocity_preflight','00:30',study/'preflight_scheduler.log')
        (study/'preflight_submission.json').write_text(json.dumps(preflight,indent=2)+'\n')
        training=submit(command,'cifar_riesz_velocity_4gpu',args.walltime,study/'scheduler.log',preflight['job_id'])
        (study/'submission.json').write_text(json.dumps(training,indent=2)+'\n')
        print(json.dumps(dict(study=str(study),preflight_job=preflight['job_id'],training_job=training['job_id'])))
    else:print(study)


if __name__=='__main__':main()
