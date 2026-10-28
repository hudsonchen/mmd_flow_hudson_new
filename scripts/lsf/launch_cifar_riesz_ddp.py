"""Freeze, validate, and submit direct-velocity Riesz training or continuation."""
import argparse
import datetime
import hashlib
import json
import math
import os
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
    p.add_argument('--reference-time-power',type=float,default=1.)
    p.add_argument('--solver',choices=['euler','heun'],default='heun')
    p.add_argument('--sample-steps',type=int,default=50)
    p.add_argument('--gpus',type=int,choices=[1,4],default=4)
    p.add_argument('--queue',default='longgpu')
    p.add_argument('--host',default='longgpu01')
    p.add_argument('--required-gpu',default='RTX 2080 Ti')
    p.add_argument('--gpu-memory',type=int,default=9)
    p.add_argument('--gpu-mig')
    p.add_argument('--resume',type=Path)
    p.add_argument('--allow-evaluation-change',action='store_true')
    p.add_argument('--dependency-job')
    p.add_argument('--walltime',default='120:00')
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.reference_time_power) or args.reference_time_power < 0:
        p.error('Invalid reference-time exponent')
    study=new_run_path(Path('cifar')/args.name).resolve()
    study.mkdir(parents=True,exist_ok=False)
    source=study/'source'
    files=['cifar_riesz_velocity.py','velocity_model.py','riesz_loss.py','backbone.py','data.py','fid.py','_fid_inception.py',
           'run_storage.py','scripts/lsf/run_cifar_riesz_ddp.py','scripts/lsf/launch_cifar_riesz_ddp.py',
           'scripts/lsf/monitor_pixel_flow.py','scripts/lsf/audit_cifar_velocity_samplers.py',
           'tests/test_cifar_riesz_velocity.py','tests/test_riesz_power.py','tests/test_velocity_model.py','tests/test_run_storage.py',
           'assets/pt_inception-2015-12-05-6726825d.pth','assets/inception.sha256',
           'assets/SCORE_SDE_LICENSE','assets/PYTORCH_FID_LICENSE','SOFTMAX_RIESZ_LICENSE.txt','DRIFTING_LICENSE.txt','AGENTS.md']
    hashes={}
    for name in files:
        dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst)
        with dst.open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    logs=[]
    for pattern in ['test_cifar_riesz_velocity.py','test_riesz_power.py','test_velocity_model.py','test_run_storage.py']:
        test=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','unittest','discover',
            '-s',str(source/'tests'),'-p',pattern,'-v'],cwd=source,text=True,capture_output=True,
            env={**__import__('os').environ,'DRIFT_COMPILE':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        logs.append(test.stdout+test.stderr)
        (study/'tests.log').write_text('\n'.join(logs))
        if test.returncode:raise RuntimeError(f'Tests failed: {study}/tests.log')
    command=[str(ROOT/'.venv/bin/python'),'-m','torch.distributed.run','--standalone',
        '--nnodes=1',f'--nproc-per-node={args.gpus}',str(source/'scripts/lsf/run_cifar_riesz_ddp.py'),
        '--output',str(study/'training'),'--data-root',str(new_run_path(Path('datasets'))),
        '--steps',str(args.steps),'--batch-per-rank',str(args.batch_per_rank),
        '--reference-time-power',str(args.reference_time_power),'--solver',args.solver,
        '--sample-steps',str(args.sample_steps),'--required-gpu',args.required_gpu]
    smoke=command.copy();smoke[smoke.index('--output')+1]=str(study/'preflight');smoke+=['--smoke']
    if args.resume:
        os.link(args.resume,study/'resume.pt')
        command+=['--resume',str(study/'resume.pt')]
        if args.allow_evaluation_change:command+=['--allow-evaluation-change']
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes=hashes, command=command, smoke_command=smoke,
        queue=args.queue,host=args.host,gpus=args.gpus,required_gpu=args.required_gpu,parameters=61804419,
        global_batch=args.gpus*args.batch_per_rank,
        initialization='Riesz-only optimizer/RNG continuation' if args.resume else 'random seed 0; no FM, diffusion or reconstruction weights',
        recipe='Direct velocity; paired pixel Riesz; powers 0.8/1.0/1.5 summed without RMS normalization',
        time_sampling='Meta sigmoid(1.2-1.2*N(0,1)), capped at 0.999; one shared time for all global particles',
        reference_time_weight=f'(1-t)^{args.reference_time_power} / h; no denoiser division in the model',
        meta_time_reference='https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/examples/image/training/train_loop.py',
        optimizer='Existing Riesz Adam schedule and EMA; AMP added for 2080 Ti capacity',
        sampling=f'{args.sample_steps} uniform {args.solver} steps; {args.sample_steps*(2 if args.solver=="heun" else 1)} network calls',
        distributed_loss=f'Global {args.gpus*args.batch_per_rank}-particle field; attach local queries; multiply loss by {args.gpus} before DDP backward',
        validation=dict(cpu_tests_passed=True,gpu_preflight='required before training'),
        checkpoint='Atomic latest every 1000 updates, best at completed FID; optimizer/scaler/all-rank RNG included',
        fid='FID-10K against all 50000 training images every 5000 updates; preview every 1000',
        storage=str(study),resume_origin=str(args.resume) if args.resume else None)
    (study/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.submit:
        gpu=f'num={args.gpus}:mode=shared:gmem={args.gpu_memory}G'
        gpu+=f':mig={args.gpu_mig}' if args.gpu_mig else ':j_exclusive=yes'
        resources=['-q',args.queue,'-m',args.host,'-n',str(max(4,2*args.gpus)),'-R',
            'span[hosts=1] rusage[mem=16GB]','-gpu',gpu,
            '-env','all,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2']
        def submit(cmd,name,wall,log,dependency=None):
            invocation=['bsub',*resources,'-J',name,'-W',wall,'-oo',str(log)]
            if dependency:invocation+=['-w',f'done({dependency})']
            result=subprocess.run([*invocation,*cmd],text=True,capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr+result.stdout)
            return dict(job_id=re.search(r'<(\d+)>',result.stdout).group(1),
                        stdout=result.stdout,command=[*invocation,*cmd])
        preflight=submit(smoke,'cifar_riesz_velocity_preflight','00:30',study/'preflight_scheduler.log',args.dependency_job)
        (study/'preflight_submission.json').write_text(json.dumps(preflight,indent=2)+'\n')
        training=submit(command,f'cifar_velocity_tw{args.reference_time_power:g}',args.walltime,study/'scheduler.log',preflight['job_id'])
        (study/'submission.json').write_text(json.dumps(training,indent=2)+'\n')
        print(json.dumps(dict(study=str(study),preflight_job=preflight['job_id'],training_job=training['job_id'])))
    else:print(study)


if __name__=='__main__':main()
