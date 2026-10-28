"""Freeze and submit a four-2080-Ti FM preflight followed by fresh training."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_storage import new_run_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--steps',type=int,default=250000)
    p.add_argument('--microbatch',type=int,default=32)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--walltime',default='120:00')
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    study=new_run_path(Path('cifar')/args.name).resolve()
    study.mkdir(parents=True,exist_ok=False)
    source=study/'source'
    files=['cifar_flow_matching.py','backbone.py','data.py','fid.py','_fid_inception.py',
           'run_storage.py','scripts/lsf/run_cifar_fm_ddp.py','scripts/lsf/launch_cifar_fm_ddp.py',
           'scripts/lsf/monitor_pixel_flow.py','tests/test_cifar_flow_matching.py',
           'assets/pt_inception-2015-12-05-6726825d.pth','assets/inception.sha256',
           'assets/SCORE_SDE_LICENSE','AGENTS.md']
    hashes={}
    for name in files:
        dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst)
        with dst.open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    references=study/'references';references.mkdir()
    api='https://api.github.com/repos/facebookresearch/flow_matching/commits/main'
    with urllib.request.urlopen(api,timeout=30) as response:
        commit=json.load(response)['sha']
    upstream=['examples/image/README.md','examples/image/train_arg_parser.py',
              'examples/image/training/train_loop.py','examples/image/training/edm_time_discretization.py',
              'examples/image/models/ema.py','LICENSE']
    reference_rows=[]
    for path in upstream:
        url=f'https://raw.githubusercontent.com/facebookresearch/flow_matching/{commit}/{path}'
        dst=references/path;dst.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(url,timeout=30) as response:dst.write_bytes(response.read())
        reference_rows.append(dict(url=url,path=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    test=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','unittest','discover',
        '-s',str(source/'tests'),'-p','test_cifar_flow_matching.py','-v'],cwd=source,
        text=True,capture_output=True)
    (study/'tests.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(f'Tests failed: {study}/tests.log')
    command=[str(ROOT/'.venv/bin/python'),'-m','torch.distributed.run','--standalone',
        '--nnodes=1','--nproc-per-node=4',str(source/'scripts/lsf/run_cifar_fm_ddp.py'),
        '--output',str(study/'training'),'--data-root',str(new_run_path(Path('datasets'))),
        '--steps',str(args.steps),'--microbatch',str(args.microbatch),
        '--accumulation',str(args.accumulation)]
    smoke=command.copy();smoke[smoke.index('--output')+1]=str(study/'preflight');smoke+=['--smoke']
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes=hashes,recipe_sources=reference_rows,upstream_commit=commit,
        command=command,smoke_command=smoke,queue='longgpu',host='longgpu01',gpus=4,
        required_gpu='RTX 2080 Ti',parameters=61804419,global_batch=4*args.microbatch*args.accumulation,
        initialization='random seed 0, no pretrained models or transferred weights',
        recipe='Conditional flow matching with independent Gaussian coupling, skewed times, AdamW and EMA',
        sampling='50 Heun intervals / 100 NFE with EDM-derived grid; 100-interval audit at update 5000, 100/250 at final',
        deviations_from_upstream=['Required DDPM backbone, including dropout 0.1 and time mapping 999*(1-t)',
            'Exact Gaussian t=0 integration endpoint added to EDM-derived time grid',
            'Global batch 256 via four GPUs and accumulation; gradient clipping at 1',
            '250000 update budget by default; not the published 3000-epoch run',
            'FID-10K unquantized pixels for local comparisons, not upstream FID-50K'],
        claims='Strong published recipe adapted to required architecture; no claim of reproduced SOTA FID',
        validation=dict(cpu_tests_passed=True,gpu_preflight='required before training'),
        checkpoint='Full model, EMA, AdamW, GradScaler and per-rank RNG every 1000 updates; atomic latest/best',
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
        preflight=submit(smoke,'cifar_fm_ddp_preflight','00:30',study/'preflight_scheduler.log')
        (study/'preflight_submission.json').write_text(json.dumps(preflight,indent=2)+'\n')
        training=submit(command,'cifar_fm_ddpm_unet_4gpu',args.walltime,study/'scheduler.log',preflight['job_id'])
        (study/'submission.json').write_text(json.dumps(training,indent=2)+'\n')
        print(json.dumps(dict(study=str(study),preflight_job=preflight['job_id'],training_job=training['job_id'])))
    else:print(study)


if __name__=='__main__':main()
