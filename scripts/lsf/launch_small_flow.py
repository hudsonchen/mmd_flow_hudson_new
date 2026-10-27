"""Freeze source and submit a reference-flow experiment; default to RTX 2080 Ti."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_storage import new_run_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--walltime', default='02:00')
    p.add_argument('--queue',default='longgpu')
    p.add_argument('--host',default='longgpu01')
    p.add_argument('--gpu-memory',type=int,help='Requested GPU memory in GB')
    p.add_argument('--gpu-mig',choices=['1/1','2/2','3/3','4/4','7/7'],
                   help='Explicit GPU/compute instance size on MIG-enabled hosts')
    p.add_argument('--exclusive',action='store_true')
    p.add_argument('--cpus',type=int,default=2)
    p.add_argument('--memory-gb',type=int,default=4)
    p.add_argument('--dataset', choices=['mog','mnist','fashionmnist','cifar10'], default='mog')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--gradient-diagnostic', action='store_true')
    args, extra = p.parse_known_args()
    if args.cpus<1 or args.memory_gb<1 or (args.gpu_memory is not None and args.gpu_memory<1):p.error('Resource requests must be positive')
    if args.gpu_mig and args.exclusive:p.error('MIG instances require shared compute mode; omit --exclusive')
    if sum([args.audit,args.smoke,args.gradient_diagnostic])>1:p.error('Choose only one audit, smoke, or gradient diagnostic mode')
    folder={'mog':'toy','cifar10':'cifar'}.get(args.dataset,args.dataset)
    study = new_run_path(Path(folder)/args.name).resolve()
    study.mkdir(parents=True,exist_ok=False)
    source = study/'source'
    files = ['small_flow.py','backbone.py','loss.py','sinkhorn.py','run_storage.py','reference_dit.py','flow_path.py',
             'scripts/lsf/run_small_flow.py','scripts/lsf/launch_small_flow.py','tests/test_small_flow.py','AGENTS.md']
    files+=['tests/test_reference_dit.py']
    files += [str(p.relative_to(ROOT)) for p in ROOT.glob('*LICENSE*')]
    if (ROOT/'assets/SCORE_SDE_LICENSE').exists(): files.append('assets/SCORE_SDE_LICENSE')
    if args.audit:
        files+=['scripts/lsf/audit_small_flow.py']
    if args.gradient_diagnostic:
        if args.dataset=='mog':p.error('Gradient diagnostic requires an image dataset')
        files+=['scripts/lsf/diagnose_image_reference_gradients.py']
    if args.dataset!='mog':
        files+=['scripts/lsf/run_small_image_flow.py','scripts/lsf/smoke_small_image_flow.py','scripts/lsf/audit_small_image_flow.py','data.py','fid.py','_fid_inception.py','perceptual.py',
                'assets/inception.sha256','assets/pt_inception-2015-12-05-6726825d.pth']
    hashes = {}
    for name in files:
        dest = source/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if '--fork' in extra:
        index = extra.index('--fork')+1
        parent = Path(extra[index]).resolve()
        shutil.copy2(parent,study/'parent.pt')
        extra[index] = str(study/'parent.pt')
        hashes['parent.pt'] = hashlib.sha256((study/'parent.pt').read_bytes()).hexdigest()
    if '--dit-prime-checkpoint' in extra:
        index=extra.index('--dit-prime-checkpoint')+1
        parent=Path(extra[index]).resolve()
        shutil.copy2(parent,study/'dit_reconstruction.pt')
        extra[index]=str(study/'dit_reconstruction.pt')
        with (study/'dit_reconstruction.pt').open('rb') as stream:hashes['dit_reconstruction.pt']=hashlib.file_digest(stream,'sha256').hexdigest()
    if '--vgg-cache' in extra:
        index=extra.index('--vgg-cache')+1
        cache=Path(extra[index]).resolve()
        weights=cache/'hub/checkpoints/vgg16-397923af.pth'
        if not weights.is_file():raise FileNotFoundError('VGG cache must already contain verified pretrained weights')
        dest=study/'perceptual_cache/hub/checkpoints'/weights.name
        dest.parent.mkdir(parents=True)
        shutil.copy2(weights,dest)
        with dest.open('rb') as stream: hashes[str(dest.relative_to(study))]=hashlib.file_digest(stream,'sha256').hexdigest()
        extra[index]=str(study/'perceptual_cache')
    if args.gradient_diagnostic:
        for flag,name in [('--checkpoint','diagnostic_checkpoint.pt'),('--recipe','recipe.json')]:
            if flag in extra:
                index=extra.index(flag)+1
                shutil.copy2(Path(extra[index]).resolve(),study/name)
                extra[index]=str(study/name)
                with (study/name).open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    runner = 'run_small_flow.py' if args.dataset=='mog' else 'run_small_image_flow.py'
    if args.audit:
        runner='audit_small_flow.py' if args.dataset=='mog' else 'audit_small_image_flow.py'
    if args.smoke:
        if args.dataset=='mog': raise ValueError('Image preflight requires an image dataset')
        runner='smoke_small_image_flow.py'
    if args.gradient_diagnostic:runner='diagnose_image_reference_gradients.py'
    command = [str(ROOT/'.venv/bin/python'),'-u',str(source/'scripts/lsf'/runner),
               '--output',str(study/('audit' if args.audit else 'training')),*extra]
    if args.dataset!='mog':
        command+=['--dataset',args.dataset,'--data-root',str(new_run_path(Path('datasets')))]
    coupling=(extra[extra.index('--reference-coupling')+1] if '--reference-coupling' in extra
        else 'paired' if '--objective' in extra and extra[extra.index('--objective')+1]=='coupled' else 'independent')
    manifest = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,
        hashes=hashes,queue=args.queue,host=args.host,cpus=args.cpus,memory_gb=args.memory_gb,gpu_memory=args.gpu_memory,gpu_mig=args.gpu_mig,
        outside_definition='distance to nearest center > 3*0.12',
        dataset=args.dataset,reference='affine Gaussian-to-data path',
        reference_particle_coupling='recorded per checkpoint' if args.audit else coupling,
        initialization_policy='Riesz: random, reconstruction-primed, or Riesz-only ancestry; no FM weights or distillation',
        fid='not applicable to 2D MoG' if args.dataset=='mog' else 'FID-10K every 5000 updates by default',
        fid_reference='all 50000 training images by default; independent test audit separate' if args.dataset=='cifar10' else 'fixed training subset; independent test audit separate')
    (study/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    gpu='num=1:mode='+('exclusive_process' if args.exclusive else 'shared')
    if args.gpu_memory:gpu+=f':gmem={args.gpu_memory}G'
    if args.gpu_mig:gpu+=f':mig={args.gpu_mig}'
    result = subprocess.run(['bsub','-q',args.queue,'-m',args.host,'-J','small_reference_flow',
        '-n',str(args.cpus),'-R',f'span[hosts=1] rusage[mem={args.memory_gb}GB]','-gpu',gpu,
        '-W',args.walltime,'-oo',str(study/'scheduler.log'),*command],text=True,capture_output=True,check=True)
    (study/'submission.json').write_text(json.dumps(dict(stdout=result.stdout,stderr=result.stderr,
        job_id=re.search(r'<(\d+)>',result.stdout).group(1)),indent=2)+'\n')
    print(study)
    print(result.stdout)


if __name__ == '__main__':
    main()
