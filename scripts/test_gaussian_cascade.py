"""Controlled study of sequential Riesz-MMD transport to eight Gaussians."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
os.environ['DRIFT_COMPILE'] = '0'
import numpy as np
import torch
from loss import _riesz_topk_softmax_loss_impl
from run_storage import new_run_path

RECIPES = ('mmd_actual', 'mmd_ideal', 'mmd_direct', 'softmax_actual', 'softmax_fourier_actual', 'rbf_actual', 'rbf_fourier_actual')
RBF_SIGMAS = (.1, .2, .4, .8, 1.6)


class Map(torch.nn.Module):
    def __init__(self, width=64, fourier=False):
        super().__init__()
        frequencies=(torch.randn(2,64)*torch.tensor([.5,1.,2.,4.]).repeat_interleave(16)
                     if fourier else torch.empty(2,0))
        self.register_buffer('frequencies',frequencies,persistent=fourier)
        self.net = torch.nn.Sequential(torch.nn.Linear(2+2*frequencies.shape[1], width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 2))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        phase=x@self.frequencies
        features=torch.cat((x,phase.sin(),phase.cos()),dim=-1) if self.frequencies.numel() else x
        return x + self.net(features)


def centers():
    angle = torch.arange(8)*2*math.pi/8
    return 2*torch.stack([angle.cos(), angle.sin()], dim=1)


def sample_data(count, rng):
    return centers()[torch.randint(8, (count,), generator=rng)] + .12*torch.randn(count, 2, generator=rng)


def sample_path(count, t, rng):
    noise = torch.randn(count, 2, generator=rng)
    clean = sample_data(count, rng)
    return (1-t)*noise+t*clean, noise


@torch.no_grad()
def compose(maps, x):
    for model in maps:
        x = model(x)
    return x.detach()


def source_batch(recipe, previous, stage, count, rng, device='cpu'):
    ideal, noise = sample_path(count, stage/10, rng)
    return compose(previous, noise.to(device)) if recipe.endswith('actual') else ideal.to(device)


def energy(x, y):
    """Unbiased empirical energy distance (Riesz kernel -||x-y||).

    Its finite-sample estimate can be negative. Targets are independent.
    """
    xx = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
    yy = torch.cdist(y, y, compute_mode='donot_use_mm_for_euclid_dist')
    xy = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    n, m = len(x), len(y)
    return 2*xy.mean()-xx.sum()/(n*(n-1))-yy.sum()/(m*(m-1))


def gaussian_mmd(x, y):
    """Unbiased MMD squared for an equal mixture of Gaussian kernels."""
    scales = x.new_tensor(RBF_SIGMAS).square()*2
    def kernel(a,b):
        distance2 = (a[:,None,:]-b[None,:,:]).square().sum(-1)
        return torch.exp(-distance2[...,None]/scales).mean(-1)
    xx,yy,xy=kernel(x,x),kernel(y,y),kernel(x,y)
    n,m=len(x),len(y)
    return ((xx.sum()-xx.diag().sum())/(n*(n-1))
            +(yy.sum()-yy.diag().sum())/(m*(m-1))-2*xy.mean())


@torch.no_grad()
def diagnostics(x, target, t):
    angles = torch.arange(128)*math.pi/128
    directions = torch.stack([angles.cos(), angles.sin()], 1)
    a = (x@directions.T).sort(dim=0).values
    b = (target@directions.T).sort(dim=0).values
    result = dict(energy_unbiased=float(energy(x, target)),
                  sliced_w2=float((a-b).square().mean().sqrt()))
    if t==1:
        distance, assignment = torch.cdist(x, centers()).min(1)
        masses = torch.bincount(assignment, minlength=8).float()/len(x)
        result.update(off_mode_fraction=float((distance > 3*.12).float().mean()),
            covered_modes=int((masses>.02).sum()), mode_mass_l1=float((masses-1/8).abs().sum()),
            nearest_center_rms=float(distance.square().mean().sqrt()))
    return result


def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)


def worker(study, recipe, seed, updates, device='cpu', final_updates=None, softmax_tau=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    study = Path(study)
    output = study/f'{recipe}_seed{seed}'
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    write_json(output/'status.json', dict(status='running', recipe=recipe, seed=seed))
    maps = [Map(209)] if recipe=='mmd_direct' else [Map(64,fourier='fourier' in recipe) for _ in range(10)]
    if device == 'cuda':
        torch.cuda.set_device(seed % torch.cuda.device_count())
    maps = [model.to(device) for model in maps]
    write_json(output/'hardware.json',dict(device=device,
        model=torch.cuda.get_device_name() if device=='cuda' else 'CPU',
        torch_version=torch.__version__))
    parameter_count = sum(p.numel() for m in maps for p in m.parameters())
    train_rng = torch.Generator().manual_seed(1000+seed)
    eval_noise = torch.randn(2048, 2, generator=torch.Generator().manual_seed(2000+seed))
    rows, train_rows = [], []
    saved = {'initial':eval_noise.numpy()}
    total_updates = 0
    stages = 1 if recipe=='mmd_direct' else 10
    for stage in range(stages):
        model = maps[stage]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        t = 1. if stages==1 else (stage+1)/10
        count = 10*updates if stages==1 else updates
        if stages==10 and stage==9 and final_updates is not None:
            count=final_updates
        accumulated = 0.
        for step in range(1,count+1):
            source = source_batch(recipe,maps[:stage],stage,128,train_rng,device)
            target,_ = sample_path(256,t,train_rng)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source)
            field_info=None
            if recipe.startswith('softmax'):
                value,field_info = _riesz_topk_softmax_loss_impl(prediction[None],target[None],
                    tau_list=(softmax_tau,) if softmax_tau is not None else (.2,.05,.02),normalize_per_temperature=False,
                    no_field_rms_normalization=True,repulsion_weight=1.,epsilon=1e-6)
                value = value.mean()
            elif recipe.startswith('rbf'):
                value = gaussian_mmd(prediction,target)
            else:
                value = energy(prediction,target)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            accumulated += float(value.detach())
            total_updates += 1
            if step%100==0:
                train_row=dict(stage=stage+1, update=total_updates,
                    training_loss=accumulated/100,elapsed_seconds=time.monotonic()-start)
                if field_info is not None:
                    train_row.update({f'last_batch_{k}':float(field_info[f'riesz_topk_softmax_{k}'])
                        for k in ['tau_count','force_rms','positive_ess','generated_ess']})
                train_rows.append(train_row)
                accumulated = 0.
            if step%500==0:
                write_json(output/'status.json',dict(status='running',recipe=recipe,seed=seed,
                    stage=stage+1,update=total_updates,elapsed_seconds=time.monotonic()-start))
            if step%5000==0:
                write_csv(output/'training_metrics.csv',train_rows)
            if recipe=='mmd_direct' and step%updates==0 and step<count:
                reference,_ = sample_path(2048,1.,torch.Generator().manual_seed(3009+seed*100))
                actual = compose([model],eval_noise.to(device)).cpu()
                milestone=dict(recipe=recipe,seed=seed,update=total_updates,
                    elapsed_seconds=time.monotonic()-start,**diagnostics(actual,reference,1.))
                write_json(output/f'metrics_update_{total_updates:07d}.json',milestone)
                torch.save(dict(recipe=recipe,seed=seed,update=total_updates,
                    maps=[model.state_dict()],widths=[209],optimizer=optimizer.state_dict(),
                    train_rng=train_rng.get_state(),torch_rng=torch.get_rng_state()),
                    output/f'checkpoint_update_{total_updates:07d}.pt')
                print(json.dumps(milestone),flush=True)
        model.eval().requires_grad_(False)
        assert all(not p.requires_grad for m in maps[:stage+1] for p in m.parameters())
        reference,_ = sample_path(2048,t,torch.Generator().manual_seed(3000+seed*100+round(10*t)-1))
        ideal_source,_ = sample_path(2048,(0 if stages==1 else stage/10),
                                     torch.Generator().manual_seed(4000+seed*100+stage))
        actual = compose(maps[:stage+1],eval_noise.to(device)).cpu()
        local = compose([model],ideal_source.to(device)).cpu()
        row=dict(recipe=recipe,seed=seed,stage=stage+1,target_time=t,update=total_updates,
            elapsed_seconds=time.monotonic()-start,**diagnostics(actual,reference,t),
            ideal_input_sliced_w2=diagnostics(local,reference,t)['sliced_w2'])
        rows.append(row)
        saved[f'stage_{stage+1:02d}']=actual.numpy()
        saved[f'target_{stage+1:02d}']=reference.numpy()
        torch.save(dict(recipe=recipe,seed=seed,stage=stage+1,update=total_updates,
            maps=[m.state_dict() for m in maps[:stage+1]], widths=[209] if stages==1 else [64]*(stage+1),
            optimizer=optimizer.state_dict(),train_rng=train_rng.get_state(),torch_rng=torch.get_rng_state()),
            output/f'checkpoint_stage_{stage+1:02d}.pt')
        write_csv(output/'stage_metrics.csv',rows)
        write_csv(output/'training_metrics.csv',train_rows)
        print(json.dumps(row),flush=True)
    np.savez_compressed(output/'samples.npz',**saved)
    final=dict(**rows[-1],parameter_count=parameter_count)
    write_json(output/'final.json',final)
    write_json(output/'status.json',dict(status='complete',**final))
    return final


def plots(study, finals, recipes, seed=0):
    os.environ['MPLCONFIGDIR']=str(study/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#385b9c','#ce6353','#539465','#9569ac','#bf943d']
    fig, axes=plt.subplots(1,1+len(recipes),figsize=(3.2*(1+len(recipes)),3.4),sharex=True,sharey=True)
    reference,_=sample_path(2048,1,torch.Generator().manual_seed(3009+seed*100))
    panels=[('Target mixture',reference.numpy())]
    for recipe in recipes:
        data=np.load(study/f'{recipe}_seed{seed}/samples.npz')
        panels.append((recipe.replace('_',' '),data['stage_01' if recipe=='mmd_direct' else 'stage_10']))
    for i,(title,points) in enumerate(panels):
        axes[i].scatter(points[:,0],points[:,1],s=2,alpha=.35,color='black' if i==0 else colors[(i-1)%len(colors)],rasterized=True)
        axes[i].set(title=title,xlim=(-3.5,3.5),ylim=(-3.5,3.5),aspect='equal')
    fig.suptitle(f'Eight-Gaussian transport: held-out samples, seed {seed}')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(study/f'final_samples.{suffix}',dpi=180)
    plt.close(fig)
    cascade_recipe=next((r for r in ('mmd_actual','rbf_actual','rbf_fourier_actual','softmax_actual','softmax_fourier_actual') if r in recipes),None)
    if cascade_recipe is None:
        return
    data=np.load(study/f'{cascade_recipe}_seed{seed}/samples.npz')
    fig,axes=plt.subplots(2,11,figsize=(19,4),sharex=True,sharey=True)
    for k in range(11):
        target=data['initial'] if k==0 else data[f'target_{k:02d}']
        actual=data['initial'] if k==0 else data[f'stage_{k:02d}']
        for row,points in enumerate((target,actual)):
            axes[row,k].scatter(points[:768,0],points[:768,1],s=1,alpha=.3,color='black' if row==0 else colors[0],rasterized=True)
            axes[row,k].set(xlim=(-3.5,3.5),ylim=(-3.5,3.5),aspect='equal',xticks=[],yticks=[])
        axes[0,k].set_title(f't={k/10:.1f}')
    axes[0,0].set_ylabel('Target')
    axes[1,0].set_ylabel('Actual cascade')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(study/f'cascade_stages.{suffix}',dpi=180)
    plt.close(fig)


def run(study, updates, workers, recipes, device, seeds, final_updates=None, softmax_tau=None):
    study=Path(study)
    write_json(study/'status.json',dict(status='running'))
    finals=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(worker,str(study),recipe,seed,updates,device,final_updates,softmax_tau):(recipe,seed)
               for recipe in recipes for seed in seeds}
        for task in as_completed(tasks):
            finals.append(task.result())
            write_csv(study/'summary.csv',finals)
    torch.set_num_threads(1)
    floors=[]
    for seed in seeds:
        a,_=sample_path(2048,1,torch.Generator().manual_seed(3009+seed*100))
        b,_=sample_path(2048,1,torch.Generator().manual_seed(9000+seed))
        floors.append(diagnostics(a,b,1))
    summary={}
    for recipe in recipes:
        rows=[r for r in finals if r['recipe']==recipe]
        summary[recipe]={k:dict(mean=float(np.mean([r[k] for r in rows])),
            std=float(np.std([r[k] for r in rows],ddof=1 if len(rows)>1 else 0))) for k in
            ['energy_unbiased','sliced_w2','off_mode_fraction','covered_modes','mode_mass_l1','nearest_center_rms','elapsed_seconds']}
        summary[recipe]['parameter_count']=rows[0]['parameter_count']
    write_json(study/'aggregate.json',dict(recipes=summary,independent_target_sample_floor=floors))
    plots(study,finals,recipes,seeds[0])
    write_json(study/'status.json',dict(status='complete',cases=len(finals)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name')
    parser.add_argument('--run-study',type=Path)
    parser.add_argument('--updates-per-stage',type=int,default=1000)
    parser.add_argument('--final-stage-updates',type=int)
    parser.add_argument('--softmax-tau',type=float,help='One fixed softmax temperature; omit to retain the legacy three-temperature mixture.')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--recipes',nargs='+',choices=RECIPES,default=['mmd_actual'])
    parser.add_argument('--study-note',default='Initial comparison')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--queue')
    parser.add_argument('--host',default='longgpu01')
    parser.add_argument('--walltime',default='01:00')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.softmax_tau is not None and (not math.isfinite(args.softmax_tau) or args.softmax_tau<=0):
        parser.error('--softmax-tau must be finite and positive')
    if args.run_study:
        run(args.run_study,args.updates_per_stage,args.workers,args.recipes,args.device,args.seeds,args.final_stage_updates,args.softmax_tau)
        return
    if not args.name:parser.error('--name is required')
    study=new_run_path(Path('toy')/args.name).resolve()
    study.mkdir(parents=True,exist_ok=False)
    files=['scripts/test_gaussian_cascade.py','loss.py','run_storage.py','tests/test_gaussian_cascade.py']
    hashes={}
    for name in files:
        dest=study/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
        hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write_json(study/'source/sha256.json',hashes)
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dimension=2,mixture_modes=8,radius=2.,component_std=.12,stages=10,
        updates_per_stage=args.updates_per_stage,
        nominal_equal_stage_total_updates=10*args.updates_per_stage,
        final_stage_updates=args.final_stage_updates or args.updates_per_stage,
        total_updates_by_recipe={r:(10*args.updates_per_stage if r=='mmd_direct' else
            9*args.updates_per_stage+(args.final_stage_updates or args.updates_per_stage)) for r in args.recipes},
        batch_size=128,target_batch_size=256,learning_rate=.001,seeds=args.seeds,recipes=args.recipes,
        gaussian_kernel_sigmas=RBF_SIGMAS,
        softmax_temperatures=[args.softmax_tau] if args.softmax_tau is not None else [.2,.05,.02],
        softmax_distance_scaling='Repository convention: divide distances by global batch-derived mean distance; tau itself is constant.',
        softmax_field_rms_normalization=False,
        training_objective_by_recipe={r:('unbiased multiscale Gaussian MMD squared' if r.startswith('rbf')
            else 'detached softmax Riesz surrogate' if r.startswith('softmax')
            else 'unbiased energy-distance Riesz MMD') for r in args.recipes},
        fourier_features='For fourier recipes: 64 fixed random Gaussian frequencies, 16 at each scale 0.5, 1, 2, 4 radians/unit; sine/cosine plus raw input. No target-center features.',
        study_note=args.study_note,
        device=args.device,queue=args.queue or ('longgpu' if args.device=='cuda' else 'long_normal'),
        host=args.host if args.device=='cuda' else None,
        walltime=args.walltime,direct_milestone_every=args.updates_per_stage,
        stage_width=64,direct_width=209,evaluation_samples=2048,sliced_w2_directions=128,
        metric='Unbiased energy distance; negative estimates are possible',
        stage_training='Freeze all previous maps; fresh Gaussian inputs each update, no cached pool',
        direct_control='Approximately matched total parameter count and equal optimizer updates; architecture and compute differ',
        fid='Not applicable to 2D distributions; evaluate held-out distributions after each stage')
    write_json(study/'manifest.json',manifest)
    python=str(ROOT/'.venv/bin/python')
    with (study/'validation.log').open('w') as stream:
        subprocess.run([python,'-c',"import sys; sys.path.append('/appl/python-3.12/lib/python3.12/site-packages'); import unittest; r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests',pattern='test_gaussian_cascade.py')); raise SystemExit(not r.wasSuccessful())"],
            cwd=study/'source',stdout=stream,stderr=subprocess.STDOUT,check=True)
    command=[python,'-u',str(study/'source/scripts/test_gaussian_cascade.py'),'--run-study',str(study),
             '--updates-per-stage',str(args.updates_per_stage),'--workers',str(args.workers),
             '--device',args.device,'--recipes',*args.recipes,'--seeds',*[str(s) for s in args.seeds]]
    if args.final_stage_updates is not None:
        command += ['--final-stage-updates',str(args.final_stage_updates)]
    if args.softmax_tau is not None:
        command += ['--softmax-tau',str(args.softmax_tau)]
    if args.submit:
        submission=['bsub','-q',manifest['queue'],'-J','gaussian_staged_mmd','-n',str(args.workers),
            '-R','span[hosts=1] rusage[mem=4GB]','-W',args.walltime,'-oo',str(study/'scheduler.log')]
        if args.device=='cuda':
            submission += ['-m',args.host,'-gpu',f'num={min(3,args.workers)}:mode=shared']
        submission += command
        result=subprocess.run(submission,text=True,capture_output=True)
        write_json(study/'deployment.json',dict(command=submission,stdout=result.stdout,
                   stderr=result.stderr,returncode=result.returncode))
        print(result.stdout,result.stderr,flush=True)
        result.check_returncode()
    else:
        print('Prepared:',study)
    print(study,flush=True)


if __name__=='__main__':main()
