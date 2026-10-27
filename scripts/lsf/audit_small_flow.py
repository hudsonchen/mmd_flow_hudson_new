"""Fresh-sample MoG audit, including solver sensitivity and component spread."""
import argparse
import json
import hashlib
import io
from itertools import product
import os
from pathlib import Path
import sys
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import numpy as np
import torch
from small_flow import ToyVelocity,PotentialVelocity,DenoiserVelocity,sample,validate_riesz_parent
from scripts.lsf.run_small_flow import metrics,draw,centers,write_json,analytic_sliced_w2
from run_storage import new_run_path


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=50000)
    p.add_argument('--steps',nargs='+',type=int)
    p.add_argument('--solver',choices=['recorded','heun','euler','both'],default='recorded')
    p.add_argument('--checkpoint',default='best.pt')
    p.add_argument('--noise-seed',type=int,default=424242)
    p.add_argument('--target-seed',type=int,default=7654321)
    args=p.parse_args()
    output=new_run_path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib_cache')
    torch.set_num_threads(2)
    noise=torch.randn(args.count,2,generator=torch.Generator().manual_seed(args.noise_seed)).cuda()
    rows=[]
    points={}
    for study in args.studies:
        payload=(study/'training'/args.checkpoint).read_bytes()
        (output/(study.name+'_checkpoint.pt')).write_bytes(payload)
        checkpoint=torch.load(io.BytesIO(payload),map_location='cuda',weights_only=False)
        if checkpoint['config']['objective']!='fm': validate_riesz_parent(checkpoint)
        ancestry=[]
        ancestor=checkpoint
        ancestor_payload=payload
        while True:
            ancestry.append(dict(objective=ancestor['config']['objective'],update=ancestor['update'],
                reference_coupling=ancestor['config'].get('reference_coupling',
                    'paired' if ancestor['config']['objective']=='coupled' else 'independent'),
                sha256=hashlib.sha256(ancestor_payload).hexdigest(),parent=str(ancestor['config'].get('fork'))))
            parent=ancestor['config'].get('fork')
            if not parent: break
            ancestor_payload=Path(parent).read_bytes()
            ancestor=torch.load(io.BytesIO(ancestor_payload),map_location='cpu',weights_only=False)
        cls=PotentialVelocity if checkpoint['config'].get('potential',False) else ToyVelocity
        model=cls(checkpoint['config']['width'],checkpoint['config'].get('spatial_fourier',False)).cuda()
        if checkpoint['config'].get('parameterization')=='denoiser': model=DenoiserVelocity(model)
        model.load_state_dict(checkpoint['ema'])
        step_list=args.steps or [checkpoint['config']['sample_steps']]
        solvers=(['euler','heun'] if args.solver=='both' else
                 [checkpoint['config']['solver'] if args.solver=='recorded' else args.solver])
        for steps,solver in product(step_list,solvers):
            if checkpoint['config'].get('parameterization')=='denoiser' and solver!='euler':
                raise ValueError('Denoiser parameterization is evaluated with Euler')
            x=sample(model,noise,steps,solver).cpu()
            sigma=checkpoint['config'].get('sample_noise',0.)
            if sigma:
                x=x+sigma*torch.randn(x.shape,generator=torch.Generator().manual_seed(args.noise_seed+1))
            row=dict(study=study.name,checkpoint_update=checkpoint['update'],sample_steps=steps,
                     parameterization=checkpoint['config'].get('parameterization','velocity'),affinity=checkpoint['config'].get('affinity','row'),
                     endpoint_tail_scale=checkpoint['config'].get('endpoint_tail_scale',0.),endpoint_tail_max_weight=checkpoint['config'].get('endpoint_tail_max_weight',4.),
                     solver=solver,network_evaluations=(2 if solver=='heun' else 1)*steps,
                     parameters=sum(p.numel() for p in model.parameters()),sample_noise=sigma,checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                     ancestry=ancestry,lineage_has_flow_matching=any(a['objective']=='fm' for a in ancestry),
                     uses_paired_reference_velocity_loss=any(a['objective'] in ['fm','reference-hybrid'] for a in ancestry),
                     reference_coupling=ancestry[0]['reference_coupling'],
                     uses_paired_query_reference_particles=any(a['reference_coupling'] in ['paired','paired-loo'] for a in ancestry),
                     sliced_w2_analytic_target=analytic_sliced_w2(x),
                     **metrics(x,seed=args.target_seed))
            f=row['off_mode_fraction']
            # Wilson interval, not a claim about training-seed uncertainty.
            z=1.96; n=len(x)
            center=(f+z*z/(2*n))/(1+z*z/n)
            radius=z*((f*(1-f)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
            row['outside_95ci']=[center-radius,center+radius]
            rows.append(row)
            if steps==step_list[-1]:
                points[study.name+('_'+solver if len(solvers)>1 else '')]=x.numpy()
            print(json.dumps(row),flush=True)
    target=draw(args.count,'cpu',torch.Generator().manual_seed(78901))
    write_json(output/'audit.json',dict(count=args.count,noise_seed=args.noise_seed,target_seed=args.target_seed,
        checkpoint_selection=args.checkpoint,
        target=dict(**metrics(target,seed=args.target_seed),sliced_w2_analytic_target=analytic_sliced_w2(target)),rows=rows))
    np.savez_compressed(output/'samples.npz',target=target.numpy(),**points)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('Target',target.numpy())]+[(k,v) for k,v in points.items()]
    cols=min(4,len(panels)); nrows=(len(panels)+cols-1)//cols
    fig,axes=plt.subplots(nrows,cols,figsize=(4*cols,4*nrows),squeeze=False)
    for ax,(name,x) in zip(axes.flat,panels):
        ax.scatter(x[:10000,0],x[:10000,1],s=1,alpha=.2,rasterized=True)
        ax.set(title=name.replace('reference_','').replace('_20261026','').replace('_',' '),
               xlim=(-3,3),ylim=(-3,3),aspect='equal')
        ax.title.set_fontsize(8)
    for ax in list(axes.flat)[len(panels):]: ax.axis('off')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160)
    fig.savefig(output/'comparison.pdf')


if __name__=='__main__':main()
