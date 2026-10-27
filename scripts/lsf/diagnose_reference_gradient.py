"""Measure reference-field Monte Carlo gradients against exact Gaussian marginals."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import numpy as np
import torch
from run_storage import new_run_path
from small_flow import marginal_loss


class GaussianVelocity(torch.nn.Module):
    def __init__(self,time,rate,offset=0.):
        super().__init__()
        self.time=time
        self.offset=torch.nn.Parameter(torch.tensor(float(offset)))
        self.rate=torch.nn.Parameter(torch.tensor(float(rate)))

    def forward(self,x,t):
        return .3+self.offset+self.rate*(x-self.time*.3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--dimension',type=int,default=16)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--repeats',type=int,default=128)
    p.add_argument('--couplings',nargs='+',choices=['independent','paired','paired-loo'],default=['independent','paired','paired-loo'])
    args=p.parse_args()
    output=new_run_path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    sigma,horizon=.3,.1
    rows=[];gradients={}
    for time in [.2,.5,.8]:
        variance=(1-time)**2+(time*sigma)**2
        next_variance=(1-time-horizon)**2+((time+horizon)*sigma)**2
        rate=(math.sqrt(next_variance/variance)-1)/horizon
        assert math.isclose((1+horizon*rate)**2*variance,next_variance,rel_tol=1e-12)
        for perturbation,offset,rate_shift in [('population_correct',0.,0.),('mean_error',.1,0.),('scale_error',0.,.25)]:
            for coupling in args.couplings:
                values=[]
                for repeat in range(args.repeats):
                    generator=torch.Generator().manual_seed(1234+repeat)
                    noise=torch.randn(args.batch,args.dimension,generator=generator)
                    clean=.3+sigma*torch.randn(args.batch,args.dimension,generator=generator)
                    target_noise=torch.randn(args.batch,args.dimension,generator=generator)
                    target=.3+sigma*torch.randn(args.batch,args.dimension,generator=generator)
                    if coupling!='independent': target,target_noise=clean,noise
                    model=GaussianVelocity(time,rate+rate_shift,offset)
                    loss=marginal_loss(model,clean,noise,target,target_noise,time,horizon,
                        affinity='row-column',taus=[.2,.05,.02],normalization='temperature',
                        exclude_target_diagonal=coupling=='paired-loo')
                    loss.backward()
                    values.append([float(model.offset.grad),float(model.rate.grad)])
                values=np.asarray(values)
                key=f't{time}_{perturbation}_{coupling}'
                gradients[key]=values
                mean=values.mean(0);std=values.std(0,ddof=1)
                row=dict(time=time,perturbation=perturbation,coupling=coupling,exact_rate=rate,
                    gradient_mean=mean.tolist(),gradient_std=std.tolist(),
                    gradient_mean_standard_error=(std/math.sqrt(args.repeats)).tolist())
                rows.append(row)
                print(json.dumps(row),flush=True)
    np.savez_compressed(output/'gradients.npz',**gradients)
    files=['small_flow.py','backbone.py','loss.py','sinkhorn.py','run_storage.py',str(Path(__file__).relative_to(ROOT))]
    files += [str(p.relative_to(ROOT)) for p in ROOT.glob('*LICENSE*')]
    hashes={}
    for name in files:
        destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
        hashes[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    report=dict(config=vars(args),data_mean=.3,data_std=sigma,horizon=horizon,rows=rows,
        gradient_coordinates=['constant_velocity_offset','radial_velocity_rate'],
        exact_reference='For Gaussian noise and independent Gaussian data, v(x)=mu+a*(x-t*mu), a=(sqrt(variance(t+h)/variance(t))-1)/h maps the current marginal exactly to the next one under one Euler step.',
        interpretation='No model training. Population-correct transport is an analytic diagnostic, not an FM checkpoint. Nonzero mean gradient there measures finite-particle surrogate bias. Gradient variability is across common-noise Monte Carlo trials. Gaussian findings do not prove image-model behavior.',
        source_hashes=hashes,runtime=dict(torch=torch.__version__,numpy=np.__version__))
    (output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
