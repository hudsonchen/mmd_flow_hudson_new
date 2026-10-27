"""Train and evaluate one small reference-flow experiment on an allocated GPU."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from functools import lru_cache

# The cluster Python executable can resolve to the OS prefix after maintenance.
# Append shared packages after stdlib (PYTHONPATH would shadow dataclasses).
if Path('/appl/python-3.12/lib/python3.12/site-packages').is_dir():
    sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DRIFT_COMPILE'] = '0'
import numpy as np
import torch
from small_flow import ToyVelocity, PotentialVelocity, DenoiserVelocity, sample, riesz_loss, marginal_loss, validate_riesz_parent, coupled_reference_loss, reference_centered_loss


def write_json(path, value):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def centers(device):
    angle = torch.arange(8, device=device)*math.pi/4
    return 2*torch.stack([angle.cos(), angle.sin()], 1)


def draw(n, device, generator=None):
    return centers(device)[torch.randint(8, (n,), device=device, generator=generator)] + .12*torch.randn(n, 2, device=device, generator=generator)


@lru_cache(maxsize=4)
def analytic_projection_quantiles(count):
    """Exact-mixture projection quantiles; exploit the ring's eightfold symmetry."""
    device='cuda' if torch.cuda.is_available() else 'cpu'
    angles=torch.arange(17,device=device)*math.pi/128
    directions=torch.stack([angles.cos(),angles.sin()],1)
    means=(centers(device)@directions.T).T
    probability=(torch.arange(count,device=device)+.5)/count
    low=torch.full((count,17),-4.,device=device)
    high=torch.full_like(low,4.)
    for _ in range(27):
        middle=(low+high)/2
        cdf=(.5+.5*torch.erf((middle[:,:,None]-means[None])/(.12*math.sqrt(2)))).mean(-1)
        low=torch.where(cdf<probability[:,None],middle,low)
        high=torch.where(cdf>=probability[:,None],middle,high)
    index=torch.arange(128,device=device)%32
    index=torch.minimum(index,32-index)
    return ((low+high)/2)[:,index].cpu()


@torch.no_grad()
def analytic_sliced_w2(points):
    x=points.cpu()
    angles=torch.arange(128)*math.pi/128
    directions=torch.stack([angles.cos(),angles.sin()],1)
    projected=(x@directions.T).sort(0).values
    return float((projected-analytic_projection_quantiles(len(x))).square().mean().sqrt())


@torch.no_grad()
def metrics(points, seed=9834):
    x = points.cpu()
    d, mode = torch.cdist(x, centers('cpu')).min(1)
    mass = torch.bincount(mode, minlength=8)/len(x)
    target = draw(len(x), 'cpu', torch.Generator().manual_seed(seed))
    angles = torch.arange(128)*math.pi/128
    directions = torch.stack([angles.cos(), angles.sin()], 1)
    sw = ((x@directions.T).sort(0).values-(target@directions.T).sort(0).values).square().mean().sqrt()
    # Include component spread: mode-center collapse must not pass.
    residual = x-centers('cpu')[mode]
    radius=d.sort().values
    cdf=1-torch.exp(-radius.square()/(2*.12**2))
    n=len(radius)
    radial_ks=torch.maximum((torch.arange(1,n+1)/n-cdf).max(),(cdf-torch.arange(n)/n).max())
    return dict(off_mode_fraction=float((d>.36).float().mean()), covered_modes=int((mass>.02).sum()),
        mode_mass_l1=float((mass-.125).abs().sum()), nearest_center_rms=float(d.square().mean().sqrt()),
        component_coordinate_std=float(residual.std(0).mean()), sliced_w2=float(sw),radial_ks=float(radial_ks),
        within_mode_coordinate_std=float((d[d<.36].square().mean()/2).sqrt()))


@torch.no_grad()
def evaluate(model, args, output, step):
    generator = torch.Generator(device='cuda').manual_seed(100000+args.seed)
    noise = torch.randn(args.eval_samples, 2, generator=generator, device='cuda')
    points = sample(model, noise, args.sample_steps, args.solver)
    if args.sample_noise:
        points=points+args.sample_noise*torch.randn(points.shape,device='cuda',generator=generator)
    row = dict(update=step, **metrics(points))
    np.save(output/f'samples_{step:07d}.npy', points.cpu().numpy())
    write_json(output/f'metrics_{step:07d}.json', row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objective', choices=['fm','marginal','endpoint','anchored','path-endpoint','coupled','reference-centered','reference-hybrid'], default='fm')
    p.add_argument('--reference-weight',type=float,default=1.)
    p.add_argument('--affinity',choices=['row','row-column'],default='row')
    p.add_argument('--parameterization',choices=['velocity','denoiser'],default='velocity')
    p.add_argument('--paired-reference-supports',action='store_true')
    p.add_argument('--reference-coupling',choices=['independent','paired','paired-loo'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--target-batch', type=int)
    p.add_argument('--independent-negatives', action='store_true')
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--eval-samples', type=int, default=10000)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--solver', choices=['euler','heun'], default='heun')
    p.add_argument('--sample-noise',type=float,default=0.)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--spatial-fourier', action='store_true')
    p.add_argument('--potential', action='store_true')
    p.add_argument('--kinetic-weight',type=float,default=0.)
    p.add_argument('--fork', type=Path)
    p.add_argument('--tau', type=float, nargs='+', default=[.1])
    p.add_argument('--normalization', choices=['none','temperature','global'], default='none')
    p.add_argument('--repulsion-weight',type=float,default=1.)
    p.add_argument('--endpoint-tail-scale',type=float,default=0.,help='Positive normalized nearest-target distance scale enables bounded query weighting in endpoint Riesz loss')
    p.add_argument('--endpoint-tail-max-weight',type=float,default=4.)
    p.add_argument('--horizon', type=float, default=.1)
    p.add_argument('--anchor-weight', type=float, default=1.)
    p.add_argument('--riesz-weight', type=float, default=.1)
    p.add_argument('--marginal-weight', type=float, default=.1)
    p.add_argument('--train-rollout-steps', type=int, default=10)
    args = p.parse_args()
    if not math.isfinite(args.endpoint_tail_scale) or args.endpoint_tail_scale<0 or not math.isfinite(args.endpoint_tail_max_weight) or args.endpoint_tail_max_weight<1:
        p.error('Invalid endpoint-tail weighting settings')
    if args.endpoint_tail_scale and (args.affinity!='row-column' or args.objective not in ['endpoint','path-endpoint','anchored','reference-hybrid']):
        p.error('Endpoint-tail weighting requires a row-column endpoint Riesz objective')
    args.reference_coupling=args.reference_coupling or ('paired' if args.objective=='coupled' else 'independent')
    if args.objective=='coupled' and args.reference_coupling=='independent':
        p.error('The coupled objective requires paired reference particles')
    if args.reference_coupling!='independent' and args.objective not in ['coupled','path-endpoint']:
        p.error('Reference coupling requires a coupled or path-endpoint objective')
    if args.reference_coupling=='paired-loo' and args.affinity!='row-column':
        p.error('Matched-target exclusion requires row-column affinity')
    if args.parameterization=='denoiser' and args.solver!='euler':
        p.error('Denoiser parameterization requires Euler to avoid the singular t=1 velocity')
    if args.objective=='reference-centered' and args.repulsion_weight!=1:
        p.error('Reference control variate requires repulsion weight one')
    if args.objective=='reference-centered' and args.affinity!='row':
        p.error('Reference control variate currently requires row affinity')
    target_batch=args.target_batch or 2*args.batch
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    # Execute kernels, including backward, before committing to training.
    probe = torch.randn(32, 32, device='cuda', requires_grad=True)
    (probe@probe.T).square().mean().backward()
    torch.cuda.synchronize()
    write_json(output/'hardware.json', dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        torch=torch.__version__, cuda=torch.version.cuda, backward_probe_passed=True))
    model = (PotentialVelocity if args.potential else ToyVelocity)(args.width,args.spatial_fourier).cuda()
    if args.parameterization=='denoiser': model=DenoiserVelocity(model)
    if args.fork:
        parent = torch.load(args.fork, map_location='cuda', weights_only=False)
        if args.objective!='fm': validate_riesz_parent(parent)
        model.load_state_dict(parent['ema'])
    ema = copy.deepcopy(model).requires_grad_(False)
    reference = copy.deepcopy(model).requires_grad_(False) if args.fork else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.monotonic()
    best = float('inf')
    for step in range(1, args.steps+1):
        clean = draw(args.batch, 'cuda')
        noise = torch.randn_like(clean)
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        optimizer.zero_grad(set_to_none=True)
        if args.objective == 'fm':
            t = torch.rand(args.batch, 1, device='cuda')
            x = (1-t)*noise+t*clean
            loss = (model(x, t[:,0])-(clean-noise)).square().mean()
        elif args.objective=='reference-centered':
            t=float(torch.rand(()))*.999
            positive=draw(target_batch,'cuda')
            positive_noise=torch.randn_like(positive)
            negative=positive if args.paired_reference_supports else draw(target_batch,'cuda')
            negative_noise=positive_noise if args.paired_reference_supports else torch.randn_like(negative)
            loss=reference_centered_loss(model,clean,noise,positive,positive_noise,negative,
                negative_noise,t,args.horizon,taus=args.tau,normalization=args.normalization)
        elif args.objective == 'coupled':
            t=float(torch.rand(()))*.999
            loss=coupled_reference_loss(model,clean,noise,t,args.horizon,
                taus=args.tau,normalization=args.normalization,repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                exclude_target_diagonal=args.reference_coupling=='paired-loo')
        elif args.objective == 'marginal':
            t = float(torch.rand(()))*.999
            loss = marginal_loss(model, clean, noise, draw(target_batch,'cuda'),
                torch.randn(target_batch,2,device='cuda'),t,args.horizon,
                taus=args.tau, normalization=args.normalization,repulsion_weight=args.repulsion_weight,affinity=args.affinity)
        else:
            prediction = sample(model, noise, args.train_rollout_steps, args.solver)
            if args.sample_noise: prediction=prediction+args.sample_noise*torch.randn_like(prediction)
            negative=None
            if args.independent_negatives:
                with torch.no_grad():
                    negative=sample(model,torch.randn(target_batch,2,device='cuda'),args.train_rollout_steps,args.solver)
                    if args.sample_noise: negative=negative+args.sample_noise*torch.randn_like(negative)
            loss = args.riesz_weight*riesz_loss(prediction, draw(target_batch,'cuda'),
                taus=args.tau, normalization=args.normalization,negative=negative,repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                query_tail_scale=args.endpoint_tail_scale,query_tail_max_weight=args.endpoint_tail_max_weight)[0]
            if args.objective=='reference-hybrid':
                # Joint analytic path supervision from the first update. No FM
                # checkpoint or teacher is used; report this as a hybrid loss.
                t=torch.rand(args.batch,1,device='cuda')
                loss=loss+args.reference_weight*(model((1-t)*noise+t*clean,t[:,0])-(clean-noise)).square().mean()
            if args.objective=='path-endpoint':
                t=float(torch.rand(()))*.999
                reference_target,reference_noise=((clean,noise) if args.reference_coupling!='independent' else
                    (draw(target_batch,'cuda'),torch.randn(target_batch,2,device='cuda')))
                loss=loss+args.marginal_weight*marginal_loss(model,clean,noise,reference_target,reference_noise,t,args.horizon,
                    taus=args.tau,normalization=args.normalization,repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                    exclude_target_diagonal=args.reference_coupling=='paired-loo')
            if args.objective == 'anchored':
                if reference is None:
                    raise ValueError('Anchored training requires --fork')
                t = torch.rand(args.batch, 1, device='cuda')
                x = (1-t)*noise+t*clean
                with torch.no_grad():
                    velocity = reference(x,t[:,0])
                loss = loss+args.anchor_weight*(model(x,t[:,0])-velocity).square().mean()
        if args.kinetic_weight:
            time_batch=torch.rand(args.batch,1,device='cuda')
            state=(1-time_batch)*noise+time_batch*clean
            loss=loss+args.kinetic_weight*model(state,time_batch[:,0]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(), model.parameters()):
                a.lerp_(b, .001 if step>1000 else .01)
        if step%100==0:
            write_json(output/'status.json', dict(status='running',update=step,loss=float(loss),grad=float(grad),seconds=time.monotonic()-start))
        if step%args.eval_every==0 or step==args.steps:
            row = evaluate(ema,args,output,step)
            row['seconds'] = time.monotonic()-start
            checkpoint = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                lineage_has_flow_matching=args.objective=='fm',
                update=step, config=vars(args), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
            torch.save(checkpoint, output/'latest.pt.tmp')
            (output/'latest.pt.tmp').replace(output/'latest.pt')
            if row['off_mode_fraction'] < best:
                best = row['off_mode_fraction']
                torch.save(checkpoint, output/'best.pt.tmp')
                (output/'best.pt.tmp').replace(output/'best.pt')
                write_json(output/'best.json',row)
            print(json.dumps(row),flush=True)
    write_json(output/'final.json',row)
    write_json(output/'status.json',dict(status='complete',**row))


if __name__ == '__main__':
    main()
