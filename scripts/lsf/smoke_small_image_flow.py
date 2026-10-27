"""GPU memory/runtime preflight; no optimizer updates or trained image models."""
import argparse
import copy
import os
from pathlib import Path
import sys
import time
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import torch
from small_flow import ImageVelocity,DenoiserVelocity,sample,image_riesz_loss,marginal_loss,image_geometry,vgg_endpoint_riesz_loss,local_reference_hybrid_loss
from scripts.lsf.run_small_image_flow import load_images
from scripts.lsf.run_small_flow import write_json
from reference_dit import image_base_from_config,initialize_from_reconstruction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=['mnist','fashionmnist','cifar10'],default='mnist')
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--architecture',choices=['unet','dit'],default='unet')
    p.add_argument('--dit-width',type=int,default=384)
    p.add_argument('--dit-depth',type=int,default=12)
    p.add_argument('--dit-heads',type=int,default=6)
    p.add_argument('--dit-patch-size',type=int,default=2)
    p.add_argument('--dit-prime-checkpoint',type=Path)
    p.add_argument('--deep',action='store_true')
    p.add_argument('--spatial-fourier',action='store_true')
    p.add_argument('--target-batch',type=int)
    p.add_argument('--parameterization',choices=['velocity','denoiser'],default='velocity')
    p.add_argument('--affinity',choices=['row','row-column'],default='row')
    p.add_argument('--pixel-statistics-weight',type=float,default=0.)
    p.add_argument('--spatial-feature-weight',type=float,default=0.)
    p.add_argument('--spatial-feature-locations',type=int,default=16)
    p.add_argument('--reference-time-power',type=float,default=0.)
    p.add_argument('--local-pixel-scales',type=int,nargs='+')
    p.add_argument('--include-self-negative',action='store_true')
    p.add_argument('--distance-smoothing',type=float,default=0.)
    p.add_argument('--normalization',choices=['none','temperature','global'],default='temperature')
    p.add_argument('--horizon',type=float,default=.1)
    p.add_argument('--checkpointing',choices=['on','off','both'],default='both')
    p.add_argument('--train-rollout-steps',type=int,default=20)
    p.add_argument('--solver',choices=['euler','heun'],default='heun')
    p.add_argument('--feature-space',choices=['pixels','primed'],default='pixels')
    p.add_argument('--vgg-endpoint-weight',type=float,default=0.)
    p.add_argument('--vgg-pool-size',type=int,default=8)
    p.add_argument('--vgg-cache',type=Path)
    p.add_argument('--local-only',action='store_true',help='Benchmark single-transition Riesz objectives; no endpoint rollout')
    p.add_argument('--local-diagnostics',type=int,default=0,help='Independent batches for local gradient comparison with analytic reference-velocity MSE; no optimizer updates')
    p.add_argument('--local-hybrid',action='store_true',help='Include one-call reference-denoising/Riesz hybrid in local benchmark')
    args=p.parse_args()
    if args.local_diagnostics<0 or (args.local_diagnostics and not args.local_only):p.error('Local diagnostics require --local-only and nonnegative repeats')
    if args.local_pixel_scales and not args.local_only:p.error('Pixel-scale preflight requires --local-only')
    if args.local_hybrid and (not args.local_only or args.parameterization!='denoiser'):p.error('Hybrid benchmark requires a local-only denoiser')
    if args.local_only and (args.vgg_endpoint_weight or args.feature_space!='pixels' or args.affinity!='row-column'):
        p.error('Local comparison requires pixel features, row-column affinity, and no VGG endpoint loss')
    shape=image_geometry(args.dataset)
    if args.parameterization=='denoiser' and args.solver!='euler':p.error('Denoiser requires Euler')
    if args.architecture=='dit' and (args.dataset!='cifar10' or args.feature_space!='pixels' or args.spatial_feature_weight):
        p.error('DiT preflight requires CIFAR and pixel/VGG endpoint features')
    if args.dit_prime_checkpoint and (args.architecture!='dit' or args.parameterization!='denoiser'):
        p.error('DiT priming requires the DiT denoiser')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark=True
    train,_=load_images(args.dataset,args.data_root)
    train=train.flatten(1).cuda().float()/127.5-1
    clean=train[:args.batch]
    target=train[args.batch:args.batch+(args.target_batch or 2*args.batch)]
    noise=torch.randn_like(clean)
    base=image_base_from_config(vars(args)).cuda()
    priming=initialize_from_reconstruction(base,args.dit_prime_checkpoint) if args.dit_prime_checkpoint else None
    model=DenoiserVelocity(base) if args.parameterization=='denoiser' else base
    # Random frozen features benchmark memory/compute only, never model quality.
    encoder=copy.deepcopy(model).requires_grad_(False) if args.feature_space=='primed' else None
    vgg=None
    if args.vgg_endpoint_weight:
        if args.dataset!='cifar10' or args.vgg_cache is None:p.error('VGG preflight requires CIFAR and a cached encoder')
        from perceptual import VGGPerceptualLoss
        vgg=VGGPerceptualLoss(args.vgg_cache,'cuda').requires_grad_(False).eval()
    rows=[]
    settings=([] if args.architecture=='dit' and args.parameterization=='denoiser' else [('fm',False)])
    settings += [('path-endpoint',v) for v in ([True,False] if args.checkpointing=='both' else [args.checkpointing=='on'])]
    if args.local_only:settings=[(name,False) for name in ['local-independent','local-paired','local-paired-loo']]
    if args.include_self_negative or args.distance_smoothing:
        if not args.local_only:p.error('Self normalization/smoothing preflight requires --local-only')
        settings=[('local-paired',False)]
    if args.local_hybrid:settings.append(('local-hybrid',False))
    for objective,checkpointing in settings:
        times=[]
        for repeat in range(3):
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.monotonic()
            if objective=='local-hybrid':
                loss,_=local_reference_hybrid_loss(model,clean,noise,clean,noise,.5,.1,
                    reference_weight=1.,riesz_weight=.1,time_weight_power=args.reference_time_power,
                    taus=[.2,.05,.02],normalization='temperature',affinity='row-column',exclude_target_diagonal=True)
            elif objective.startswith('local-'):
                paired=objective!='local-independent'
                local_target=clean if paired else target
                local_noise=noise if paired else torch.randn_like(local_target)
                loss=marginal_loss(model,clean,noise,local_target,local_noise,.5,args.horizon,
                    include_self_negative=args.include_self_negative,distance_smoothing=args.distance_smoothing,
                    pixel_scales=args.local_pixel_scales,image_shape=shape,
                    taus=[.2,.05,.02],normalization=args.normalization,affinity=args.affinity,
                    time_weight_power=args.reference_time_power,exclude_target_diagonal=objective=='local-paired-loo')
            elif objective=='fm':
                loss=(base(.5*(clean+noise),.5)-(clean-noise)).square().mean()
            else:
                pred=sample(model,noise,args.train_rollout_steps,args.solver,checkpoint_steps=checkpointing)
                loss=image_riesz_loss(pred,target,encoder=encoder,taus=[.2,.05,.02],normalization='temperature',
                    affinity=args.affinity,statistics_weight=args.pixel_statistics_weight,
                    spatial_feature_weight=args.spatial_feature_weight,spatial_feature_locations=args.spatial_feature_locations,
                    image_shape=shape)
                if vgg is not None:
                    vgg_loss=vgg_endpoint_riesz_loss(pred,target,vgg,pool_size=args.vgg_pool_size,
                        taus=[.2,.05,.02],normalization='temperature',affinity=args.affinity)
                    loss=(loss+args.vgg_endpoint_weight*vgg_loss)/(1+args.vgg_endpoint_weight)
                loss=loss+.1*marginal_loss(model,clean,noise,target,torch.randn_like(target),.5,.1,
                    taus=[.2,.05,.02],normalization='temperature',affinity=args.affinity,time_weight_power=args.reference_time_power)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if vgg is not None and any(p.grad is not None for p in vgg.parameters()):
                raise RuntimeError('VGG weights received gradients')
            torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            if repeat:times.append(elapsed)
        row=dict(objective=objective,checkpointing=checkpointing,seconds_per_update=sum(times)/len(times),
                 peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,gradient_norm=float(grad))
        rows.append(row)
        print(row,flush=True)
    previews=[]
    if args.architecture=='dit':
        from torchvision.utils import save_image
        with torch.no_grad():
            for steps in [1,5,10,20]:
                raw=sample(model,noise[:32],steps,args.solver)
                if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite primed DiT preview')
                save_image(((raw+1)/2).clamp(0,1).reshape(-1,*shape),str(args.output/f'initial_euler{steps}.png'),nrow=8)
                previews.append(dict(steps=steps,variance=float(raw.var(0).mean()),
                    raw_pixel_outside_fraction=float((raw.abs()>1).float().mean())))
    diagnostics=[]
    if args.local_diagnostics:
        parameters=tuple(model.parameters())
        def gradient(value):
            values=torch.autograd.grad(value,parameters)
            result=torch.cat([v.flatten() for v in values]).detach()
            if not torch.isfinite(result).all():raise RuntimeError('Nonfinite local diagnostic gradient')
            return result
        for repeat in range(args.local_diagnostics):
            torch.manual_seed(20261027+repeat)
            clean=train[torch.randint(len(train),(args.batch,),device='cuda')]
            target=train[torch.randint(len(train),(args.batch,),device='cuda')]
            noise=torch.randn_like(clean);target_noise=torch.randn_like(target)
            for t in [.01,.2,.5,.9,.99]:
                source=(1-t)*noise+t*clean
                # Analytic loss direction is measured only: no FM teacher, weights or updates.
                fm_gradient=gradient((model(source,t)-(clean-noise)).square().mean())
                for name in ['independent','paired','paired-loo']:
                    paired=name!='independent'
                    value=marginal_loss(model,clean,noise,clean if paired else target,
                        noise if paired else target_noise,t,args.horizon,
                        pixel_scales=args.local_pixel_scales,image_shape=shape,
                        taus=[.2,.05,.02],normalization='temperature',affinity='row-column',
                        time_weight_power=args.reference_time_power,exclude_target_diagonal=name=='paired-loo')
                    local_gradient=gradient(value)
                    diagnostics.append(dict(repeat=repeat,time=t,coupling=name,
                        gradient_norm=float(local_gradient.norm()),analytic_reference_mse_gradient_norm=float(fm_gradient.norm()),
                        cosine_with_analytic_reference_mse=float(torch.dot(local_gradient,fm_gradient)/
                            (local_gradient.norm()*fm_gradient.norm()).clamp_min(1e-12))))
            print(dict(local_diagnostic_repeat=repeat),flush=True)
    write_json(args.output/'smoke.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),optimizer_updates=0,batch=args.batch,
        width=args.width,rollout_steps=args.train_rollout_steps,rows=rows,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        image_shape=shape,resident_training_images=len(train),cuda=torch.version.cuda,
        priming=priming,initial_previews=previews,vgg_metadata=vgg.metadata if vgg is not None else None,
        local_gradient_diagnostics=diagnostics,
        diagnostic_caveat='Analytic velocity-MSE gradients are diagnostic only; no teacher or optimizer updates. Alignment is not evidence of FID quality.'))


if __name__=='__main__': main()
