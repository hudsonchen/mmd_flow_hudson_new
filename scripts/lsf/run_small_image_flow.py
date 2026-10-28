"""MNIST, FashionMNIST and CIFAR-10 reference flows with matched FID evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import numpy as np
import torch
import torch.nn.functional as F
from small_flow import ImageVelocity, DenoiserVelocity, sample, marginal_loss, riesz_loss, validate_riesz_parent, coupled_reference_loss, image_riesz_loss, reference_centered_loss
from small_flow import restore_image_checkpoint,verify_reconstruction_encoder,image_geometry,vgg_endpoint_riesz_loss,local_reference_hybrid_loss
from reference_dit import image_base_from_config,initialize_from_reconstruction
from scripts.lsf.run_small_flow import write_json
from fid import CIFAR10FID, feature_statistics


def load_images(dataset, root):
    if dataset=='cifar10':
        from data import download_cifar10,load_cifar10_split
        archive=download_cifar10(Path(root))
        return load_cifar10_split(archive,True)[0],load_cifar10_split(archive,False)[0]
    image_geometry(dataset)
    from torchvision.datasets import MNIST, FashionMNIST
    cls = MNIST if dataset=='mnist' else FashionMNIST
    if dataset=='fashionmnist':
        cls.mirrors=['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/']
    train,test = cls(str(root),train=True,download=True),cls(str(root),train=False,download=True)
    return train.data,test.data


def rgb32(images,image_shape=(1,28,28)):
    x = images.reshape(-1,*image_shape).float()
    if images.dtype==torch.uint8:
        x=x/255
    if image_shape==(3,32,32):return x.contiguous()
    return F.interpolate(x,size=(32,32),mode='bilinear',align_corners=False).expand(-1,3,-1,-1).contiguous()


@torch.no_grad()
def save_training_preview(model,output,update,args):
    """Fixed independent noise for early sampler diagnostics; never an FID estimate."""
    shape=image_geometry(args.dataset)
    generator=torch.Generator(device='cuda').manual_seed(8765432)
    noise=torch.randn(32,math.prod(shape),device='cuda',generator=generator)
    raw=sample(model,noise,args.sample_steps,args.solver)
    if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite training preview')
    from torchvision.utils import save_image
    save_image(((raw+1)/2).clamp(0,1).reshape(-1,*shape),
        str(output/f'preview_{update:07d}.png'),nrow=8)
    write_json(output/f'preview_{update:07d}.json',dict(update=update,count=32,seed=8765432,
        device='cuda',solver=args.solver,steps=args.sample_steps,
        raw_pixel_variance=float(raw.var(0).mean()),
        near_boundary_pixel_fraction=float((raw.abs()>.98).float().mean()),
        raw_pixel_outside_fraction=float((raw.abs()>1).float().mean()),
        evaluation='EMA sampling diagnostic only; not FID or a diversity guarantee'))


class ImageEvaluation:
    def __init__(self, real, test, args):
        self.args=args
        self.image_shape=image_geometry(getattr(args,'dataset','mnist'))
        count=getattr(args,'fid_reference_samples',None)
        if count is None:count=len(real) if self.image_shape[0]==3 else args.eval_samples
        index=torch.randperm(len(real),generator=torch.Generator().manual_seed(3456))[:count]
        self.fid=CIFAR10FID(rgb32(real[index],self.image_shape),device='cuda',batch_size=getattr(args,'fid_batch_size',64))
        self.test=test

    @torch.no_grad()
    def run(self, model, output, update, final=False, decoder=None):
        args=self.args
        generator=torch.Generator(device='cuda').manual_seed(100000+args.seed)
        raw=[]
        for start in range(0,args.eval_samples,128):
            noise=torch.randn(min(128,args.eval_samples-start),math.prod(self.image_shape),device='cuda',generator=generator)
            generated=sample(model,noise,args.sample_steps,args.solver)
            if decoder is not None:
                generated=getattr(decoder,'reconstruct',decoder)(generated,1.)
            raw.append(generated.cpu())
        raw=torch.cat(raw)
        pixels=((raw+1)/2).clamp(0,1)
        stats=self.fid.statistics(rgb32(pixels,self.image_shape))
        row=dict(update=update,fid=self.fid.score(stats),generated_count=len(pixels),
                 reference_count=self.fid.reference[2],raw_pixel_outside_fraction=float((raw.abs()>1).float().mean()))
        if final:
            test_stats=self.fid.statistics(rgb32(self.test[:args.eval_samples],self.image_shape))
            from fid import frechet_distance
            row['test_fid']=frechet_distance(*test_stats[:2],*stats[:2])
        from torchvision.utils import save_image
        save_image(pixels[:100].reshape(-1,*self.image_shape),str(output/f'samples_{update:07d}.png'),nrow=10)
        np.save(output/f'samples_{update:07d}.npy',(pixels.reshape(-1,*self.image_shape).numpy()*255).round().astype(np.uint8))
        write_json(output/f'metrics_{update:07d}.json',row)
        return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=['mnist','fashionmnist','cifar10'],required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--objective',choices=['fm','marginal','anchored-local','anchored-endpoint','endpoint','path-endpoint','coupled','reference-centered','reference-hybrid','local-hybrid'],default='fm')
    p.add_argument('--reference-weight',type=float,default=1.)
    p.add_argument('--affinity',choices=['row','row-column'],default='row')
    p.add_argument('--parameterization',choices=['velocity','denoiser'],default='velocity')
    p.add_argument('--endpoint-batch',type=int)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--paired-reference-supports',action='store_true')
    p.add_argument('--reference-coupling',choices=['independent','paired','paired-loo'],
                   help='Particle coupling for the path-endpoint next-marginal Riesz loss')
    p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--lr',type=float,default=.0002)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--target-batch',type=int)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--architecture',choices=['unet','dit','ddpm-unet'],default='unet')
    p.add_argument('--dit-width',type=int,default=384)
    p.add_argument('--dit-depth',type=int,default=12)
    p.add_argument('--dit-heads',type=int,default=6)
    p.add_argument('--dit-patch-size',type=int,default=2)
    p.add_argument('--dit-prime-checkpoint',type=Path)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--spatial-fourier',action='store_true')
    p.add_argument('--deep',action='store_true')
    p.add_argument('--random-flip',action='store_true',help='Independent horizontal flips for training images and reconstruction priming')
    p.add_argument('--prime-steps',type=int,default=0)
    p.add_argument('--prime-mask-fraction',type=float,default=0.)
    p.add_argument('--reprime-features',action='store_true',help='Train a fresh reconstruction encoder, then restore only the Riesz generator EMA from --fork')
    p.add_argument('--feature-space',choices=['pixels','primed'],default='pixels')
    p.add_argument('--vgg-endpoint-weight',type=float,default=0.,help='Relative weight of multiscale VGG endpoint Riesz; leaves the reference loss in pixel space')
    p.add_argument('--vgg-pool-size',type=int,default=8)
    p.add_argument('--vgg-cache',type=Path)
    p.add_argument('--decode-primed',action='store_true',help='Train and sample through the frozen reconstruction decoder')
    p.add_argument('--pixel-statistics-weight',type=float,default=0.)
    p.add_argument('--pixel-gradient-weight',type=float,default=0.,help='Riesz matching of signed spatial-derivative features')
    p.add_argument('--spatial-feature-weight',type=float,default=0.,help='Per-location and global-statistic Riesz fields in the frozen encoder')
    p.add_argument('--spatial-feature-locations',type=int,default=16)
    p.add_argument('--reference-whitening-floor',type=float,default=0.,help='Positive floor enables per-pixel reference standard-deviation geometry')
    p.add_argument('--eval-every',type=int,default=5000)
    p.add_argument('--preview-every',type=int,default=0,help='Optional 32-image EMA diagnostics, including update zero; no extra FID')
    p.add_argument('--eval-samples',type=int,default=10000)
    p.add_argument('--fid-reference-samples',type=int,help='Default: all CIFAR images in the reference split, or eval-samples for grayscale datasets')
    p.add_argument('--fid-batch-size',type=int,default=64)
    p.add_argument('--sample-steps',type=int,default=20)
    p.add_argument('--solver',choices=['euler','heun'],default='heun')
    p.add_argument('--train-rollout-steps',type=int,default=20)
    p.add_argument('--rollout-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--fork',type=Path)
    p.add_argument('--tau',type=float,nargs='+',default=[.1])
    p.add_argument('--normalization',choices=['none','temperature','global'],default='none')
    p.add_argument('--repulsion-weight',type=float,default=1.)
    p.add_argument('--horizon',type=float,default=.1)
    p.add_argument('--reference-time-power',type=float,default=0.,help='Multiply local Riesz loss by (1-t)^power; power1 cancels the denoiser Jacobian singularity')
    p.add_argument('--local-pixel-scales',type=int,nargs='+',help='Average local Riesz losses on fixed pooled pixel resolutions; no learned encoder')
    p.add_argument('--include-self-negative',action='store_true',help='Include zero-displacement self negatives in local paired Riesz affinity normalization')
    p.add_argument('--distance-powers',type=float,nargs='+',default=[1.],help='Sum Riesz kernel forces for powers p in (r/scale)^p; local row-column loss only')
    p.add_argument('--distance-smoothing',type=float,default=0.,help='Local paired Riesz smooth-norm radius relative to mean batch distance')
    p.add_argument('--anchor-weight',type=float,default=1.)
    p.add_argument('--riesz-weight',type=float,default=.1)
    p.add_argument('--marginal-weight',type=float,default=.1)
    args=p.parse_args()
    if any(not math.isfinite(p) or not 0<p<=2 for p in args.distance_powers) or len(set(args.distance_powers))!=len(args.distance_powers):
        p.error('--distance-powers requires distinct finite powers in (0, 2]')
    if args.distance_powers!=[1.] and (args.objective!='coupled' or args.affinity!='row-column'):
        p.error('Multiple/nonunit distance powers require --objective coupled --affinity row-column')
    if args.preview_every<0:p.error('--preview-every must be nonnegative')
    if args.preview_every and args.decode_primed:p.error('Training previews currently require direct image sampling')
    args.image_channels,args.image_size,_=image_geometry(args.dataset)
    if args.local_pixel_scales is not None:
        if (args.objective not in ['marginal','coupled','path-endpoint']
                or len(set(args.local_pixel_scales))!=len(args.local_pixel_scales)
                or min(args.local_pixel_scales)<1 or max(args.local_pixel_scales)>args.image_size):
            p.error('Local pixel scales require a marginal, coupled, or path-endpoint loss and unique valid image resolutions')
    if args.objective=='local-hybrid':
        if args.parameterization!='denoiser' or args.feature_space!='pixels' or args.decode_primed or args.pixel_statistics_weight:
            p.error('Local hybrid requires direct denoiser sampling and pixel features')
        if any(not math.isfinite(w) or w<=0 for w in [args.reference_weight,args.riesz_weight]):
            p.error('Both hybrid loss weights must be positive')
        args.training_rule='joint_paired_reference_denoising_mse_and_local_riesz; one_network_call; no_rollout_loss'
    if args.threads<1:p.error('Threads must be positive')
    if args.architecture=='dit' and (args.dataset!='cifar10' or args.feature_space!='pixels' or args.spatial_feature_weight or args.spatial_fourier or args.decode_primed):
        p.error('DiT requires CIFAR and pixel/VGG endpoint features; U-Net reconstruction features and decoder are unsupported')
    if args.architecture=='dit' and args.prime_steps:
        p.error('Use an audited --dit-prime-checkpoint for DiT reconstruction initialization')
    if args.dit_prime_checkpoint and (args.architecture!='dit' or args.parameterization!='denoiser' or args.objective=='fm' or args.fork or args.prime_steps):
        p.error('DiT priming source requires a fresh Riesz denoiser stage, without a fork or additional priming')
    if not math.isfinite(args.vgg_endpoint_weight) or args.vgg_endpoint_weight<0 or args.vgg_pool_size<1:
        p.error('Invalid VGG weight or pooling size')
    if args.vgg_endpoint_weight and (args.dataset!='cifar10' or args.vgg_cache is None or args.objective not in ['endpoint','path-endpoint','anchored-endpoint','reference-hybrid']):
        p.error('VGG features require CIFAR, an explicit cache, and an endpoint objective')
    if args.fid_batch_size<1 or (args.fid_reference_samples is not None and args.fid_reference_samples<2):
        p.error('Invalid FID batch or reference sample count')
    if args.random_flip and args.reference_whitening_floor:
        p.error('Reference whitening with augmented image statistics is not implemented')
    if args.reprime_features and (not args.fork or args.prime_steps<1 or args.feature_space!='primed' or args.objective not in ['endpoint','path-endpoint','anchored-endpoint','reference-hybrid']):
        p.error('Repriming features requires a fork, positive priming steps, primed features and a Riesz endpoint objective')
    if args.prime_steps and args.fork and not args.reprime_features:
        p.error('Combining priming with a fork requires explicit --reprime-features')
    args.reference_coupling=args.reference_coupling or ('paired' if args.objective=='coupled' else 'independent')
    if not math.isfinite(args.distance_smoothing) or args.distance_smoothing<0:
        p.error('Distance smoothing must be finite and nonnegative')
    if args.include_self_negative or args.distance_smoothing:
        if (args.objective!='coupled' or args.reference_coupling!='paired' or args.affinity!='row-column'
                or args.repulsion_weight!=1 or args.feature_space!='pixels'):
            p.error('Self normalization/smoothing trials require paired coupled pixel Riesz with row-column affinities and unit repulsion')
        args.training_rule='paired_pixel_riesz_with_direct_matched_positive; no_velocity_mse'
    if args.objective=='coupled' and args.reference_coupling=='independent':
        p.error('The coupled objective requires paired reference particles')
    if args.reference_coupling!='independent' and args.objective not in ['coupled','path-endpoint','local-hybrid']:
        p.error('Reference coupling requires a coupled or path-endpoint objective')
    if args.reference_coupling=='paired-loo' and args.affinity!='row-column':
        p.error('Matched-target exclusion requires row-column affinity')
    if not math.isfinite(args.reference_time_power) or args.reference_time_power<0:
        p.error('Reference time power must be finite and nonnegative')
    if args.reference_time_power and args.objective not in ['marginal','coupled','path-endpoint','anchored-local','local-hybrid']:
        p.error('Reference time weighting requires a local marginal Riesz objective')
    if not math.isfinite(args.pixel_statistics_weight) or args.pixel_statistics_weight<0:
        p.error('Pixel statistics weight must be finite and nonnegative')
    if not math.isfinite(args.pixel_gradient_weight) or args.pixel_gradient_weight<0:
        p.error('Pixel gradient weight must be finite and nonnegative')
    if args.pixel_gradient_weight and args.objective not in ['endpoint','path-endpoint','anchored-endpoint','reference-hybrid']:
        p.error('Pixel gradient matching requires an endpoint Riesz objective')
    if not math.isfinite(args.spatial_feature_weight) or args.spatial_feature_weight<0 or args.spatial_feature_locations<1:
        p.error('Spatial feature weight must be finite and nonnegative, with positive location count')
    if args.spatial_feature_weight and (args.feature_space!='primed' or args.affinity!='row-column' or args.objective not in ['endpoint','path-endpoint','anchored-endpoint','reference-hybrid']):
        p.error('Spatial feature matching requires primed features, row-column affinities and an endpoint objective')
    if not math.isfinite(args.reference_whitening_floor) or args.reference_whitening_floor<0:
        p.error('Reference whitening floor must be finite and nonnegative')
    if args.reference_whitening_floor and args.objective in ['fm','reference-centered']:
        p.error('Reference whitening is currently supported by marginal and endpoint Riesz objectives')
    if args.parameterization=='denoiser' and args.solver!='euler':
        p.error('Denoiser parameterization requires Euler to avoid the singular t=1 velocity')
    if args.decode_primed and args.feature_space!='primed':
        p.error('Frozen reconstruction decoding requires --feature-space primed')
    if args.decode_primed and args.objective not in ['endpoint','path-endpoint','anchored-endpoint','reference-hybrid']:
        p.error('Training through the reconstruction decoder requires an endpoint objective')
    if not 0<=args.prime_mask_fraction<1: p.error('--prime-mask-fraction must be in [0,1)')
    if args.endpoint_batch is not None and not 2<=args.endpoint_batch<=args.batch:
        p.error('--endpoint-batch must be between 2 and --batch')
    if args.objective=='reference-centered':
        if args.affinity!='row': p.error('Reference control variate currently requires row affinity')
        if args.repulsion_weight!=1: p.error('Reference centering requires equal attraction and repulsion')
        if args.feature_space!='pixels': p.error('Reference centering currently operates in pixel space')
    output=args.output
    parent=torch.load(args.fork,map_location='cpu',weights_only=False) if args.fork else None
    if parent is not None and args.objective!='fm':validate_riesz_parent(parent)
    if parent is not None and parent['config'].get('dataset')!=args.dataset:
        p.error('Parent checkpoint dataset must match the requested dataset')
    if parent is not None and parent['config'].get('architecture','unet')!=args.architecture:
        p.error('Parent checkpoint architecture must match')
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib_cache')
    write_json(output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=True
    train,test=load_images(args.dataset,args.data_root)
    write_json(output/'data.json',dict(train_count=len(train),test_count=len(test),
        train_pixels_sha256=hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        test_pixels_sha256=hashlib.sha256(test.numpy().tobytes()).hexdigest(),
        image_shape=image_geometry(args.dataset),
        evaluation='FID, native RGB 32->299, pytorch-fid Inception2048' if args.dataset=='cifar10' else 'FID, grayscale repeated RGB, bilinear 28->32->299, pytorch-fid Inception2048',
        requested_generated_count=args.eval_samples,
        selection_reference_count=min(len(train),args.fid_reference_samples or (len(train) if args.dataset=='cifar10' else args.eval_samples)),
        selection_reference='fixed training reference; held-out test FID reported at final update'))
    evaluator=ImageEvaluation(train,test,args)
    train=(train.flatten(1).cuda().float()/127.5)-1
    def draw_training(count):
        images=train[torch.randint(len(train),(count,),device='cuda')]
        if args.random_flip:
            images=images.reshape(-1,*image_geometry(args.dataset))
            flip=torch.rand(len(images),1,1,1,device='cuda')<.5
            images=torch.where(flip,images.flip(-1),images).flatten(1)
        return images
    data_variance=train.var(0,unbiased=False) if args.reference_whitening_floor else None
    def reference_scale(time):
        if data_variance is None: return None
        # Exact marginal variances of the independent Gaussian/data affine path.
        return ((1-time)**2+time*time*data_variance).sqrt().clamp_min(args.reference_whitening_floor)
    pixel_scale=reference_scale(1.)
    if pixel_scale is not None:
        write_json(output/'reference_geometry.json',dict(data_variance=data_variance.cpu().tolist(),
            endpoint_scale=pixel_scale.cpu().tolist(),floor=args.reference_whitening_floor,
            formula='std(t) = sqrt((1-t)^2 + t^2 * data_variance), clamped below by floor'))
    model=image_base_from_config(vars(args)).cuda()
    if args.dit_prime_checkpoint:
        proof=initialize_from_reconstruction(model,args.dit_prime_checkpoint)
        args.dit_priming_sha256=proof['sha256']
        write_json(output/'initialization.json',proof)
        write_json(output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.parameterization=='denoiser': model=DenoiserVelocity(model)
    transport_rng=(torch.get_rng_state(),torch.cuda.get_rng_state()) if args.reprime_features else None
    encoder=None
    if args.prime_steps:
        prime_optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for step in range(1,args.prime_steps+1):
            clean=draw_training(args.batch)
            prime_optimizer.zero_grad(set_to_none=True)
            reconstruction_input=clean
            if args.prime_mask_fraction:
                grid=args.image_size//4
                mask=(torch.rand(len(clean),1,grid,grid,device='cuda')>=args.prime_mask_fraction).to(clean.dtype)
                mask=F.interpolate(mask,size=(args.image_size,args.image_size),mode='nearest').expand(-1,args.image_channels,-1,-1).flatten(1)
                reconstruction_input=clean*mask
            reconstruction=getattr(model,'reconstruct',model)
            value=(reconstruction(reconstruction_input,torch.rand(len(clean),device='cuda'))-clean).square().mean()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            prime_optimizer.step()
            if step%100==0:
                write_json(output/'status.json',dict(status='reconstruction_priming',update=step,mse=float(value)))
            if step%5000==0 or step==args.prime_steps:
                write_json(output/f'priming_{step:07d}.json',dict(stage='reconstruction_priming',update=step,training_batch_mse=float(value)))
        torch.save(dict(model=model.state_dict(),ema=model.state_dict(),lineage_has_flow_matching=False,
            config=dict(objective='reconstruction',fork=None,width=args.width,spatial_fourier=args.spatial_fourier,
                        deep=args.deep,prime_mask_fraction=args.prime_mask_fraction,parameterization=args.parameterization,
                        dataset=args.dataset,image_channels=args.image_channels,image_size=args.image_size),
            update=args.prime_steps),output/'reconstruction.pt')
        if args.feature_space=='primed': encoder=copy.deepcopy(model).requires_grad_(False).eval()
        # A velocity head starts at zero. A denoiser retains its reconstruction
        # head, since clean prediction is its transport parameterization.
        if args.parameterization=='velocity':
            torch.nn.init.zeros_(model.output.weight)
            torch.nn.init.zeros_(model.output.bias)
    if args.fork:
        if args.feature_space=='primed' and not args.reprime_features:
            encoder=copy.deepcopy(model).requires_grad_(False).eval()
        restore_image_checkpoint(model,parent,encoder,keep_encoder=args.reprime_features,require_riesz=args.objective!='fm')
        if any(not torch.equal(v.detach().cpu(),parent['ema'][k].detach().cpu()) for k,v in model.state_dict().items()):
            raise ValueError('Restored generator differs from parent EMA')
        proof=verify_reconstruction_encoder(dict(config=vars(args),encoder=encoder.state_dict() if encoder is not None else None))
        write_json(output/'initialization.json',dict(generator_source='parent EMA',generator_matches_parent=True,
            parent_sha256=hashlib.sha256(args.fork.read_bytes()).hexdigest(),
            encoder_source='fresh reconstruction priming' if args.reprime_features else 'inherited encoder' if encoder is not None else 'none',
            transport_rng_preserved_across_feature_priming=args.reprime_features,
            reconstruction_encoder=proof))
    if transport_rng is not None:
        torch.set_rng_state(transport_rng[0]);torch.cuda.set_rng_state(transport_rng[1])
    if args.feature_space=='primed' and encoder is None:
        raise ValueError('Primed features require reconstruction priming or an inherited Riesz-only encoder')
    vgg=None
    if args.vgg_endpoint_weight:
        from perceptual import VGGPerceptualLoss
        vgg=VGGPerceptualLoss(args.vgg_cache,'cuda').requires_grad_(False).eval()
        args.vgg_weights_sha256=vgg.metadata['weights_sha256']
        write_json(output/'vgg_features.json',dict(**vgg.metadata,
            objective='mean_of_15_independent_Riesz_endpoint_fields',
            input_mapping='unclipped (centered endpoint + 1)/2',learned_lpips_pairwise_heads=False,
            reference_features='pixels',pool_size=args.vgg_pool_size,weight=args.vgg_endpoint_weight,
            normalization=args.normalization,affinity=args.affinity))
        write_json(output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    reference=copy.deepcopy(model).requires_grad_(False) if args.fork else None
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    # Actual model forward/backward preflight before updates.
    model(train[:2],.5).square().mean().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    write_json(output/'hardware.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,backward_probe_passed=True,parameters=sum(p.numel() for p in model.parameters()),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        multiprocessor_count=torch.cuda.get_device_properties(0).multi_processor_count))
    best=float('inf')
    if args.preview_every:save_training_preview(ema,output,0,args)
    start=time.monotonic()
    for step in range(1,args.steps+1):
        clean=draw_training(args.batch)
        noise=torch.randn_like(clean)
        for group in optimizer.param_groups:
            group['lr']=args.lr*min(step/500,1)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/args.steps)))
        optimizer.zero_grad(set_to_none=True)
        loss_parts={}
        if args.objective=='fm':
            t=torch.rand(args.batch,1,device='cuda')
            loss=(model((1-t)*noise+t*clean,t[:,0])-(clean-noise)).square().mean()
        elif args.objective=='reference-centered':
            support_size=args.target_batch or args.batch*2
            target=draw_training(support_size)
            target_noise=torch.randn_like(target)
            negative=target if args.paired_reference_supports else draw_training(support_size)
            negative_noise=target_noise if args.paired_reference_supports else torch.randn_like(negative)
            t=float(torch.rand(()))*.999
            loss=reference_centered_loss(model,clean,noise,target,target_noise,
                negative,negative_noise,t,args.horizon,taus=args.tau,
                normalization=args.normalization)
        elif args.objective=='local-hybrid':
            t=float(torch.rand(()))*.999
            target=draw_training(args.target_batch or args.batch*2) if args.reference_coupling=='independent' else clean
            target_noise=torch.randn_like(target) if args.reference_coupling=='independent' else noise
            loss,loss_parts=local_reference_hybrid_loss(model,clean,noise,target,target_noise,t,args.horizon,
                reference_weight=args.reference_weight,riesz_weight=args.riesz_weight,
                time_weight_power=args.reference_time_power,taus=args.tau,normalization=args.normalization,
                repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                coordinate_scale=reference_scale(min(1.,t+args.horizon)),
                exclude_target_diagonal=args.reference_coupling=='paired-loo')
        elif args.objective=='coupled':
            t=float(torch.rand(()))*.999
            loss=coupled_reference_loss(model,clean,noise,t,args.horizon,taus=args.tau,normalization=args.normalization,
                time_weight_power=args.reference_time_power,pixel_scales=args.local_pixel_scales,image_shape=image_geometry(args.dataset),
                repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                coordinate_scale=reference_scale(min(1.,t+args.horizon)),
                include_self_negative=args.include_self_negative,distance_smoothing=args.distance_smoothing,distance_powers=args.distance_powers,
                exclude_target_diagonal=args.reference_coupling=='paired-loo')
        else:
            target=draw_training(args.target_batch or args.batch*2)
            if args.objective in ['endpoint','anchored-endpoint','path-endpoint','reference-hybrid']:
                pred=sample(model,noise[:args.endpoint_batch],args.train_rollout_steps,args.solver,checkpoint_steps=args.rollout_checkpointing)
                if args.decode_primed: pred=getattr(encoder,'reconstruct',encoder)(pred,1.)
                loss=args.riesz_weight*image_riesz_loss(pred,target,encoder=encoder,taus=args.tau,
                    normalization=args.normalization,repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                    statistics_weight=args.pixel_statistics_weight,gradient_weight=args.pixel_gradient_weight,pixel_scale=pixel_scale,
                    spatial_feature_weight=args.spatial_feature_weight,spatial_feature_locations=args.spatial_feature_locations,
                    image_shape=image_geometry(args.dataset))
                if vgg is not None:
                    vgg_loss=vgg_endpoint_riesz_loss(pred,target,vgg,pool_size=args.vgg_pool_size,
                        taus=args.tau,normalization=args.normalization,
                        repulsion_weight=args.repulsion_weight,affinity=args.affinity)
                    loss_parts.update(base_endpoint_riesz=loss.detach(),vgg_endpoint_riesz=vgg_loss.detach())
                    loss=(loss+args.riesz_weight*args.vgg_endpoint_weight*vgg_loss)/(1+args.vgg_endpoint_weight)
                if args.objective=='reference-hybrid':
                    t=torch.rand(args.batch,1,device='cuda')
                    reference_loss=(model((1-t)*noise+t*clean,t[:,0])-(clean-noise)).square().mean()
                    loss_parts.update(weighted_riesz=loss.detach(),reference_velocity_mse=reference_loss.detach())
                    loss=loss+args.reference_weight*reference_loss
                if args.objective=='path-endpoint':
                    t=float(torch.rand(()))*.999
                    reference_target,reference_noise=((clean,noise) if args.reference_coupling!='independent'
                        else (target,torch.randn_like(target)))
                    reference_loss=marginal_loss(model,clean,noise,reference_target,reference_noise,
                        time_weight_power=args.reference_time_power,pixel_scales=args.local_pixel_scales,image_shape=image_geometry(args.dataset),
                        t=t,horizon=args.horizon,taus=args.tau,normalization=args.normalization,repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                        coordinate_scale=reference_scale(min(1.,t+args.horizon)),
                        exclude_target_diagonal=args.reference_coupling=='paired-loo')
                    loss_parts.update(weighted_endpoint_riesz=loss.detach(),reference_marginal_riesz=reference_loss.detach())
                    loss=loss+args.marginal_weight*reference_loss
            else:
                t=float(torch.rand(()))*.999
                loss=args.riesz_weight*marginal_loss(model,clean,noise,target,torch.randn_like(target),
                    t,args.horizon,time_weight_power=args.reference_time_power,taus=args.tau,normalization=args.normalization,repulsion_weight=args.repulsion_weight,affinity=args.affinity,
                    pixel_scales=args.local_pixel_scales,image_shape=image_geometry(args.dataset),
                    coordinate_scale=reference_scale(min(1.,t+args.horizon)))
            if args.objective.startswith('anchored'):
                if reference is None: raise ValueError('Reference anchoring requires --fork')
                t=torch.rand(args.batch,1,device='cuda')
                x=(1-t)*noise+t*clean
                with torch.no_grad(): v=reference(x,t[:,0])
                loss=loss+args.anchor_weight*(model(x,t[:,0])-v).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite objective')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()): a.lerp_(b,.001 if step>1000 else .01)
        if step%100==0:
            write_json(output/'status.json',dict(status='running',update=step,loss=float(loss),grad=float(grad),
                seconds=time.monotonic()-start,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                loss_parts={k:float(v) for k,v in loss_parts.items()}))
        if args.preview_every and step%args.preview_every==0:
            save_training_preview(ema,output,step,args)
        if step%1000==0 and step%args.eval_every!=0 and step!=args.steps:
            recovery=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                lineage_has_flow_matching=args.objective=='fm',encoder=encoder.state_dict() if encoder is not None else None,
                update=step,config=vars(args),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            torch.save(recovery,output/'latest.pt.tmp')
            (output/'latest.pt.tmp').replace(output/'latest.pt')
        if step%args.eval_every==0 or step==args.steps:
            write_json(output/'status.json',dict(status='evaluating',update=step,seconds=time.monotonic()-start))
            row=evaluator.run(ema,output,step,final=step==args.steps,decoder=encoder if args.decode_primed else None)
            row['seconds']=time.monotonic()-start
            checkpoint=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                lineage_has_flow_matching=args.objective=='fm',
                encoder=encoder.state_dict() if encoder is not None else None,
                update=step,config=vars(args),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            torch.save(checkpoint,output/'latest.pt.tmp')
            (output/'latest.pt.tmp').replace(output/'latest.pt')
            if row['fid']<best:
                best=row['fid']
                torch.save(checkpoint,output/'best.pt.tmp')
                (output/'best.pt.tmp').replace(output/'best.pt')
                write_json(output/'best.json',row)
            print(json.dumps(row),flush=True)
    write_json(output/'final.json',row)
    write_json(output/'status.json',dict(status='complete',**row))


if __name__=='__main__': main()
