"""Compare endpoint and local-reference parameter gradients without training."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import torch
from run_storage import new_run_path
from small_flow import ImageVelocity,DenoiserVelocity,sample,image_riesz_loss,marginal_loss,validate_riesz_parent,verify_reconstruction_encoder,image_geometry,vgg_endpoint_riesz_loss
from scripts.lsf.run_small_image_flow import load_images
from reference_dit import image_base_from_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--recipe',type=Path,help='Loss configuration for a reconstruction-only checkpoint')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--dataset',choices=['mnist','fashionmnist','cifar10'])
    p.add_argument('--data-root',type=Path,default=new_run_path('datasets'))
    p.add_argument('--vgg-cache',type=Path)
    args=p.parse_args()
    if args.batch<2 or args.repeats<1:p.error('At least two particles and one repeat are required')
    output=new_run_path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    payload=args.checkpoint.read_bytes()
    (output/'checkpoint.pt').write_bytes(payload)
    cp=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    validate_riesz_parent(cp)
    cfg=dict(cp['config'])
    if args.recipe:
        recipe=json.loads(args.recipe.read_text())
        for key in ['dataset','width','deep','spatial_fourier','parameterization']:
            if cfg.get(key)!=recipe.get(key):raise ValueError(f'Recipe architecture mismatch: {key}')
        cfg.update(recipe)
        (output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    if args.dataset and args.dataset!=cfg['dataset']:p.error('Dataset must match the checkpoint')
    if cfg.get('decode_primed') or cfg.get('reference_whitening_floor',0):
        raise ValueError('This diagnostic requires unmodified pixel transport geometry')
    shape=image_geometry(cfg['dataset'])
    model=image_base_from_config(cfg).to(args.device)
    if cfg.get('parameterization')=='denoiser':model=DenoiserVelocity(model)
    model.load_state_dict(cp['ema']);model.eval()
    encoder=None
    if cp.get('encoder') is not None:
        encoder=copy.deepcopy(model).requires_grad_(False).eval();encoder.load_state_dict(cp['encoder'])
        provenance=verify_reconstruction_encoder(cp)
    elif cp['config']['objective']=='reconstruction' and cfg.get('feature_space')=='primed':
        encoder=copy.deepcopy(model).requires_grad_(False).eval();encoder.load_state_dict(cp['model'])
        provenance=verify_reconstruction_encoder(dict(config=dict(objective='diagnostic',fork=str(output/'checkpoint.pt')),encoder=encoder.state_dict()))
    else:provenance=dict(present=False)
    data,_=load_images(cfg['dataset'],args.data_root)
    data=data.flatten(1).to(args.device).float()/127.5-1
    vgg=None
    if cfg.get('vgg_endpoint_weight',0):
        if args.vgg_cache is None:p.error('VGG diagnostic requires an explicit frozen cache')
        from perceptual import VGGPerceptualLoss
        vgg=VGGPerceptualLoss(args.vgg_cache,args.device).requires_grad_(False).eval()
    kwargs=dict(taus=cfg['tau'],normalization=cfg['normalization'],affinity=cfg.get('affinity','row'),repulsion_weight=cfg.get('repulsion_weight',1.))
    def gradient(loss):
        values=torch.autograd.grad(loss,tuple(model.parameters()))
        return torch.cat([v.flatten() for v in values]).detach()
    rows=[]
    for repeat in range(args.repeats):
        torch.manual_seed(20261027+repeat)
        def draw():
            images=data[torch.randint(len(data),(args.batch,),device=args.device)]
            if cfg.get('random_flip',False):
                images=images.reshape(-1,*shape)
                mask=torch.rand(len(images),1,1,1,device=args.device)<.5
                images=torch.where(mask,images.flip(-1),images).flatten(1)
            return images
        clean,target=draw(),draw()
        noise=torch.randn_like(clean);target_noise=torch.randn_like(target)
        pred=sample(model,noise,cfg['train_rollout_steps'],cfg['solver'],checkpoint_steps=cfg.get('rollout_checkpointing',False))
        endpoint=cfg['riesz_weight']*image_riesz_loss(pred,target,encoder=encoder,
            statistics_weight=cfg.get('pixel_statistics_weight',0),gradient_weight=cfg.get('pixel_gradient_weight',0),
            spatial_feature_weight=cfg.get('spatial_feature_weight',0),spatial_feature_locations=cfg.get('spatial_feature_locations',16),image_shape=shape,**kwargs)
        if vgg is not None:
            value=vgg_endpoint_riesz_loss(pred,target,vgg,pool_size=cfg.get('vgg_pool_size',8),**kwargs)
            endpoint=(endpoint+cfg['riesz_weight']*cfg['vgg_endpoint_weight']*value)/(1+cfg['vgg_endpoint_weight'])
        eg=gradient(endpoint);en=float(eg.norm())
        for t in [.01,.1,.3,.5,.7,.9,.95,.99,.999]:
            loss=marginal_loss(model,clean,noise,target,target_noise,t,cfg['horizon'],**kwargs)*cfg['marginal_weight']
            rg=gradient(loss);rn=float(rg.norm())
            cosine=float(torch.dot(eg,rg)/(eg.norm()*rg.norm()).clamp_min(1e-12))
            combined=eg+rg
            weighted=eg+(1-t)*rg
            rows.append(dict(repeat=repeat,time=t,endpoint_norm=en,weighted_reference_norm=rn,
                reference_to_endpoint_ratio=rn/en,remaining_weighted_ratio=(1-t)*rn/en,
                endpoint_reference_cosine=cosine,
                combined_endpoint_cosine=float(torch.dot(eg,combined)/(eg.norm()*combined.norm()).clamp_min(1e-12)),
                time_weighted_combined_endpoint_cosine=float(torch.dot(eg,weighted)/(eg.norm()*weighted.norm()).clamp_min(1e-12)),
                combined_norm=float(combined.norm()),time_weighted_combined_norm=float(weighted.norm())))
        print(json.dumps(dict(repeat=repeat,endpoint_norm=en,last_time=rows[-1])),flush=True)
    files=['small_flow.py','backbone.py','loss.py','sinkhorn.py','run_storage.py','data.py','perceptual.py','fid.py','_fid_inception.py','reference_dit.py','flow_path.py',
           'scripts/lsf/run_small_image_flow.py','scripts/lsf/run_small_flow.py',str(Path(__file__).relative_to(ROOT))]
    files += [str(p.relative_to(ROOT)) for p in ROOT.glob('*LICENSE*')]
    hashes={}
    for name in files:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        reconstruction_encoder=provenance,source_hashes=hashes,config=vars(args),model_config=cfg,
        optimizer_updates=0,device=args.device,torch=torch.__version__,rows=rows,
        vgg_weights_sha256=vgg.metadata['weights_sha256'] if vgg is not None else None,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
        interpretation='Descriptive parameter-gradient diagnostic at a fixed checkpoint with independent train-image populations. Uses the specified particle batch size and augmentation. Gradient cosine is not a quality metric; time weighting is a proposed ablation, not demonstrated quality improvement. Local fields are evaluated unweighted and multiplied by (1-t) for the candidate comparison.')
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


if __name__=='__main__':main()
