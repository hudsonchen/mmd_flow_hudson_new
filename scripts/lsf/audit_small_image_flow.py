"""Evaluate a selected image checkpoint on fresh noise and held-out test images."""
import argparse
import copy
import json
import hashlib
import io
import math
import os
from pathlib import Path
import sys
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['DRIFT_COMPILE']='0'
import torch
from small_flow import ImageVelocity,DenoiserVelocity,sample,validate_riesz_parent,verify_reconstruction_encoder,image_geometry
from scripts.lsf.run_small_image_flow import load_images,ImageEvaluation
from scripts.lsf.run_small_flow import write_json
from reference_dit import image_base_from_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=['mnist','fashionmnist','cifar10'],required=True)
    p.add_argument('--studies',nargs='+',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--preview-count',type=int,default=100)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda',help='CPU is allowed for small previews only; FID requires CUDA')
    p.add_argument('--decode-primed',action='store_true',help='Diagnostic: apply the frozen reconstruction decoder after transport')
    p.add_argument('--reference-split',choices=['train','test'],default='test')
    p.add_argument('--checkpoint',default='best.pt')
    p.add_argument('--sample-steps',type=int,nargs='+')
    p.add_argument('--fid-batch-size',type=int,default=32)
    p.add_argument('--noise-seed',type=int,help='Common generated-noise seed; default 7754321 for FID, 8765432 for previews')
    args=p.parse_args()
    shape=image_geometry(args.dataset)
    if args.sample_steps and (min(args.sample_steps)<1 or len(set(args.sample_steps))!=len(args.sample_steps)):
        p.error('Sampling step counts must be positive and distinct')
    if args.fid_batch_size<1: p.error('FID batch size must be positive')
    if args.preview_count<2:p.error('Preview count must be at least two')
    if args.device!='cuda' and not args.preview_only:p.error('FID audits require CUDA')
    noise_seed=args.noise_seed if args.noise_seed is not None else (8765432 if args.preview_only else 7754321)
    if not 0<=noise_seed<2**63: p.error('Noise seed must be in [0, 2**63)')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    if not args.preview_only:
        train,test=load_images(args.dataset,args.data_root)
        real=train if args.reference_split=='train' else test
        # ImageEvaluation adds 100000 to its seed; keep the audit's actual noise
        # seed explicit and independent of all training or sampler-selection noise.
        settings=argparse.Namespace(dataset=args.dataset,eval_samples=10000,seed=noise_seed-100000,sample_steps=1,solver='euler',fid_batch_size=args.fid_batch_size)
        evaluator=ImageEvaluation(real,test,settings)
    results=[]
    for study in args.studies:
        payload=(study/'training'/args.checkpoint).read_bytes()
        (args.output/(study.name+'_checkpoint.pt')).write_bytes(payload)
        ckpt=torch.load(io.BytesIO(payload),map_location=args.device,weights_only=False)
        cfg=ckpt['config']
        if cfg['dataset']!=args.dataset:raise ValueError('Checkpoint dataset does not match the audit reference')
        if cfg['objective']!='fm': validate_riesz_parent(ckpt)
        ancestry=[]
        ancestor,ancestor_payload=ckpt,payload
        while True:
            ancestry.append(dict(objective=ancestor['config']['objective'],update=ancestor['update'],
                reference_coupling=ancestor['config'].get('reference_coupling',
                    'paired' if ancestor['config']['objective']=='coupled' else 'independent'),
                sha256=hashlib.sha256(ancestor_payload).hexdigest(),parent=str(ancestor['config'].get('fork'))))
            parent=ancestor['config'].get('fork')
            if not parent: break
            ancestor_payload=Path(parent).read_bytes()
            ancestor=torch.load(io.BytesIO(ancestor_payload),map_location='cpu',weights_only=False)
        provenance=dict(ancestry=ancestry,checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
            reconstruction_encoder=verify_reconstruction_encoder(ckpt),
            parameterization=cfg.get('parameterization','velocity'),affinity=cfg.get('affinity','row'),
            pixel_statistics_weight=cfg.get('pixel_statistics_weight',0.),reference_whitening_floor=cfg.get('reference_whitening_floor',0.),
            pixel_gradient_weight=cfg.get('pixel_gradient_weight',0.),
            spatial_feature_weight=cfg.get('spatial_feature_weight',0.),spatial_feature_locations=cfg.get('spatial_feature_locations',16),
            reference_time_power=cfg.get('reference_time_power',0.),
            local_pixel_scales=cfg.get('local_pixel_scales'),
            include_self_negative=cfg.get('include_self_negative',False),
            distance_smoothing=cfg.get('distance_smoothing',0.),
            vgg_endpoint_weight=cfg.get('vgg_endpoint_weight',0.),vgg_pool_size=cfg.get('vgg_pool_size'),
            vgg_weights_sha256=cfg.get('vgg_weights_sha256'),
            architecture=cfg.get('architecture','unet'),dit_priming_sha256=cfg.get('dit_priming_sha256'),
            reprime_features=cfg.get('reprime_features',False),priming_updates_in_stage=cfg.get('prime_steps',0),
            lineage_has_flow_matching=any(a['objective']=='fm' for a in ancestry),
            uses_paired_reference_velocity_loss=any(a['objective'] in ['fm','reference-hybrid'] for a in ancestry))
        provenance['reference_coupling']=ancestry[0]['reference_coupling']
        provenance['uses_paired_query_reference_particles']=any(a['reference_coupling'] in ['paired','paired-loo'] for a in ancestry)
        provenance['uses_direct_matched_positive']=any(a['reference_coupling']=='paired' for a in ancestry)
        model=image_base_from_config(cfg).to(args.device)
        if cfg.get('parameterization')=='denoiser': model=DenoiserVelocity(model)
        model.load_state_dict(ckpt['ema'])
        model.eval()
        decoder=None
        if args.decode_primed or cfg.get('decode_primed',False):
            if ckpt.get('encoder') is None: raise ValueError('Decoder diagnostic requires saved reconstruction weights')
            decoder=copy.deepcopy(model).requires_grad_(False).eval()
            decoder.load_state_dict(ckpt['encoder'])
        provenance['postprocessing']=('trained_frozen_reconstruction_decoder' if cfg.get('decode_primed',False)
            else 'frozen_reconstruction_decoder' if decoder is not None else 'none')
        for steps in args.sample_steps or [cfg['sample_steps']]:
            solver=cfg.get('solver','heun')
            tag=study.name if args.sample_steps is None else f'{study.name}_{solver}{steps}'
            transport_nfe=steps*(2 if solver=='heun' else 1)
            sampling=dict(solver=solver,sample_steps=steps,transport_network_evaluations=transport_nfe,
                decoder_evaluations=int(decoder is not None),network_evaluations=transport_nfe+int(decoder is not None))
            if args.preview_only:
                generator=torch.Generator(device=args.device).manual_seed(noise_seed)
                with torch.no_grad():
                    raw=sample(model,torch.randn(args.preview_count,math.prod(shape),device=args.device,generator=generator),steps,solver)
                    if decoder is not None: raw=getattr(decoder,'reconstruct',decoder)(raw,1.)
                    if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite preview samples')
                from torchvision.utils import save_image
                save_image(((raw+1)/2).clamp(0,1).reshape(-1,*shape),str(args.output/(tag+'.png')),nrow=10)
                row=dict(study=study.name,update=ckpt['update'],noise_seed=noise_seed,**provenance,**sampling,
                    preview_only=True,no_fid_evaluated=True,preview_count=args.preview_count,sampling_device=args.device,
                    raw_pixel_outside_fraction=float((raw.abs()>1).float().mean()),
                    mean_pixel_variance_across_samples=float(raw.var(0).mean()))
                results.append(row)
                write_json(args.output/'audit.json',dict(rows=results))
                print(json.dumps(row),flush=True)
                continue
            # Reuse the same real statistics and common fresh noise at every step count.
            settings.sample_steps=steps;settings.solver=solver
            destination=args.output/tag
            destination.mkdir()
            row=evaluator.run(model,destination,ckpt['update'],decoder=decoder)
            row.update(study=study.name,dataset=args.dataset,reference_split=args.reference_split,noise_seed=noise_seed,
                parameters=sum(p.numel() for p in model.parameters()),fid_batch_size=args.fid_batch_size,
                **sampling,**provenance)
            results.append(row)
            write_json(args.output/'audit.json',dict(checkpoint_selection='best training-reference FID',reference_split=args.reference_split,
                purpose='development diagnostic' if args.reference_split=='train' else 'fresh-noise test evaluation',rows=results))
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
