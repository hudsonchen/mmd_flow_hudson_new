"""Measure raw-versus-clipped image geometry at fixed checkpoints; no training."""
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
import numpy as np
import torch
from torchvision import datasets
from small_flow import ImageVelocity,DenoiserVelocity,sample,validate_riesz_parent,verify_reconstruction_encoder
from run_storage import new_run_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--studies',type=Path,nargs='+',required=True)
    p.add_argument('--count',type=int,default=512)
    p.add_argument('--checkpoint',default='best.pt',help='Use latest.pt for early stability checks before the first FID evaluation')
    args=p.parse_args()
    if args.count<2 or args.count%2:p.error('Sample count must be positive and even')
    output=new_run_path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    rows=[]
    for study in args.studies:
        payload=(study/'training'/args.checkpoint).read_bytes()
        cp=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        cfg=cp['config']
        if cfg.get('decode_primed'):raise ValueError('This diagnostic does not apply a post-transport decoder')
        if cfg['objective']!='fm':validate_riesz_parent(cp)
        model=ImageVelocity(cfg['width'],cfg.get('spatial_fourier',False),cfg.get('deep',False))
        if cfg.get('parameterization')=='denoiser':model=DenoiserVelocity(model)
        model.load_state_dict(cp['ema']);model.eval()
        generator=torch.Generator().manual_seed(20261028)
        noise=torch.randn(args.count,784,generator=generator)
        with torch.no_grad():
            raw=torch.cat([sample(model,x,cfg['sample_steps'],cfg['solver']) for x in noise.split(32)])
        clipped=raw.clamp(-1,1);excess=(raw-clipped).abs();outside=excess>0
        row=dict(study=study.name,checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
            update=cp['update'],sample_steps=cfg['sample_steps'],solver=cfg['solver'],count=args.count,
            pixel_outside_fraction=float(outside.float().mean()),mean_absolute_excursion=float(excess.mean()),
            clipping_rms=float(excess.square().mean().sqrt()),raw_min=float(raw.min()),raw_max=float(raw.max()),
            nonzero_excursion_quantiles={str(q):float(torch.quantile(excess[outside],q)) if outside.any() else 0. for q in [.5,.9,.95,.99]},
            fraction_excursion_above_005=float((excess>.05).float().mean()),
            fraction_excursion_above_01=float((excess>.1).float().mean()))
        if cp.get('encoder') is not None:
            row['encoder_provenance']=verify_reconstruction_encoder(cp)
            encoder=copy.deepcopy(model).requires_grad_(False).eval();encoder.load_state_dict(cp['encoder'])
            cls=datasets.MNIST if cfg['dataset']=='mnist' else datasets.FashionMNIST
            data=cls(str(new_run_path('datasets')),train=True,download=False).data
            indices=torch.randperm(len(data),generator=torch.Generator().manual_seed(90213))[:args.count]
            target=data[indices].flatten(1).float()/127.5-1
            with torch.no_grad():
                def features(images):
                    batches=[encoder.features(x) for x in images.split(32)]
                    return [torch.cat([b[i] for b in batches]) for i in range(len(batches[0]))]
                a,b,c=features(raw),features(clipped),features(target)
            row['encoder_feature_shift']=[]
            for i,(x,y,z) in enumerate(zip(a,b,c)):
                scale=(z[::2]-z[1::2]).square().mean().sqrt().clamp_min(1e-8)
                row['encoder_feature_shift'].append(dict(group=i,rms=float((x-y).square().mean().sqrt()),
                    relative_to_real_pair_rms=float((x-y).square().mean().sqrt()/scale)))
        (output/(study.name+'_checkpoint.pt')).write_bytes(payload)
        np.savez_compressed(output/(study.name+'_samples.npz'),raw=raw.numpy(),clipped=clipped.numpy())
        rows.append(row);print(json.dumps(row),flush=True)
    hashes={}
    files=['small_flow.py','backbone.py','loss.py','sinkhorn.py','run_storage.py',str(Path(__file__).relative_to(ROOT))]
    files += [str(p.relative_to(ROOT)) for p in ROOT.glob('*LICENSE*')]
    for name in files:
        destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination);hashes[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    report=dict(config=vars(args),rows=rows,source_hashes=hashes,torch=torch.__version__,optimizer_updates=0,device='cpu',
        interpretation='Descriptive range/feature diagnostic on common fresh CPU noise. No FID or quality claim, no model updates, and no model is used as a teacher. Encoder shifts are normalized by real-image pair RMS, not by evaluation features.')
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


if __name__=='__main__':main()
