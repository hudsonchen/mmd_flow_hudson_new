"""Descriptive texture diagnostics on saved samples; no FID or model training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from run_storage import new_run_path


def statistics(pixels):
    x=pixels.reshape(-1,28,28).astype(np.float32)/255
    dx=x[:,:,1:]-x[:,:,:-1]
    dy=x[:,1:,:]-x[:,:-1,:]
    gradient_energy=(np.square(dx).mean((1,2))+np.square(dy).mean((1,2)))/2
    variance=x.var((1,2))
    centered=x-x.mean((1,2),keepdims=True)
    spectrum=np.abs(np.fft.fft2(centered))**2
    frequency=np.fft.fftfreq(28)
    radius=np.sqrt(frequency[:,None]**2+frequency[None,:]**2)
    high_frequency=spectrum[:,radius>.35].sum(1)/spectrum.sum((1,2)).clip(1e-12)
    border=np.ones((28,28),dtype=bool);border[2:-2,2:-2]=False
    return dict(count=len(x),mean_intensity=float(x.mean()),mean_image_variance=float(variance.mean()),
        mean_absolute_gradient=float((np.abs(dx).mean()+np.abs(dy).mean())/2),
        mean_gradient_energy=float(gradient_energy.mean()),
        mean_gradient_energy_over_image_variance=float((gradient_energy/variance.clip(1e-8)).mean()),
        mean_high_frequency_energy_fraction=float(high_frequency.mean()),
        border_mean=float(x[:,border].mean()),border_fraction_above_005=float((x[:,border]>.05).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset',choices=['mnist','fashionmnist'],required=True)
    parser.add_argument('--studies',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    output=new_run_path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    import torchvision
    cls=torchvision.datasets.MNIST if args.dataset=='mnist' else torchvision.datasets.FashionMNIST
    train=cls(str(new_run_path('datasets')),train=True,download=False).data
    index=torch.randperm(len(train),generator=torch.Generator().manual_seed(3456))[:10000]
    real=train[index].numpy()
    rows=[dict(study='real_training_reference',**statistics(real),
        selected_pixels_sha256=hashlib.sha256(real.tobytes()).hexdigest())]
    for study in args.studies:
        best=json.loads((study/'training/best.json').read_text())
        path=study/'training'/f"samples_{best['update']:07d}.npy"
        rows.append(dict(study=study.name,selected_training_reference_fid=best['fid'],
            selected_update=best['update'],samples=str(path),
            samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**statistics(np.load(path))))
    script=output/'source/scripts/lsf'/Path(__file__).name
    script.parent.mkdir(parents=True)
    shutil.copy2(__file__,script)
    for name in ['run_storage.py']:
        destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    report=dict(dataset=args.dataset,rows=rows,
        runtime=dict(torch=torch.__version__,numpy=np.__version__,torchvision=torchvision.__version__),
        definitions='Rounded uint8 sample pixels scaled to [0,1]. First differences use horizontal/vertical neighbors. High frequency means radial FFT frequency above 0.35 cycles/pixel after per-image mean removal. Border is the outer two pixels.',
        interpretation='Descriptive diagnostics, not a quality score. Real texture and garment boundaries also contribute to gradients and high-frequency energy. No images were selected using these measurements.')
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows: print(json.dumps(row),flush=True)


if __name__=='__main__': main()
