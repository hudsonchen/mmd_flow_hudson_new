"""Measure patch-period artifacts in existing sample grids; does not edit images."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data import load_dataset
from run_storage import new_run_path


def stats(images):
    x = images.astype(np.float64)/127.5-1
    means = np.stack([x[:, a::2,b::2].mean((1,2)) for a,b in [(0,0),(0,1),(1,0),(1,1)]],axis=1)
    periodic = means-means.mean(1,keepdims=True)
    spectrum = abs(np.fft.fft2(x, axes=(1,2)))**2
    spectrum[:,0,0] = 0
    nyquist = spectrum[:,16,0]+spectrum[:,0,16]+spectrum[:,16,16]
    return dict(count=len(x), patch_phase_rms=float(np.sqrt((periodic**2).mean())),
                nyquist_energy_fraction=float(nyquist.sum()/spectrum.sum()))


def main():
    base = Path('/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar')
    output = new_run_path('cifar/time_flow_diagnostic_20261026T2359')
    _, (raw, _), _ = load_dataset('cifar10',ROOT/'data')
    results = dict(real=stats(raw[:2048].reshape(-1,3,32,32).numpy().transpose(0,2,3,1)), grids=[])
    for study in ['time_dit_riesz10_pixels_rollout50_ema999_seed42_20261026',
                  'time_dit_fm10_pixels_ema999_fresh_seed42_20261026']:
        for path in sorted((base/study/'training').glob('particles_step_*.png')):
            arr = np.asarray(Image.open(path))
            # Invert save_grid's nearest-neighbor enlargement at source pixel centers.
            ys = np.floor((np.arange(272)+.5)*arr.shape[0]/272).astype(int)
            xs = np.floor((np.arange(272)+.5)*arr.shape[1]/272).astype(int)
            grid = arr[ys[:,None],xs[None,:]]
            images = np.stack([grid[r*34:r*34+32,c*34:c*34+32] for r in range(8) for c in range(8)])
            results['grids'].append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**stats(images)))
    (output/'grid_spectrum.json').write_text(json.dumps(results,indent=2)+'\n')
    (output/'source/scripts/lsf/audit_flow_grid_spectrum.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
