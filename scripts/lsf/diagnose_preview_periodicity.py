"""Measure grid artifacts in existing grayscale previews without editing images."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from run_storage import new_run_path


def statistics(pixels):
    x = pixels.astype(np.float64) / 255
    phase = np.stack([x[:, a::2, b::2].mean((1, 2)) for a, b in [(0,0),(0,1),(1,0),(1,1)]], 1)
    phase -= phase.mean(1, keepdims=True)
    spectrum = np.abs(np.fft.fft2(x, axes=(1, 2)))**2
    spectrum[:, 0, 0] = 0
    height, width = x.shape[-2:]
    nyquist = spectrum[:, height//2, 0] + spectrum[:, 0, width//2] + spectrum[:, height//2, width//2]
    return dict(count=len(x), parity_phase_rms=float(np.sqrt((phase**2).mean())),
                nyquist_energy_fraction=float(nyquist.sum() / max(spectrum.sum(), 1e-12)),
                horizontal_difference_energy=float(np.square(np.diff(x, axis=2)).mean()),
                vertical_difference_energy=float(np.square(np.diff(x, axis=1)).mean()))


def extract_grid(path, side=28, padding=2):
    array = np.asarray(Image.open(path).convert('L'))
    height, width = array.shape
    stride = side + padding
    if (height-padding) % stride or (width-padding) % stride:
        raise ValueError('Expected an unscaled torchvision grid with two-pixel padding')
    rows, columns = (height-padding)//stride, (width-padding)//stride
    metadata = json.loads(path.with_suffix('.json').read_text())
    count = metadata['count']
    if not 0 < count <= rows*columns:
        raise ValueError('Invalid preview count')
    tiles = [array[padding+r*stride:padding+r*stride+side,
                   padding+c*stride:padding+c*stride+side] for r in range(rows) for c in range(columns)]
    return np.stack(tiles[:count]), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist'], required=True)
    parser.add_argument('--grids', type=Path, nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    from torchvision.datasets import MNIST, FashionMNIST
    cls = MNIST if args.dataset == 'mnist' else FashionMNIST
    real = cls(str(new_run_path('datasets')), train=True, download=False).data
    indices = torch.randperm(len(real), generator=torch.Generator().manual_seed(3456))[:2048]
    rows = [dict(study='real_training_reference', **statistics(real[indices].numpy()))]
    for index, path in enumerate(args.grids):
        pixels, metadata = extract_grid(path)
        shutil.copy2(path, output / f'grid_{index:02d}.png')
        shutil.copy2(path.with_suffix('.json'), output / f'grid_{index:02d}.json')
        rows.append(dict(study=path.parent.parent.name, source=str(path), update=metadata['update'],
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **statistics(pixels)))
    source = output / 'source'
    source.mkdir()
    for path in [Path(__file__), ROOT / 'run_storage.py']:
        shutil.copy2(path, source / path.name)
    report = dict(dataset=args.dataset, rows=rows, optimizer_updates=0,
                  interpretation='Descriptive two-pixel-period diagnostic on saved, clipped PNG previews. '
                  'Parity RMS measures mean differences among 2x2 phases. Nyquist energy sums the three '
                  'period-two axial/checkerboard Fourier bins. Not FID, not a diversity score, and previews '
                  'at different updates are not matched training-budget comparisons.')
    (output / 'diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
