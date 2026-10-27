"""Describe component residuals in an existing fresh-sample MoG audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from run_storage import new_run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.samples, output / 'samples.npz')
    samples = np.load(output / 'samples.npz')
    angle = np.arange(8) * np.pi / 4
    centers = 2 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    rows, radii = [], {}
    for name in samples.files:
        points = samples[name].astype(np.float64)
        distances = np.linalg.norm(points[:, None] - centers[None], axis=-1)
        mode = distances.argmin(axis=1)
        residual = points - centers[mode]
        radius = np.linalg.norm(residual, axis=1)
        radii[name] = radius
        centroid = np.stack([residual[mode == index].mean(axis=0) for index in range(8)])
        rows.append(dict(study=name, count=len(points),
            centroid_shift_rms=float(np.sqrt(np.mean(np.sum(centroid ** 2, axis=1)))),
            tail_above_05=float(np.mean(radius > .5)), tail_above_1=float(np.mean(radius > 1)),
            radius_quantiles={str(q): float(np.quantile(radius, q)) for q in [.5, .9, .99, .999]},
            max_radius=float(radius.max()), coordinate_rms=float(np.sqrt(np.mean(radius ** 2) / 2)),
            core_coordinate_rms=float(np.sqrt(np.mean(radius[radius < .36] ** 2) / 2))))
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(0, 1.1, 250)
    for prefix, label, color in [('target', 'Target sample', '#333333'),
            ('reference_fm_', 'FM', '#2878b5'),
            ('riesz_row_column_mog_refine_', 'Riesz parent', '#8c4fb5'),
            ('riesz_row_column_mog_tail_control_', 'Riesz continuation control', '#ff7f0e'),
            ('riesz_row_column_mog_tail_weighted_', 'Riesz tail weighting', '#2ca02c')]:
        selected = [v for k, v in radii.items() if k.startswith(prefix)]
        if len(selected) > 1:
            label += f' ({len(selected)} checkpoints)'
        for i, radius in enumerate(selected):
            survival = (radius[:, None] > grid[None]).mean(axis=0)
            ax.plot(grid, np.where(survival > 0, survival, np.nan), color=color,
                    alpha=1 if len(selected) == 1 else .6, label=label if i == 0 else None)
    ax.axvline(.36, color='gray', ls=':', label='Outside threshold')
    ax.set(xlabel='Distance to nearest component center', ylabel='Fraction beyond distance',
           yscale='log', ylim=(1e-5, 1), title='MoG residual tails: existing 50,000-sample audit')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(output / ('residual_survival.' + suffix), dpi=180)
    plt.close(fig)
    source = output / 'source'
    source.mkdir()
    shutil.copy2(__file__, source / Path(__file__).name)
    shutil.copy2(ROOT / 'run_storage.py', source / 'run_storage.py')
    report = dict(rows=rows, samples_source=str(args.samples),
        samples_sha256=hashlib.sha256((output / 'samples.npz').read_bytes()).hexdigest(),
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
        optimizer_updates=0,
        interpretation='Descriptive nearest-component residuals on previously audited samples. '
        'Known centers are used only for evaluation. Heavier distant tails may contribute to the '
        'sliced-Wasserstein gap; this is not a causal decomposition or a new independent audit.')
    (output / 'diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
