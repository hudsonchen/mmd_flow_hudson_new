"""Plot recorded pixel-only development FIDs without evaluating any model."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_storage import new_run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--x-axis', choices=['updates', 'examples'], default='updates',
                        help='Examples counts source particles, not equal compute or unique images')
    parser.add_argument('--previews', action='store_true', help='Also snapshot the latest active-run image grids')
    args = parser.parse_args()
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    campaign = json.loads(args.campaign.read_text())
    shutil.copy2(args.campaign, output / 'campaign_snapshot.json')
    labels = dict(independent='Independent, h=0.01', independent_h01='Independent, h=0.1',
                  paired_loo='Shared particles, own target excluded',
                  paired_loo_multiscale='Shared particles + fixed pixel scales',
                  paired_loo_multiscale_b1024='Shared particles + fixed pixel scales, batch 1,024',
                  paired_positive='Shared particles, own target retained')
    colors = dict(independent='#a6a6a6', independent_h01='#dd8a24', paired_loo='#2878b5',
                  paired_loo_multiscale='#249c65', paired_loo_multiscale_b1024='#bc4b51',
                  paired_positive='#9652b1')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    rows = []
    for ax, dataset in zip(axes, ['mnist', 'fashionmnist']):
        for run in campaign['runs']:
            if run['dataset'] != dataset:
                continue
            study = Path(run['path'])
            config_path = study / 'training/config.json'
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            batch = config.get('batch')
            records = []
            for path in sorted((study / 'training').glob('metrics_*.json')):
                try:
                    data = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                records.append(data)
                rows.append(dict(dataset=dataset, variant=run['variant'], file=str(path),
                                 batch=batch, source_examples=data['update'] * batch if batch else None,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **data))
            if not records:
                continue
            stopped = (study / 'early_stop_decision.json').exists()
            variant = run['variant']
            if args.x_axis == 'examples' and not batch:
                raise ValueError(f'Missing batch size for {study}')
            factor = batch if args.x_axis == 'examples' else 1
            ax.plot([r['update'] * factor for r in records], [r['fid'] for r in records], marker='o',
                    linestyle='--' if stopped else '-', color=colors.get(variant),
                    label=labels.get(variant, variant) + (' (stopped)' if stopped else ''))
        ax.set(title='MNIST' if dataset == 'mnist' else 'FashionMNIST',
               xlabel='Source examples processed' if args.x_axis == 'examples' else 'Optimizer updates')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(FuncFormatter(
            (lambda value, position: f'{value/1e6:g}M') if args.x_axis == 'examples'
            else (lambda value, position: f'{value/1000:g}k')))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='best')
    axes[0].set_ylabel('Training-reference FID-10K ↓')
    fig.suptitle('Pixel-only random-time Riesz flow · 100-step Euler sampling · seed 0')
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'fid_progress.{suffix}', dpi=170)
    plt.close(fig)
    preview_rows = []
    if args.previews:
        for run in campaign['runs']:
            study = Path(run['path'])
            if (study / 'early_stop_decision.json').exists():
                continue
            paths = sorted((study / 'training').glob('preview_*.png'))
            if not paths:
                continue
            path = paths[-1]
            update = int(path.stem.split('_')[-1])
            dest = output / 'previews' / study.name / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            metadata = path.with_suffix('.json')
            if metadata.exists():
                shutil.copy2(metadata, dest.with_suffix('.json'))
            preview_rows.append(dict(dataset=run['dataset'], variant=run['variant'], update=update,
                                     original=str(path), snapshot=str(dest),
                                     sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        if preview_rows:
            fig, axes = plt.subplots(math.ceil(len(preview_rows)/3), 3,
                                     figsize=(12, 3 * math.ceil(len(preview_rows)/3)), squeeze=False)
            for ax in axes.flat:
                ax.axis('off')
            for ax, row in zip(axes.flat, preview_rows):
                ax.imshow(plt.imread(row['snapshot']), interpolation='nearest')
                label = labels.get(row['variant'], row['variant'])
                ax.set_title(f"{row['dataset']} · {row['update']:,} updates\n{label}", fontsize=9)
            fig.suptitle('Latest active-run previews · fixed noise · different training budgets', fontsize=12)
            fig.tight_layout()
            fig.savefig(output / 'latest_previews.png', dpi=160)
            plt.close(fig)
    (output / 'metrics_snapshot.json').write_text(json.dumps(dict(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows, x_axis=args.x_axis,
        previews=preview_rows,
        caveat='Development scores against a fixed training reference, not held-out confirmation. '
               'Updates are not equal wall-clock cost across recipes. No additional FID was computed.'), indent=2)+'\n')
    source = output / 'source'
    source.mkdir()
    for path in [Path(__file__), ROOT / 'run_storage.py']:
        shutil.copy2(path, source / path.name)
    print(output / 'fid_progress.png')


if __name__ == '__main__':
    main()
