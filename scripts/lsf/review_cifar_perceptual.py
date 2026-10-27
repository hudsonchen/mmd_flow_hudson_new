"""Snapshot and compare the October 26 CIFAR VGG/perceptual studies, read-only."""
import argparse
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
from run_storage import new_run_path


RUNS = [
    ('pixel_lpips', 'LPIPS priming + pixel Riesz', 'priming_compact_dit_lpips01_seed42_20261026_r2', 'posttraining', 49823627),
    ('repulsion05', 'Pixel Riesz, repulsion 0.5', 'priming_compact_dit_lpips01_repulsion05_seed42_20261026', 'posttraining', 49824187),
    ('additive_vgg', 'Pixel + 0.1 VGG field', 'priming_compact_dit_lpips01_vggpost01_seed42_20261026', 'posttraining', 49824383),
    ('single_vgg', 'Single concatenated VGG field', 'priming_compact_dit_vggflow_seed42_20261026', 'posttraining', 49824393),
    ('multi128', 'Multiscale VGG, 128/256', 'priming_dit_vgg_multiscale_b128_r256_fresh_seed42_20261026', 'posttraining', 49826390),
    ('multi256', 'Multiscale VGG, 256/1024', 'priming_dit_vgg_multiscale_b256_r1024_fresh_seed42_20261026', 'posttraining', 49825452),
    ('fm', 'Flow matching, Euler 10', 'time_dit_fm10_pixels_ema999_fresh_seed42_20261026', 'training', 49825501),
    ('local_vgg', 'Local ten-step VGG Riesz', 'time_dit_riesz10_vgg_seed42_20261026', 'training', 49825450),
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true', help='Refresh this review report; never modifies the original experiments')
    args = parser.parse_args()
    output = new_run_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.refresh)
    import os
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    root = new_run_path('cifar')
    records, hashes, configs, snapshots = {}, {}, {}, {}
    for key, label, name, stage, job in RUNS:
        study = root / name
        dest = output / 'evidence' / key
        dest.mkdir(parents=True, exist_ok=args.refresh)
        for relative in ['manifest.json', f'{stage}/config.json', f'{stage}/fid_metrics.json',
                         f'{stage}/metrics.csv', f'{stage}/job_status.json']:
            source = study / relative
            if source.exists():
                # Read once: curves and their hashes refer to the same snapshot.
                data = source.read_bytes()
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                hashes[str(source)] = hashlib.sha256(data).hexdigest()
        config = json.loads((dest / stage / 'config.json').read_text())
        curve = json.loads((dest / stage / 'fid_metrics.json').read_text())
        history = list(csv.DictReader(io.StringIO((dest / stage / 'metrics.csv').read_text())))
        times = {int(row['update']): float(row['elapsed_seconds']) for row in history}
        curve = [dict(row, elapsed_seconds=times.get(row['iteration'])) for row in curve]
        trained = [row for row in curve if row['iteration'] > 0]
        best = min(trained, key=lambda row: row['fid']) if trained else None
        at25 = next((row for row in curve if row['iteration'] == 25000), None)
        result = subprocess.run(['bjobs', '-l', str(job)], text=True, capture_output=True)
        scheduler = result.stdout + result.stderr
        (dest / 'scheduler.txt').write_text(scheduler)
        status = ('owner-terminated' if 'TERM_OWNER' in scheduler else
                  'running' if 'Status <RUN>' in scheduler else
                  'exited' if 'Status <EXIT>' in scheduler else 'see recorded job status')
        records[key] = dict(label=label, study=str(study), stage=stage, job_id=job,
            scheduler_status=status, best_trained=best, latest=curve[-1], at_25000=at25,
            parameter_count=config.get('parameter_count'),
            sampling_nfe=config.get('sampling_network_evaluations', config.get('architecture', {}).get('sampling_network_evaluations') if isinstance(config.get('architecture'), dict) else None),
            generated_batch=config.get('train_particles', config.get('batch_size')),
            target_batch=config.get('target_batch_size'),
            protocol={k: config.get(k) for k in ['fid_samples', 'fid_reference_split', 'fid_reference_count', 'fid_latent_seed', 'fid_weights_sha256']},
            curve=curve)
        configs[key] = config
        if best is not None:
            source = study / stage / f"particles_step_{best['iteration']:06d}.png"
            if source.exists():
                target = dest / 'best_samples.png'
                shutil.copy2(source, target)
                hashes[str(source)] = digest(target)
                snapshots[key] = target
    # Verify actual reconstruction checkpoint and every recorded copy, not just labels.
    origins = {}
    for config in configs.values():
        origin = config.get('priming_initialization')
        if origin is None:
            continue
        path = Path(origin['checkpoint'])
        actual = digest(path)
        if actual != origin['sha256']:
            raise ValueError(f'Priming checkpoint hash mismatch: {path}')
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        c = checkpoint['config']
        if c.get('training_stage') != 'reconstruction' or any(c.get(k) for k in ['init_checkpoint', 'fork_checkpoint', 'resume']):
            raise ValueError(f'Unexpected reconstruction ancestry: {path}')
        origins[str(path)] = dict(sha256=actual, stage=c['training_stage'],
            rule=c['training_rule'], lpips_weight=c['lpips_weight'],
            iteration=checkpoint['iteration'], updates=checkpoint['update_count'],
            elapsed_seconds=checkpoint['elapsed_seconds'], no_parent_weights=True)
        del checkpoint
    protocols = [record['protocol'] for record in records.values()]
    same_protocol = all(protocol == protocols[0] for protocol in protocols)
    small, large = records['multi128']['at_25000'], records['multi256']['at_25000']
    summary = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Read-only review; original runs, paths and source snapshots unchanged',
        identical_recorded_fid_protocol=same_protocol, runs=records, verified_priming_origins=origins,
        at_25000_large_batch_gain=small['fid']-large['fid'],
        at_25000_large_batch_time_ratio=large['elapsed_seconds']/small['elapsed_seconds'],
        caveats=[
            'Single-seed checkpoint-selection FID: 10000 generated versus 50000 training images, not held-out confirmation.',
            'Historical Riesz uses reconstruction + LPIPS priming (62560 extra optimizer updates); FM starts fresh.',
            'One-pass Riesz has 21.3M parameters; time-conditioned FM has 32.3M and uses 10 network evaluations.',
            'Elapsed times are recorded training-stage times including evaluation, excluding priming; not equal total compute.',
            'The additive VGG-field pilot has no measured post-update FID; update zero is not a training result.',
            'The VGG feature matching objectives do not use the learned pairwise LPIPS heads during transport.',
            'Owner-terminated jobs are not restarted by this review.',
        ])
    write_json(output / 'review.json', summary)
    columns = ['label', 'best_fid', 'best_update', 'latest_fid', 'latest_update', 'fid_25000', 'hours_25000', 'sampling_nfe', 'scheduler_status']
    with (output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in records.values():
            best, matched = r['best_trained'] or {}, r['at_25000'] or {}
            writer.writerow(dict(label=r['label'], best_fid=best.get('fid'), best_update=best.get('iteration'),
                latest_fid=r['latest']['fid'], latest_update=r['latest']['iteration'], fid_25000=matched.get('fid'),
                hours_25000=matched['elapsed_seconds']/3600 if matched else None,
                sampling_nfe=r['sampling_nfe'], scheduler_status=r['scheduler_status']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for key in ['pixel_lpips', 'single_vgg', 'multi128', 'multi256', 'fm']:
        record = records[key]
        curve = [r for r in record['curve'] if r['iteration'] > 0]
        axes[0].plot([r['iteration']/1000 for r in curve], [r['fid'] for r in curve], '.-', label=record['label'])
        axes[1].plot([r['elapsed_seconds']/3600 for r in curve], [r['fid'] for r in curve], '.-')
    for ax, xlabel in zip(axes, ['Transport updates (thousands)', 'Recorded transport-stage hours (excludes priming)']):
        ax.set(xlabel=xlabel, ylabel='FID-10K / train-50K (lower is better)', ylim=(25, 115))
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Recent CIFAR-10 runs: multiscale VGG improves Riesz; FM continues improving')
    fig.text(.5, .015, 'Different architectures, priming budgets, EMA and sampling costs; one seed each. These are not equal-compute comparisons.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(output / 'fid_comparison.png', dpi=180)
    fig.savefig(output / 'fid_comparison.pdf')
    plt.close(fig)
    chosen = [key for key in ['pixel_lpips', 'single_vgg', 'multi128', 'multi256'] if key in snapshots]
    fig, axes = plt.subplots(1, len(chosen), figsize=(4*len(chosen), 4.5), squeeze=False)
    for ax, key in zip(axes[0], chosen):
        row = records[key]
        ax.imshow(plt.imread(snapshots[key]))
        ax.set_title(f"{row['label']}\nFID {row['best_trained']['fid']:.2f} at {row['best_trained']['iteration']:,}", fontsize=9)
        ax.axis('off')
    fig.suptitle('Archived fixed-noise grids at each run’s best measured FID; all 64 samples shown')
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(output / 'sample_comparison.png', dpi=180)
    plt.close(fig)
    shutil.copy2(__file__, output / Path(__file__).name)
    hashes[str(Path(__file__).resolve())] = digest(output / Path(__file__).name)
    write_json(output / 'evidence_sha256.json', hashes)
    print(json.dumps(dict(output=str(output), identical_recorded_fid_protocol=same_protocol,
        results={k: r['best_trained'] for k, r in records.items()}), indent=2))


if __name__ == '__main__':
    main()
