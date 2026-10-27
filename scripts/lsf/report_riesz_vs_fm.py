"""Freeze recorded Riesz/FM metrics and render actual saved generated samples."""
import argparse
import datetime
import hashlib
import json
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
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = new_run_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figures').mkdir()
    (output / 'snapshots').mkdir()
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib_cache')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    specs = {
        'mnist_riesz': ('mnist', 'pixel_riesz_mnist_symmetric_w32_b256_seed0_20261027'),
        'mnist_fm': ('mnist', 'reference_fm_mnist_deep16_seed0_20261026'),
        'fashionmnist_riesz': ('fashionmnist', 'pixel_riesz_fashionmnist_symmetric_w32_b256_seed0_20261027'),
        'fashionmnist_fm': ('fashionmnist', 'reference_fm_fashionmnist_deep16_seed0_20261027'),
        'mnist_smooth': ('mnist', 'pixel_riesz_mnist_symmetric_smooth_no_rms_w32_b256_seed0_20261027'),
        'mnist_original': ('mnist', 'pixel_riesz_random_time_mnist_paired_positive_w32_b256_seed0_20261027'),
    }
    provenance = []

    def snapshot_json(source, relative):
        payload = source.read_bytes()
        value = json.loads(payload)
        dest = output / 'snapshots' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        provenance.append(dict(source=str(source), snapshot=str(dest.relative_to(output)),
                               sha256=hashlib.sha256(payload).hexdigest()))
        return value

    runs = {}
    for key, (dataset, name) in specs.items():
        study = new_run_path(dataset) / name
        training = study / 'training'
        run = dict(path=str(study), dataset=dataset)
        for name in ['config', 'hardware', 'data', 'status', 'best']:
            path = training / (name + '.json')
            if path.exists():
                run[name] = snapshot_json(path, f'{key}/{name}.json')
        run['metrics'] = [snapshot_json(path, f'{key}/{path.name}')
                          for path in sorted(training.glob('metrics_*.json'))]
        if not run['metrics']:
            raise ValueError(f'No completed evaluations for {key}')
        runs[key] = run

    audit_root = new_run_path('mnist/pixel_symmetric_5000_independent_test_audit_20261027/audit')
    audit = snapshot_json(audit_root / 'audit.json', 'independent_test_audit.json')
    assert len(audit['rows']) == 2
    assert all(row['reference_split'] == 'test' and row['noise_seed'] == 9274311
               and row['network_evaluations'] == 100 for row in audit['rows'])
    for dataset in ['mnist', 'fashionmnist']:
        left, right = runs[dataset + '_riesz'], runs[dataset + '_fm']
        assert left['data']['train_pixels_sha256'] == right['data']['train_pixels_sha256']
        assert left['data']['test_pixels_sha256'] == right['data']['test_pixels_sha256']
        assert left['config']['seed'] == right['config']['seed'] == 0

    def by_update(run):
        return {row['update']: row for row in run['metrics']}

    def grid(source, tag):
        array = np.load(source, mmap_mode='r')
        assert array.shape[0] == 10000 and array.shape[1:] == (1, 28, 28)
        assert array.dtype == np.uint8
        first = np.array(array[:64], copy=True)
        np.save(output / 'snapshots' / f'{tag}_first64.npy', first)
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        provenance.append(dict(source=str(source), source_sha256=digest,
                               snapshot=f'snapshots/{tag}_first64.npy',
                               sample_indices=list(range(64))))
        canvas = np.zeros((8 * 28 + 7 * 2, 8 * 28 + 7 * 2), dtype=np.uint8)
        for i, image in enumerate(first[:, 0]):
            row, col = divmod(i, 8)
            canvas[row * 30:row * 30 + 28, col * 30:col * 30 + 28] = image
        return canvas

    def sample_panel(tag, title, entries, noise_seed, note):
        fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
        for ax, entry in zip(axes, entries):
            label, source, update, fid = entry
            ax.imshow(grid(source, tag + '_' + str(update) + '_' + label.split()[0]),
                      cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(f'{label}\n{update:,} updates | FID {fid:.2f}', fontsize=12)
            ax.axis('off')
        fig.suptitle(title, fontsize=15)
        fig.text(.04, .025, f'First 64 saved samples; common noise seed {noise_seed}; no cherry-picking.\n{note}', fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .95))
        destination = output / 'figures' / (tag + '.png')
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        return str(destination.relative_to(output))

    sample_figures = {}
    matched_updates = {}
    for dataset, title in [('mnist', 'MNIST'), ('fashionmnist', 'FashionMNIST')]:
        a, b = runs[dataset + '_riesz'], runs[dataset + '_fm']
        am, bm = by_update(a), by_update(b)
        update = max(set(am) & set(bm))
        matched_updates[dataset] = update
        sample_figures[dataset] = sample_panel(dataset + '_matched_samples', f'{title}: matched optimizer update', [
            ('Riesz', Path(a['path']) / 'training' / f'samples_{update:07d}.npy', update, am[update]['fid']),
            ('FM', Path(b['path']) / 'training' / f'samples_{update:07d}.npy', update, bm[update]['fid'])],
            100000, '100 network evaluations each; model sizes, training batches, and recipes differ.')

    ar, af = audit['rows']
    sample_figures['audit'] = sample_panel('mnist_independent_test_samples', 'MNIST: independent held-out test audit', [
        ('Riesz', audit_root / ar['study'] / f"samples_{ar['update']:07d}.npy", ar['update'], ar['fid']),
        ('FM', audit_root / af['study'] / f"samples_{af['update']:07d}.npy", af['update'], af['fid'])],
        9274311, 'Same test reference and fresh noise; different training budgets and model sizes.')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, dataset, title in zip(axes, ['mnist', 'fashionmnist'], ['MNIST', 'FashionMNIST']):
        for suffix, label, color, style in [('riesz', 'Symmetric paired Riesz', '#16836b', '-'),
                                          ('fm', 'Flow matching', '#596b88', '-')]:
            rows = runs[dataset + '_' + suffix]['metrics']
            ax.plot([row['update'] / 1000 for row in rows], [row['fid'] for row in rows],
                    marker='o', color=color, linestyle=style, label=label)
        if dataset == 'mnist':
            rows = runs['mnist_smooth']['metrics']
            ax.plot([row['update'] / 1000 for row in rows], [row['fid'] for row in rows],
                    marker='o', color='#a46b35', linestyle='--', label='Riesz smooth / no RMS')
        ax.set(title=title, xlabel='Optimizer updates (thousands)', ylabel='Training-reference FID-10K')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Recorded evaluations only | lower FID is better | one training seed', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'figures/fid_curves.png', dpi=180)
    plt.close(fig)

    lines = ['# Riesz flow versus flow matching: performance and generated images', '',
             f'**Snapshot:** {now}. This report freezes completed evaluations; it is not a live dashboard.', '',
             '## Main findings', '']
    for dataset, title in [('mnist', 'MNIST'), ('fashionmnist', 'FashionMNIST')]:
        a, b = runs[dataset + '_riesz'], runs[dataset + '_fm']
        update = matched_updates[dataset]
        ra, rb = by_update(a)[update], by_update(b)[update]
        ba, bb = min(a['metrics'], key=lambda row: row['fid']), min(b['metrics'], key=lambda row: row['fid'])
        lines += [f"- **{title}:** at {update:,} updates, symmetric paired Riesz scores **{ra['fid']:.2f}**, "
                  f"versus **{rb['fid']:.2f}** for FM. Best recorded scores are "
                  f"{ba['fid']:.2f} at {ba['update']:,} for Riesz and {bb['fid']:.2f} at {bb['update']:,} for FM."]
    lines += [f"- **Independent MNIST test check:** the preserved {ar['update']:,}-update Riesz checkpoint scored "
              f"**{ar['fid']:.2f}**, versus **{af['fid']:.2f}** for the {af['update']:,}-update FM checkpoint, "
              'using the same fresh noise and held-out reference.', '',
              '**Interpretation:** these are comparisons of the deployed recipes. Riesz uses a larger network, '
              'a larger batch, a different learning rate, and a different output parameterization. '
              'The results do not isolate the loss or establish equal-compute superiority. All training runs shown use seed 0.', '',
              '## Evaluation protocol', '',
              '- Development FID: 10,000 generated images against a fixed 10,000-image training subset.',
              '- Reference subset: the same dataset and selection seed 3456 for both methods.',
              '- Generated development noise: seed 100000, reused across the displayed runs and checkpoints.',
              '- Features: pytorch-fid Inception, 2,048 dimensions; grayscale repeated to RGB, bilinear resizing 28 -> 32 -> 299.',
              '- Sampling: Riesz uses 100 Euler steps; FM uses 50 Heun steps. Both use 100 network evaluations, but the networks have different sizes.',
              '- Original development Inception batches: 32 for the new Riesz runs and 64 for the older FM runs. The independent audit uses 32 for both.',
              '- Independent audit: all 10,000 MNIST test images and common fresh noise seed 9274311.',
              '- Figures use actual saved generator outputs. No new models or FID evaluations were run to create this report.', '',
              '## FID versus optimizer updates', '',
              '![FID curves](figures/fid_curves.png)', '']
    for dataset, title in [('mnist', 'MNIST'), ('fashionmnist', 'FashionMNIST')]:
        a, b = by_update(runs[dataset + '_riesz']), by_update(runs[dataset + '_fm'])
        lines += [f'### {title}', '', '| Updates | Symmetric paired Riesz | Flow matching |',
                  '| ---: | ---: | ---: |']
        for update in sorted(set(a) | set(b)):
            av = f"{a[update]['fid']:.2f}" if update in a else 'Not yet evaluated'
            bv = f"{b[update]['fid']:.2f}" if update in b else 'Not evaluated'
            lines.append(f'| {update:,} | {av} | {bv} |')
        lines += ['', f'**Generated images at {matched_updates[dataset]:,} updates:**', '',
                  f'![{title} generated images]({sample_figures[dataset]})', '',
                  'Each panel shows the first 64 samples from the saved 10,000-image evaluation array, '
                  'in original order. Individual images were not selected for appearance. '
                  'The scalar FID comes from the original full evaluation, not from these grids.', '']

    lines += ['## Independent held-out MNIST comparison', '',
              '| Method | Checkpoint update | Test FID-10K | Parameters |',
              '| --- | ---: | ---: | ---: |',
              f"| Symmetric paired Riesz | {ar['update']:,} | **{ar['fid']:.2f}** | {ar['parameters']:,} |",
              f"| Flow matching | {af['update']:,} | {af['fid']:.2f} | {af['parameters']:,} |", '',
              'The Riesz checkpoint was frozen after its first scheduled development evaluation. '
              'Both checkpoints were then evaluated with the same fresh noise and test reference. '
              'This test score belongs to the 5,000-update Riesz checkpoint; do not attach it to a later checkpoint.', '',
              f"![Independent test samples]({sample_figures['audit']})", '',
              'FashionMNIST Riesz does not yet have an independent held-out audit in this report. '
              'Its comparisons above are development scores.', '',
              '## Model and training differences', '',
              '| Setting | Symmetric paired Riesz | Flow matching baseline |',
              '| --- | --- | --- |',
              '| Backbone | Deep time-conditioned U-Net, width 32 | Deep time-conditioned U-Net, width 16 |',
              '| Parameters | 1,370,881 | 381,441 |',
              '| Training batch | 256 | 128 |',
              '| Peak Adam learning rate | 0.0002 | 0.0005 |',
              '| Planned update budget / cosine schedule | 100,000 | 20,000 |',
              '| Network output | Clean-image prediction, converted to velocity | Velocity |',
              '| Training times | One random time shared by the interaction batch | Random time per example |',
              '| Sampling | Euler, 100 steps | Heun, 50 steps |',
              '| Sampling network evaluations | 100 | 100 |',
              '| Initialization | Random | Random |',
              '| Pretrained encoder in generator loss | None | None |', '',
              'Both use a linear Gaussian-to-image reference path, 500-update learning-rate warmup, '
              'gradient clipping at norm 1, and EMA sampling. At an equal number of updates, Riesz processes '
              'twice as many source examples because its batch is twice as large. '
              'Wall times were affected by shared-GPU scheduling, so this report makes no controlled throughput claim.', '',
              'The Riesz loss retains each particle\'s own paired reference target and includes self negatives '
              'in affinity normalization. It is distinct from the earlier leave-one-out and independent-population runs. '
              'There is no FM weight initialization or added reference-velocity MSE term.', '',
              '## Improvement within the MNIST Riesz recipe', '',
              '| Recipe | 5,000-update FID | 10,000-update FID |',
              '| --- | ---: | ---: |']
    for key, label in [('mnist_original', 'Original paired-positive'),
                       ('mnist_riesz', 'Symmetric paired'),
                       ('mnist_smooth', 'Symmetric + smoothed distance + no RMS')]:
        metric = by_update(runs[key])
        values = [f"{metric[update]['fid']:.2f}" if update in metric else 'Pending' for update in [5000, 10000]]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    lines += ['', 'The symmetric-only run preserves the original paired-positive architecture, batch, '
              'optimizer schedule, initialization, and sampler. The main effective change is inclusion of '
              'zero-displacement self negatives in affinity normalization. The smoothed/no-RMS variant changes '
              'additional settings and is reported separately.', '',
              '## Snapshot status and provenance', '',
              '| Run | Recorded state | Last reported update |', '| --- | --- | ---: |']
    for key in ['mnist_riesz', 'mnist_smooth', 'mnist_fm', 'fashionmnist_riesz', 'fashionmnist_fm']:
        state = runs[key]['status']
        lines.append(f"| {key} | {state['status']} | {state['update']:,} |")
    lines += ['', 'Riesz training and the persistent metric reviewers continue independently of this frozen report. '
              'Scores that arrive after the snapshot are not included.', '',
              '- [Frozen metrics, configurations, statuses, and source hashes](manifest.json)',
              '- [Independent test audit metadata](snapshots/independent_test_audit.json)',
              '- [Report-generation source](source/report_riesz_vs_fm.py)',
              '- [Implementation walkthrough](/home/zonghaoc/mmd_flow_hudson_new/PIXEL_RIESZ_IMPLEMENTATION.md)', '',
              '### Source studies', '']
    for key, run in runs.items():
        lines.append(f"- `{key}`: [{Path(run['path']).name}]({run['path']})")
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (output / 'manifest.json').write_text(json.dumps(dict(
        created_at=now, scope='MNIST and FashionMNIST pixel-only paired-reference Riesz versus FM',
        runs=runs, independent_audit=audit, matched_image_updates=matched_updates,
        provenance=provenance, sample_selection='First 64 samples in original order; no cherry-picking',
        new_fid_evaluations=0), indent=2) + '\n')
    (output / 'source').mkdir()
    shutil.copy2(__file__, output / 'source/report_riesz_vs_fm.py')
    shutil.copy2(ROOT / 'run_storage.py', output / 'source/run_storage.py')
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
