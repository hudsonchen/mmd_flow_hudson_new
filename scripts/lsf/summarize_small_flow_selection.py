"""Verify and collect the selected small-flow benchmarks and training provenance."""
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DRIFT_COMPILE'] = '0'
import torch
from run_storage import new_run_path
from small_flow import validate_riesz_parent


def main():
    torch.set_num_threads(2)
    destination = new_run_path('toy/reference_flow_campaign_20261026')
    inputs, models = {}, {}

    def read(path):
        payload = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def collect(relative, dataset, prefixes=None):
        path = new_run_path(relative)
        audit = read(path)
        rows = [r for r in audit['rows'] if prefixes is None or r['study'].startswith(prefixes)]
        for row in rows:
            name = row['study']
            archived = path.parent / (name + '_checkpoint.pt')
            payload = archived.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row['checkpoint_sha256']:
                raise ValueError(f'Checkpoint changed: {name}')
            cp = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
            role = 'fm_baseline' if cp['config']['objective'] == 'fm' else 'pure_riesz'
            if role == 'pure_riesz':
                validate_riesz_parent(cp)
                if row['lineage_has_flow_matching'] or row['uses_paired_reference_velocity_loss']:
                    raise ValueError(f'Non-pure selected lineage: {name}')
                encoder = row.get('reconstruction_encoder', {})
                if encoder.get('present') and not encoder.get('unchanged'):
                    raise ValueError(f'Unverified reconstruction encoder: {name}')
            if name not in models:
                stages, seen = [], set()
                ancestor, ancestor_payload = cp, payload
                while True:
                    cfg = ancestor['config']
                    manifest_path = Path(cfg['output']).parent / 'manifest.json'
                    manifest = read(manifest_path)
                    stages.append(dict(checkpoint_sha256=hashlib.sha256(ancestor_payload).hexdigest(),
                        selected_update=ancestor['update'], config=cfg, manifest=str(manifest_path),
                        recorded_training_command=manifest['command']))
                    parent = cfg.get('fork')
                    if not parent:
                        break
                    parent = Path(parent).resolve()
                    if parent in seen:
                        raise ValueError('Cyclic checkpoint ancestry')
                    seen.add(parent)
                    ancestor_payload = parent.read_bytes()
                    ancestor = torch.load(io.BytesIO(ancestor_payload), map_location='cpu', weights_only=False)
                models[name] = dict(dataset=dataset, role=role, checkpoint_archive=str(archived),
                    checkpoint_sha256=digest, stages=list(reversed(stages)),
                    selected_transport_updates=sum(s['selected_update'] for s in stages),
                    lineage_stage_training_budgets=sum(s['config']['steps'] for s in stages),
                    reconstruction_priming_updates=sum(s['config'].get('prime_steps', 0) for s in stages),
                    evaluations=[])
            elif models[name]['checkpoint_sha256'] != digest:
                raise ValueError('Conflicting selected checkpoints')
            models[name]['evaluations'].append(dict(audit=str(path), result=row))
        return audit, rows

    mog, mog_rows = collect('toy/mog_tail_three_seed_independent_confirmation_20261027/audit/audit.json',
        'mog', ('reference_fm_fourier_mog_', 'riesz_row_column_mog_tail_weighted_refine_'))
    mnist, mnist_rows = collect('mnist/mnist_pixel_moments_final_fresh_test_audit_20261027/audit/audit.json', 'mnist')
    _, fashion_fm = collect('fashionmnist/fashion_fm_independent_noise_test_audit_20261027/audit/audit.json', 'fashionmnist')
    _, fashion_riesz = collect('fashionmnist/fashion_encoder20k_independent_noise_test_audit_20261027/audit/audit.json', 'fashionmnist')
    if len(mog_rows) != 6 or len(mnist_rows) != 2 or len(fashion_fm) != 1 or len(fashion_riesz) != 2:
        raise ValueError('Incomplete confirmation audits')
    mnist_protocols = {(r['dataset'], r['reference_split'], r['noise_seed'], r['generated_count'], r['reference_count'], r['fid_batch_size']) for r in mnist_rows}
    if len(mnist_protocols) != 1 or next(iter(mnist_protocols))[:2] != ('mnist', 'test'):
        raise ValueError('MNIST evaluation protocols differ')
    fashion_rows = fashion_fm + fashion_riesz
    if {(r['reference_split'], r['noise_seed'], r['generated_count'], r['reference_count'], r['fid_batch_size']) for r in fashion_rows} != {('test', 8865432, 10000, 10000, 32)}:
        raise ValueError('FashionMNIST evaluation protocols differ')
    if mog['count'] != 200000 or mog['noise_seed'] != 8855123:
        raise ValueError('MoG confirmation protocol differs')
    benchmarks = []
    for prefix, label in [('reference_fm_', 'fm'), ('riesz_', 'riesz')]:
        rows = [r for r in mog_rows if r['study'].startswith(prefix)]
        benchmarks.append(dict(dataset='mog', method=label, training_seeds=len(rows), samples_per_checkpoint=mog['count'],
            mean_outside_percent=100*statistics.mean(r['off_mode_fraction'] for r in rows),
            mean_analytic_sliced_w2=statistics.mean(r['sliced_w2_analytic_target'] for r in rows),
            all_eight_modes=all(r['covered_modes'] == 8 for r in rows),
            all_outside_upper_95ci_below_5_percent=all(r['outside_95ci'][1] < .05 for r in rows),
            network_evaluations=rows[0]['network_evaluations']))
    for dataset, rows in [('mnist', mnist_rows), ('fashionmnist', fashion_rows)]:
        for row in rows:
            benchmarks.append(dict(dataset=dataset, method='fm' if row['study'].startswith('reference_fm') else 'riesz',
                training_seeds=1, fid=row['fid'], network_evaluations=row['network_evaluations'],
                reference_split=row['reference_split'], noise_seed=row['noise_seed']))
    controls = []
    for name in ['riesz_denoiser_fashionmnist_spatial_time_weighted_refine_seed0_20261027',
                 'riesz_denoiser_fashionmnist_spatial_continuation_control_seed0_20261027']:
        folder = new_run_path(Path('fashionmnist') / name / 'training')
        status = read(folder / 'status.json')
        controls.append(dict(study=name, status=status, best=read(folder/'best.json') if (folder/'best.json').exists() else None))
    report = dict(benchmarks=benchmarks, selected_models=models, controls=controls,
        control_runs_complete=all(c['status']['status'] == 'complete' for c in controls),
        interpretation='MoG: three training seeds and independent 200000-sample confirmation. Images: one training seed each, '
        'held-out reference and fresh noise; FashionMNIST confirmation noise is independent of sampler selection. '
        'Small FashionMNIST FID differences support comparable measured quality, not statistical superiority. '
        'Priming and training compute differ; inference evaluations do not measure training efficiency. '
        'Selected model ancestry excludes every FM-initialized or velocity-regression-hybrid Riesz experiment.',
        input_hashes=inputs)
    source_payload = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source_payload).hexdigest()
    source = destination / 'selection_report_sources' / source_hash
    source.mkdir(parents=True, exist_ok=True)
    files = [str(Path(__file__).relative_to(ROOT)), 'run_storage.py', 'small_flow.py', 'backbone.py', 'loss.py', 'sinkhorn.py']
    files += [str(p.relative_to(ROOT)) for p in ROOT.glob('*LICENSE*')]
    if (ROOT / 'assets/SCORE_SDE_LICENSE').exists():
        files.append('assets/SCORE_SDE_LICENSE')
    hashes = {}
    for name in files:
        payload = (ROOT / name).read_bytes()
        archived = source / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    report['report_source_sha256'] = source_hash
    report['report_source_directory'] = str(source)
    report['report_source_hashes'] = hashes
    temporary = destination / 'selected_models.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, default=str) + '\n')
    temporary.replace(destination / 'selected_models.json')
    print(json.dumps(dict(benchmarks=benchmarks, control_runs_complete=report['control_runs_complete']), indent=2))


if __name__ == '__main__':
    main()
