"""Review live pixel-Riesz trials and preserve checkpoints before bounded early stops.

This supervisor evaluates recorded diagnostics, not images or new FID samples.
It never launches training, changes a running recipe, or treats a plateau as
proof of convergence. Visual reviews remain separate artifacts.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_storage import new_run_path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def assess(records, status, previews, baseline_fid=None):
    """Pure decision logic: three >10% regressions, only after 20K updates."""
    records = sorted(records, key=lambda row: row['update'])
    warnings = []
    nonfinite = []
    for kind, row, keys in [('status', status or {}, ['loss', 'grad', 'fid']),
                            *[('fid', row, ['fid']) for row in records],
                            *[('preview', row, ['raw_pixel_variance', 'raw_pixel_outside_fraction'])
                              for row in previews]]:
        nonfinite += [f'{kind}: nonfinite {key}' for key in keys
                      if key in row and not math.isfinite(row[key])]
    if nonfinite:
        return dict(action='stop_nonfinite', reason='; '.join(nonfinite), warnings=nonfinite)
    best = min(records, key=lambda row: row['fid']) if records else None
    latest = records[-1] if records else None
    if len(previews) >= 2 and all(row.get('update', 0) >= 1000 and
                                  row.get('raw_pixel_variance', 1.) < 1e-5
                                  for row in previews[-2:]):
        warnings.append('Very low raw variance in two previews; visual collapse review needed')
    if previews and previews[-1].get('raw_pixel_variance', 0.) > 100:
        warnings.append('Large raw pixel variance; visual divergence review needed')
    result = dict(action='continue', reason='No sustained material regression detected',
                  warnings=warnings, latest=latest, best=best,
                  baseline_fid=baseline_fid,
                  latest_minus_baseline=(latest['fid'] - baseline_fid
                                         if latest and baseline_fid is not None else None))
    if len(records) >= 4 and latest['update'] >= 20000:
        trailing = records[-3:]
        if best['update'] < trailing[0]['update'] and all(row['fid'] > 1.1 * best['fid'] for row in trailing):
            result.update(action='stop_regression', reason=(
                f"Three consecutive FIDs exceed best {best['fid']:.4f} at "
                f"{best['update']} by >10%, and latest update is >=20000. "
                'Resource-allocation early stop; not proof that recovery is impossible.'))
    return result


def preserve_and_stop(study, job_id, destination, assessment):
    """Only issue bkill after readable, hashed best/latest checkpoints exist."""
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ['best.pt', 'latest.pt']:
        source = study / 'training' / name
        if not source.is_file():
            raise RuntimeError(f'Cannot stop without preserving {source}')
        # Trainers atomically replace completed checkpoint files.
        temporary = destination / (name + '.tmp')
        shutil.copy2(source, temporary)
        with temporary.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        temporary.replace(destination / name)
    for pattern in ['metrics_*.json', 'config.json', 'status.json', 'best.json', 'preview_*.json']:
        for source in (study / 'training').glob(pattern):
            shutil.copy2(source, destination / source.name)
    decision = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    job_id=job_id, reason=assessment['reason'], assessment=assessment,
                    archive=str(destination), hashes=hashes, checkpoints_preserved=True,
                    supervisor='supervise_pixel_flow.py')
    write_json(destination / 'decision_before_stop.json', decision)
    result = subprocess.run(['bkill', job_id], capture_output=True, text=True, timeout=30)
    decision['scheduler_stop'] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    write_json(destination / 'decision.json', decision)
    if result.returncode == 0:
        write_json(study / 'early_stop_decision.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Recorded FM training-reference best.json; comparison only')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--stop-on-regression', action='store_true')
    parser.add_argument('--stop-on-nonfinite', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 5 or args.hours <= 0:
        parser.error('Interval must be >=5 seconds and hours positive')
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / 'supervisor_source.py')
    shutil.copy2(ROOT / 'run_storage.py', output / 'run_storage_source.py')
    baseline = read_json(args.baseline)
    if baseline is None or not math.isfinite(baseline['fid']):
        parser.error('A finite recorded baseline score is required')
    write_json(output / 'policy.json', dict(
        campaign=str(args.campaign), baseline_path=str(args.baseline), baseline=baseline,
        interval_seconds=args.interval, hours=args.hours,
        stop_on_regression=args.stop_on_regression, stop_on_nonfinite=args.stop_on_nonfinite,
        regression_rule='Latest update >=20000 and three consecutive FIDs >1.1 * earlier best',
        visual_policy='Record preview paths and numerical warnings; does not inspect images',
        scope='Only runs explicitly registered in this campaign; no new training or recipe edits'))
    seen = {}
    stop_attempted = set()
    start = time.monotonic()
    while True:
        campaign = read_json(args.campaign)
        if campaign is None:
            raise RuntimeError('Campaign became unreadable')
        ids = []
        for run in campaign['runs']:
            submission = read_json(Path(run['path']) / 'submission.json') or {}
            if submission.get('job_id'):
                ids.append(str(submission['job_id']))
        scheduler = subprocess.run(['bjobs', '-w', *ids], capture_output=True, text=True, timeout=30) if ids else None
        states = {parts[0]: parts[2] for line in (scheduler.stdout if scheduler else '').splitlines()
                  if len(parts := line.split()) >= 3 and parts[0] in ids}
        rows = []
        for run in campaign['runs']:
            study = Path(run['path'])
            training = study / 'training'
            status = read_json(training / 'status.json')
            records = [row for file in sorted(training.glob('metrics_*.json'))
                       if (row := read_json(file)) is not None]
            files = sorted(training.glob('preview_*.json'))
            previews = [row for file in files[-2:] if (row := read_json(file)) is not None]
            assessment = assess(records, status, previews, baseline['fid'])
            submission = read_json(study / 'submission.json') or {}
            job_id = str(submission.get('job_id', ''))
            state = states.get(job_id, 'unknown')
            stopped = read_json(study / 'early_stop_decision.json')
            terminal = bool(stopped) or state in ['DONE', 'EXIT'] or (status or {}).get('status') == 'complete'
            if state == 'EXIT' and not stopped:
                assessment['warnings'].append('Unexpected job exit: inspect scheduler.log')
            age = time.time() - (training / 'status.json').stat().st_mtime if status else None
            if state == 'RUN' and age is not None and age > 1200:
                assessment['warnings'].append('No status change for >20 minutes; inspect training/evaluation')
            row = dict(study=study.name, path=str(study), job_id=job_id,
                       scheduler_state=state, status=status, status_age_seconds=age,
                       latest_preview=str(files[-1].with_suffix('.png')) if files else None,
                       assessment=assessment, terminal=terminal)
            key = (state, (status or {}).get('status'), len(records), len(files),
                   assessment['action'], tuple(assessment['warnings']))
            if seen.get(str(study)) != key:
                event = dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **row)
                with (output / 'reviews.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
                seen[str(study)] = key
            enabled = ((assessment['action'] == 'stop_regression' and args.stop_on_regression)
                       or (assessment['action'] == 'stop_nonfinite' and args.stop_on_nonfinite))
            if enabled and not terminal and state == 'RUN' and job_id not in stop_attempted:
                stop_attempted.add(job_id)
                try:
                    row['stop_decision'] = preserve_and_stop(study, job_id, output / 'preserved' / study.name, assessment)
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    row['stop_error'] = str(error)
            rows.append(row)
        write_json(output / 'latest.json', dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                               runs=rows, scheduler_stderr=scheduler.stderr if scheduler else None))
        if args.once or (rows and all(row['terminal'] for row in rows)) or time.monotonic() - start >= args.hours * 3600:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
