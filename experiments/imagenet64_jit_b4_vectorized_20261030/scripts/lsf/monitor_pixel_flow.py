"""Record live campaign status and previews; flag problems for human review."""
import argparse
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_storage import new_run_path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Training may be replacing a status file while we read it.
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=float, default=15)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 1 or args.hours <= 0:
        parser.error('Interval and duration must be positive')
    campaign = read_json(args.campaign)
    if campaign is None:
        parser.error('Cannot read campaign')
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / 'monitor_source.py')
    observed = set()
    status_seen = {}
    scheduler_seen = {}
    start = time.monotonic()

    def emit(event):
        event['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        if event.get('warnings'):
            with (output / 'alerts.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
        if event['kind'] != 'status' or event.get('warnings'):
            print(json.dumps(event), flush=True)

    while True:
        # Re-read so deliberately registered follow-up runs are included.
        campaign = read_json(args.campaign) or campaign
        rows = []
        for run in campaign['runs']:
            path = Path(run['path'])
            training = path / 'training'
            status = read_json(training / 'status.json')
            submission = read_json(path / 'submission.json') or {}
            row = dict(study=path.name, path=str(path), job_id=submission.get('job_id'), status=status)
            row['early_stop'] = read_json(path / 'early_stop_decision.json')
            if status is not None and status != status_seen.get(str(path)):
                warnings = [f'nonfinite {key}' for key in ['loss', 'grad', 'fid']
                            if key in status and not math.isfinite(status[key])]
                emit(dict(kind='status', **row, warnings=warnings))
                status_seen[str(path)] = status
            previews = sorted(training.glob('preview_*.json'))
            row['latest_preview'] = str(previews[-1].with_suffix('.png')) if previews else None
            for file in [*previews, *sorted(training.glob('metrics_*.json')), training / 'final.json']:
                if str(file) in observed:
                    continue
                payload = read_json(file)
                if payload is None:
                    continue
                warnings = []
                for key in ['fid', 'raw_pixel_variance', 'raw_pixel_outside_fraction']:
                    if key in payload and not math.isfinite(payload[key]):
                        warnings.append(f'nonfinite {key}')
                if payload.get('update', 0) >= 1000:
                    variance = payload.get('raw_pixel_variance')
                    if variance is not None and variance < 1e-5:
                        warnings.append('very low across-sample pixel variance; inspect for collapse')
                    if variance is not None and variance > 100:
                        warnings.append('very large raw pixel variance; inspect for divergence')
                emit(dict(kind=file.stem.split('_')[0], study=path.name, file=str(file),
                          image=str(file.with_suffix('.png')) if file.name.startswith('preview_') else None,
                          metrics=payload, warnings=warnings))
                observed.add(str(file))
            rows.append(row)
        ids = [str(row['job_id']) for row in rows if row['job_id']]
        scheduler = None
        if ids:
            try:
                result = subprocess.run(['bjobs', '-w', *ids], text=True, capture_output=True, timeout=15)
                scheduler = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                states = {parts[0]: parts[2] for line in result.stdout.splitlines()
                          if len(parts := line.split()) >= 3 and parts[0] in ids}
                for row in rows:
                    job_id = str(row['job_id'])
                    state = states.get(job_id, scheduler_seen.get(job_id))
                    row['scheduler_state'] = state
                    row['outcome'] = ('stopped_early' if row['early_stop'] and state not in ['RUN', 'PEND']
                                      else 'failed' if state == 'EXIT'
                                      else 'complete' if (row['status'] or {}).get('status') == 'complete'
                                      else state or 'unknown')
                    if state is not None and state != scheduler_seen.get(job_id):
                        emit(dict(kind='scheduler', study=row['study'], job_id=job_id, state=state,
                                  outcome=row['outcome'],
                                  warnings=['job exited unexpectedly; inspect scheduler log']
                                  if row['outcome'] == 'failed' else []))
                        scheduler_seen[job_id] = state
            except (OSError, subprocess.TimeoutExpired) as error:
                scheduler = dict(error=str(error))
        summary = dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       runs=rows, scheduler=scheduler,
                       policy='Records and flags only; no automatic cancellation or tuning. Images require visual review.')
        temporary = output / 'latest.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(output / 'latest.json')
        if args.once or all(row.get('outcome') in ['complete', 'stopped_early', 'failed'] for row in rows):
            break
        if time.monotonic() - start >= args.hours * 3600:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
