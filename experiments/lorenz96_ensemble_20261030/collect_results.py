"""Collect completed artifacts, verify hashes, replay scores and render notebook.

With --wait, monitor only this study every 45 seconds for up to 12 hours.
Never launch/cancel jobs or treat a missing scheduler entry as completion.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from core import scores

ROOT=Path(__file__).resolve().parent
LOCAL=ROOT.parents[1]/'results/lorenz96/ensemble_20261030'
REMOTE='/SAN/intelsys/imagenet_mmd_flow/mmd_flow_hudson_new/results/lorenz96/ensemble_20261030'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def record(**kwargs):
    value=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
    temp=LOCAL/'collector_status.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(LOCAL/'collector_status.json')
    print(json.dumps(value),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(LOCAL/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    code=("from pathlib import Path; import json; p=Path("+repr(REMOTE)+
          "); result={str(f.relative_to(p)):json.loads(f.read_text()) for f in "
          "[p/'preflight/status.json',p/'production/status.json',p/'production/ready.json'] if f.exists()}; print(json.dumps(result))")
    import shlex
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','cs-cluster','python3 -c '+shlex.quote(code)]
    while True:
        try:
            result=subprocess.run(command,text=True,capture_output=True,timeout=35,check=True)
            state=json.loads(result.stdout)
            if any(s.get('stage')=='failed' for s in state.values()):
                raise RuntimeError('Remote worker failed: '+json.dumps(state))
            if state.get('production/ready.json',{}).get('complete'):
                break
            record(stage='waiting',remote=state)
        except (subprocess.SubprocessError,json.JSONDecodeError) as e:
            record(stage='connection_unverified',error=str(e))
            if not args.wait:
                raise
        if not args.wait:
            return
        if time.monotonic()-start>12*3600:
            raise TimeoutError('Collection timed out; remote job was not modified')
        time.sleep(45)
    record(stage='transferring')
    subprocess.run(['rsync','-a','--checksum',f'cs-cluster:{REMOTE}/',str(LOCAL)+'/'],check=True)
    production=LOCAL/'production'
    ready=json.loads((production/'ready.json').read_text())
    audit=json.loads((production/'audit.json').read_text())
    assert audit['passed']
    assert sha(production/'audit.json')==ready['audit_sha256']
    assert sha(production/'summary.json')==ready['summary_sha256']
    assert sha(production/'data.npz')==audit['data_sha256']
    c=json.loads((production/'config.json').read_text())
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert sha(ROOT/name)==digest,name
    replay=[]
    with np.load(production/'data.npz') as data:
        for name,digest in audit['sha256'].items():
            p=production/name;assert sha(p)==digest,name
            meta=json.loads(p.with_suffix('.json').read_text())
            assert sha(p.parent/f'checkpoint_{meta["step"]:06d}.pt')==meta['checkpoint_sha256']
            split=meta['split'];refs=data[split+'_future'].shape[2]//2
            with np.load(p) as values:
                # Replay one fixed held-out context at every lead for every
                # artifact from transferred samples, not the saved summary.
                for li in range(len(c['lead_times'])):
                    actual=scores(values['ensemble'][:1,li],data[split+'_future'][:1,li,:refs],data['event_threshold'][li])
                    for key,v in actual.items():
                        np.testing.assert_allclose(v,values[key][:1,li],rtol=1e-10,atol=1e-10)
            replay.append(name)
    record(stage='rendering',verified_evaluations=len(replay))
    import nbformat
    from nbclient import NotebookClient
    notebook=ROOT/'results.ipynb'
    nb=nbformat.read(notebook,as_version=4)
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,notebook)
    validation=dict(passed=True,remote_audit_sha256=sha(production/'audit.json'),
                    replayed_artifacts=replay,notebook=str(notebook),
                    figure_sha256={p.name:sha(p) for p in (production/'figures').iterdir()})
    (LOCAL/'local_verification.json').write_text(json.dumps(validation,indent=2)+'\n')
    record(stage='complete',local_verification=str(LOCAL/'local_verification.json'))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        record(stage='failed',error=repr(error))
        raise
