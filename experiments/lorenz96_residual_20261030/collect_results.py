"""Monitor only this study, collect audited results, replay and render figures."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
from study import sha, write_json, evaluate_arrays

ROOT=Path(__file__).resolve().parent
LOCAL=ROOT.parents[1]/'results/lorenz96/residual_20261030'
REMOTE='/SAN/intelsys/imagenet_mmd_flow/mmd_flow_hudson_new/results/lorenz96/residual_20261030'


def record(stage,**kwargs):
    value=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
    write_json(LOCAL/'collector_status.json',value);print(json.dumps(value),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(LOCAL/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=("from pathlib import Path; import json; p=Path("+repr(REMOTE)+
        "); print(json.dumps({str(f.relative_to(p)):json.loads(f.read_text()) for f in "
        "[p/'launch_status.json',p/'preflight/status.json',p/'production/status.json',p/'production/ready.json'] if f.exists()}))")
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','cs-cluster','python3 -c '+shlex.quote(code)]
    started=time.monotonic()
    while True:
        try:
            reply=subprocess.run(command,check=True,text=True,capture_output=True,timeout=35)
            state=json.loads(reply.stdout)
            stopped=any(v.get('stage') in ('failed','diagnostic_gate_failed') for v in state.values())
            if stopped or state.get('production/ready.json',{}).get('complete'):break
            record('waiting',remote=state)
        except (subprocess.SubprocessError,json.JSONDecodeError) as error:
            record('connection_unverified',error=str(error))
            if not args.wait:raise
        if not args.wait:return
        if time.monotonic()-started>13*3600:raise TimeoutError('Collector timed out; cluster job was not modified')
        time.sleep(45)
    record('transferring',remote=state)
    subprocess.run(['rsync','-a','--checksum',f'cs-cluster:{REMOTE}/',str(LOCAL)+'/'],check=True)
    if stopped:
        record('worker_stopped',remote=state);return
    out=LOCAL/'production'
    audit=json.loads((out/'audit.json').read_text());ready=json.loads((out/'ready.json').read_text())
    assert audit['passed'] and audit['architecture_identical']
    assert sha(out/'audit.json')==ready['audit_sha256']
    assert sha(out/'summary.json')==ready['summary_sha256']
    assert sha(out/'data.npz')==audit['data_sha256']
    assert sha(ROOT/'source_manifest.json')==audit['source_manifest_sha256']
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    config=json.loads((out/'config.json').read_text());raw=dict(np.load(out/'data.npz'))
    replay=[]
    for name,digest in audit['sha256'].items():
        path=out/name;assert sha(path)==digest,name
        if name=='baselines.npz':continue
        meta=json.loads(path.with_suffix('.json').read_text())
        assert sha(path.parent/f'checkpoint_{meta["step"]:06d}.pt')==meta['checkpoint_sha256']
        with np.load(path) as values:
            actual=evaluate_arrays(values['ensemble'][:1],raw,config,'test',raw['test_future'].shape[2]//2)
            for k,v in actual.items():np.testing.assert_allclose(v,values[k][:1],rtol=1e-9,atol=1e-9)
        replay.append(name)
    record('rendering',replayed_artifacts=len(replay))
    import nbformat
    from nbclient import NotebookClient
    path=ROOT/'results.ipynb';nb=nbformat.read(path,as_version=4)
    NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    write_json(LOCAL/'local_verification.json',dict(passed=True,audit_sha256=sha(out/'audit.json'),
        replayed_artifacts=replay,notebook=str(path),
        figure_sha256={p.name:sha(p) for p in (out/'figures').iterdir()}))
    record('complete',verification=str(LOCAL/'local_verification.json'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        record('failed',error=repr(error));raise
