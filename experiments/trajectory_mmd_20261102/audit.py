"""Replay artifact identities and finite coverage; no distribution-quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(path):
    receipt=json.loads((path/'audit.json').read_text())
    config=json.loads((path/'config.json').read_text())
    assert receipt['passed'] and receipt['source_hashes_verified']
    assert json.loads((path/'ready.json').read_text())['ready']
    assert json.loads((path/'history.json').read_text())[-1]['update']==config['updates']
    for name,expected in receipt['files'].items():
        assert hashlib.sha256((path/name).read_bytes()).hexdigest()==expected,name
    for name,expected in config['source_hashes'].items():
        assert hashlib.sha256((path/'source'/name).read_bytes()).hexdigest()==expected,name
    for file in path.glob('*.npz'):
        with np.load(file) as arrays:
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files),file
    marginal=json.loads((path/'marginals.json').read_text())
    assert [row['t'] for row in marginal]==[0,.25,.5,.75,1]
    refinement=json.loads((path/'solver_refinement.json').read_text())
    assert set(refinement)==set(map(str,[config['train_steps'],config['eval_steps'],2*config['eval_steps']]))
    return {'run':str(path),'passed':True,'quality_accepted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+')
    args=p.parse_args()
    for run in args.runs:
        print(json.dumps(audit(Path(run))))
