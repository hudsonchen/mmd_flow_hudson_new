"""Fresh-noise evaluation of completed MoG runs, separate from monitoring draws."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from core import Velocity, mixture
from run import generate, metrics, sha, write

p=argparse.ArgumentParser()
p.add_argument('runs',nargs='+')
p.add_argument('--n',type=int,default=8192)
args=p.parse_args()
torch.set_num_threads(1)
for name in args.runs:
    path=Path(name)
    assert json.loads((path/'ready.json').read_text())['ready']
    config=json.loads((path/'config.json').read_text())
    assert config['dataset']=='mog'
    output=path/'fresh_validation'
    output.mkdir(exist_ok=False)
    checkpoint=path/'checkpoint_latest.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['update']==config['updates']
    model=Velocity('mog',config['width'],config.get('linear_skip',False)).eval()
    model.load_state_dict(state['model'])
    g=torch.Generator().manual_seed(907000+config['seed'])
    z=torch.randn(args.n,2,generator=g)
    target,other=mixture(args.n,g),mixture(args.n,g)
    previous=None
    rows={}
    for steps in (64,128):
        x=generate(model,z,steps)
        rows[str(steps)]=metrics(x,target,'mog')
        if previous is not None:
            rows[str(steps)]['paired_rms_change']=float((x-previous).square().mean().sqrt())
        previous=x
    result={'seed':config['seed'],'n':args.n,'evaluation_seed':907000+config['seed'],
        'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(Path(__file__)),
        'solver':rows,'real_real':metrics(other,target,'mog')}
    write(output/'metrics.json',result)
    np.savez_compressed(output/'samples.npz',generated=x.numpy(),reference=target.numpy(),noise=z.numpy())
    write(output/'audit.json',{'passed':True,'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(result),flush=True)
