"""Inspect raw pixel tails; never interpret clipped displays as ODE outputs."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from run import exact_mmd, sha, write

p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');args=p.parse_args()
torch.set_num_threads(1)
for name in args.runs:
    path=Path(name)
    assert json.loads((path/'ready.json').read_text())['ready']
    config=json.loads((path/'config.json').read_text())
    source=path/f"samples_{config['updates']:06d}.npz"
    samples=np.load(source)
    x=torch.from_numpy(samples['generated'])
    y=torch.from_numpy(samples['reference'])
    rms=x.double().square().flatten(1).mean(1).sqrt()
    result={'sample_sha256':sha(source),'n':len(x),
        'exact_mmd2_float64':exact_mmd(x,y,(1.4,2.8,5.6,11.2)),
        'pixel_mean':float(x.double().mean()),'pixel_std':float(x.double().std()),
        'maximum_absolute_pixel':float(x.abs().max()),
        'image_rms_quantiles':dict(zip(['median','p90','p99','max'],rms.quantile(torch.tensor([.5,.9,.99,1.],dtype=torch.double)).tolist())),
        'fraction_image_rms_above_2':float((rms>2).double().mean()),
        'real_image_rms_max':float(y.double().square().flatten(1).mean(1).sqrt().max()),
        'interpretation':'Raw values, without clipping; real normalized pixels are bounded by 1 in absolute value.'}
    folder=path/'raw_tail_diagnostic';folder.mkdir(exist_ok=False)
    shutil.copy2(__file__,folder/'diagnose_mnist.py')
    shutil.copy2(Path(__file__).parent/'run.py',folder/'run.py')
    write(folder/'metrics.json',result)
    write(folder/'audit.json',{'passed':True,'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
    print(json.dumps(result),flush=True)
