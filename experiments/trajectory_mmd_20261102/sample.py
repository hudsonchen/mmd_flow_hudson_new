"""Standalone generation: needs only a checkpoint and fresh Gaussian noise."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision.utils import save_image
from core import Velocity, integrate

p=argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('output')
p.add_argument('--device',default='cpu')
p.add_argument('--n',type=int,default=256)
p.add_argument('--steps',type=int,default=128)
p.add_argument('--seed',type=int,default=7001)
a=p.parse_args()
torch.set_num_threads(1)
state=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
config=state['config']
model=Velocity(config['dataset'],config['width']).to(a.device).eval()
model.load_state_dict(state['model'])
shape=(2,) if config['dataset']=='mog' else (1,28,28)
g=torch.Generator(device=a.device).manual_seed(a.seed)
z=torch.randn(a.n,*shape,device=a.device,generator=g)
with torch.no_grad():
    x=torch.cat([integrate(model,b,steps=a.steps) for b in z.split(128)])
path=Path(a.output)
if path.exists():
    raise FileExistsError(path)
np.savez_compressed(path,generated=x.cpu().numpy(),noise=z.cpu().numpy())
if config['dataset']=='mnist':
    save_image(((x[:64]+1)/2).clamp(0,1),path.with_suffix('.png'),nrow=8)
