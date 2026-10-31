"""Shared cyclic architecture, hybrid residual forecasts and proper diagnostics."""
import math
import numpy as np
import torch
from torch import nn
from legacy_core import integrate, riesz_loss, scores as base_scores


class ForecastNet(nn.Module):
    """Exactly this class and these inputs are used by every objective."""
    def __init__(self, width=48, depth=3):
        super().__init__()
        self.register_buffer('frequency', math.pi * 2**torch.arange(4).float())
        self.input = nn.Conv1d(13, width, 3, padding=1, padding_mode='circular')
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(),
            nn.Conv1d(width,width,3,padding=1,padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(width,width,3,padding=1,padding_mode='circular')) for _ in range(depth)])
        self.output = nn.Conv1d(width,1,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, context, control, lead, t):
        b,n,d=x.shape
        t=torch.as_tensor(t,device=x.device,dtype=x.dtype).expand(b,n,1)
        phase=t*self.frequency
        features=torch.cat([lead[:,None,None].expand(b,n,1),t,phase.sin(),phase.cos()],-1)
        spatial=torch.stack([x,context[:,None].expand_as(x),control[:,None].expand_as(x)],-2)
        inputs=torch.cat([spatial,features[:,:,:,None].expand(b,n,10,d)],-2)
        z=self.input(inputs.reshape(b*n,13,d))
        for block in self.blocks:
            z=z+block(z)/math.sqrt(len(self.blocks))
        return self.output(torch.nn.functional.silu(z)).reshape(b,n,d)


def energy_loss(draws, targets):
    """Unbiased energy score, beta=1, averaged over conditioning groups."""
    n=draws.shape[1]
    if n<2:
        raise ValueError('At least two independent generator samples are required')
    cross=torch.cdist(draws,targets,compute_mode='donot_use_mm_for_euclid_dist').mean()
    within=torch.cdist(draws,draws,compute_mode='donot_use_mm_for_euclid_dist').sum((-2,-1))
    return cross-.5*(within/(n*(n-1))).mean()


def objective(model, clean, noise, context, control, lead, t, method, c):
    if method=='energy_score':
        draws=noise+model(noise,context,control,lead,0.)
        return energy_loss(draws,clean)/math.sqrt(clean.shape[-1])
    source=(1-t)*noise+t*clean
    v=model(source,context,control,lead,t)
    if method=='flow_matching':
        return (v-(clean-noise)).square().mean()
    if method not in ('riesz','riesz_no_repulsion'):
        raise ValueError(method)
    h=min(c['horizon'],1-t)
    loss,_=riesz_loss(source+h*v,(1-t-h)*noise+(t+h)*clean,
        c['taus'],c['powers'],repulsion=float(method=='riesz'))
    return loss*(1-t)**c['time_power']/h


@torch.inference_mode()
def sample(model, noise, context, control, lead, method, steps=20):
    if method=='energy_score':
        return noise+model(noise,context,control,lead,0.)
    x=noise.clone()
    for k in range(steps):
        v=model(x,context,control,lead,k/steps)
        predicted=x+v/steps
        v2=model(predicted,context,control,lead,(k+1)/steps)
        x=x+(v+v2)/(2*steps)
    if not torch.isfinite(x).all():
        raise RuntimeError('Nonfinite generated samples')
    return x


def variogram(ensemble, reference):
    """Standard p=0.5 variogram score, averaged over unordered spatial pairs."""
    i,j=np.triu_indices(ensemble.shape[-1],1)
    pred=np.sqrt(np.abs(ensemble[:,:,i]-ensemble[:,:,j])).mean(1)
    total=np.zeros(len(ensemble))
    for lo in range(0,reference.shape[1],32):
        truth=np.sqrt(np.abs(reference[:,lo:lo+32,i]-reference[:,lo:lo+32,j]))
        total+=((truth-pred[:,None])**2).sum((1,2))
    return total/(reference.shape[1]*len(i))


def shuffle_members(x, seed=371):
    """Preserve each coordinate's empirical marginal, destroy joint pairing."""
    rng=np.random.default_rng(seed)
    out=np.empty_like(x)
    for ci in range(len(x)):
        for di in range(x.shape[-1]):
            out[ci,:,di]=x[ci,rng.permutation(x.shape[1]),di]
    return out


def scores(x,y,threshold,extended=True):
    result=base_scores(x,y,threshold)
    if extended:
        x,y=np.asarray(x,np.float64),np.asarray(y,np.float64)
        result['variogram']=variogram(x,y)
        result['shuffled_variogram']=variogram(shuffle_members(x),y)
        result['shuffle_variogram_difference']=result['shuffled_variogram']-result['variogram']
        # Rank histogram bins include ranks 0 and M. Store contextwise frequencies.
        ranks=np.zeros(y.shape,dtype=np.int32)
        for member in range(x.shape[1]):
            ranks+=(x[:,member:member+1]<y)
        bins=np.minimum((ranks*10)//(x.shape[1]+1),9)
        for k in range(10):
            result[f'rank_bin_{k}']=(bins==k).mean((1,2))
    return result
