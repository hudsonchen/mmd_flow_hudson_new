"""Independent-population Gaussian MMD in full, pooled and cropped raw pixels."""
import math
import torch
from torch.nn import functional as F


def squared_distances(x,y):
    return (x.square().sum(1)[:,None]+y.square().sum(1)[None]-2*x@y.T).clamp_min(0)


def off_diagonal(k):
    return (k.sum()-k.diagonal().sum())/(len(k)*(len(k)-1))


def marginal_loss(x,clean,t,scale=1,include_constant=False):
    x=x.flatten(1);y=(clean*t).flatten(1);dim=x.shape[1]
    xx,xy=squared_distances(x,x),squared_distances(x,y)
    yy=squared_distances(y,y) if include_constant else None
    variance=(1-t)**2/scale**2
    terms=[]
    for bandwidth in (.05,.1,.2,.4):
        length=dim*((t*bandwidth)**2+variance)
        kxx=(-xx/(2*length)).exp()
        kxy=(-xy/(2*(length+variance))+.5*dim*math.log(length/(length+variance))).exp()
        value=off_diagonal(kxx)-2*kxy.mean()
        if include_constant:
            kyy=(-yy/(2*(length+2*variance))+.5*dim*math.log(length/(length+2*variance))).exp()
            value=value+off_diagonal(kyy)
        terms.append(value)
    return torch.stack(terms).mean()


def objective(x,clean,t,generator,patch_weight=.25):
    global_terms=[marginal_loss(F.avg_pool2d(x,s) if s>1 else x,
        F.avg_pool2d(clean,s) if s>1 else clean,t,s) for s in (1,2,4)]
    full=torch.stack(global_terms).mean()
    if patch_weight==0:
        return full
    crops=[]
    for size in (7,14):
        row,col=torch.randint(29-size,(2,),generator=generator).tolist()
        crops.append(marginal_loss(x[:,:,row:row+size,col:col+size],
            clean[:,:,row:row+size,col:col+size],t))
    return (1-patch_weight)*full+patch_weight*torch.stack(crops).mean()
