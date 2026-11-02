"""A bounded clean-image predictor parameterizes a continuous pixel-space ODE."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self, cin, cout, time_dim=96):
        super().__init__()
        self.norm1=nn.GroupNorm(math.gcd(8,cin),cin)
        self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.norm2=nn.GroupNorm(math.gcd(8,cout),cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.time=nn.Linear(time_dim,2*cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()

    def forward(self,x,time):
        h=self.conv1(F.silu(self.norm1(x)))
        scale,shift=self.time(time).chunk(2,1)
        h=self.norm2(h)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        return (self.skip(x)+self.conv2(F.silu(h)))/math.sqrt(2)


class CleanPredictor(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.time=nn.Sequential(nn.Linear(9,96),nn.SiLU(),nn.Linear(96,96))
        self.input=nn.Conv2d(3,width,3,padding=1)
        self.enc1=Block(width,width)
        self.down1=nn.Conv2d(width,2*width,3,2,1)
        self.enc2=Block(2*width,2*width)
        self.down2=nn.Conv2d(2*width,4*width,3,2,1)
        self.mid1=Block(4*width,4*width)
        self.mid2=Block(4*width,4*width)
        self.dec2=Block(6*width,2*width)
        self.dec1=Block(3*width,width)
        self.output=nn.Conv2d(width,1,3,padding=1)
        nn.init.normal_(self.output.weight,std=.001)
        nn.init.zeros_(self.output.bias)
        coordinate=torch.linspace(-1,1,28)
        row,col=torch.meshgrid(coordinate,coordinate,indexing='ij')
        self.register_buffer('coordinates',torch.stack([row,col])[None])

    def forward(self,x,t):
        phase=x.new_tensor([1.,2.,4.,8.])*math.pi*float(t)
        time=self.time(torch.cat([x.new_tensor([float(t)]),phase.sin(),phase.cos()])[None]).expand(len(x),-1)
        a=self.enc1(self.input(torch.cat([x,self.coordinates.expand(len(x),-1,-1,-1)],1)),time)
        b=self.enc2(self.down1(a),time)
        h=self.mid2(self.mid1(self.down2(b),time),time)
        h=self.dec2(torch.cat([F.interpolate(h,size=b.shape[-2:],mode='nearest'),b],1),time)
        h=self.dec1(torch.cat([F.interpolate(h,size=a.shape[-2:],mode='nearest'),a],1),time)
        # Keep initial particles distinguishable during a long contraction path.
        # This is an input skip inside the bounded predictor, not a training target.
        return torch.tanh(x+self.output(F.silu(h)))


def velocity(model,x,t,epsilon=.001):
    return (model(x,t)-x)/(1-t+epsilon)


def integrate(model,z,end=1.,steps=8,epsilon=.001,checkpoint_steps=False):
    """Exponential midpoint in s=log((1+eps)/(1-t+eps)).

    The continuous transformed equation is dy/ds=D_theta(y,t(s))-y.
    Both stages are convex combinations, so every solver grid preserves
    |y_t| <= exp(-s)|z| + (1-exp(-s)). No output or state clipping is used.
    """
    if steps<1 or epsilon<=0 or not 0<=end<=1:
        raise ValueError('Invalid integration configuration')
    horizon=math.log((1+epsilon)/(1-float(end)+epsilon))
    step=horizon/steps
    decay=math.exp(-step)
    half=math.exp(-step/2)
    x=z
    for k in range(steps):
        def advance(state,start=k*step):
            t=(1+epsilon)*(-math.expm1(-start))
            t_mid=(1+epsilon)*(-math.expm1(-(start+step/2)))
            middle=half*state+(1-half)*model(state,t)
            return decay*state+(1-decay)*model(middle,t_mid)
        x=checkpoint(advance,x,use_reentrant=False) if checkpoint_steps else advance(x)
    return x
