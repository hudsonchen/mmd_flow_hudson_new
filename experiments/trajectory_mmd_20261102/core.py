"""Direct marginal MMD through a continuous, time-conditioned particle ODE."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Velocity(nn.Module):
    def __init__(self, dataset, width=64):
        super().__init__()
        self.dataset = dataset
        dim = 2 if dataset == 'mog' else 784
        # Globally connected image velocity: all state coordinates remain pixels.
        self.net = nn.Sequential(nn.Linear(dim + 9, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
            nn.SiLU(), nn.Linear(width, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        flat = x.flatten(1)
        time = flat.new_full((len(x), 1), float(t))
        phase = time * flat.new_tensor([1., 2., 4., 8.]) * math.pi
        return self.net(torch.cat([flat, time, phase.sin(), phase.cos()], 1)).reshape_as(x)


def integrate(model, z, end=1., steps=16, checkpoint_steps=False):
    """Heun discretization; no state detachment, resets, or terminal clipping."""
    if steps < 1 or not 0 <= end <= 1:
        raise ValueError('Require steps >= 1 and end in [0, 1]')
    x = z
    dt = float(end) / steps
    for i in range(steps):
        def advance(state, t=i * dt):
            v = model(state, t)
            return state + .5 * dt * (v + model(state + dt * v, t + dt))
        x = checkpoint(advance, x, use_reentrant=False) if checkpoint_steps else advance(x)
    return x


def distances(x, y):
    return (x.square().sum(1)[:, None] + y.square().sum(1)[None] - 2*x@y.T).clamp_min(0)


def offdiag(k):
    if len(k) < 2:
        raise ValueError('U-statistic needs at least two observations')
    return (k.sum() - k.diagonal().sum()) / (len(k) * (len(k)-1))


def gaussian_mmd(x, y, bandwidths=(.2, .4, .8, 1.6)):
    x, y = x.flatten(1), y.flatten(1)
    xx, yy, xy = distances(x, x), distances(y, y), distances(x, y)
    return torch.stack([offdiag((-xx/(2*b*b)).exp()) +
        offdiag((-yy/(2*b*b)).exp()) - 2*(-xy/(2*b*b)).exp().mean()
        for b in bandwidths]).mean()


def pixel_marginal_mmd(x, clean, t, scales=(1, 2, 4)):
    """Exact Gaussian-noise integration for R_t=t*clean+(1-t)*N(0,I).

    Full U-statistic, including the reference-reference term. Pooling is fixed,
    nonoverlapping and linear; projected Gaussian variance is (1-t)^2/scale^2.
    A positive full-image Gaussian component keeps the kernel characteristic.
    """
    values = []
    for scale in scales:
        a = (F.avg_pool2d(x, scale) if scale > 1 else x).flatten(1)
        b = (F.avg_pool2d(clean, scale) if scale > 1 else clean).flatten(1) * t
        dim = a.shape[1]
        xx, yy, xy = distances(a, a), distances(b, b), distances(a, b)
        variance = (1-t)**2 / scale**2
        for bandwidth in (.05, .1, .2, .4):
            length2 = dim * ((t*bandwidth)**2 + variance)
            kxx = (-xx/(2*length2)).exp()
            kxy = (-xy/(2*(length2+variance)) +
                .5*dim*math.log(length2/(length2+variance))).exp()
            kyy = (-yy/(2*(length2+2*variance)) +
                .5*dim*math.log(length2/(length2+2*variance))).exp()
            values.append(offdiag(kxx) + offdiag(kyy) - 2*kxy.mean())
    return torch.stack(values).mean()


def centers(device):
    angle = torch.arange(8, device=device) * math.pi/4
    return 2*torch.stack([angle.cos(), angle.sin()], 1)


def mixture(n, generator):
    device = generator.device
    return centers(device)[torch.randint(8, (n,), generator=generator, device=device)] + \
        .15*torch.randn(n, 2, generator=generator, device=device)


def objective(dataset, generated, clean, t, reference_generator):
    if dataset == 'mnist':
        return pixel_marginal_mmd(generated, clean, t)
    noise = torch.randn(clean.shape, generator=reference_generator, device=clean.device)
    return gaussian_mmd(generated, t*clean + (1-t)*noise)
