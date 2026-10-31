"""Lorenz-96, conditional paired-reference Riesz training, and proper scores."""
import math
import numpy as np
import torch
from torch import nn


def rhs(x, forcing=8.0):
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + forcing


def rk4(x, dt, forcing=8.0):
    k1 = rhs(x, forcing)
    k2 = rhs(x + dt * k1 / 2, forcing)
    k3 = rhs(x + dt * k2 / 2, forcing)
    k4 = rhs(x + dt * k3, forcing)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def integrate(x, leads, dt=.005, forcing=8.0):
    steps = [round(t/dt) for t in leads]
    if any(abs(s*dt-t) > 1e-10 for s, t in zip(steps, leads)):
        raise ValueError('Lead times must lie on the solver grid')
    x = np.asarray(x, dtype=np.float64).copy()
    out = []
    for k in range(1, max(steps)+1):
        x = rk4(x, dt, forcing)
        if k in steps:
            out.append(x.copy())
    return np.stack(out, axis=-2)


def riesz_loss(prediction, target, taus=(.2, .05, .02), powers=(.8, 1., 1.5), repulsion=1.):
    """Independent interaction population per context; no cross-context forces.

    Batched algebraic extension of the existing Tiny ImageNet Riesz field.
    repulsion=0 removes only the negative displacement force, retaining the
    full affinity, cross-mass balancing, scale and self-negative convention.
    """
    with torch.no_grad():
        q, p = prediction.detach().float(), target.detach().float()
        n = q.shape[-2]
        distance = torch.cdist(q, torch.cat([q, p], -2),
                               compute_mode='donot_use_mm_for_euclid_dist')
        scale = distance.mean((-2, -1), keepdim=True).clamp_min(1e-3)
        scale_inputs = (scale/math.sqrt(q.shape[-1])).clamp_min(1e-3)
        dn, dp = distance[..., :n], distance[..., n:]
        ns, ps = dn.clamp_min(1e-6), dp.clamp_min(1e-6)
        wp, wn = torch.zeros_like(dp), torch.zeros_like(dn)
        for power in powers:
            powered = distance/scale if power == 1 else distance.square().pow(power/2)/scale.pow(power)
            kp, kn = 1/ps, 1/ns
            if power != 1:
                kp = kp*power*(ps/scale).pow(power-1)
                kn = kn*power*(ns/scale).pow(power-1)
            for tau in taus:
                logits = -powered/tau
                affinity = (logits.softmax(-1)*logits.softmax(-2)).sqrt()
                neg, pos = affinity[..., :n], affinity[..., n:]
                wp += pos*neg.sum(-1, keepdim=True)*kp/tau
                wn += neg*pos.sum(-1, keepdim=True)*kn/tau
        wp.masked_fill_(dp == 0, 0)
        wn.masked_fill_(dn == 0, 0)
        paired = torch.arange(min(n, p.shape[-2]), device=q.device)
        paired_w = wp[..., paired, paired].clone()
        wp[..., paired, paired] = 0
        center = q.mean(-2, keepdim=True)
        qc, pc = q-center, p-center
        attraction = wp @ pc - qc*wp.sum(-1, keepdim=True)
        attraction[..., paired, :] += paired_w[..., None]*(p[..., paired, :]-q[..., paired, :])
        negative = wn @ qc - qc*wn.sum(-1, keepdim=True)
        field = attraction - repulsion*negative
        goal = q/scale_inputs + field
    return ((prediction/scale_inputs-goal).square().mean(), field)


class ForecastFlow(nn.Module):
    def __init__(self, dimension=40, width=256, depth=3):
        super().__init__()
        self.register_buffer('freq', 2**torch.arange(8, dtype=torch.float32)*math.pi)
        self.input = nn.Linear(2*dimension+18, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
            nn.SiLU(), nn.Linear(width, width)) for _ in range(depth)])
        self.output = nn.Linear(width, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, context, lead, t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).expand(*x.shape[:-1], 1)
        context = context[:, None, :].expand_as(x)
        lead = lead[:, None, None].expand(*x.shape[:-1], 1)
        phase = t*self.freq
        z = torch.cat([x, context, lead, t, phase.sin(), phase.cos()], -1)
        z = torch.nn.functional.silu(self.input(z))
        for block in self.blocks:
            z = z + block(z)/math.sqrt(len(self.blocks))
        return self.output(z)


def velocity(model, x, context, lead, t, floor=.05):
    return (model(x, context, lead, t)-x)/max(1-float(t), floor)


def objective(model, clean, noise, context, lead, t, method, config):
    source = (1-t)*noise+t*clean
    v = velocity(model, source, context, lead, t, config['denominator_floor'])
    if method == 'flow_matching':
        return (v-(clean-noise)).square().mean()
    if method not in ('riesz', 'riesz_no_repulsion'):
        raise ValueError(method)
    h = min(config['horizon'], 1-t)
    prediction = source+h*v
    target = (1-t-h)*noise+(t+h)*clean
    loss, _ = riesz_loss(prediction, target, config['taus'], config['powers'],
                        repulsion=float(method == 'riesz'))
    return loss*(1-t)**config['time_power']/h


@torch.inference_mode()
def sample(model, noise, context, lead, steps=50, floor=.05):
    x = noise.clone()
    for k in range(steps):
        t = k/steps
        v = velocity(model, x, context, lead, t, floor)
        predicted = x+v/steps
        if k < steps-1:
            v2 = velocity(model, predicted, context, lead, (k+1)/steps, floor)
            x = x+(v+v2)/(2*steps)
        else:
            x = predicted
    if not torch.isfinite(x).all():
        raise RuntimeError('Nonfinite generated ensemble')
    return x


def scores(ensemble, reference, event_threshold):
    """Per-context scores; references are independent future realizations.

    Fair (U-statistic) CRPS/energy scores remove ensemble-size bias. Coverage
    is central 90% empirical-quantile coverage and has finite-ensemble effects.
    Brier scores average over reference events; no reference is a condition.
    """
    x, y = np.asarray(ensemble, np.float64), np.asarray(reference, np.float64)
    n, m, d = x.shape
    # Average over all reference outcomes without forming a C*M*R*D tensor.
    crps = np.zeros(n)
    energy = np.zeros(n)
    for k in range(y.shape[1]):
        delta = x-y[:, k:k+1]
        crps += np.abs(delta).mean((1, 2))/y.shape[1]
        energy += np.linalg.norm(delta, axis=-1).mean(1)/y.shape[1]
    if m > 1:
        for k in range(m):
            delta = x-x[:, k:k+1]
            crps -= np.abs(delta).sum((1, 2))/(2*m*(m-1)*d)
            energy -= np.linalg.norm(delta, axis=-1).sum(1)/(2*m*(m-1))
    lower, upper = np.quantile(x, [.05, .95], axis=1)
    coverage = ((y >= lower[:, None]) & (y <= upper[:, None])).mean((1, 2))
    xm, ym = x.mean(1), y.mean(1)
    xc, yc = x-xm[:, None], y-ym[:, None]
    cx = np.einsum('cmi,cmj->cij', xc, xc)/max(m-1, 1)
    cy = np.einsum('cri,crj->cij', yc, yc)/(y.shape[1]-1)
    sx = np.sqrt(np.maximum(np.diagonal(cx, axis1=1, axis2=2), 1e-12))
    sy = np.sqrt(np.maximum(np.diagonal(cy, axis1=1, axis2=2), 1e-12))
    corx, cory = cx/(sx[:, :, None]*sx[:, None, :]), cy/(sy[:, :, None]*sy[:, None, :])
    offdiag = ~np.eye(d, dtype=bool)
    px, py = (x.max(-1)>event_threshold).mean(1), (y.max(-1)>event_threshold).mean(1)
    return dict(crps=crps, energy_score=energy,
        coverage90=coverage, width90=(upper-lower).mean(1),
        mean_rmse=np.sqrt(((xm-ym)**2).mean(1)),
        variance_ratio=np.diagonal(cx,axis1=1,axis2=2).mean(1)/np.diagonal(cy,axis1=1,axis2=2).mean(1),
        covariance_rmse=np.sqrt(((cx-cy)**2).mean((1,2))),
        correlation_rmse=np.sqrt(((corx-cory)[:, offdiag]**2).mean(1)),
        brier_extreme=px**2-2*px*py+py,
        extreme_probability_error=np.abs(px-py))
