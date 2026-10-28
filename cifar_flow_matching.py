"""Continuous flow matching on the unchanged, time-conditioned DDPM backbone.

Recipe: facebookresearch/flow_matching examples/image (accessed 2026-10-27).
Independent Gaussian/data coupling, skewed times, EMA, EDM-spaced Heun ODE.
This is an adaptation to our backbone, not a reproduction of Meta's FID.
"""
from contextlib import nullcontext
import numpy as np
import torch

from backbone import CIFAR10UNetParticleGenerator


class DDPMFlow(torch.nn.Module):
    def __init__(self, base_channels=128, num_res_blocks=4):
        super().__init__()
        self.backbone = CIFAR10UNetParticleGenerator(
            base_channels, num_res_blocks, time_conditioned=True)

    def forward(self, x, t):
        # Same embedding coordinate as the Riesz U-Net; unrestricted velocity head.
        if t.ndim == 0:
            t = t.expand(len(x))
        return self.backbone(x, 999 * (1 - t))


def skewed_times(count, device, generator=None):
    # sigma ~ LogNormal(-1.2, 1.2), t=1/(1+sigma), Meta CIFAR recipe.
    normal = torch.randn(count, device=device, generator=generator)
    return torch.sigmoid(1.2 - 1.2 * normal).clamp(0.0001, 1.)


def flow_target(clean, noise, t):
    weight = t.reshape(-1, *([1] * (clean.ndim - 1)))
    return noise.lerp(clean, weight), clean - noise


def edm_time_grid(intervals, device='cpu'):
    """EDM rho=7 spacing mapped to flow time; exact t=0 and t=1 endpoints.

    Unlike the upstream example's initial t=1/81, include the Gaussian endpoint.
    `intervals` always means actual Heun steps; NFE is exactly 2*intervals.
    """
    if intervals < 3:
        raise ValueError('At least three integration intervals are required')
    sigma = torch.linspace(80. ** (1/7), .002 ** (1/7), intervals - 1,
                           dtype=torch.float64, device=device).pow(7)
    return torch.cat([sigma.new_zeros(1), 1 / (1 + sigma), sigma.new_ones(1)])


@torch.inference_mode()
def sample_heun(model, noise, intervals=50, amp=False):
    previous_mode = model.training
    model.eval()
    grid = edm_time_grid(intervals, noise.device)
    x = noise.float().clone()
    try:
        for left, right in zip(grid[:-1], grid[1:]):
            dt = (right - left).float()
            with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
                first = model(x, left.float()).float()
                second = model(x + dt * first, right.float()).float()
            x = x + .5 * dt * (first + second)
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite Heun samples')
        return x
    finally:
        model.train(previous_mode)


@torch.no_grad()
def update_ema(model, ema, update, decay=.999):
    effective = min(decay, (1 + update) / (10 + update))
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.lerp_(current, 1 - effective)


class FeatureMoments:
    """Mergeable float64 sufficient statistics for distributed FID."""
    def __init__(self, dimension=2048):
        self.count = 0
        self.sum = np.zeros(dimension, dtype=np.float64)
        self.outer = np.zeros((dimension, dimension), dtype=np.float64)

    def add(self, features):
        values = np.asarray(features, dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite Inception features')
        self.count += len(values)
        self.sum += values.sum(0)
        self.outer += values.T @ values

    def statistics(self):
        if self.count < 2:
            raise ValueError('At least two feature vectors are required')
        mean = self.sum / self.count
        cov = (self.outer - self.count * np.outer(mean, mean)) / (self.count - 1)
        return mean, (cov + cov.T) / 2, self.count
