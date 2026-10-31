"""Time-conditioned CIFAR-10 velocity U-Net and skewed reference times.

The state-dict layout is unchanged from existing velocity Riesz checkpoints.
"""
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
