"""Official JiT backbone with direct RGB velocity and paired Riesz utilities."""
import math
import torch
from vendor.jit.model_jit import JiT_B_16


class JiTVelocity(torch.nn.Module):
    """Conditional velocity: retain official null class and all JiT-B/16 layers.

    Only input resolution and output interpretation differ from the released
    network. No clean-image division, pretrained weights, or feature encoder.
    """
    def __init__(self):
        super().__init__()
        self.backbone = JiT_B_16(input_size=64, num_classes=1000)

    def forward(self, x, t, labels=None):
        if t.ndim == 0:
            t = t.expand(len(x))
        if labels is None:
            labels = torch.full((len(x),), 1000, device=x.device, dtype=torch.long)
        return self.backbone(x, t, labels)


def sample_time(device, mean=-.8, std=.8, generator=None):
    """One JiT logit-normal time shared by the global interacting population."""
    return torch.sigmoid(torch.randn((), device=device, generator=generator)*std+mean).clamp(.0001,.999)


def learning_rate(update, peak, warmup):
    """JiT's linear warmup then constant LR, expressed in optimizer updates."""
    return peak * min((update+1)/max(warmup,1),1.)


@torch.inference_mode()
def sample_velocity(model, noise, steps=50, amp=True, solver='heun', labels=None,
                    cfg=2.9, cfg_interval=(.1,1.)):
    if steps < 1 or solver not in ('euler','heun'):
        raise ValueError('Invalid sampling grid')
    previous = model.training
    model.eval()
    x = noise.float().clone()
    if labels is None:
        labels = torch.full((len(x),),1000,device=x.device,dtype=torch.long)

    def velocity(state, time):
        t = state.new_tensor(time)
        conditional = model(state,t,labels).float()
        if cfg == 1 or not cfg_interval[0] <= time <= cfg_interval[1]:
            return conditional
        unconditional = model(state,t,torch.full_like(labels,1000)).float()
        return unconditional + cfg*(conditional-unconditional)

    try:
        for k in range(steps):
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
                v = velocity(x,k/steps)
                if solver == 'heun':
                    v2 = velocity(x+v/steps,(k+1)/steps)
                    v = (v+v2)/2
            x = x+v/steps
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite velocity samples')
        return x
    finally:
        model.train(previous)
