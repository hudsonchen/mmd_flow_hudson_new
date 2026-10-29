"""Official JiT backbone with direct or clean-image-derived RGB velocity."""
import math
import torch
from vendor.jit.model_jit import JiT


def clean_to_velocity(clean, state, time, denominator_floor=.05):
    """Official JiT x-to-v conversion, computed in FP32 for training and sampling."""
    time = torch.as_tensor(time, device=state.device, dtype=torch.float32)
    if time.ndim == 1:
        time = time.reshape(-1, *([1] * (state.ndim-1)))
    return (clean.float()-state.float()) / (1-time).clamp_min(denominator_floor)


class JiTVelocity(torch.nn.Module):
    """Conditional velocity interface with selectable network prediction space.

    X-prediction uses the official JiT denominator floor in both training and
    sampling. No pretrained weights or feature encoder are loaded.
    """
    def __init__(self, patch_size=16, prediction='velocity', denominator_floor=.05,
                 gradient_checkpointing=False):
        super().__init__()
        if patch_size not in (4,16) or prediction not in ('velocity','x'):
            raise ValueError('Unsupported patch size or prediction parameterization')
        if not 0 < denominator_floor <= 1:
            raise ValueError('Denominator floor must be in (0,1]')
        self.prediction = prediction
        self.denominator_floor = denominator_floor
        self.backbone = JiT(depth=12, hidden_size=768, num_heads=12,
            bottleneck_dim=128, in_context_len=32, in_context_start=4,
            patch_size=patch_size, input_size=64, num_classes=1000)
        self.backbone.gradient_checkpointing = gradient_checkpointing

    def forward(self, x, t, labels=None):
        if t.ndim == 0:
            t = t.expand(len(x))
        if labels is None:
            labels = torch.full((len(x),), 1000, device=x.device, dtype=torch.long)
        output = self.backbone(x, t, labels)
        if self.prediction == 'x':
            return clean_to_velocity(output, x, t, self.denominator_floor)
        return output


def sampling_nfe(steps, solver, prediction='velocity'):
    return steps if solver == 'euler' else 2*steps-int(prediction == 'x')


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
                # Official JiT uses Euler for the final interval with x-prediction.
                if solver == 'heun' and not (getattr(model, 'prediction', None) == 'x' and k == steps-1):
                    v2 = velocity(x+v/steps,(k+1)/steps)
                    v = (v+v2)/2
            x = x+v/steps
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite velocity samples')
        return x
    finally:
        model.train(previous)
