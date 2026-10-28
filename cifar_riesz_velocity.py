"""Direct-velocity CIFAR Riesz training with global detached particle fields."""
import math
import torch
from cifar_flow_matching import DDPMFlow, skewed_times
from small_flow import riesz_loss


def training_time(device, policy='skewed', generator=None):
    """One shared time per interaction batch, excluding a zero-length horizon."""
    if policy == 'skewed':
        return skewed_times(1, device, generator).squeeze(0).clamp(max=.999)
    if policy == 'uniform':
        return torch.rand((), device=device, generator=generator) * .999
    raise ValueError('Unknown training time distribution')


def global_particle_loss(local_prediction, predictions, targets, rank, time, horizon,
                         powers=(.8, 1., 1.5), taus=(.2, .05, .02), time_power=1.):
    """Reattach only this rank's predictions to an otherwise detached global field.

    All ranks construct exactly the same global particle population, scale,
    affinities and force. The caller multiplies the backward loss by world size
    to undo DDP's gradient averaging. No support/field derivative is taken,
    matching the existing detached-target Riesz surrogate.
    """
    if not 0 <= time < 1 or not 0 < horizon <= 1:
        raise ValueError('Invalid reference time or horizon')
    if not math.isfinite(time_power) or time_power < 0:
        raise ValueError('Invalid reference time weight')
    population = [x.detach() for x in predictions]
    population[rank] = local_prediction
    loss, info = riesz_loss(torch.cat(population).flatten(1), torch.cat(targets).detach().flatten(1),
        taus=taus, normalization='none', affinity='row-column', include_self_negative=True,
        distance_powers=powers)
    h = min(horizon, 1-time)
    return loss * (1-time)**time_power / h, info


@torch.inference_mode()
def sample_velocity(model, noise, steps=100, amp=False):
    """Uniform Euler sampling; direct velocity is defined at both endpoints."""
    previous_mode = model.training
    model.eval()
    x = noise.float().clone()
    try:
        for k in range(steps):
            t = x.new_tensor(k/steps)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                velocity = model(x, t).float()
            x = x + velocity/steps
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite direct-velocity samples')
        return x
    finally:
        model.train(previous_mode)
