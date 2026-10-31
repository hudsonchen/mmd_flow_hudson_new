"""Only the training objective differs between the two matched experiments."""
import torch
from cifar_riesz_velocity import global_particle_loss


def training_loss(velocity, source, clean, noise, time, horizon, objective,
                  powers=(.8, 1., 1.5), taus=(.2, .05, .02), time_power=1.):
    if objective == 'flow_matching':
        # Exact conditional velocity of x_t=(1-t)*noise+t*clean.
        return (velocity.float() - (clean-noise)).square().mean()
    if objective != 'riesz':
        raise ValueError('Unknown objective')
    h = min(horizon, 1-time)
    prediction = source + h*velocity.float()
    target = (1-time-h)*noise + (time+h)*clean
    loss, _ = global_particle_loss(prediction, [prediction.detach()],
        [target.detach()], 0, time, horizon, powers, taus, time_power)
    return loss
