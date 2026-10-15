"""Drifting-style detached regression targets, with optional step/noise ablations."""
import torch

from loss import _riesz_topk_softmax_loss_impl


def softmax_target_loss(features, target, *, tau_list,
                        normalize_per_temperature, field_step=1.0,
                        noise_levels=(0.0,), noise_rng=None):
    losses = []
    for sigma in noise_levels:
        # Independent Gaussian noise for the two populations, with no clipping.
        if sigma:
            x = features + sigma * torch.randn(features.shape, device=features.device,
                                              generator=noise_rng)
            y = target.detach() + sigma * torch.randn(target.shape, device=target.device,
                                                      generator=noise_rng)
        else:
            x, y = features, target.detach()

        with torch.no_grad():
            _, info = _riesz_topk_softmax_loss_impl(
                x.detach().unsqueeze(0), y.unsqueeze(0), tau_list=tau_list,
                normalize_per_temperature=normalize_per_temperature,
                epsilon=1e-6, return_field=True)
            scale = info['scale_inputs']
            goal = x.detach() + field_step * scale * info['frozen_velocity'].squeeze(0)
            if not torch.isfinite(goal).all():
                raise RuntimeError('Nonfinite softmax regression target')
        losses.append(((x - goal) / scale).square().mean())
    return torch.stack(losses).mean(), {}
