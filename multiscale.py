"""Pixel-only coarse and aligned-patch signals for one-pass generators."""
import math
import torch
from torch.nn import functional as F
from loss import _riesz_topk_softmax_loss_impl, velocity


def objective_surrogate(features, target, chunk_size, *, objective, tau_list,
                        normalize_per_temperature):
    target = target.detach()
    if objective == 'softmax-riesz':
        # Preserve upstream's batch-distance and per-coordinate normalization.
        value, _ = _riesz_topk_softmax_loss_impl(
            features.unsqueeze(0), target.unsqueeze(0), tau_list=tau_list,
            normalize_per_temperature=normalize_per_temperature, epsilon=1e-6)
        return value.mean()
    if objective == 'riesz':
        # RMS pixel geometry avoids distance growth solely from view dimension.
        features = features / math.sqrt(features.shape[-1])
        target = target / math.sqrt(target.shape[-1])
        gradient = -velocity(features.detach(), target, chunk_size) / len(features)
        return (features * gradient).sum()
    raise ValueError(f'Unknown objective: {objective}')


def multiscale_loss(images, target, image_shape, chunk_size, *,
                    objective='softmax-riesz', tau_list=(0.2, 0.05, 0.02),
                    normalize_per_temperature=True, coarse_sizes=(8, 16),
                    patch_sizes=(7, 14), crops_per_size=1, patch_rng=None,
                    global_weight=1.0, coarse_weight=1.0, patch_weight=1.0):
    """Return a weighted loss and detached group diagnostics.

    Crop coordinates are shared across generated/real populations, without
    pairing images. Average views within groups, then normalize group weights.
    Ordinary Riesz values are gradient surrogates, not MMD energy measurements.
    """
    channels, height, width = image_shape
    coarse_sizes, patch_sizes = tuple(coarse_sizes), tuple(patch_sizes)
    weights = (global_weight, coarse_weight, patch_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('View weights must be finite, nonnegative, and have positive sum')
    if (coarse_weight and not coarse_sizes) or (patch_weight and not patch_sizes):
        raise ValueError('An enabled group needs at least one view size')
    if crops_per_size < 1 or any(s < 1 or s > min(height, width)
                                for s in (*coarse_sizes, *patch_sizes)):
        raise ValueError('Invalid crop count or view size for this image shape')
    generated = images.reshape(-1, channels, height, width)
    real = target.detach().reshape(-1, channels, height, width)

    def loss(x, y):
        return objective_surrogate(x.flatten(1), y.flatten(1), chunk_size,
                                   objective=objective, tau_list=tau_list,
                                   normalize_per_temperature=normalize_per_temperature)

    groups = {}
    if global_weight:
        groups['global'] = loss(generated, real)
    if coarse_weight:
        groups['coarse'] = torch.stack([
            loss(F.adaptive_avg_pool2d(generated, (size, size)),
                 F.adaptive_avg_pool2d(real, (size, size)))
            for size in coarse_sizes]).mean()
    if patch_weight:
        patches = []
        for size in patch_sizes:
            for _ in range(crops_per_size):
                top = int(torch.randint(height - size + 1, (), generator=patch_rng))
                left = int(torch.randint(width - size + 1, (), generator=patch_rng))
                patches.append(loss(generated[:, :, top:top + size, left:left + size],
                                    real[:, :, top:top + size, left:left + size]))
        groups['patch'] = torch.stack(patches).mean()
    total = sum(groups[name] * weight for name, weight in
                zip(('global', 'coarse', 'patch'), weights) if weight) / sum(weights)
    return total, {**{name: value.detach() for name, value in groups.items()},
                   'combined': total.detach()}
