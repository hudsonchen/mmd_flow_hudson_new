"""Detached power-distance Riesz field for paired pixel-space velocity training.

Uses joint positive/negative row-column softmax affinities and cross-mass
balancing (DRIFTING_LICENSE.txt), with summed power/temperature fields.
Self negatives participate in normalization; their displacement is zero.
No field RMS normalization, pretrained features, or auxiliary objective.
"""
import math
import torch


def riesz_loss(prediction, target, taus=(.2, .05, .02), distance_powers=(.8, 1., 1.5)):
    """Return the detached-field surrogate and field diagnostics.

Only the query term is differentiated. Distances/scales, support particles,
affinities, and target field are detached, as in the original velocity runs.
"""
    if (not distance_powers or any(not math.isfinite(p) or not 0 < p <= 2 for p in distance_powers)
            or len(set(distance_powers)) != len(distance_powers)):
        raise ValueError('Distance powers must be distinct finite numbers in (0, 2]')
    if len(prediction) < 2 or len(target) < 1:
        raise ValueError('Need at least two queries and one target')
    if not taus or any(not math.isfinite(t) or t <= 0 for t in taus):
        raise ValueError('Temperatures must be finite and positive')
    with torch.no_grad():
        query = prediction.detach().float()
        positive = target.detach().float()
        support = torch.cat([query, positive])
        distances = torch.cdist(query, support, compute_mode='donot_use_mm_for_euclid_dist')
        scale = distances.mean().clamp_min(1e-3)
        scale_inputs = (scale / math.sqrt(query.shape[-1])).clamp_min(1e-3)
        field = torch.zeros_like(query)
        for power in distance_powers:
            # Keep arithmetic identical to the frozen velocity implementation.
            powered_distance = (distances / scale if power == 1.
                                else distances.square().pow(power / 2) / scale.pow(power))
            for tau in taus:
                logits = -powered_distance / tau
                affinity = (logits.softmax(-1) * logits.softmax(-2)).sqrt()
                neg, pos = affinity[:, :len(query)], affinity[:, len(query):]
                pos_coeff = pos * neg.sum(-1, keepdim=True)
                neg_coeff = neg * pos.sum(-1, keepdim=True)
                chunks = []
                for start in range(0, len(query), 32):
                    q = query[start:start + 32]
                    dp = positive[None] - q[:, None]
                    dn = query[None] - q[:, None]
                    positive_norm = dp.norm(dim=-1)
                    negative_norm = dn.norm(dim=-1)
                    positive_weight = pos_coeff[start:start + 32] / positive_norm.clamp_min(1e-6)
                    negative_weight = neg_coeff[start:start + 32] / negative_norm.clamp_min(1e-6)
                    if power != 1.:
                        positive_weight = positive_weight * power * (positive_norm.clamp_min(1e-6) / scale).pow(power - 1)
                        negative_weight = negative_weight * power * (negative_norm.clamp_min(1e-6) / scale).pow(power - 1)
                    attraction = (dp * positive_weight[..., None]).sum(1)
                    repulsion = (dn * negative_weight[..., None]).sum(1)
                    chunks.append((attraction - repulsion) / tau)
                field += torch.cat(chunks)
        goal = query / scale_inputs + field
    return (prediction / scale_inputs - goal).square().mean(), dict(
        frozen_velocity=field, scale_inputs=scale_inputs)
