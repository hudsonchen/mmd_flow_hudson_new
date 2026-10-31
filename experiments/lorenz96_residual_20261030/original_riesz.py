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
        # Reuse direct distances: the paired positive can be very close to its
        # query, so a squared-norm GEMM distance would lose significant digits.
        negative_norm, positive_norm = distances[:, :len(query)], distances[:, len(query):]
        positive_safe = positive_norm.clamp_min(1e-6)
        negative_safe = negative_norm.clamp_min(1e-6)
        positive_weights = torch.zeros_like(positive_norm)
        negative_weights = torch.zeros_like(negative_norm)
        for power in distance_powers:
            powered_distance = (distances / scale if power == 1.
                                else distances.square().pow(power / 2) / scale.pow(power))
            positive_kernel = 1 / positive_safe
            negative_kernel = 1 / negative_safe
            if power != 1.:
                positive_kernel = positive_kernel * power * (positive_safe / scale).pow(power - 1)
                negative_kernel = negative_kernel * power * (negative_safe / scale).pow(power - 1)
            for tau in taus:
                logits = -powered_distance / tau
                affinity = (logits.softmax(-1) * logits.softmax(-2)).sqrt()
                neg, pos = affinity[:, :len(query)], affinity[:, len(query):]
                pos_coeff = pos * neg.sum(-1, keepdim=True)
                neg_coeff = neg * pos.sum(-1, keepdim=True)
                positive_weights.add_(pos_coeff * positive_kernel / tau)
                negative_weights.add_(neg_coeff * negative_kernel / tau)
        # Zero displacements have exactly zero force. Removing their weights
        # avoids cancellation of large singular coefficients in the GEMMs.
        positive_weights.masked_fill_(positive_norm == 0, 0)
        negative_weights.masked_fill_(negative_norm == 0, 0)
        # Evaluate the close paired-positive term directly for endpoint accuracy.
        paired = torch.arange(min(len(query), len(positive)), device=query.device)
        paired_weights = positive_weights[paired, paired].clone()
        positive_weights[paired, paired] = 0
        # sum_j w_ij (y_j - q_i) = (W Y)_i - q_i sum_j w_ij.
        # Centering also makes this robust to a common large coordinate offset.
        center = query.mean(0, keepdim=True)
        centered_query, centered_positive = query - center, positive - center
        field = positive_weights @ centered_positive - negative_weights @ centered_query
        field += centered_query * (negative_weights.sum(1) - positive_weights.sum(1))[:, None]
        field[paired] += paired_weights[:, None] * (positive[paired] - query[paired])
        goal = query / scale_inputs + field
    return (prediction / scale_inputs - goal).square().mean(), dict(
        frozen_velocity=field, scale_inputs=scale_inputs)
