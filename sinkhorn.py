"""Pixel-space W-Flow targets (arXiv:2605.11755, Algorithms 1 and 2).

Independent implementation of balanced log-domain Sinkhorn and the two-batch
debiased velocity. Batch-distance scaling and velocity RMS normalization follow
the official release's quadratic-cost path:
https://github.com/hanjq17/W-Flow/blob/main/drift_loss_ot.py
No encoder, CFG, diagonal masking, or differentiation through OT is used.
"""
import math

import torch


@torch.no_grad()
def sinkhorn_plan(source, support, *, epsilon, iterations=10):
    """Uniform balanced coupling for cost ||x-y||^2/2, with fixed iterations.

    Finite iteration counts need not exactly satisfy both marginals. Return
    the actual plan so callers can measure this approximation error.
    """
    if (source.ndim != 2 or support.ndim != 2 or source.shape[1] != support.shape[1]
            or not len(source) or not len(support)):
        raise ValueError('Sinkhorn needs nonempty [particles, dimension] populations')
    if not math.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
        raise ValueError('Sinkhorn epsilon and iteration count must be positive')
    cost = 0.5 * torch.cdist(source, support).square()
    log_kernel = -cost / epsilon
    log_a, log_b = -math.log(len(source)), -math.log(len(support))
    log_v = support.new_zeros(len(support))
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return (log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)).exp()


@torch.no_grad()
def sinkhorn_barycenter(source, support, *, epsilon, iterations=10):
    plan = sinkhorn_plan(source, support, epsilon=epsilon, iterations=iterations)
    rows = plan.sum(dim=1, keepdim=True)
    # Normalize actual row masses, as in the release, rather than assuming
    # exact convergence after the fixed iteration budget.
    projected = plan @ support / rows.clamp_min(1e-12)
    row_error = (rows.flatten() - 1 / len(source)).abs().max()
    col_error = (plan.sum(dim=0) - 1 / len(support)).abs().max()
    return projected, torch.maximum(row_error, col_error)


def sinkhorn_target_loss(features, target, reference, *, epsilon=0.05, iterations=10):
    """Regress toward x + T_real(x) - T_independent_generated(x).

    `reference` must be a separately sampled, detached generator population.
    The two maps share the same batch scale and entropic regularization.
    This scalar loss is a regression surrogate, not a Sinkhorn energy estimate.
    """
    if reference is None:
        raise ValueError('W-Flow requires an independent generated reference batch')
    with torch.no_grad():
        x, y, ref = features.detach().float(), target.detach().float(), reference.detach().float()
        dimension = x.shape[1]
        support = torch.cat((x, ref, y))
        distance_rms = (torch.cdist(x, support).square().mean() / (1 + 1e-8)).sqrt()
        scale = (distance_rms / math.sqrt(dimension)).clamp_min(1e-3)
        xs, ys, rs = x / scale, y / scale, ref / scale
        regularization = epsilon * dimension
        attraction, real_error = sinkhorn_barycenter(
            xs, ys, epsilon=regularization, iterations=iterations)
        repulsion, generated_error = sinkhorn_barycenter(
            xs, rs, epsilon=regularization, iterations=iterations)
        velocity = attraction - repulsion
        velocity_rms = velocity.square().mean().clamp_min(1e-8).sqrt()
        goal = xs + velocity / velocity_rms
        if not torch.isfinite(goal).all():
            raise RuntimeError('Nonfinite Sinkhorn regression target')
        info = dict(sinkhorn_velocity_raw_rms=velocity.square().mean().sqrt(),
                    sinkhorn_marginal_max_error=torch.maximum(real_error, generated_error))
    return (features.float() / scale - goal).square().mean(), info
