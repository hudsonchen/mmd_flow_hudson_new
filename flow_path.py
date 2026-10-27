"""Affine path and time-conditioned DiT for distributional transport.

The prescribed path is the affine noise-to-data path used by flow matching:
https://facebookresearch.github.io/flow_matching/generated/flow_matching.path.AffineProbPath.html
Training uses Riesz-softmax marginal matching, not conditional velocity MSE.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint

from backbone import CIFAR10DiTParticleGenerator
from loss import _riesz_topk_softmax_loss_impl, raw_field_diagnostics


def interpolate(clean, noise, t):
    """Noise at t=0, centered image data at t=1; no clipping along the path."""
    if clean.shape != noise.shape:
        raise ValueError('Clean images and noise must have the same shape')
    t = torch.as_tensor(t, device=clean.device, dtype=clean.dtype)
    if t.ndim == 1:
        if len(t) != len(clean):
            raise ValueError('One time per image is required')
        t = t.reshape(-1, *([1] * (clean.ndim - 1)))
    return (1 - t) * noise + t * clean


class TimeConditionedDiT(CIFAR10DiTParticleGenerator):
    """Predict unrestricted image-space velocity with time-dependent adaLN gates."""
    def __init__(self, width=384, depth=12, heads=6, patch_size=2):
        super().__init__(width, depth, heads, patch_size)
        self.width = width
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, width))
        self.time_modulations = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(width, 6 * width))
            for _ in range(depth)])
        for modulation in self.time_modulations:
            torch.nn.init.zeros_(modulation[-1].weight)
            torch.nn.init.zeros_(modulation[-1].bias)

    def forward(self, state, time):
        if state.ndim != 2 or state.shape[1] != 3072:
            raise ValueError('Time-conditioned DiT expects flattened CIFAR states')
        time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
        if time.ndim == 0:
            time = time.expand(len(state))
        if time.shape != (len(state),):
            raise ValueError('Time must be scalar or have one value per image')
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            self.width // 2, device=state.device, dtype=state.dtype) / (self.width // 2))
        phase = time[:, None] * 1000 * frequencies[None]
        conditioning = self.time_embedding(torch.cat([phase.cos(), phase.sin()], dim=1))
        tokens = self.patch_embed(state.reshape(-1, 3, 32, 32)).flatten(2).transpose(1, 2)
        tokens = tokens + self.positions
        for block, modulation in zip(self.blocks, self.time_modulations):
            values = block.modulation[None] + modulation(conditioning).reshape(-1, 6, self.width)
            shift1, scale1, gate1, shift2, scale2, gate2 = [v[:, None] for v in values.unbind(1)]
            batch, count, width = tokens.shape
            normalized = block.norm1(tokens) * (1 + scale1) + shift1
            qkv = block.qkv(normalized).reshape(batch, count, 3, block.heads, width // block.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attended = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            tokens = tokens + gate1 * block.projection(attended.transpose(1, 2).reshape(batch, count, width))
            tokens = tokens + gate2 * block.mlp(block.norm2(tokens) * (1 + scale2) + shift2)
        return self.unpatchify(self.head(self.norm(tokens))).flatten(1)


class DiscreteFlow:
    def __init__(self, steps=10):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('Flow steps must be a positive integer')
        self.steps = steps

    def transition(self, model, state, index):
        """Map path marginal index to index+1, using one residual network call."""
        if not 0 <= index < self.steps:
            raise ValueError('Transition index is outside the time grid')
        return state + model(state, index / self.steps) / self.steps

    def differentiable_sample(self, model, noise, *, checkpoint_steps=True):
        """Full sampler graph, with optional recomputation to bound activation memory.

        Never detach intermediate states: the endpoint field must update every
        transition. Non-reentrant checkpoints also work when noise has no gradient.
        """
        state = noise.detach()
        for index in range(self.steps):
            def advance(current, k=index):
                return self.transition(model, current, k)
            state = (checkpoint(advance, state, use_reentrant=False)
                     if checkpoint_steps else advance(state))
        return state

    @torch.no_grad()
    def sample(self, model, noise, return_trajectory=False):
        training = model.training
        model.eval()
        try:
            state = noise
            trajectory = [state] if return_trajectory else None
            for index in range(self.steps):
                state = self.transition(model, state, index)
                if not torch.isfinite(state).all():
                    raise RuntimeError(f'Nonfinite state at sampling transition {index}')
                if trajectory is not None:
                    trajectory.append(state)
            return trajectory if return_trajectory else state
        finally:
            model.train(training)


def rollout_fraction_at_step(update, maximum, warmup_steps):
    """Linear exposure ramp, measured in updates of the current training stage."""
    if update < 0 or warmup_steps < 0 or not 0 <= maximum <= 1:
        raise ValueError('Invalid rollout schedule')
    return maximum * (min(update / warmup_steps, 1.) if warmup_steps else 1.)


@torch.no_grad()
def mixed_source_states(model, clean, noise, index, *, flow_steps=10, rollout_fraction=0.):
    """Replace a fraction of path particles by detached current-model rollouts.

    The first particles are an IID subset of the sampled batch. Only their
    prefixes are evaluated; the differentiated transition is computed later.
    At index zero both sources are exactly Gaussian noise.
    """
    if not 0 <= rollout_fraction <= 1 or not 0 <= index < flow_steps:
        raise ValueError('Invalid rollout fraction or transition index')
    source = interpolate(clean, noise, index / flow_steps)
    count = int(len(source) * rollout_fraction + .5) if index else 0
    if count:
        state = noise[:count]
        training = model.training
        model.eval()
        try:
            flow = DiscreteFlow(flow_steps)
            for k in range(index):
                state = flow.transition(model, state, k)
                if not torch.isfinite(state).all():
                    raise RuntimeError(f'Nonfinite training rollout at transition {k}')
        finally:
            model.train(training)
        source[:count] = state
    return source.detach(), count / len(source)


def flow_matching_loss(model, clean, noise, time, *, flow_steps=10):
    """Conditional affine-path velocity MSE, per example, plus detached step RMS.

    The target y-z is paired with x_t=(1-t)z+t*y. Rollout states do not have
    this conditional target and must not be substituted here. flow_steps only
    sets the displacement diagnostic; it does not scale the velocity loss.
    """
    clean, noise = clean.detach(), noise.detach()
    state = interpolate(clean, noise, time)
    prediction = model(state, time)
    per_example = (prediction - (clean - noise)).square().flatten(1).mean(1)
    displacement = prediction.detach().square().flatten(1).mean(1).sqrt() / flow_steps
    return per_example, displacement


def transition_loss(model, source_clean, source_noise, target_clean, target_noise, index,
                    *, flow_steps=10, encoder=None, pool_size=4, tau_list=(.2, .05, .02),
                    repulsion_weight=1., source_state=None):
    """Unpaired Riesz matching of a predicted next state and the next path marginal.

    All particles in this call have the same time index. Source and target
    populations use independently sampled clean images and Gaussian noise.
    """
    flow = DiscreteFlow(flow_steps)
    source = (interpolate(source_clean.detach(), source_noise.detach(), index / flow_steps)
              if source_state is None else source_state.detach())
    if source.shape != source_clean.shape:
        raise ValueError('Source state must match the source batch shape')
    target = interpolate(target_clean.detach(), target_noise.detach(), (index + 1) / flow_steps)
    prediction = flow.transition(model, source, index)
    value, diagnostics = _marginal_riesz_loss(prediction, target, encoder=encoder,
        pool_size=pool_size, tau_list=tau_list, repulsion_weight=repulsion_weight)
    diagnostics['displacement_rms'] = (prediction.detach() - source).square().mean().sqrt()
    return value, diagnostics


def continuous_transition_loss(model, source_clean, source_noise, target_clean, target_noise, time,
                               *, transport_horizon=.1, encoder=None, pool_size=4,
                               tau_list=(.2, .05, .02), repulsion_weight=1.):
    """Riesz marginal matching at a continuous time shared by the whole population.

    t is sampled uniformly on [0,1), h=min(transport_horizon,1-t).
    Match x_t+h*v(x_t,t) to independent samples of p_(t+h). The training
    horizon is independent of the numerical sampler's step count. This is a
    finite-interval marginal objective, not conditional velocity regression.
    """
    time = float(time)
    if not math.isfinite(time) or not 0 <= time < 1:
        raise ValueError('Continuous source time must be in [0,1)')
    if not math.isfinite(transport_horizon) or not 0 < transport_horizon <= 1:
        raise ValueError('Transport horizon must be in (0,1]')
    delta = min(transport_horizon, 1-time)
    source = interpolate(source_clean.detach(), source_noise.detach(), time)
    target = interpolate(target_clean.detach(), target_noise.detach(), time+delta)
    prediction = source + delta*model(source, time)
    value, diagnostics = _marginal_riesz_loss(prediction, target, encoder=encoder,
        pool_size=pool_size, tau_list=tau_list, repulsion_weight=repulsion_weight)
    diagnostics.update(displacement_rms=(prediction.detach()-source).square().mean().sqrt(),
        source_time=prediction.new_tensor(time), target_time=prediction.new_tensor(time+delta),
        transport_delta=prediction.new_tensor(delta))
    return value, diagnostics


def _marginal_riesz_loss(prediction, target, *, encoder, pool_size, tau_list, repulsion_weight):
    if encoder is not None:
        # This affine conversion is not clipped: intermediate path states can
        # leave the clean-image range. The same map is applied to both sets.
        generated_features = encoder.flow_features((prediction + 1) / 2, pool_size)
        with torch.no_grad():
            real_features = encoder.flow_features((target + 1) / 2, pool_size)
    else:
        generated_features, real_features = prediction, target
    value, info = _riesz_topk_softmax_loss_impl(
        generated_features.unsqueeze(0), real_features.unsqueeze(0), tau_list=tau_list,
        normalize_per_temperature=False, no_field_rms_normalization=True,
        repulsion_weight=repulsion_weight, epsilon=1e-6)
    diagnostics = raw_field_diagnostics(info)
    return value.mean(), diagnostics


def patch_phase_rms(state):
    """RMS of per-image/channel mean differences across the four 2x2 phases."""
    images = state.reshape(-1, 3, 32, 32)
    phases = torch.stack([images[:, :, a::2, b::2].mean((-1, -2))
                          for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))], dim=-1)
    return (phases - phases.mean(-1, keepdim=True)).square().mean().sqrt()


def endpoint_riesz_loss(model, noise, target_clean, *, flow_steps=10,
                        tau_list=(.2, .05, .02), repulsion_weight=1., checkpoint_steps=True):
    """Match complete noise-to-image rollouts to real images with a separate field.

    Pixel-space Riesz only: no paired target, teacher states, clipping, or
    intermediate detach. The full particle population participates in the field.
    """
    prediction = DiscreteFlow(flow_steps).differentiable_sample(
        model, noise, checkpoint_steps=checkpoint_steps)
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Nonfinite endpoint rollout')
    value, info = _riesz_topk_softmax_loss_impl(
        prediction.unsqueeze(0), target_clean.detach().unsqueeze(0), tau_list=tau_list,
        normalize_per_temperature=False, no_field_rms_normalization=True,
        repulsion_weight=repulsion_weight, epsilon=1e-6)
    diagnostics = {'endpoint_' + k: v for k, v in raw_field_diagnostics(info).items()}
    with torch.no_grad():
        diagnostics.update(endpoint_patch_phase_rms=patch_phase_rms(prediction),
                           endpoint_outside_range=(prediction.abs() > 1).float().mean())
    return value.mean(), diagnostics


class FlowSampler(torch.nn.Module):
    """Adapter for the existing FID evaluator; clipping occurs only at final output."""
    def __init__(self, model, flow_steps=10):
        super().__init__()
        self.model = model
        self.flow = DiscreteFlow(flow_steps)

    def forward(self, noise):
        state = self.flow.sample(self.model, noise)
        return ((state + 1) / 2).clamp(0, 1)
