"""One-pass, guidance-conditioned training shared by all three particle fields."""
import math
import torch

from multiscale import multiscale_loss, objective_surrogate


def sample_cfg(rng, count, minimum=1.0, maximum=4.0, power=5.0, no_guidance_fraction=0.0):
    """Sample density proportional to scale**(-power), as in the releases."""
    if (not all(math.isfinite(v) for v in (minimum, maximum, power, no_guidance_fraction))
            or minimum < 1 or maximum < minimum or not 0 <= no_guidance_fraction <= 1):
        raise ValueError('Invalid guidance sampling range, power, or fraction')
    u = torch.rand(count, generator=rng)
    exponent = 1 - power
    scales = (torch.exp(math.log(minimum) + u * math.log(maximum / minimum))
              if abs(exponent) < 1e-6 else
              (minimum ** exponent + u * (maximum ** exponent - minimum ** exponent)) ** (1 / exponent))
    return torch.where(torch.rand(count, generator=rng) < no_guidance_fraction, 1., scales)


def guided_generator_step(model, optimizer, class_pools, unconditional_uint8, *,
                          rng, latent_rng, reference_rng, particles, latent_dim,
                          positives, unconditional_samples, groups, cfg_min, cfg_max,
                          cfg_power, cfg_no_guidance_fraction, objective, tau_list,
                          normalize_per_temperature, chunk_size, multiscale_options,
                          sinkhorn_epsilon, sinkhorn_iters, max_grad_norm):
    """Average independent class/scale groups before one optimizer/EMA update.

    Groups run sequentially to bound activation memory. Real unconditional
    samples are drawn from the full training distribution, including the
    selected class. Sinkhorn's generated reference uses the same class/scale
    and independent fresh noise.
    """
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    classes = sorted(class_pools)
    selected = torch.randint(len(classes), (groups,), generator=rng)
    scales = sample_cfg(rng, groups, cfg_min, cfg_max, cfg_power, cfg_no_guidance_fraction)
    totals = {}
    for index, scale in zip(selected.tolist(), scales.tolist()):
        label = classes[index]
        pool = class_pools[label]
        ids = torch.randperm(len(pool), generator=rng)[:positives]
        positive = pool[ids.to(device)]
        ids = torch.randint(len(unconditional_uint8), (unconditional_samples,), generator=rng)
        unconditional = unconditional_uint8[ids].to(device=device, dtype=torch.float32) / 255
        labels = torch.full((particles,), label, device=device, dtype=torch.long)
        latent = torch.randn(particles, latent_dim, generator=latent_rng).to(device)
        images = model(latent, labels=labels, cfg_scale=scale)
        reference = None
        if objective == 'sinkhorn':
            with torch.no_grad():
                z = torch.randn(particles, latent_dim, generator=reference_rng).to(device)
                reference = model(z, labels=labels, cfg_scale=scale)
        if multiscale_options is None:
            values = {}
            loss = objective_surrogate(
                images, positive, chunk_size, objective=objective, tau_list=tau_list,
                normalize_per_temperature=normalize_per_temperature, reference=reference,
                sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=sinkhorn_iters,
                diagnostics=values, unconditional=unconditional, cfg_scale=scale)
            info = {name: torch.stack(items).mean() for name, items in values.items()}
            info['combined'] = loss.detach()
        else:
            loss, info = multiscale_loss(
                images, positive, chunk_size=chunk_size, objective=objective,
                tau_list=tau_list, normalize_per_temperature=normalize_per_temperature,
                reference_images=reference, unconditional_images=unconditional, cfg_scale=scale,
                sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iters=sinkhorn_iters,
                **multiscale_options)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite guided regression loss')
        (loss / groups).backward()
        for name, value in info.items():
            totals[name] = totals.get(name, 0.) + float(value) / groups
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm or float('inf'),
                                         error_if_nonfinite=True)
    optimizer.step()
    totals.update(gradient_norm=float(norm), cfg_scale_mean=float(scales.mean()))
    return totals
