"""Train a time-conditioned DiT with Riesz marginal matching or velocity flow matching."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch

from data import load_dataset
from fid import CIFAR10FID, WEIGHTS
from flow_path import (DiscreteFlow, FlowSampler, TimeConditionedDiT, transition_loss,
                       mixed_source_states, rollout_fraction_at_step, flow_matching_loss,
                       endpoint_riesz_loss, patch_phase_rms, continuous_transition_loss)
from loss import mean_distance, mmd_squared
from main import training_learning_rate, update_ema
from run_storage import new_run_path
from visualization import save_grid, save_trajectory


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--flow-steps', type=int, default=10)
    parser.add_argument('--objective', choices=['riesz', 'flow-matching'], default='riesz')
    parser.add_argument('--riesz-time-mode', choices=['discrete', 'continuous'], default='discrete')
    parser.add_argument('--transport-horizon', type=float, default=.1,
                        help='Maximum training interval for continuous-time Riesz; independent of --flow-steps')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--target-batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--min-learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--rollout-fraction', type=float, default=0.,
                        help='Maximum fraction of source particles from detached current-model rollouts')
    parser.add_argument('--rollout-warmup-steps', type=int, default=5000)
    parser.add_argument('--endpoint-weight', type=float, default=0.,
                        help='Additional pixel Riesz loss through the complete sampler; 0 preserves legacy training')
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--softmax-feature-space', choices=['pixels', 'vgg'], default='vgg')
    parser.add_argument('--lpips-cache', type=Path)
    parser.add_argument('--perceptual-pool-size', type=int, default=4)
    parser.add_argument('--repulsion-weight', type=float, default=1.)
    parser.add_argument('--tau-list', type=float, nargs='+', default=[.2, .05, .02])
    parser.add_argument('--dit-width', type=int, default=384)
    parser.add_argument('--dit-depth', type=int, default=12)
    parser.add_argument('--dit-heads', type=int, default=6)
    parser.add_argument('--dit-patch-size', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--keep-checkpoint-every', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--fid-samples', type=int, default=10000)
    parser.add_argument('--fid-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--resume', type=Path)
    recovery.add_argument('--fork-checkpoint', type=Path,
                          help='Start a new stage from live weights; reset Adam, EMA, RNG and update count')
    args = parser.parse_args()
    if any(getattr(args, name) < 1 for name in [
            'steps', 'flow_steps', 'batch_size', 'target_batch_size', 'save_every',
            'checkpoint_every', 'log_every', 'fid_batch_size', 'threads', 'perceptual_pool_size']):
        parser.error('Step counts, batch sizes, and intervals must be positive')
    if (args.fid_samples < 2 or args.warmup_steps < 0 or args.keep_checkpoint_every < 0
            or args.rollout_warmup_steps < 0):
        parser.error('Invalid FID count, warmup, or retention interval')
    if (not math.isfinite(args.learning_rate) or not math.isfinite(args.min_learning_rate)
            or not 0 <= args.min_learning_rate <= args.learning_rate or args.learning_rate <= 0
            or not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not math.isfinite(args.repulsion_weight) or args.repulsion_weight < 0
            or not math.isfinite(args.rollout_fraction) or not 0 <= args.rollout_fraction <= 1
            or not math.isfinite(args.endpoint_weight) or args.endpoint_weight < 0
            or any(not math.isfinite(t) or t <= 0 for t in args.tau_list)):
        parser.error('Invalid learning rate, EMA, clipping, repulsion, or temperatures')
    if args.softmax_feature_space == 'vgg' and args.lpips_cache is None:
        parser.error('VGG flow requires --lpips-cache')
    if args.objective == 'flow-matching' and (args.softmax_feature_space != 'pixels' or args.rollout_fraction):
        parser.error('Flow matching requires pixel velocity MSE and prescribed path inputs (no rollout mixing)')
    if args.endpoint_weight and (args.objective != 'riesz' or args.softmax_feature_space != 'pixels'):
        parser.error('Endpoint Riesz requires the pixel-space Riesz objective')
    if not math.isfinite(args.transport_horizon) or not 0 < args.transport_horizon <= 1:
        parser.error('Transport horizon must be finite and in (0,1]')
    if args.riesz_time_mode == 'continuous' and (
            args.objective != 'riesz' or args.rollout_fraction or args.endpoint_weight):
        parser.error('Continuous-time Riesz uses affine inputs and one model call; disable rollout mixing and endpoint loss')
    if not args.resume:
        args.output = new_run_path(args.output)
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists() and not args.resume:
        raise FileExistsError('Existing run; choose a fresh output or resume')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    _, (raw, _), (test, _) = load_dataset('cifar10', args.data_root)
    clean = raw.to(device=device, dtype=torch.float32).flatten(1) / 127.5 - 1
    heldout = test[:512].to(device=device, dtype=torch.float32).flatten(1) / 255
    eval_real = (clean[:512] + 1) / 2
    yy_train = mean_distance(eval_real, eval_real, 64)
    yy_test = mean_distance(heldout, heldout, 64)
    model = TimeConditionedDiT(args.dit_width, args.dit_depth, args.dit_heads, args.dit_patch_size).to(device)
    initialization = dict(initialization='fresh_parameters_no_onepass_checkpoint')
    if args.fork_checkpoint:
        checkpoint = torch.load(args.fork_checkpoint, map_location='cpu', weights_only=False)
        for key in ('flow_steps', 'dit_width', 'dit_depth', 'dit_heads', 'dit_patch_size'):
            if checkpoint['arguments'][key] != getattr(args, key):
                raise ValueError(f'Fork architecture mismatch for {key}')
        model.load_state_dict(checkpoint['model'])
        initialization = dict(initialization='fork_live_weights_reset_optimizer_and_ema',
            parent_checkpoint=str(args.fork_checkpoint.resolve()),
            parent_update=checkpoint['update_count'], parent_config=checkpoint['config'])
        del checkpoint
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.9, .999))
    encoder = None
    if args.softmax_feature_space == 'vgg':
        from perceptual import VGGPerceptualLoss
        encoder = VGGPerceptualLoss(args.lpips_cache, device)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                 if k not in ('resume', 'fork_checkpoint')}
    is_flow_matching = args.objective == 'flow-matching'
    continuous_riesz = args.objective == 'riesz' and args.riesz_time_mode == 'continuous'
    config = dict(
        **arguments, architecture='time_conditioned_compact_dit',
        parameter_count=sum(p.numel() for p in model.parameters()),
        time_conditioning='sinusoidal_embedding_and_per_block_adaln', output_activation='identity_velocity',
        **initialization,
        path='x(t)=(1-t)*Gaussian_noise+t*centered_real_image',
        path_reference='https://facebookresearch.github.io/flow_matching/generated/flow_matching.path.AffineProbPath.html',
        time_grid=[k / args.flow_steps for k in range(args.flow_steps + 1)],
        training_time_grid=(None if is_flow_matching or continuous_riesz
                            else [k / args.flow_steps for k in range(args.flow_steps + 1)]),
        time_batching=('independent_uniform_continuous_time_per_image' if is_flow_matching
                       else 'one_uniform_continuous_time_per_minibatch' if continuous_riesz
                       else 'one_shared_transition_index_per_minibatch'),
        source_and_target=('paired_affine_state_and_velocity_y_minus_z' if is_flow_matching
                           else 'independent_real_images_and_independent_noise'),
        training_rule=('conditional_flow_matching_velocity_MSE' if is_flow_matching
                       else 'continuous_time_finite_interval_Riesz_marginal_matching' if continuous_riesz
                       else 'local_Riesz_plus_full_rollout_endpoint_Riesz' if args.endpoint_weight
                       else 'Riesz_softmax_match_predicted_next_marginal_to_affine_next_marginal'),
        endpoint_training=('separate_pixel_Riesz_field_full_sampler_backprop_checkpointed_each_transition'
                           if args.endpoint_weight else 'disabled'),
        continuous_time_interval=('h=min(transport_horizon,1-t), t~Uniform[0,1)' if continuous_riesz else None),
        teacher_forcing=True, paired_velocity_supervision=is_flow_matching,
        transition_metric_count_unit='images_in_time_bin' if is_flow_matching else 'optimizer_updates',
        training_source=('affine_path_and_detached_live_model_rollouts' if args.rollout_fraction
                         else 'affine_path_only'),
        rollout_schedule='linear_ramp_to_maximum_over_stage_updates',
        sampling='noise_then_flow_steps_residual_updates_without_intermediate_clipping',
        sampling_network_evaluations=args.flow_steps,
        final_pixel_mapping='clamp((state+1)/2,0,1)_only_for_evaluation',
        feature_input_mapping=('(state+1)/2_without_clipping_for_both_populations' if encoder is not None
                               else 'centered_pixel_states_without_clipping'),
        softmax_field_normalization='none',
        fid_reference_split='train', fid_reference_count=len(raw), fid_latent_seed=args.seed + 12345,
        fid_weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        frozen_encoder=encoder.metadata if encoder is not None else None)
    data_rng = torch.Generator().manual_seed(args.seed + 1)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
    diagnostic_rows, transition_rows, evaluation_rows, fid_history = [], [], [], []
    totals, time_totals, interval_count = {}, {}, 0
    start, elapsed_offset = 0, 0.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        previous = checkpoint['arguments']
        legacy_defaults = dict(rollout_fraction=0., rollout_warmup_steps=5000, objective='riesz', endpoint_weight=0.,
                               riesz_time_mode='discrete', transport_horizon=.1)
        for key, value in arguments.items():
            if key in ('threads', 'device', 'save_every'):
                continue
            if previous.get(key, legacy_defaults.get(key)) != value:
                raise ValueError(f'Resume configuration mismatch for {key}')
        if checkpoint['config']['frozen_encoder'] != config['frozen_encoder']:
            raise ValueError('Resume frozen encoder mismatch')
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, elapsed_offset = checkpoint['update_count'], checkpoint['elapsed_seconds']
        data_rng.set_state(checkpoint['data_rng'])
        noise_rng.set_state(checkpoint['noise_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        diagnostic_rows, transition_rows = checkpoint['diagnostic_rows'], checkpoint['transition_rows']
        evaluation_rows, fid_history = checkpoint['evaluation_rows'], checkpoint['fid_history']
        totals, time_totals, interval_count = checkpoint['totals'], checkpoint['time_totals'], checkpoint['interval_count']
        for key in ('initialization', 'parent_checkpoint', 'parent_update', 'parent_config'):
            if key in checkpoint['config']:
                config[key] = checkpoint['config'][key]
        del checkpoint
    write_json(args.output / 'config.json', config)
    evaluator = CIFAR10FID(raw, device=device, batch_size=args.fid_batch_size)
    sampler = FlowSampler(ema, args.flow_steps).eval()
    preview_noise = torch.randn(64, 3072, generator=torch.Generator().manual_seed(args.seed + 4)).to(device)
    started = time.monotonic()
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}

    def elapsed():
        return elapsed_offset + time.monotonic() - started

    def flush_metrics():
        write_rows(args.output / 'training_diagnostics.csv', diagnostic_rows)
        write_rows(args.output / 'transition_metrics.csv', transition_rows)
        write_rows(args.output / 'metrics.csv', evaluation_rows)
        write_json(args.output / 'fid_metrics.json', fid_history)

    def save_checkpoint(step):
        checkpoint = dict(arguments=arguments, config=config, model=model.state_dict(), ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(), update_count=step, elapsed_seconds=elapsed(),
            data_rng=data_rng.get_state(), noise_rng=noise_rng.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            diagnostic_rows=diagnostic_rows, transition_rows=transition_rows, evaluation_rows=evaluation_rows,
            fid_history=fid_history, totals=totals, time_totals=time_totals, interval_count=interval_count)
        path = args.output / 'checkpoint_latest.pt'
        torch.save(checkpoint, path.with_suffix('.pt.tmp'))
        path.with_suffix('.pt.tmp').replace(path)
        if args.keep_checkpoint_every and (step % args.keep_checkpoint_every == 0 or step == args.steps):
            retained = args.output / f'checkpoint_step_{step:06d}.pt'
            if not retained.exists():
                import os
                os.link(path, retained)

    def evaluate(step):
        value = evaluator.evaluate_generator(sampler, 3072, count=args.fid_samples, seed=args.seed + 12345)
        with torch.no_grad():
            trajectory = DiscreteFlow(args.flow_steps).sample(ema, preview_noise, return_trajectory=True)
            images = ((trajectory[-1] + 1) / 2).clamp(0, 1)
            pixel_train = mmd_squared(images, eval_real, yy_train, 64)
            pixel_test = mmd_squared(images, heldout, yy_test, 64)
        fid_history.append(dict(iteration=step, fid=value))
        evaluation_rows.append(dict(iteration=step, update=step, train_subset_mmd_squared=pixel_train,
            heldout_mmd_squared=pixel_test, fid=value, sampling_steps=args.flow_steps,
            elapsed_seconds=elapsed()))
        if args.endpoint_weight or continuous_riesz:
            evaluation_rows[-1].update(patch_phase_rms=float(patch_phase_rms(2*images-1)),
                reference_patch_phase_rms=float(patch_phase_rms(2*eval_real-1)))
        save_grid(images, args.output / f'particles_step_{step:06d}.png', image_shape=(3, 32, 32))
        save_trajectory([((s[:8] + 1) / 2).reshape(-1, 3, 32, 32).cpu() for s in trajectory],
                        list(range(args.flow_steps + 1)), args.output, 8, 'step')
        flush_metrics()
        print(f'step={step} FID={value:.6f} NFE={args.flow_steps} elapsed={elapsed():.1f}s', flush=True)

    try:
        flush_metrics()
        if not args.resume:
            evaluate(0)
            save_checkpoint(0)
        for step in range(start + 1, args.steps + 1):
            if continuous_riesz:
                source_time = float(torch.rand((), generator=data_rng))
                # Ten bins summarize coverage only; never round the model's time.
                index = min(int(source_time*10), 9)
            elif not is_flow_matching:
                index = int(torch.randint(args.flow_steps, (), generator=data_rng))
            src_ids = torch.randint(len(clean), (args.batch_size,), generator=data_rng).to(device)
            source = clean[src_ids]
            if not is_flow_matching:
                tgt_ids = torch.randint(len(clean), (args.target_batch_size,), generator=data_rng).to(device)
                target = clean[tgt_ids]
            def flip(images):
                mask = torch.rand(len(images), 1, 1, 1, device=device, generator=noise_rng) < .5
                images = images.reshape(-1, 3, 32, 32)
                return torch.where(mask, images.flip(-1), images).flatten(1)
            source = flip(source)
            if not is_flow_matching:
                target = flip(target)
            source_noise = torch.randn(source.shape, device=device, generator=noise_rng)
            if not is_flow_matching:
                target_noise = torch.randn(target.shape, device=device, generator=noise_rng)
            lr = training_learning_rate(step - 1, args.learning_rate, args.warmup_steps,
                schedule='cosine', minimum=args.min_learning_rate, start=0, total=args.steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            source_state = None
            if args.rollout_fraction:
                fraction = rollout_fraction_at_step(step, args.rollout_fraction, args.rollout_warmup_steps)
                source_state, actual_fraction = mixed_source_states(model, source, source_noise, index,
                    flow_steps=args.flow_steps, rollout_fraction=fraction)
            if is_flow_matching:
                times = torch.rand(len(source), device=device, generator=noise_rng)
                per_example_mse, displacements = flow_matching_loss(
                    model, source, source_noise, times, flow_steps=args.flow_steps)
                value = per_example_mse.mean()
                info = dict(velocity_mse=value.detach(), displacement_rms=displacements.mean())
            elif continuous_riesz:
                value, info = continuous_transition_loss(model, source, source_noise, target, target_noise,
                    source_time, transport_horizon=args.transport_horizon, encoder=encoder,
                    pool_size=args.perceptual_pool_size, tau_list=tuple(args.tau_list),
                    repulsion_weight=args.repulsion_weight)
            else:
                value, info = transition_loss(model, source, source_noise, target, target_noise, index,
                    flow_steps=args.flow_steps, encoder=encoder, pool_size=args.perceptual_pool_size,
                    tau_list=tuple(args.tau_list), repulsion_weight=args.repulsion_weight, source_state=source_state)
            if args.rollout_fraction:
                info['rollout_fraction'] = actual_fraction
                info['scheduled_rollout_fraction'] = fraction
            if not torch.isfinite(value):
                raise RuntimeError(f'Nonfinite transition loss at update {step}')
            value.backward()
            # Release local activations before building the checkpointed endpoint
            # graph; both losses accumulate into the same single optimizer update.
            if args.endpoint_weight:
                endpoint_noise = torch.randn(source.shape, device=device, generator=noise_rng)
                endpoint_value, endpoint_info = endpoint_riesz_loss(model, endpoint_noise, target,
                    flow_steps=args.flow_steps, tau_list=tuple(args.tau_list),
                    repulsion_weight=args.repulsion_weight)
                if not torch.isfinite(endpoint_value):
                    raise RuntimeError(f'Nonfinite endpoint loss at update {step}')
                (args.endpoint_weight * endpoint_value).backward()
                info.update(endpoint_info, endpoint_surrogate=endpoint_value.detach(),
                            combined_surrogate=value.detach() + args.endpoint_weight * endpoint_value.detach())
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            update_ema(model, ema, args.ema_decay)
            info.update(surrogate=value.detach(), grad_norm_before_clip=grad.detach(),
                        clip_fraction=(grad > args.max_grad_norm).float())
            info = {k: float(v) for k, v in info.items()}
            for key, v in info.items():
                totals[key] = totals.get(key, 0.) + v
            if is_flow_matching:
                bins = (times * args.flow_steps).long().cpu()
                losses = per_example_mse.detach().cpu()
                shifts = displacements.cpu()
                for k in bins.unique().tolist():
                    mask = bins == k
                    bin_totals = time_totals.setdefault(k, dict(count=0, surrogate=0., displacement_rms=0.))
                    bin_totals['count'] += int(mask.sum())
                    bin_totals['surrogate'] += float(losses[mask].sum())
                    bin_totals['displacement_rms'] += float(shifts[mask].sum())
            else:
                bin_totals = time_totals.setdefault(index, dict(count=0, surrogate=0., displacement_rms=0.))
                bin_totals['count'] += 1
                bin_totals['surrogate'] += info['surrogate']
                bin_totals['displacement_rms'] += info['displacement_rms']
                if continuous_riesz:
                    for key in ('source_time', 'target_time', 'transport_delta'):
                        bin_totals[key] = bin_totals.get(key, 0.) + info[key]
                if args.rollout_fraction:
                    bin_totals['rollout_fraction'] = bin_totals.get('rollout_fraction', 0.) + actual_fraction
            interval_count += 1
            if step % args.log_every == 0 or step == args.steps or stop_requested:
                diagnostic_rows.append(dict(update=step, interval_updates=interval_count, learning_rate=lr,
                    **{k: v / interval_count for k, v in totals.items()}))
                for k, values in sorted(time_totals.items()):
                    transition_rows.append(dict(update=step, transition=k, count=values['count'],
                        **{key: value / values['count'] for key, value in values.items() if key != 'count'}))
                label = 'velocity_mse' if is_flow_matching else 'surrogate'
                print(f'step={step} {label}={totals["surrogate"]/interval_count:.6f} '
                      + (f'endpoint_surrogate={totals["endpoint_surrogate"]/interval_count:.6f} '
                         if args.endpoint_weight else '') +
                      f'{"time_bins" if continuous_riesz else "transitions"}={sorted(time_totals)} '
                      f'elapsed={elapsed():.1f}s', flush=True)
                totals, time_totals, interval_count = {}, {}, 0
                flush_metrics()
            if step % args.save_every == 0 or step == args.steps:
                evaluate(step)
            if step % args.checkpoint_every == 0 or step == args.steps or stop_requested:
                save_checkpoint(step)
            if stop_requested:
                raise SystemExit(130)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
