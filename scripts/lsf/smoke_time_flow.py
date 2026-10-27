"""Exercise all ten time bins, the selected training objective, and ten-step sampling."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from data import load_dataset
from flow_path import (DiscreteFlow, TimeConditionedDiT, transition_loss, mixed_source_states,
                       flow_matching_loss, endpoint_riesz_loss, continuous_transition_loss)
from main import update_ema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lpips-cache', type=Path)
    parser.add_argument('--softmax-feature-space', choices=['pixels', 'vgg'], default='vgg')
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--rollout-fraction', type=float, default=0.)
    parser.add_argument('--endpoint-weight', type=float, default=0.)
    parser.add_argument('--fork-checkpoint', type=Path)
    parser.add_argument('--objective', choices=['riesz', 'flow-matching'], default='riesz')
    parser.add_argument('--riesz-time-mode', choices=['discrete', 'continuous'], default='discrete')
    parser.add_argument('--transport-horizon', type=float, default=.1)
    parser.add_argument('--flow-steps', type=int, default=10)
    args = parser.parse_args()
    if args.objective == 'flow-matching' and (args.softmax_feature_space != 'pixels' or args.rollout_fraction):
        parser.error('Flow matching requires pixels and no rollout mixing')
    if args.flow_steps < 1 or not math.isfinite(args.transport_horizon) or not 0 < args.transport_horizon <= 1:
        parser.error('Invalid sampling steps or transport horizon')
    if args.riesz_time_mode == 'continuous' and (
            args.objective != 'riesz' or args.rollout_fraction or args.endpoint_weight):
        parser.error('Continuous-time Riesz requires no rollout mixing or endpoint loss')
    if not math.isfinite(args.endpoint_weight) or args.endpoint_weight < 0:
        parser.error('Endpoint weight must be finite and nonnegative')
    if args.endpoint_weight and (args.objective != 'riesz' or args.softmax_feature_space != 'pixels'):
        parser.error('Endpoint Riesz requires the pixel-space Riesz objective')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    _, (raw, _), _ = load_dataset('cifar10', args.data_root)
    images = raw[:384].to(device='cuda', dtype=torch.float32).flatten(1) / 127.5 - 1
    model = TimeConditionedDiT().cuda()
    if args.fork_checkpoint:
        checkpoint = torch.load(args.fork_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    encoder = None
    if args.softmax_feature_space == 'vgg':
        from perceptual import VGGPerceptualLoss
        encoder = VGGPerceptualLoss(args.lpips_cache, 'cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    records = []
    torch.cuda.synchronize()
    started = time.monotonic()
    continuous_riesz = args.riesz_time_mode == 'continuous'
    coverage = [0., .113, .237, .361, .489, .523, .647, .771, .893, .999] if continuous_riesz else range(args.flow_steps)
    for index in coverage:
        optimizer.zero_grad(set_to_none=True)
        source, target = images[:128], images[128:]
        noise = torch.randn_like(source)
        if args.objective == 'flow-matching':
            # Cover each tenth of the continuous training-time distribution.
            times = (index + torch.rand(len(source), device='cuda')) / args.flow_steps
            per_example, displacement = flow_matching_loss(model, source, noise, times, flow_steps=args.flow_steps)
            value = per_example.mean()
            info = dict(displacement_rms=displacement.mean())
            actual_fraction = 0.
        elif continuous_riesz:
            value, info = continuous_transition_loss(model, source, noise, target,
                torch.randn_like(target), index, transport_horizon=args.transport_horizon, encoder=encoder)
            actual_fraction = 0.
        else:
            state, actual_fraction = mixed_source_states(model, source, noise, index,
                flow_steps=args.flow_steps, rollout_fraction=args.rollout_fraction)
            assert not state.requires_grad
            value, info = transition_loss(model, source, noise, target,
                torch.randn_like(target), index, flow_steps=args.flow_steps, encoder=encoder, source_state=state)
        value.backward()
        endpoint_record = {}
        if args.endpoint_weight:
            endpoint_value, endpoint_info = endpoint_riesz_loss(model, torch.randn_like(source), target,
                                                               flow_steps=args.flow_steps)
            (args.endpoint_weight * endpoint_value).backward()
            assert torch.isfinite(endpoint_value)
            assert all(torch.isfinite(v) for v in endpoint_info.values())
            endpoint_record = dict(endpoint_loss=endpoint_value.item(),
                                   **{k:float(v) for k,v in endpoint_info.items()})
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        update_ema(model, ema, args.ema_decay)
        assert torch.isfinite(value)
        assert all(torch.isfinite(v) for v in info.values())
        records.append(dict(**({'source_time': index, 'target_time':float(info['target_time'])}
                               if continuous_riesz else {'transition':index}), loss=value.item(), grad_norm=grad.item(),
                            rollout_fraction=actual_fraction, **endpoint_record))
    torch.cuda.synchronize()
    seconds_per_update = (time.monotonic() - started) / len(coverage)
    assert model.time_embedding[0].weight.grad.abs().sum() > 0
    if encoder is not None:
        assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    trajectory = DiscreteFlow(args.flow_steps).sample(ema, torch.randn(64, 3072, device='cuda'), True)
    assert len(trajectory) == args.flow_steps+1 and all(torch.isfinite(x).all() for x in trajectory)
    result = dict(status='passed', gpu=torch.cuda.get_device_name(),
                  objective=args.objective, paired_velocity_supervision=args.objective == 'flow-matching',
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  flow_steps=args.flow_steps, feature_space=args.softmax_feature_space, source_batch=128,
                  riesz_time_mode=args.riesz_time_mode, transport_horizon=args.transport_horizon,
                  target_batch=128 if args.objective == 'flow-matching' else 256,
                  ema_decay=args.ema_decay, rollout_fraction=args.rollout_fraction,
                  endpoint_weight=args.endpoint_weight, endpoint_backprop_steps=args.flow_steps if args.endpoint_weight else 0,
                  seconds_per_update=seconds_per_update, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  transitions=records, teacher_frozen=encoder is not None, trajectory_states=len(trajectory))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
