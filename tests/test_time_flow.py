"""Time-conditioned transport, unpaired marginal targets, rollout, and recovery."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from flow_path import (DiscreteFlow, FlowSampler, TimeConditionedDiT, interpolate, transition_loss,
                       mixed_source_states, rollout_fraction_at_step, flow_matching_loss,
                       endpoint_riesz_loss, patch_phase_rms, continuous_transition_loss)
from loss import _riesz_topk_softmax_loss_impl
import train_time_flow


class TimeFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_continuous_time_is_not_rounded_and_uses_one_model_call(self):
        class Velocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.speed = torch.nn.Parameter(torch.tensor(.3))
                self.calls = []
            def forward(self, state, time):
                self.calls.append((state.detach().clone(), time))
                return state*self.speed
        model = Velocity()
        src, sn = torch.randn(3,3072,requires_grad=True), torch.randn(3,3072,requires_grad=True)
        tgt, tn = torch.randn(5,3072,requires_grad=True), torch.randn(5,3072,requires_grad=True)
        for time in (0., .237, .999):
            with self.subTest(time=time):
                model.calls.clear()
                model.zero_grad()
                value, info = continuous_transition_loss(model,src,sn,tgt,tn,time,transport_horizon=.1)
                delta=min(.1,1-time)
                state=interpolate(src.detach(),sn.detach(),time)
                next_state=state+delta*state*model.speed
                next_target=interpolate(tgt.detach(),tn.detach(),time+delta)
                expected,_=_riesz_topk_softmax_loss_impl(next_state[None],next_target[None],
                    tau_list=(.2,.05,.02),no_field_rms_normalization=True,epsilon=1e-6)
                torch.testing.assert_close(value,expected.mean())
                self.assertEqual(len(model.calls),1)
                self.assertEqual(model.calls[0][1],time)
                torch.testing.assert_close(model.calls[0][0],state)
                self.assertAlmostEqual(float(info['target_time']),time+delta,places=6)
                value.backward()
                self.assertTrue(torch.isfinite(model.speed.grad))
                self.assertGreater(abs(float(model.speed.grad)),0)
                for tensor in (src,sn,tgt,tn):self.assertIsNone(tensor.grad)

    def test_continuous_time_rejects_invalid_intervals_and_grid_dependent_options(self):
        model=TimeConditionedDiT(width=24,depth=1,heads=3)
        data=torch.randn(3,3072)
        for time,horizon in [(1.,.1),(-.1,.1),(float('nan'),.1),(.3,0),(.3,2),(.3,float('nan'))]:
            with self.assertRaises(ValueError):
                continuous_transition_loss(model,data,data,data,data,time,transport_horizon=horizon)
        for extra in [['--rollout-fraction','.5'],['--endpoint-weight','1'],['--objective','flow-matching']]:
            with patch('sys.argv',['train_time_flow.py','--output','/tmp/unused_continuous',
                    '--softmax-feature-space','pixels','--riesz-time-mode','continuous',*extra]):
                with self.assertRaises(SystemExit):train_time_flow.parse_args()

    def test_endpoint_backpropagates_every_transition_without_input_gradients(self):
        class PerStepVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.speeds = torch.nn.Parameter(torch.linspace(.1, .3, 10))
            def forward(self, state, time):
                return state * self.speeds[round(time*10)]
        model = PerStepVelocity()
        noise = torch.randn(3, 3072, requires_grad=True)
        target = torch.randn(5, 3072, requires_grad=True)
        value, info = endpoint_riesz_loss(model, noise, target)
        value.backward()
        self.assertIsNone(noise.grad)
        self.assertIsNone(target.grad)
        self.assertTrue((model.speeds.grad.abs()>0).all())
        self.assertTrue(all(torch.isfinite(v) for v in info.values()))
        with torch.no_grad():
            sampled = DiscreteFlow(10).sample(model, noise)
            expected, _ = _riesz_topk_softmax_loss_impl(sampled[None], target[None],
                tau_list=(.2,.05,.02), no_field_rms_normalization=True, epsilon=1e-6)
        torch.testing.assert_close(value, expected.mean())
        permuted, _ = endpoint_riesz_loss(model, noise, target.flip(0))
        torch.testing.assert_close(value, permuted)

    def test_checkpointed_endpoint_matches_dit_outputs_and_gradients(self):
        model = TimeConditionedDiT(width=24, depth=1, heads=3)
        noise, target = torch.randn(2,3072), torch.randn(3,3072)
        plain, _ = endpoint_riesz_loss(model, noise, target, checkpoint_steps=False)
        plain.backward()
        gradients = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        checked, _ = endpoint_riesz_loss(model, noise, target, checkpoint_steps=True)
        checked.backward()
        torch.testing.assert_close(plain, checked)
        for name, param in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(param.grad, gradients[name], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(DiscreteFlow(10).differentiable_sample(model,noise),
                                   DiscreteFlow(10).sample(model,noise))

    def test_patch_phase_metric_detects_checkerboard(self):
        flat = torch.ones(2,3,32,32)*.2
        self.assertAlmostEqual(float(patch_phase_rms(flat)),0.,places=7)
        flat[:,:,::2,::2] += .1
        self.assertGreater(float(patch_phase_rms(flat)),.04)

    def test_endpoint_rejects_incompatible_or_invalid_options(self):
        for options in (['--endpoint-weight','nan'], ['--endpoint-weight','-1'],
                        ['--endpoint-weight','1','--objective','flow-matching'],
                        ['--endpoint-weight','1','--softmax-feature-space','vgg','--lpips-cache','/unused']):
            with self.subTest(options=options), patch('sys.argv', [
                    'train_time_flow.py','--output','/tmp/unused_endpoint','--softmax-feature-space','pixels',*options]):
                with self.assertRaises(SystemExit):
                    train_time_flow.parse_args()

    def test_path_endpoints_and_ten_step_rollout(self):
        clean, noise = torch.randn(3, 3072), torch.randn(3, 3072)
        torch.testing.assert_close(interpolate(clean, noise, 0), noise, rtol=0, atol=0)
        torch.testing.assert_close(interpolate(clean, noise, 1), clean, rtol=0, atol=0)
        class Oracle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, state, time):
                self.calls.append(time)
                return clean - noise
        model = Oracle()
        trajectory = DiscreteFlow(10).sample(model, noise, True)
        self.assertEqual(model.calls, [k / 10 for k in range(10)])
        self.assertEqual(len(trajectory), 11)
        for k, state in enumerate(trajectory):
            torch.testing.assert_close(state, interpolate(clean, noise, k / 10))
        self.assertTrue(model.training)
        self.assertGreater(trajectory[0].abs().max().item(), 1)
        samples = FlowSampler(model, 10)(noise)
        torch.testing.assert_close(samples, ((clean + 1) / 2).clamp(0, 1))

    def test_time_modulation_learns_and_batch_independence(self):
        model = TimeConditionedDiT(width=24, depth=2, heads=3)
        state = torch.randn(3, 3072)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            time = torch.tensor([0., .4, .9])
            loss = (model(state, time) - time[:, None]).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(model.time_embedding[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.time_modulations[0][-1].weight.grad.abs().sum().item(), 0)
        self.assertFalse(torch.allclose(model(state, 0.), model(state, .9)))
        torch.testing.assert_close(model(state, .4)[:1], model(state[:1], .4))
        self.assertEqual(model(state, .4).shape, state.shape)
        with torch.no_grad():
            model.head.bias.fill_(-10)
        self.assertTrue((model(state, .4) < 0).any())

    def test_transition_is_next_marginal_riesz_not_paired_regression(self):
        model = TimeConditionedDiT(width=24, depth=1, heads=3)
        source, source_noise = torch.rand(3, 3072), torch.randn(3, 3072)
        target, target_noise = torch.rand(5, 3072, requires_grad=True), torch.randn(5, 3072)
        index = 4
        value, _ = transition_loss(model, source, source_noise, target, target_noise, index)
        expected_state = interpolate(source, source_noise, .4)
        expected_next = expected_state + .1 * model(expected_state, .4)
        expected_target = interpolate(target.detach(), target_noise, .5)
        expected, _ = _riesz_topk_softmax_loss_impl(
            expected_next.unsqueeze(0), expected_target.unsqueeze(0), tau_list=(.2, .05, .02),
            no_field_rms_normalization=True, epsilon=1e-6)
        torch.testing.assert_close(value, expected.mean())
        permuted, _ = transition_loss(model, source, source_noise, target.flip(0), target_noise.flip(0), index)
        torch.testing.assert_close(value, permuted)
        value.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_vgg_receives_unclipped_states_and_detached_targets(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []
            def flow_features(self, images, pool):
                self.inputs.append((images.detach().clone(), torch.is_grad_enabled()))
                return images[:, :16]
        encoder = Encoder()
        model = TimeConditionedDiT(width=24, depth=1, heads=3)
        value, _ = transition_loss(model, torch.zeros(3, 3072), torch.full((3, 3072), 4.),
            torch.zeros(5, 3072), torch.full((5, 3072), -4.), 0, encoder=encoder)
        value.backward()
        self.assertTrue(encoder.inputs[0][1])
        self.assertFalse(encoder.inputs[1][1])
        self.assertGreater(encoder.inputs[0][0].max().item(), 1)
        self.assertLess(encoder.inputs[1][0].min().item(), 0)

    def test_mixed_sources_use_detached_live_prefix_and_train_only_final_transition(self):
        class Velocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.speed = torch.nn.Parameter(torch.tensor(.3))
                self.calls = []
            def forward(self, state, time):
                self.calls.append((time, torch.is_grad_enabled(), self.training, len(state)))
                return torch.ones_like(state) * self.speed
        model = Velocity()
        clean = torch.randn(4, 12, requires_grad=True)
        noise = torch.randn_like(clean)
        state, fraction = mixed_source_states(model, clean, noise, 3, rollout_fraction=.5)
        self.assertEqual(fraction, .5)
        self.assertFalse(state.requires_grad)
        self.assertTrue(model.training)
        self.assertEqual(model.calls, [(k / 10, False, False, 2) for k in range(3)])
        torch.testing.assert_close(state[:2], noise[:2] + .09)
        torch.testing.assert_close(state[2:], interpolate(clean[2:], noise[2:], .3))
        target = torch.randn(5, 12, requires_grad=True)
        value, _ = transition_loss(model, clean, noise, target, torch.randn_like(target),
                                   3, source_state=state)
        value.backward()
        self.assertEqual(model.calls[-1], (.3, True, True, 4))
        self.assertIsNone(clean.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(model.speed.grad))
        self.assertGreater(model.speed.grad.abs().item(), 0)
        calls = len(model.calls)
        initial, fraction = mixed_source_states(model, clean, noise, 0, rollout_fraction=1.)
        torch.testing.assert_close(initial, noise)
        self.assertEqual(fraction, 0.)
        self.assertEqual(len(model.calls), calls)

    def test_rollout_exposure_schedule(self):
        self.assertEqual([rollout_fraction_at_step(k, .5, 5000) for k in (0, 2500, 5000, 9000)],
                         [0., .25, .5, .5])
        self.assertEqual(rollout_fraction_at_step(1, .5, 0), .5)

    def test_flow_matching_uses_paired_velocity_without_step_scaling(self):
        clean = torch.randn(3, 3072, requires_grad=True)
        noise = torch.randn(3, 3072, requires_grad=True)
        times = torch.tensor([0., .37, .99])
        class Oracle(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.error = torch.nn.Parameter(torch.tensor(.5))
            def forward(self, state, time):
                torch.testing.assert_close(state, interpolate(clean, noise, times))
                torch.testing.assert_close(time, times)
                return clean.detach() - noise.detach() + self.error
        model = Oracle()
        losses, steps = flow_matching_loss(model, clean, noise, times)
        torch.testing.assert_close(losses, torch.full((3,), .25))
        losses.mean().backward()
        torch.testing.assert_close(model.error.grad, torch.tensor(1.))
        self.assertIsNone(clean.grad)
        self.assertIsNone(noise.grad)
        self.assertFalse(steps.requires_grad)
        other_losses, other_steps = flow_matching_loss(model, clean, noise, times, flow_steps=5)
        torch.testing.assert_close(other_losses, losses)
        torch.testing.assert_close(other_steps, 2 * steps)

    def test_flow_matching_rejects_rollout_mixing(self):
        with patch('sys.argv', ['train_time_flow.py', '--output', '/tmp/unused_fm',
                   '--objective', 'flow-matching', '--softmax-feature-space', 'pixels',
                   '--rollout-fraction', '.5']):
            with self.assertRaises(SystemExit):
                train_time_flow.parse_args()

    def run_training(self, output, extra=()):
        data = torch.arange(8 * 3072).reshape(8, 3072).remainder(256).byte()
        labels = torch.zeros(8, dtype=torch.long)
        argv = ['train_time_flow.py', '--output', str(output), '--steps', '3',
                '--flow-steps', '10', '--dit-width', '24', '--dit-depth', '1', '--dit-heads', '3',
                '--batch-size', '3', '--target-batch-size', '5', '--softmax-feature-space', 'pixels',
                '--save-every', '3', '--checkpoint-every', '1', '--log-every', '1',
                '--fid-samples', '2', '--threads', '1', '--device', 'cpu', *extra]
        with patch('sys.argv', argv), patch.object(train_time_flow, 'load_dataset', return_value=(
                Path('/unused'), (data, labels), (data, labels))), \
                patch.object(train_time_flow, 'CIFAR10FID') as evaluator, \
                patch.object(train_time_flow, 'save_grid'), patch.object(train_time_flow, 'save_trajectory'):
            evaluator.return_value.evaluate_generator.return_value = 100.
            train_time_flow.main()
        return torch.load(output / 'checkpoint_latest.pt', weights_only=False)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.run_training(root / 'full')
            original = train_time_flow.transition_loss
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('interrupt transition training')
                return original(*args, **kwargs)
            with patch.object(train_time_flow, 'transition_loss', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupt transition'):
                    self.run_training(root / 'resume')
            resumed = self.run_training(root / 'resume', ['--resume', str(root / 'resume/checkpoint_latest.pt'),
                                                        '--save-every', '2'])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            self.assertEqual(full['diagnostic_rows'], resumed['diagnostic_rows'])
            self.assertEqual(full['transition_rows'], resumed['transition_rows'])
            self.assertFalse(resumed['config']['paired_velocity_supervision'])
            self.assertEqual(resumed['config']['sampling_network_evaluations'], 10)
            with self.assertRaisesRegex(ValueError, 'mismatch for flow_steps'):
                self.run_training(root / 'resume', ['--resume', str(root / 'resume/checkpoint_latest.pt'),
                                                   '--flow-steps', '5'])

    def test_rollout_resume_and_fork_initialization(self):
        extra = ['--rollout-fraction', '.5', '--rollout-warmup-steps', '2', '--ema-decay', '.999']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.run_training(root / 'full', extra)
            original = train_time_flow.transition_loss
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('interrupt mixed training')
                return original(*args, **kwargs)
            with patch.object(train_time_flow, 'transition_loss', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupt mixed'):
                    self.run_training(root / 'resume', extra)
            resumed = self.run_training(root / 'resume', [*extra, '--resume',
                                       str(root / 'resume/checkpoint_latest.pt')])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            self.assertEqual(full['diagnostic_rows'], resumed['diagnostic_rows'])
            self.assertEqual(full['transition_rows'], resumed['transition_rows'])
            parent = root / 'full/checkpoint_latest.pt'
            child = self.run_training(root / 'fork', [*extra, '--fork-checkpoint', str(parent)])
            initial = torch.load(root / 'fork/checkpoint_step_000000.pt', weights_only=False)
            self.assertEqual(initial['update_count'], 0)
            self.assertEqual(initial['optimizer']['state'], {})
            self.assertEqual(child['config']['parent_update'], 3)
            self.assertIsNone(child['config']['frozen_encoder'])
            for name in full['model']:
                torch.testing.assert_close(initial['model'][name], full['model'][name], rtol=0, atol=0)
                torch.testing.assert_close(initial['ema_model'][name], full['model'][name], rtol=0, atol=0)
            recovered = self.run_training(root / 'fork', [*extra, '--resume',
                                         str(root / 'fork/checkpoint_latest.pt')])
            self.assertEqual(child['config'], recovered['config'])

    def test_flow_matching_resume_and_objective_isolation(self):
        extra = ['--objective', 'flow-matching', '--ema-decay', '.999']
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train_time_flow, 'transition_loss', side_effect=AssertionError('Riesz called')), \
                patch.object(train_time_flow, 'mixed_source_states', side_effect=AssertionError('rollout called')):
            root = Path(directory)
            full = self.run_training(root / 'full', extra)
            original = train_time_flow.flow_matching_loss
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('interrupt FM training')
                return original(*args, **kwargs)
            with patch.object(train_time_flow, 'flow_matching_loss', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupt FM'):
                    self.run_training(root / 'resume', extra)
            resumed = self.run_training(root / 'resume', [*extra, '--resume',
                                       str(root / 'resume/checkpoint_latest.pt')])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            self.assertEqual(full['diagnostic_rows'], resumed['diagnostic_rows'])
            self.assertEqual(full['transition_rows'], resumed['transition_rows'])
            self.assertEqual(sum(r['count'] for r in full['transition_rows']), 9)
            self.assertTrue(full['config']['paired_velocity_supervision'])
            self.assertEqual(full['config']['sampling_network_evaluations'], 10)
            self.assertEqual(full['config']['time_batching'], 'independent_uniform_continuous_time_per_image')
            self.assertIn('velocity_mse', full['diagnostic_rows'][0])
            self.assertIsNone(full['config']['frozen_encoder'])
            with self.assertRaisesRegex(ValueError, 'mismatch for objective'):
                self.run_training(root / 'resume', ['--ema-decay', '.999', '--resume',
                                                  str(root / 'resume/checkpoint_latest.pt')])

    def test_endpoint_combined_training_resume_matches_uninterrupted(self):
        extra = ['--endpoint-weight','1','--rollout-fraction','.5','--ema-decay','.999']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.run_training(root/'full', extra)
            original = train_time_flow.endpoint_riesz_loss
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls==2:
                    raise RuntimeError('interrupt endpoint training')
                return original(*args, **kwargs)
            with patch.object(train_time_flow,'endpoint_riesz_loss',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupt endpoint'):
                    self.run_training(root/'resume', extra)
            resumed = self.run_training(root/'resume', [*extra,'--resume',str(root/'resume/checkpoint_latest.pt')])
            for key in ('model','ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name],resumed[key][name],rtol=0,atol=0)
            self.assertEqual(full['diagnostic_rows'],resumed['diagnostic_rows'])
            self.assertEqual(full['transition_rows'],resumed['transition_rows'])
            self.assertIn('endpoint_surrogate',resumed['diagnostic_rows'][-1])
            self.assertIn('patch_phase_rms',resumed['evaluation_rows'][-1])
            self.assertFalse(resumed['config']['paired_velocity_supervision'])
            with self.assertRaisesRegex(ValueError,'mismatch for endpoint_weight'):
                self.run_training(root/'resume',['--rollout-fraction','.5','--ema-decay','.999',
                    '--resume',str(root/'resume/checkpoint_latest.pt')])

    def test_continuous_training_resume_and_sampling_steps_independence(self):
        extra=['--riesz-time-mode','continuous','--transport-horizon','.1','--ema-decay','.999']
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train_time_flow,'transition_loss',side_effect=AssertionError('Discrete loss called')), \
                patch.object(train_time_flow,'mixed_source_states',side_effect=AssertionError('Rollout called')), \
                patch.object(train_time_flow,'endpoint_riesz_loss',side_effect=AssertionError('Endpoint called')):
            root=Path(directory)
            full=self.run_training(root/'full',extra)
            different_sampler=self.run_training(root/'different_sampler',[*extra,'--flow-steps','3'])
            for key in ('model','ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name],different_sampler[key][name],rtol=0,atol=0)
            self.assertEqual(full['diagnostic_rows'],different_sampler['diagnostic_rows'])
            self.assertIsNone(full['config']['training_time_grid'])
            self.assertFalse(full['config']['paired_velocity_supervision'])
            original=train_time_flow.continuous_transition_loss
            calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('interrupt continuous training')
                return original(*args,**kwargs)
            with patch.object(train_time_flow,'continuous_transition_loss',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupt continuous'):
                    self.run_training(root/'resume',extra)
            resumed=self.run_training(root/'resume',[*extra,'--resume',str(root/'resume/checkpoint_latest.pt')])
            for key in ('model','ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name],resumed[key][name],rtol=0,atol=0)
            self.assertEqual(full['diagnostic_rows'],resumed['diagnostic_rows'])
            self.assertEqual(full['transition_rows'],resumed['transition_rows'])
            self.assertTrue(any(abs(r['source_time']*10-round(r['source_time']*10))>1e-4
                                for r in full['diagnostic_rows']))
            for options,key in [(['--riesz-time-mode','discrete'],'riesz_time_mode'),
                                (['--transport-horizon','.2'],'transport_horizon')]:
                with self.assertRaisesRegex(ValueError,'mismatch for '+key):
                    self.run_training(root/'resume',[*extra,*options,'--resume',str(root/'resume/checkpoint_latest.pt')])


if __name__ == '__main__':
    unittest.main()
