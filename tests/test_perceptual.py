"""Check perceptual gradients, frozen teachers, normalization, and CLI boundaries."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import main
from perceptual import VGGPerceptualLoss
from priming import reconstruction_step


class TinyLPIPS(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 2, 1)
        self.dropout = torch.nn.Dropout(.5)
        self.scaling_layer = torch.nn.Identity()

    def net(self, images):
        feature = self.features(images)
        return tuple(torch.nn.functional.adaptive_avg_pool2d(feature, side)
                     for side in (32, 16, 8, 4, 2))

    def forward(self, prediction, target, normalize=False):
        if not normalize:
            raise AssertionError('LPIPS must convert [0,1] inputs')
        a = self.features(prediction * 2 - 1)
        b = self.features(target * 2 - 1)
        return self.dropout((a - b).square()).mean((1, 2, 3), keepdim=True)


class PerceptualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_frozen_encoder_preserves_rng_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as cache, patch('lpips.LPIPS', TinyLPIPS):
            before = torch.get_rng_state().clone()
            hub = torch.hub.get_dir()
            perceptual = VGGPerceptualLoss(cache)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(hub, torch.hub.get_dir())
            self.assertFalse(perceptual.metric.training)
            prediction = torch.rand(2, 3072, requires_grad=True)
            target = torch.rand(2, 3072, requires_grad=True)
            value = perceptual(prediction, target)
            value.backward()
            self.assertGreater(prediction.grad.abs().sum().item(), 0)
            self.assertIsNone(target.grad)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in perceptual.parameters()))
            self.assertEqual(perceptual(prediction.detach(), prediction.detach()).item(), 0.)

    def test_weighted_update_and_zero_weight_compatibility(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sigmoid())
        expected = copy.deepcopy(model)
        clean, noise = torch.rand(3, 4), torch.randn(3, 4)
        def perceptual(a, b):
            return (a - b).abs().mean()
        prediction = expected(.2 * (2 * clean - 1) + .8 * noise)
        mse = (prediction - clean).square().mean()
        loss = mse + .1 * perceptual(prediction, clean)
        reference = torch.optim.SGD(expected.parameters(), lr=.1)
        loss.backward()
        reference.step()
        actual = reconstruction_step(model, clean, noise,
                                     torch.optim.SGD(model.parameters(), lr=.1),
                                     perceptual_loss=perceptual, perceptual_weight=.1)
        self.assertAlmostEqual(actual['reconstruction_total'], loss.item())
        self.assertAlmostEqual(actual['reconstruction_mse'], mse.item())
        for a, b in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)
        with patch('torch.nn.Module.forward', side_effect=AssertionError('unused')):
            info = reconstruction_step(model, clean, noise,
                                       torch.optim.SGD(model.parameters(), lr=.1),
                                       perceptual_loss=torch.nn.Module(), perceptual_weight=0.)
        self.assertEqual(set(info), {'reconstruction_mse'})

    def test_invalid_perceptual_stage_rejected(self):
        for extra in [[], ['--training-stage', 'reconstruction', '--dataset', 'mnist']]:
            with patch('sys.argv', ['main.py', '--lpips-weight', '.1', *extra]):
                with self.assertRaises(SystemExit):
                    main.parse_args()

    def test_vgg_distance_and_field_gradient(self):
        from loss import _riesz_topk_softmax_loss_impl
        with tempfile.TemporaryDirectory() as cache, patch('lpips.LPIPS', TinyLPIPS):
            encoder = VGGPerceptualLoss(cache)
            x = torch.rand(3, 3072, requires_grad=True)
            y = torch.rand(5, 3072, requires_grad=True)
            layers = encoder.distribution_features(x, pool_size=4, batch_size=2)
            self.assertEqual([list(f.shape) for f in layers], [[3, 32]] * 4 + [[3, 8]])
            phi = encoder.flow_features(x)
            expected_distance_squared = sum((layer[0] - layer[1]).square().sum() / (side**2 * 5)
                                            for layer, side in zip(layers, (4, 4, 4, 4, 2)))
            torch.testing.assert_close((phi[0] - phi[1]).square().sum(), expected_distance_squared)
            with torch.no_grad():
                real = encoder.flow_features(y.detach())
            value, info = _riesz_topk_softmax_loss_impl(
                phi.unsqueeze(0), real.unsqueeze(0), no_field_rms_normalization=True, return_field=True)
            gradient, target_gradient = torch.autograd.grad(value.mean(), (x, y), retain_graph=True,
                                                            allow_unused=True)
            expected_feature_gradient = -2 * info['frozen_velocity'].squeeze(0) / (phi.numel() * info['scale_inputs'])
            expected_image_gradient = torch.autograd.grad(phi, x, expected_feature_gradient, retain_graph=True)[0]
            torch.testing.assert_close(gradient, expected_image_gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            self.assertIsNone(target_gradient)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
            permuted, _ = _riesz_topk_softmax_loss_impl(
                phi.unsqueeze(0), real.flip(0).unsqueeze(0), no_field_rms_normalization=True)
            torch.testing.assert_close(value, permuted)
            torch.testing.assert_close(gradient, torch.autograd.grad(permuted.mean(), x)[0])

    def test_generator_uses_only_vgg_riesz_flow(self):
        from loss import _riesz_topk_softmax_loss_impl
        with tempfile.TemporaryDirectory() as cache, patch('lpips.LPIPS', TinyLPIPS):
            encoder = VGGPerceptualLoss(cache)
            model = torch.nn.Sequential(torch.nn.Linear(4, 3072), torch.nn.Sigmoid())
            expected = copy.deepcopy(model)
            z, y = torch.randn(3, 4), torch.rand(5, 3072)
            generated = encoder.flow_features(expected(z))
            with torch.no_grad():
                target = encoder.flow_features(y)
            value, _ = _riesz_topk_softmax_loss_impl(
                generated.unsqueeze(0), target.unsqueeze(0), no_field_rms_normalization=True,
                tau_list=(.2, .05, .02), normalize_per_temperature=True, epsilon=1e-6)
            value.mean().backward()
            torch.optim.SGD(expected.parameters(), lr=.001).step()
            with patch.object(main, 'multiscale_loss', side_effect=AssertionError('no pixel objective')):
                info = main.generator_step(model, z, y, torch.optim.SGD(model.parameters(), lr=.001), 8,
                    objective='softmax-riesz', no_field_rms_normalization=True,
                    feature_encoder=encoder, softmax_feature_space='vgg')
            for a, b in zip(model.parameters(), expected.parameters()):
                torch.testing.assert_close(a, b)
            self.assertIn('raw_attraction_rms', info)
            self.assertIn('raw_self_repulsion_rms', info)
            self.assertNotIn('perceptual_weighted', info)
            with patch.object(encoder, 'flow_features', side_effect=AssertionError('unused')):
                main.generator_step(model, z, y, torch.optim.SGD(model.parameters(), lr=.001), 8,
                    objective='softmax-riesz', no_field_rms_normalization=True,
                    feature_encoder=encoder, softmax_feature_space='pixels')

    def test_multiscale_branches_gradients_and_independent_scaling(self):
        from perceptual import multiscale_vgg_loss
        with tempfile.TemporaryDirectory() as cache, patch('lpips.LPIPS', TinyLPIPS):
            encoder = VGGPerceptualLoss(cache)
            x = torch.rand(3, 3072, requires_grad=True)
            y = torch.rand(7, 3072, requires_grad=True)
            features = encoder.multiscale_features(x, pool_size=8, batch_size=2)
            real = encoder.multiscale_features(y, pool_size=8, batch_size=3)
            self.assertEqual(len(features), 15)
            self.assertEqual(features['layer1_spatial'].shape, (3, 128))
            self.assertEqual(features['layer1_patch_stats'].shape, (3, 64))
            self.assertEqual(features['layer5_global_stats'].shape, (3, 4))
            options = dict(tau_list=(.2, .05), normalize_per_temperature=False,
                           no_field_rms_normalization=True, repulsion_weight=1.)
            value, info = multiscale_vgg_loss(features, real, **options)
            gradient = torch.autograd.grad(value, (x, y), allow_unused=True, retain_graph=True)
            self.assertIsNone(gradient[1])
            self.assertTrue(torch.isfinite(gradient[0]).all())
            self.assertGreater(gradient[0].abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in encoder.parameters()))
            self.assertIn('raw_layer1_spatial_positive_ess', info)
            # A scale change in one branch must not alter another branch's geometry.
            scaled, scaled_real = dict(features), dict(real)
            scaled['layer1_spatial'] = 3 * scaled['layer1_spatial']
            scaled_real['layer1_spatial'] = 3 * scaled_real['layer1_spatial']
            other, _ = multiscale_vgg_loss(scaled, scaled_real, **options)
            torch.testing.assert_close(value, other)
            torch.testing.assert_close(gradient[0], torch.autograd.grad(other, x)[0])

    def test_multiscale_fork_and_exact_resume(self):
        import test_dit
        helper = test_dit.DiTTests()
        with tempfile.TemporaryDirectory() as directory, patch('lpips.LPIPS', TinyLPIPS):
            root = Path(directory)
            common = ['--softmax-feature-space', 'vgg', '--lpips-cache', str(root / 'cache'),
                      '--no-field-rms-normalization']
            original = main.generator_step
            def fail_after_first(*args, **kwargs):
                if (root / 'source/checkpoint_latest.pt').exists():
                    checkpoint = torch.load(root / 'source/checkpoint_latest.pt', weights_only=False)
                    if checkpoint['update_count'] == 1:
                        raise RuntimeError('stop source')
                return original(*args, **kwargs)
            with patch.object(main, 'generator_step', side_effect=fail_after_first):
                with self.assertRaisesRegex(RuntimeError, 'stop source'):
                    helper.run_training(root / 'source', common)
            fork_options = common + ['--fork-checkpoint', str(root / 'source/checkpoint_latest.pt'),
                                     '--vgg-multiscale', '--perceptual-pool-size', '8',
                                     '--train-particles', '4', '--target-batch-size', '5']
            full = helper.run_training(root / 'full', fork_options)
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('stop fork')
                return original(*args, **kwargs)
            with patch.object(main, 'generator_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'stop fork'):
                    helper.run_training(root / 'resume', fork_options)
            resume_options = common + ['--vgg-multiscale', '--perceptual-pool-size', '8',
                                      '--train-particles', '4', '--target-batch-size', '5',
                                      '--resume', str(root / 'resume/checkpoint_latest.pt')]
            recovered = helper.run_training(root / 'resume', resume_options)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], recovered[key][name], rtol=0, atol=0)
            self.assertIn('target_batch_size', full['config']['tuning_initialization']['changes'])
            with self.assertRaisesRegex(ValueError, 'mismatch for vgg_multiscale'):
                helper.run_training(root / 'resume', common + [
                    '--perceptual-pool-size', '8', '--train-particles', '4', '--target-batch-size', '5',
                    '--resume', str(root / 'resume/checkpoint_latest.pt')])

    def test_feature_posttraining_resume_and_configuration_guards(self):
        import test_dit
        helper = test_dit.DiTTests()
        with tempfile.TemporaryDirectory() as directory, patch('lpips.LPIPS', TinyLPIPS):
            root = Path(directory)
            common = ['--softmax-feature-space', 'vgg', '--lpips-cache', str(root / 'cache'),
                      '--no-field-rms-normalization']
            full = helper.run_training(root / 'full', common)
            original = main.generator_step
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('feature interruption')
                return original(*args, **kwargs)
            with patch.object(main, 'generator_step', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'feature interruption'):
                    helper.run_training(root / 'resume', common)
            resumed = helper.run_training(root / 'resume', common + [
                '--resume', str(root / 'resume/checkpoint_latest.pt')])
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            self.assertEqual(full['config']['softmax_feature_space'], 'vgg')
            self.assertFalse(full['config']['distributional_perceptual']['additional_supervision'])
            self.assertNotIn('train_perceptual_surrogate', (root / 'full/metrics.csv').read_text())
            self.assertIn('raw_force_rms', (root / 'full/training_diagnostics.csv').read_text())
            with self.assertRaisesRegex(ValueError, 'mismatch for softmax_feature_space'):
                helper.run_training(root / 'resume', common + ['--softmax-feature-space', 'pixels',
                    '--resume', str(root / 'resume/checkpoint_latest.pt')])


if __name__ == '__main__':
    unittest.main()
