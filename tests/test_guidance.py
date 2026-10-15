"""Normalization invariants, official CFG parity, and conditional integration."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from backbone import CIFAR10UNetParticleGenerator
from guidance import guided_generator_step, sample_cfg
from loss import _softmax_toward, drifting_loss, sinkhorn_target_loss
from multiscale import objective_surrogate


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_near_constant_unit_directions_match_float64(self):
        x = .5 + 1e-5 * torch.randn(1, 32, 3072)
        y = .5 + 1e-5 * torch.randn(1, 35, 3072)
        actual, _ = _softmax_toward(x, y, torch.ones(1, 35), .2, 1e-8)
        # Independent direct-displacement calculation, without barycentric cancellation.
        displacement = y.double()[:, None] - x.double()[:, :, None]
        distance = displacement.norm(dim=-1)
        expected = (torch.softmax(-distance / .2, -1)[..., None]
                    * displacement / distance[..., None]).sum(2)
        torch.testing.assert_close(actual.double(), expected, rtol=2e-4, atol=2e-7)

    def test_scale_one_ignores_unconditional_support_for_all_fields(self):
        x, y, r, u = [torch.randn(n, 6) for n in (4, 7, 5, 3)]
        for objective in ('drifting', 'softmax-riesz', 'sinkhorn'):
            def evaluate(unconditional):
                z = x.clone().requires_grad_()
                value = objective_surrogate(
                    z, y, 8, objective=objective, tau_list=(.2, .05, .02),
                    normalize_per_temperature=True, reference=r,
                    unconditional=unconditional, cfg_scale=1.)
                return value, torch.autograd.grad(value, z)[0]
            a, ga = evaluate(None)
            b, gb = evaluate(u)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_official_sinkhorn_cfg_loss_and_gradient(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/sinkhorn_cfg_official.json').read_text())
        for case in fixture['cases']:
            x = torch.tensor(case['x'], requires_grad=True)
            y, r, u = [torch.tensor(case[name], requires_grad=True)
                       for name in ('y', 'reference', 'unconditional')]
            value, _ = sinkhorn_target_loss(x, y, r, unconditional=u, cfg_scale=case['scale'])
            value.backward()
            torch.testing.assert_close(value, torch.tensor(case['loss']), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(x.grad, torch.tensor(case['gradient']), rtol=3e-4, atol=3e-6)
            self.assertTrue(all(z.grad is None for z in (y, r, u)))

    def test_guided_drifting_uses_reference_population_weights(self):
        x = torch.randn(4, 6, requires_grad=True)
        y, u = torch.randn(7, 6), torch.randn(3, 6)
        actual = objective_surrogate(x, y, 8, objective='drifting', tau_list=(.2, .05),
                                     normalize_per_temperature=True, unconditional=u, cfg_scale=2.5)
        reference, _ = drifting_loss(x[None], y[None], u[None],
                                     weight_neg=torch.full((1, 3), 1.5 * 3 / 3), tau_list=(.2, .05))
        torch.testing.assert_close(actual, reference.mean())

    def test_normalization_scale_and_dimension_invariance(self):
        x, y, r, u = [torch.rand(n, 8) for n in (4, 7, 5, 3)]
        for objective in ('drifting', 'softmax-riesz', 'sinkhorn'):
            def evaluate(multiplier=1., repeat=1):
                z = (x * multiplier).repeat(1, repeat).requires_grad_()
                value = objective_surrogate(
                    z, (y * multiplier).repeat(1, repeat), 8, objective=objective,
                    tau_list=(.2, .05, .02), normalize_per_temperature=True,
                    reference=(r * multiplier).repeat(1, repeat),
                    unconditional=(u * multiplier).repeat(1, repeat), cfg_scale=2.)
                gradient, = torch.autograd.grad(value, z)
                return value, gradient.reshape(len(x), repeat, -1).sum(1)
            value, gradient = evaluate()
            scaled, scaled_grad = evaluate(multiplier=3.)
            repeated, repeated_grad = evaluate(repeat=3)
            torch.testing.assert_close(value, scaled, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(gradient, 3 * scaled_grad, rtol=2e-3, atol=2e-5)
            torch.testing.assert_close(value, repeated, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(gradient, repeated_grad, rtol=2e-3, atol=2e-5)

    def test_softmax_cfg_uses_population_averages(self):
        # At fixed geometry, duplicating a population leaves its average unchanged.
        # The batch-derived geometry scale can itself depend on population counts.
        x, u = torch.randn(1, 4, 6), torch.randn(1, 3, 6)
        a, _ = _softmax_toward(x, u, torch.full((1, 3), 2.), .2, 1e-8)
        b, _ = _softmax_toward(x, u.repeat(1, 2, 1), torch.full((1, 6), 2.), .2, 1e-8)
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
        x, y, u = [torch.randn(n, 6) for n in (4, 7, 3)]
        def gradient(unconditional):
            z = x.clone().requires_grad_()
            value = objective_surrogate(z, y, 8, objective='softmax-riesz', tau_list=(.2,),
                                         normalize_per_temperature=True,
                                         unconditional=unconditional, cfg_scale=3.)
            return torch.autograd.grad(value, z)[0]
        torch.testing.assert_close(gradient(u), gradient(u.repeat(2, 1)), rtol=2e-4, atol=2e-5)

    def test_training_artifacts_and_exact_resume_for_all_fields(self):
        import contextlib
        import io
        import fid
        import main as trainer
        images = torch.randint(256, (40, 3072), dtype=torch.uint8)
        labels = torch.arange(10).repeat_interleave(4)
        dataset = (Path('/tmp/cifar-10-python.tar.gz'), (images, labels), (images, labels))

        class TinyFID:
            def __init__(self, *args, **kwargs):
                pass
            def evaluate_generator(self, model, latent_dim, *, count, seed, class_ids, cfg_scale):
                z = torch.randn(count, latent_dim, generator=torch.Generator().manual_seed(seed))
                labels = torch.tensor(class_ids)[torch.arange(count) % len(class_ids)]
                return float(trainer.evaluation_images(model, z, labels=labels, cfg_scale=cfg_scale).mean())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / 'weights'; weights.write_bytes(b'test-weights')
            with (patch.object(trainer, 'load_dataset', return_value=dataset),
                  patch.object(fid, 'CIFAR10FID', TinyFID), patch.object(fid, 'WEIGHTS', weights),
                  patch.object(trainer, 'save_trajectory'), patch.object(trainer, 'plot_mmd'),
                  contextlib.redirect_stdout(io.StringIO())):
                for objective in ('drifting', 'softmax-riesz', 'sinkhorn'):
                    output = root / objective
                    argv = ['main.py', '--cfg', '--objective', objective, '--target-size', '0',
                            '--target-batch-size', '2', '--particles', '2', '--base-channels', '8',
                            '--num-res-blocks', '1', '--steps', '2', '--save-every', '1',
                            '--checkpoint-every', '1', '--fid-samples', '2', '--eval-size', '4',
                            '--cfg-groups', '2', '--cfg-uncond-samples', '2', '--cfg-eval-scales', '1', '2',
                            '--multiscale-loss', '--coarse-sizes', '8', '--patch-sizes', '7',
                            '--threads', '1', '--device', 'cpu', '--output', str(output)]
                    with patch.object(sys, 'argv', argv):
                        trainer.main()
                    expected = torch.load(output / 'checkpoint_latest.pt', weights_only=False)
                    expected_fids = json.loads((output / 'fid_cfg_metrics.json').read_text())
                    # Crash after the first durable update, then resume that checkpoint.
                    resumed = root / (objective + '_resumed')
                    resume_argv = argv[:-1] + [str(resumed)]
                    original_step = torch.optim.Adam.step
                    calls = 0
                    def crash(optimizer, *args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 2:
                            raise RuntimeError('simulated interruption')
                        return original_step(optimizer, *args, **kwargs)
                    with patch.object(sys, 'argv', resume_argv), patch.object(torch.optim.Adam, 'step', crash):
                        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                            trainer.main()
                    checkpoint = resumed / 'checkpoint_latest.pt'
                    with patch.object(sys, 'argv', resume_argv + ['--resume', str(checkpoint)]):
                        trainer.main()
                    actual = torch.load(checkpoint, weights_only=False)
                    self.assertEqual(actual['update_count'], 2)
                    for key, value in expected['model'].items():
                        torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)
                    self.assertEqual(json.loads((resumed / 'fid_cfg_metrics.json').read_text()), expected_fids)
                    with (resumed / 'metrics.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual([int(row['update']) for row in rows], [0, 1, 2])
                    self.assertTrue(all(row['normalized_field_rms'] for row in rows[1:]))
                    self.assertTrue((resumed / 'particles_step_000002_cfg_2.png').is_file())

    def test_cfg_sampling_and_conditioned_backbone(self):
        rng = torch.Generator().manual_seed(5)
        scales = sample_cfg(rng, 1000)
        self.assertTrue(((scales >= 1) & (scales <= 4)).all())
        self.assertTrue((sample_cfg(rng, 10, no_guidance_fraction=1.) == 1).all())
        model = CIFAR10UNetParticleGenerator(8, 1, num_classes=10).eval()
        # Open the near-zero branches to test conditioning independently of initialization.
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, std=.05)
        z = torch.randn(1, 3072).repeat(3, 1)
        result = model(z, labels=torch.tensor([0, 1, 0]), cfg_scale=torch.tensor([1., 1., 3.]))
        self.assertGreater(float((result[0] - result[1]).abs().max()), 1e-5)
        self.assertGreater(float((result[0] - result[2]).abs().max()), 1e-5)
        result.mean().backward()
        self.assertGreater(float(model.class_embedding.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.cfg_embedding[0].weight.grad.abs().sum()), 0.)

    def test_real_training_step_all_three_fields(self):
        for objective in ('drifting', 'softmax-riesz', 'sinkhorn'):
            model = CIFAR10UNetParticleGenerator(8, 1, num_classes=10)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            pools = {0: torch.rand(4, 3072), 1: torch.rand(4, 3072)}
            before = model.output[2].weight.detach().clone()
            for multiscale in (None, dict(image_shape=(3, 32, 32), coarse_sizes=(8,),
                                         patch_sizes=(7,), patch_rng=torch.Generator().manual_seed(3))):
                info = guided_generator_step(
                    model, optimizer, pools, torch.randint(256, (10, 3072), dtype=torch.uint8),
                    rng=torch.Generator().manual_seed(1), latent_rng=torch.Generator().manual_seed(2),
                    reference_rng=torch.Generator().manual_seed(3), particles=2, latent_dim=3072,
                    positives=3, unconditional_samples=2, groups=2, cfg_min=2., cfg_max=2.,
                    cfg_power=5., cfg_no_guidance_fraction=0., objective=objective,
                    tau_list=(.2, .05, .02), normalize_per_temperature=True, chunk_size=8,
                    multiscale_options=multiscale, sinkhorn_epsilon=.05, sinkhorn_iters=10,
                    max_grad_norm=2.)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in info.values()))
                self.assertEqual(info['cfg_scale_mean'], 2.)
                self.assertIn('scale_inputs', info)
            self.assertFalse(torch.equal(before, model.output[2].weight))

    def test_fid_passes_balanced_classes_and_guidance_without_training_rng(self):
        import fid
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, z, *, labels, cfg_scale):
                self.calls.append((labels.tolist(), cfg_scale))
                return z
        model = RecordingModel()
        evaluator = fid.CIFAR10FID.__new__(fid.CIFAR10FID)
        evaluator.batch_size, evaluator.device, evaluator.inception = 3, 'cpu', None
        evaluator.score = lambda stats: 0.
        before = torch.get_rng_state().clone()
        with patch.object(fid, 'feature_statistics', side_effect=lambda batches, *_: list(batches)):
            evaluator.evaluate_generator(model, 4, count=8, class_ids=[2, 5], cfg_scale=2.)
        self.assertEqual([label for labels, _ in model.calls for label in labels], [2, 5] * 4)
        self.assertTrue(all(scale == 2. for _, scale in model.calls))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
