"""Check pixel-view gradients, aligned crops, RNG isolation and one-pass updates."""
import math
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

from backbone import ParticleGenerator
from loss import backpropagate_objective
from main import generator_step, parse_args
from multiscale import multiscale_loss, objective_surrogate


class MultiscaleTests(unittest.TestCase):
    def test_riesz_gradient_matches_explicit_multiview_energy(self):
        torch.manual_seed(73)
        generated = torch.rand(3, 3 * 8 * 8, dtype=torch.float64, requires_grad=True)
        real = torch.rand(5, 3 * 8 * 8, dtype=torch.float64, requires_grad=True)
        loss, diagnostics = multiscale_loss(
            generated, real, (3, 8, 8), 2, objective='riesz', coarse_sizes=(2, 4),
            patch_sizes=(3, 5), patch_rng=torch.Generator().manual_seed(79))
        actual, = torch.autograd.grad(loss, generated, retain_graph=True)
        x, y = generated.reshape(-1, 3, 8, 8), real.detach().reshape(-1, 3, 8, 8)

        def energy(a, b):
            a, b = a.flatten(1), b.flatten(1)
            return (torch.linalg.vector_norm(a[:, None] - b[None], dim=-1).mean()
                    - 0.5 * torch.linalg.vector_norm(a[:, None] - a[None], dim=-1).mean()) / math.sqrt(a.shape[1])

        coarse = sum(energy(F.adaptive_avg_pool2d(x, size), F.adaptive_avg_pool2d(y, size))
                     for size in (2, 4)) / 2
        rng = torch.Generator().manual_seed(79)
        crops = []
        for size in (3, 5):
            top = int(torch.randint(8 - size + 1, (), generator=rng))
            left = int(torch.randint(8 - size + 1, (), generator=rng))
            crops.append(energy(x[:, :, top:top + size, left:left + size],
                                y[:, :, top:top + size, left:left + size]))
        reference = (energy(x, y) + coarse + sum(crops) / 2) / 3
        expected, = torch.autograd.grad(reference, generated)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        loss.backward()
        self.assertIsNone(real.grad)
        self.assertTrue(all(not value.requires_grad for value in diagnostics.values()))

    def test_global_only_softmax_preserves_existing_gradient(self):
        torch.manual_seed(83)
        generated = torch.rand(3, 3 * 8 * 8, requires_grad=True)
        baseline = generated.detach().clone().requires_grad_(True)
        real = torch.rand(5, 3 * 8 * 8)
        loss, _ = multiscale_loss(generated, real, (3, 8, 8), 2,
                                  coarse_sizes=(), patch_sizes=(),
                                  coarse_weight=0, patch_weight=0)
        loss.backward()
        backpropagate_objective(baseline, real, 2, objective='softmax-riesz')
        torch.testing.assert_close(generated.grad, baseline.grad)

    def test_patch_rng_and_population_permutation(self):
        torch.manual_seed(89)
        x, y = torch.rand(3, 3 * 8 * 8), torch.rand(5, 3 * 8 * 8)
        state = torch.random.get_rng_state().clone()
        options = dict(image_shape=(3, 8, 8), chunk_size=2, coarse_sizes=(2, 4),
                       patch_sizes=(3, 5), crops_per_size=2)
        a, _ = multiscale_loss(x, y, patch_rng=torch.Generator().manual_seed(97), **options)
        b, _ = multiscale_loss(x, y.flip(0), patch_rng=torch.Generator().manual_seed(97), **options)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_rms_geometry_invariant_to_replicated_coordinates(self):
        gradients = []
        for dimension in (1, 19):
            x = torch.tensor([[0.1], [0.8]], dtype=torch.float64, requires_grad=True)
            y = torch.tensor([[0.3], [0.5], [0.9]], dtype=torch.float64)
            value = objective_surrogate(x.expand(-1, dimension), y.expand(-1, dimension), 2,
                                        objective='riesz', tau_list=(0.2,),
                                        normalize_per_temperature=True)
            gradients.append(torch.autograd.grad(value, x)[0])
        torch.testing.assert_close(*gradients)

    def test_combined_loss_updates_one_generator_with_one_forward(self):
        for objective in ('riesz', 'softmax-riesz'):
            model = ParticleGenerator(4, (8,), 3 * 8 * 8)
            before = [p.detach().clone() for p in model.parameters()]
            with patch.object(model, 'forward', wraps=model.forward) as forward:
                diagnostics = generator_step(
                    model, torch.randn(3, 4), torch.rand(5, 3 * 8 * 8),
                    torch.optim.Adam(model.parameters(), lr=1e-3), 2, objective=objective,
                    multiscale_options=dict(image_shape=(3, 8, 8), coarse_sizes=(2, 4),
                                            patch_sizes=(3, 5), patch_rng=torch.Generator().manual_seed(101)))
                self.assertEqual(forward.call_count, 1)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(set(diagnostics), {'global', 'coarse', 'patch', 'combined'})

    def test_cli_and_invalid_views(self):
        with patch('sys.argv', ['main.py', '--multiscale-loss']):
            args = parse_args()
        self.assertTrue(args.multiscale_loss)
        self.assertEqual(args.coarse_sizes, [8, 16])
        self.assertEqual(args.patch_sizes, [7, 14])
        for options in [dict(coarse_sizes=(33,)), dict(crops_per_size=0),
                        dict(global_weight=0, coarse_weight=0, patch_weight=0),
                        dict(patch_weight=float('nan'))]:
            with self.assertRaises(ValueError):
                multiscale_loss(torch.rand(3, 3072), torch.rand(5, 3072), (3, 32, 32), 2, **options)
