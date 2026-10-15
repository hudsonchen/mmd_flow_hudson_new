"""Numerical checks for target semantics, experiment isolation, and restart state."""
import contextlib
import csv
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import main
from loss import _riesz_topk_softmax_loss_impl
from multiscale import multiscale_loss
from target_updates import softmax_target_loss


class TargetUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.x = torch.rand(5, 12, requires_grad=True)
        self.y = torch.rand(7, 12, requires_grad=True)
        self.options = dict(tau_list=(0.2, 0.05, 0.02), normalize_per_temperature=True)

    def test_default_matches_existing_loss_and_gradient(self):
        for normalize in (True, False):
            options = dict(self.options, normalize_per_temperature=normalize)
            expected, _ = _riesz_topk_softmax_loss_impl(
                self.x.unsqueeze(0), self.y.unsqueeze(0), epsilon=1e-6, **options)
            actual, _ = softmax_target_loss(self.x, self.y, **options)
            torch.testing.assert_close(actual, expected.mean())
            expected_gradient = torch.autograd.grad(expected.mean(), self.x)[0]
            actual_gradient = torch.autograd.grad(actual, self.x)[0]
            torch.testing.assert_close(actual_gradient, expected_gradient)
            self.assertIsNone(self.y.grad)

    def test_explicit_step_scales_gradient(self):
        full, _ = softmax_target_loss(self.x, self.y, **self.options)
        small, _ = softmax_target_loss(self.x, self.y, field_step=0.1, **self.options)
        g_full = torch.autograd.grad(full, self.x)[0]
        g_small = torch.autograd.grad(small, self.x)[0]
        torch.testing.assert_close(g_small, 0.1 * g_full)

    def test_noise_uses_independent_rng_and_is_reproducible(self):
        targets = []
        def field(gen, pos, **kwargs):
            targets.append(pos.clone())
            return _riesz_topk_softmax_loss_impl(gen, pos, **kwargs)
        rng = torch.Generator().manual_seed(20)
        global_state = torch.get_rng_state().clone()
        with patch('target_updates._riesz_topk_softmax_loss_impl', field):
            loss, _ = softmax_target_loss(
                self.x, self.y, noise_levels=(0.1,),
                noise_rng=rng, **self.options)
        repeated, _ = softmax_target_loss(
            self.x, self.y, noise_levels=(0.1,),
            noise_rng=torch.Generator().manual_seed(20), **self.options)
        self.assertEqual(len(targets), 1)
        self.assertFalse(torch.equal(targets[0].squeeze(0), self.y))
        torch.testing.assert_close(loss, repeated, rtol=0, atol=0)
        torch.testing.assert_close(global_state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, self.x)[0]).all())

    def test_multiscale_variants_have_finite_gradients(self):
        for variant in ({}, {'global_weight': 4}, {'patch_weight': 0},
                        {'target_options': {'noise_levels': (0, 0.05, 0.15),
                                            'noise_rng': torch.Generator().manual_seed(3)}}):
            x = torch.rand(4, 3 * 16 * 16, requires_grad=True)
            y = torch.rand(6, 3 * 16 * 16)
            value, info = multiscale_loss(
                x, y, (3, 16, 16), 4, coarse_sizes=(4, 8), patch_sizes=(3, 7),
                patch_rng=torch.Generator().manual_seed(2), **variant)
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(torch.autograd.grad(value, x)[0]).all())


class ExperimentTests(unittest.TestCase):
    def test_array_tasks_change_one_factor(self):
        root = Path(__file__).resolve().parents[1]
        parsed = []
        for task in range(1, 9):
            result = subprocess.run(
                ['bash', str(root / 'scripts/cs_cluster/run_cifar_ablation_h100.sh'), '--dry-run'],
                env={**os.environ, 'SGE_TASK_ID': str(task), 'JOB_ID': 'test'},
                check=True, capture_output=True, text=True)
            words = shlex.split(result.stdout.splitlines()[1])
            with patch.object(sys, 'argv', ['main.py'] + words[3:]):
                parsed.append(vars(main.parse_args()))
        permitted = [set(), {'learning_rate'}, {'learning_rate'}, {'particles'},
                     {'global_loss_weight'}, {'patch_loss_weight'}, {'noise_levels'},
                     {'field_step'}]
        for args, expected in zip(parsed, permitted):
            changed = {k for k in args if args[k] != parsed[0][k]} - {'output'}
            self.assertEqual(changed, expected)
            self.assertEqual(args['epochs'], 500)
            self.assertEqual(args['tau_list'], [0.2, 0.05, 0.02])

    def test_interrupted_training_resumes_identically(self):
        torch.set_num_threads(1)
        rng = torch.Generator().manual_seed(100)
        dataset = (Path('/tmp/synthetic_mnist'), (torch.randint(256, (8, 784), generator=rng, dtype=torch.uint8),
                           torch.zeros(8, dtype=torch.long)),
                   (torch.randint(256, (4, 784), generator=rng, dtype=torch.uint8),
                    torch.zeros(4, dtype=torch.long)))
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'reference'
            interrupted = Path(directory) / 'interrupted'
            def run(output, resume=False):
                args = ['main.py', '--dataset', 'mnist', '--architecture', 'mlp',
                        '--latent-dim', '4', '--hidden-dims', '8', '--particles', '3',
                        '--target-size', '0', '--target-batch-size', '4', '--eval-size', '4',
                        '--epochs', '2', '--save-every', '1', '--checkpoint-every', '1',
                        '--ema-decay', '0.9', '--device', 'cpu', '--threads', '1',
                        '--multiscale-loss',
                        '--noise-levels', '0', '0.05', '--output', str(output)]
                if resume:
                    args += ['--resume', str(output / 'checkpoint_latest.pt')]
                with patch.object(sys, 'argv', args), patch.object(main, 'load_dataset', return_value=dataset), \
                        patch.object(main, 'save_grid'), patch.object(main, 'save_trajectory'), \
                        patch.object(main, 'plot_mmd'), contextlib.redirect_stdout(io.StringIO()):
                    main.main()
            run(reference)
            real_step = main.generator_step
            calls = 0
            def fail_after_checkpoint(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('simulated interruption')
                return real_step(*args, **kwargs)
            with patch.object(main, 'generator_step', fail_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    run(interrupted)
            run(interrupted, resume=True)
            expected = torch.load(reference / 'checkpoint_latest.pt', weights_only=False)
            actual = torch.load(interrupted / 'checkpoint_latest.pt', weights_only=False)
            for group in ('model', 'ema_model'):
                for key in expected[group]:
                    torch.testing.assert_close(actual[group][key], expected[group][key], rtol=0, atol=0)
            self.assertEqual(actual['update_count'], expected['update_count'])
            for key in ('data_rng', 'latent_rng', 'noise_rng', 'patch_rng', 'torch_rng'):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for output in (reference, interrupted):
                with (output / 'metrics.csv').open() as stream:
                    self.assertEqual([int(r['epoch']) for r in csv.DictReader(stream)], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
