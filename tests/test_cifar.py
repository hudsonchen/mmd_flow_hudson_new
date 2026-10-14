"""Check CIFAR channel ordering, RGB rendering, and the larger generator's training."""
import tempfile
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from data import decode_binary_batch
from backbone import CIFAR10UNetParticleGenerator, SpatialAttention, ResidualBlock
from main import generator_step, parse_args, update_ema, evaluation_images
from visualization import save_grid, save_trajectory


class CIFAR10Tests(unittest.TestCase):
    def test_binary_channel_order_and_rgb_artifacts(self):
        records = np.zeros((2, 3073), dtype=np.uint8)
        records[:, 0] = [0, 9]
        records[0, 1:1025] = 255  # red
        records[1, 1025:2049] = 255  # green
        images, labels = decode_binary_batch(records.tobytes())
        self.assertEqual(images.shape, (2, 3072))
        self.assertEqual(labels.tolist(), [0, 9])
        pixels = images.float() / 255
        with tempfile.TemporaryDirectory(prefix='cifar_rgb_test_') as directory:
            root = Path(directory)
            save_grid(pixels, root / 'grid.png', image_shape=(3, 32, 32))
            with Image.open(root / 'grid.png') as grid:
                self.assertEqual(grid.mode, 'RGB')
                self.assertEqual(grid.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(grid.getpixel((121, 0)), (0, 255, 0))
            save_trajectory([pixels.reshape(2, 3, 32, 32)], [0], root, 2, 'epoch')
            with Image.open(root / 'trajectory.png') as trajectory:
                self.assertEqual(trajectory.mode, 'RGB')
                self.assertEqual(trajectory.getpixel((80, 25)), (255, 0, 0))
        for data in [b'', b'bad', bytes([10]) + bytes(3072)]:
            with self.assertRaises(RuntimeError):
                decode_binary_batch(data)

    def test_cifar_cli_defaults(self):
        with patch('sys.argv', ['main.py', '--dataset', 'cifar10']):
            args = parse_args(default_dataset='cifar10')
        self.assertEqual(args.dataset, 'cifar10')
        self.assertEqual(args.base_channels, 128)
        self.assertEqual(args.num_res_blocks, 4)
        self.assertEqual(args.latent_dim, 3072)
        self.assertEqual(args.objective, 'softmax-riesz')
        self.assertEqual(args.ema_decay, 0.999)

    def test_ema_tracks_updates_without_changing_training(self):
        torch.manual_seed(57)
        model = CIFAR10UNetParticleGenerator(base_channels=8, num_res_blocks=1)
        model.eval()
        control = copy.deepcopy(model)
        averaged = copy.deepcopy(model).requires_grad_(False).eval()
        before = [p.clone() for p in averaged.parameters()]
        latent, target = torch.randn(3, 3072), torch.rand(4, 3072)
        for network in (model, control):
            generator_step(network, latent, target,
                           torch.optim.Adam(network.parameters(), lr=2e-4),
                           2, objective='softmax-riesz')
        update_ema(model, averaged, 0.999)
        for raw, baseline, old, ema in zip(model.parameters(), control.parameters(),
                                            before, averaged.parameters()):
            torch.testing.assert_close(raw, baseline, rtol=0, atol=0)
            torch.testing.assert_close(ema, old * 0.999 + raw.detach() * 0.001)
            self.assertFalse(ema.requires_grad)
            self.assertIsNone(ema.grad)
        self.assertTrue(torch.isfinite(averaged(latent)).all())

    def test_ddpmpp_topology_and_dropout_evaluation(self):
        model = CIFAR10UNetParticleGenerator(base_channels=8, num_res_blocks=4)
        self.assertEqual([len(stage) for stage in model.encoders], [4] * 4)
        self.assertEqual([len(stage) for stage in model.decoders], [5] * 4)
        self.assertEqual(sum(isinstance(m, SpatialAttention) for m in model.modules()), 6)
        self.assertEqual(sum(isinstance(m, ResidualBlock) for m in model.modules()), 44)
        latent = torch.randn(2, 3072)
        state = torch.random.get_rng_state().clone()
        first = evaluation_images(model, latent)
        torch.testing.assert_close(first, evaluation_images(model, latent))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(first.shape, (2, 3072))
        self.assertTrue(torch.isfinite(first).all())

    def test_ddpmpp_training_with_dropout(self):
        model = CIFAR10UNetParticleGenerator(base_channels=8, num_res_blocks=1)
        generator_step(model, torch.randn(3, 3072), torch.rand(4, 3072),
                       torch.optim.Adam(model.parameters(), lr=2e-4),
                       2, objective='softmax-riesz')
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_attention_unet_trains_with_both_objectives(self):
        for objective in ['softmax-riesz', 'riesz']:
            with self.subTest(objective=objective):
                torch.manual_seed(57)
                model = CIFAR10UNetParticleGenerator(base_channels=8, num_res_blocks=1)
                model.eval()
                latent = torch.randn(3, 3072)
                target = torch.rand(4, 3072, requires_grad=True)
                with torch.no_grad():
                    initial = model(latent)
                    torch.testing.assert_close(initial[:1], model(latent[:1]), atol=1e-6, rtol=1e-5)
                self.assertEqual(initial.shape, (3, 3072))
                generator_step(model, latent, target, torch.optim.Adam(model.parameters(), lr=2e-4),
                               2, objective=objective)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                with torch.no_grad():
                    updated = model(latent)
                self.assertTrue(torch.isfinite(updated).all())
                self.assertTrue(((updated >= 0) & (updated <= 1)).all())
                self.assertFalse(torch.equal(updated, initial))
                self.assertIsNone(target.grad)
                self.assertIsNone(latent.grad)


if __name__ == '__main__':
    unittest.main()
