"""Numerical FID checks and isolation of evaluation from training randomness."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fid import CIFAR10FID, feature_statistics, frechet_distance


class SmallFeatures(torch.nn.Module):
    def forward(self, images):
        return [images.mean(dim=(2, 3), keepdim=True)]


class FIDTests(unittest.TestCase):
    def test_streaming_statistics(self):
        images = torch.randint(0, 256, (11, 3, 32, 32), dtype=torch.uint8)
        mean, cov, count = feature_statistics(images.split(4), SmallFeatures(), 'cpu')
        features = (images.double() / 255).mean(dim=(2, 3)).numpy()
        np.testing.assert_allclose(mean, features.mean(0), atol=1e-7)
        np.testing.assert_allclose(cov, np.cov(features, rowvar=False), atol=1e-9)
        self.assertEqual(count, 11)
        with self.assertRaises(ValueError):
            feature_statistics([torch.full((2, 3, 32, 32), float('nan'))], SmallFeatures(), 'cpu')
        with self.assertRaises(ValueError):
            feature_statistics([images[:1]], SmallFeatures(), 'cpu')

    def test_analytic_and_singular_distances(self):
        mean = np.zeros(3)
        cov = np.diag([1., 4., 0.])
        other = np.diag([4., 9., 0.])
        self.assertAlmostEqual(frechet_distance(mean, cov, mean, other), 2.)
        self.assertAlmostEqual(frechet_distance(mean, cov, mean, cov), 0.)
        self.assertAlmostEqual(frechet_distance(mean, cov, np.ones(3), cov), 3.)

    def test_generator_rng_and_mode_preserved(self):
        with patch('fid.load_inception', return_value=SmallFeatures()):
            evaluator = CIFAR10FID(torch.rand(5, 3072), device='cpu', batch_size=3)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3072), torch.nn.Sigmoid())
        state = torch.random.get_rng_state().clone()
        first = evaluator.evaluate_generator(model, 4, count=7, seed=31)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        second = evaluator.evaluate_generator(model, 4, count=7, seed=31)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
