"""Check the mathematical field, collision convention, and Euler energy descent."""

import unittest

import torch

from backbone import ParticleGenerator, UNetParticleGenerator
from main import epoch_batches, generator_step, snapshot_steps
from loss import mean_away, mean_distance, mmd_squared, velocity, _riesz_topk_softmax_loss_impl


class RieszFlowTests(unittest.TestCase):
    def test_unet_standard_riesz_matches_direct_mmd_gradient(self):
        torch.manual_seed(41)
        model = UNetParticleGenerator(base_channels=4).double()
        latent = torch.randn(3, 784, dtype=torch.float64)
        target = torch.rand(4, 784, dtype=torch.float64, requires_grad=True)
        images = model(latent)
        energy = (torch.linalg.vector_norm(images[:, None] - target[None], dim=-1).mean()
                  - 0.5 * torch.linalg.vector_norm(images[:, None] - images[None], dim=-1).mean())
        expected = torch.autograd.grad(energy, tuple(model.parameters()))
        before = [p.detach().clone() for p in model.parameters()]
        generator_step(model, latent, target, torch.optim.SGD(model.parameters(), lr=1e-3),
                       2, objective="riesz")
        for param, old, gradient in zip(model.parameters(), before, expected):
            torch.testing.assert_close(param.grad, gradient)
            torch.testing.assert_close(param, old - 1e-3 * gradient)
        self.assertIsNone(target.grad)
        self.assertIsNone(latent.grad)

    def test_unet_generation_is_batch_independent_and_trains_with_softmax(self):
        torch.manual_seed(31)
        model = UNetParticleGenerator(base_channels=8)
        latent = torch.randn(3, 784)
        target = torch.rand(4, 784, requires_grad=True)
        with torch.no_grad():
            before_images = model(latent)
            single_image = model(latent[:1])
        self.assertEqual(before_images.shape, (3, 784))
        torch.testing.assert_close(before_images[:1], single_image, atol=1e-6, rtol=1e-5)
        before_parameters = [p.detach().clone() for p in model.parameters()]
        generator_step(model, latent, target, torch.optim.Adam(model.parameters(), lr=1e-3),
                       2, objective="softmax-riesz")
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(any(not torch.equal(p, old)
                            for p, old in zip(model.parameters(), before_parameters)))
        with torch.no_grad():
            after_images = model(latent)
        self.assertTrue(torch.isfinite(after_images).all())
        self.assertTrue(((after_images >= 0) & (after_images <= 1)).all())
        self.assertFalse(torch.equal(before_images, after_images))
        self.assertIsNone(target.grad)
        self.assertIsNone(latent.grad)

    def test_softmax_generator_update_matches_upstream_gradient(self):
        torch.manual_seed(15)
        model = ParticleGenerator(3, (7,))
        latent = torch.randn(5, 3)
        target = torch.rand(6, 784, requires_grad=True)
        loss, _ = _riesz_topk_softmax_loss_impl(
            model(latent).unsqueeze(0), target.unsqueeze(0),
            tau_list=(0.2, 0.05, 0.02), normalize_per_temperature=True, epsilon=1e-6)
        expected = torch.autograd.grad(loss.mean(), tuple(model.parameters()))
        before = [p.detach().clone() for p in model.parameters()]
        generator_step(model, latent, target, torch.optim.SGD(model.parameters(), lr=1e-3),
                       2, objective="softmax-riesz")
        for param, old, gradient in zip(model.parameters(), before, expected):
            torch.testing.assert_close(param.grad, gradient)
            torch.testing.assert_close(param, old - 1e-3 * gradient)
        self.assertIsNone(target.grad)
        self.assertIsNone(latent.grad)

    def test_generator_parameter_gradient_matches_direct_energy(self):
        torch.manual_seed(12)
        model = ParticleGenerator(3, (7,)).double()
        latent = torch.randn(5, 3, dtype=torch.float64)
        target = torch.rand(6, 784, dtype=torch.float64)
        images = model(latent)
        energy = (torch.linalg.vector_norm(images[:, None] - target[None], dim=-1).mean()
                  - 0.5 * torch.linalg.vector_norm(images[:, None] - images[None], dim=-1).mean())
        expected = torch.autograd.grad(energy, tuple(model.parameters()))
        before = [p.detach().clone() for p in model.parameters()]
        latent_before = latent.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        generator_step(model, latent, target, optimizer, 2)
        for param, old, gradient in zip(model.parameters(), before, expected):
            torch.testing.assert_close(param.grad, gradient)
            torch.testing.assert_close(param, old - 1e-3 * gradient)
        torch.testing.assert_close(latent, latent_before)
        self.assertIsNone(latent.grad)
        generated = model(latent)
        self.assertEqual(generated.shape, (5, 784))
        self.assertTrue(((generated >= 0) & (generated <= 1)).all())
        yy = mean_distance(target, target, 2)
        self.assertLess(mmd_squared(generated.detach(), target, yy, 2),
                        mmd_squared(images.detach(), target, yy, 2))

    def test_epoch_covers_all_targets_including_partial_batch(self):
        batches = list(epoch_batches(60000, 256, torch.Generator().manual_seed(0)))
        self.assertEqual(len(batches), 235)
        self.assertEqual(len(batches[-1]), 96)
        self.assertTrue(torch.equal(torch.cat(batches).sort().values, torch.arange(60000)))
        self.assertEqual(sorted(snapshot_steps(100, 8)), [0, 13, 25, 38, 50, 63, 75, 88, 100])

    def test_field_matches_wasserstein_gradient(self):
        rng = torch.Generator().manual_seed(7)
        x = torch.randn(5, 4, generator=rng, dtype=torch.float64, requires_grad=True)
        y = torch.randn(7, 4, generator=rng, dtype=torch.float64)
        xy = torch.linalg.vector_norm(x[:, None] - y[None], dim=-1).mean()
        xx = torch.linalg.vector_norm(x[:, None] - x[None], dim=-1).mean()
        energy = xy - 0.5 * xx
        gradient, = torch.autograd.grad(energy, x)
        torch.testing.assert_close(velocity(x.detach(), y, 2), -len(x) * gradient)

    def test_exact_collisions_and_chunking(self):
        x = torch.tensor([[0., 0.], [0., 0.], [3., 4.]], dtype=torch.float64)
        expected = torch.tensor([[-0.2, -4 / 15], [-0.2, -4 / 15], [0.4, 8 / 15]],
                                dtype=torch.float64)
        torch.testing.assert_close(mean_away(x, x, 1), expected)
        torch.testing.assert_close(mean_away(x, x, 100), expected)
        torch.testing.assert_close(velocity(x, x, 2), torch.zeros_like(x))
        self.assertAlmostEqual(mmd_squared(x, x, mean_distance(x, x, 2), 2), 0.)

    def test_small_euler_step_lowers_energy(self):
        rng = torch.Generator().manual_seed(19)
        x = torch.randn(8, 3, generator=rng, dtype=torch.float64)
        y = torch.randn(11, 3, generator=rng, dtype=torch.float64)
        yy = mean_distance(y, y, 3)
        before = mmd_squared(x, y, yy, 3)
        after = mmd_squared(x + 0.01 * velocity(x, y, 3), y, yy, 3)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
