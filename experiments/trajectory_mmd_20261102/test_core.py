"""Scientific checks for the estimator and the differentiated flow."""
import unittest
import torch
from core import Velocity, integrate, gaussian_mmd, pixel_marginal_mmd


class ScientificChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_gram_and_permutations(self):
        x = torch.randn(7, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(9, 2, dtype=torch.double)
        k = lambda a, b: torch.exp(-torch.cdist(a, b).square()/2)
        xx, yy = k(x,x), k(y,y)
        expected = (xx.sum()-xx.trace())/42 + (yy.sum()-yy.trace())/72 - 2*k(x,y).mean()
        actual = gaussian_mmd(x, y, (1.,))
        torch.testing.assert_close(actual, expected)
        p, q = torch.randperm(7), torch.randperm(9)
        permuted = gaussian_mmd(x[p], y[q], (1.,))
        torch.testing.assert_close(permuted, actual)
        torch.testing.assert_close(torch.autograd.grad(actual,x)[0], torch.autograd.grad(permuted,x)[0])

    def test_solver_and_full_gradient(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(.4, dtype=torch.double))
            def forward(self,x,t):
                return self.a*x
        model = Linear()
        z = torch.randn(5,2,dtype=torch.double)
        result = integrate(model,z,steps=32)
        exact = z*model.a.exp()
        error = (result-exact).abs().max()
        refined = (integrate(model,z,steps=64)-exact).abs().max()
        self.assertLess(float(refined), float(error)/3.8)
        gradient = torch.autograd.grad(result.sum(), model.a)[0]
        torch.testing.assert_close(gradient, exact.sum(), atol=2e-4, rtol=2e-4)
        checked = integrate(model,z,steps=32,checkpoint_steps=True)
        torch.testing.assert_close(result,checked)
        torch.testing.assert_close(gradient,torch.autograd.grad(checked.sum(),model.a)[0])

    def test_pixel_endpoint_and_gradient(self):
        x = torch.randn(5,1,4,4,dtype=torch.double,requires_grad=True)
        y = torch.randn(6,1,4,4,dtype=torch.double)
        torch.testing.assert_close(pixel_marginal_mmd(x,y,1.,scales=(1,)),
            gaussian_mmd(x,y,tuple(4*b for b in (.05,.1,.2,.4))))
        self.assertTrue(torch.autograd.gradcheck(lambda a:pixel_marginal_mmd(a,y,.61,scales=(1,)), (x,)))
        value=pixel_marginal_mmd(x,y,.61,scales=(1,))
        permuted=pixel_marginal_mmd(x[torch.randperm(5)],y[torch.randperm(6)],.61,scales=(1,))
        torch.testing.assert_close(value,permuted)
        torch.testing.assert_close(torch.autograd.grad(value,x)[0],torch.autograd.grad(permuted,x)[0])

    def test_learned_skip_can_contract_every_pixel(self):
        model=Velocity('mnist',width=16,linear_skip=True).double()
        z=torch.randn(3,1,28,28,dtype=torch.double)
        torch.testing.assert_close(integrate(model,z,steps=4),z)
        with torch.no_grad():
            model.skip[-1].bias.fill_(-1.)
        value=integrate(model,z,steps=128)
        torch.testing.assert_close(value,z*torch.exp(torch.tensor(-1.)),atol=2e-5,rtol=2e-5)

    def test_integrated_noise_matches_monte_carlo(self):
        x = torch.randn(8,1,2,2,dtype=torch.double)
        y = torch.randn(11,1,2,2,dtype=torch.double)
        t = .45
        bands = tuple((4*((t*b)**2+(1-t)**2))**.5 for b in (.05,.1,.2,.4))
        exact = pixel_marginal_mmd(x,y,t,scales=(1,))
        estimates = torch.stack([gaussian_mmd(x,t*y+(1-t)*torch.randn_like(y),bands) for _ in range(1000)])
        self.assertLess(float(abs(estimates.mean()-exact)), 5*float(estimates.std()/len(estimates)**.5))

    def test_stable_field_growth_bound(self):
        model=Velocity('mnist',width=16,stable_field=True).double()
        with torch.no_grad():
            model.net[-1].weight.normal_(std=10.)
        x=100*torch.randn(4,1,28,28,dtype=torch.double)
        v=model(x,.7)
        # Coordinatewise outward speed cannot exceed the bounded residual speed.
        self.assertTrue(bool((x.sign()*v <= 4+1e-10).all()))
        self.assertTrue(torch.isfinite(torch.autograd.grad(v.square().mean(),model.net[-1].weight)[0]).all())


if __name__ == '__main__':
    unittest.main()
