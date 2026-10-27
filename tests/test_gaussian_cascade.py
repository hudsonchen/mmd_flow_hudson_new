import unittest
import torch
from scripts.test_gaussian_cascade import Map, compose, source_batch, sample_path, energy, centers, gaussian_mmd, RBF_SIGMAS


class GaussianCascadeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_path_endpoints_and_analytic_moments(self):
        for t in (0.,.3,1.):
            x,_=sample_path(40000,t,torch.Generator().manual_seed(4))
            # Uniform ring has covariance radius^2/2 = 2 per coordinate.
            expected=(1-t)**2+t*t*(2+.12**2)
            self.assertLess(float(x.mean(0).abs().max()),.03)
            torch.testing.assert_close(x.var(0),torch.full((2,),expected),rtol=.025,atol=.01)

    def test_actual_source_is_composed_and_frozen(self):
        class Shift(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shift=torch.nn.Parameter(torch.tensor(3.))
            def forward(self,x):return x+self.shift
        prior=Shift()
        actual=source_batch('mmd_actual',[prior],1,16,torch.Generator().manual_seed(1))
        ideal,noise=sample_path(16,.1,torch.Generator().manual_seed(1))
        torch.testing.assert_close(actual,noise+3)
        self.assertFalse(actual.requires_grad)
        self.assertFalse(torch.allclose(actual,ideal))
        current=Map(8)
        energy(current(actual),torch.randn(20,2)).backward()
        self.assertIsNone(prior.shift.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in current.parameters()),0)

    def test_energy_translation_gradient_points_toward_target(self):
        base=torch.tensor([[-1.,0.],[0.,1.],[1.,0.],[0.,-1.]])
        shift=torch.tensor([4.,0.],requires_grad=True)
        loss=energy(base+shift,base)
        loss.backward()
        self.assertGreater(float(shift.grad[0]),0)
        self.assertAlmostEqual(float(shift.grad[1]),0,places=5)
        torch.testing.assert_close(energy(base+shift,base),energy(base,base+shift))

    def test_map_initializes_to_identity_and_parameter_match(self):
        x=torch.randn(10,2)
        torch.testing.assert_close(Map()(x),x,rtol=0,atol=0)
        staged=10*sum(p.numel() for p in Map().parameters())
        direct=sum(p.numel() for p in Map(209).parameters())
        self.assertLess(abs(direct-staged)/staged,.01)

    def test_gaussian_mmd_matches_explicit_unbiased_estimator(self):
        x=torch.tensor([[0.,0.],[1.,0.],[0.,1.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[.3,.2],[-.2,-.4]],dtype=torch.float64)
        def k(a,b):
            return sum(torch.exp(-(a-b).square().sum()/(2*s*s)) for s in RBF_SIGMAS)/len(RBF_SIGMAS)
        expected=sum(k(a,b) for i,a in enumerate(x) for j,b in enumerate(x) if i!=j)/6
        expected=expected+sum(k(a,b) for i,a in enumerate(y) for j,b in enumerate(y) if i!=j)/2
        expected=expected-2*sum(k(a,b) for a in x for b in y)/6
        actual=gaussian_mmd(x,y)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])

    def test_fourier_map_starts_at_identity_and_freezes_frequencies(self):
        model=Map(fourier=True)
        x=torch.randn(16,2)
        torch.testing.assert_close(model(x),x,rtol=0,atol=0)
        frequency=model.frequencies.clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        gaussian_mmd(model(x),torch.randn(20,2)).backward()
        optimizer.step()
        torch.testing.assert_close(model.frequencies,frequency,rtol=0,atol=0)
        self.assertFalse(model.frequencies.requires_grad)
        self.assertGreater(float((model(x)-x).abs().sum()),0)


if __name__=='__main__':unittest.main()
