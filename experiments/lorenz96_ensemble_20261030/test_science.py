import unittest
import numpy as np
import torch
from core import rhs, integrate, riesz_loss, scores, ForecastFlow, objective, sample
from original_riesz import riesz_loss as original


class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_batched_loss_field_and_gradient_match_original(self):
        torch.manual_seed(19)
        for t in [.001,.3,.9,.999]:
            clean,noise,v=torch.randn(3,3,17,40)
            h=min(.01,1-t)
            q=((1-t)*noise+t*clean+h*v).requires_grad_()
            p=(1-t-h)*noise+(t+h)*clean
            actual,field=riesz_loss(q,p)
            parts=[original(q[i],p[i]) for i in range(3)]
            expected=torch.stack([x[0] for x in parts]).mean()
            expected_field=torch.stack([x[1]['frozen_velocity'] for x in parts])
            ag,=torch.autograd.grad(actual,q,retain_graph=True)
            eg,=torch.autograd.grad(expected,q)
            for a,b in [(actual,expected),(field,expected_field),(ag,eg)]:
                rel=(a-b).double().norm()/b.double().norm().clamp_min(1e-15)
                self.assertLess(float(rel),2e-4)

    def test_contexts_do_not_interact_and_ablation_only_changes_force(self):
        torch.manual_seed(2)
        q,p=torch.randn(2,2,9,40)
        _,full=riesz_loss(q,p)
        _,alone=riesz_loss(q[:1],p[:1])
        torch.testing.assert_close(full[:1],alone)
        _,attract=riesz_loss(q,p,repulsion=0)
        self.assertGreater(float((full-attract).norm()),0)
        _,coincident=riesz_loss(q,q)
        self.assertLess(float(coincident.abs().max()),1e-4)

    def test_lorenz_equilibrium_symmetry_energy_and_solver(self):
        np.testing.assert_allclose(rhs(np.full((2,40),8.)),0.)
        rng=np.random.default_rng(3);x=rng.normal(size=(3,40))
        np.testing.assert_allclose(rhs(np.roll(x,4,-1)),np.roll(rhs(x),4,-1))
        nonlinear=rhs(x)-(-x+8)
        np.testing.assert_allclose((nonlinear*x).sum(-1),0,atol=1e-12)
        a=integrate(x,[.2],.01);b=integrate(x,[.2],.005);c=integrate(x,[.2],.0025)
        self.assertLess(np.max(abs(b-c)),np.max(abs(a-b))/8)

    def test_proper_score_analytic_and_overdispersion(self):
        # Fair CRPS for {0,2} evaluated at truth 1 is 1 - 1 = 0.
        x=np.array([[[0.,0.],[2.,2.]]]);y=np.tile(np.array([[[.9,.9],[1.1,1.1]]]),(1,4,1))
        v=scores(x,y,1.)
        self.assertAlmostEqual(v['crps'][0],0.)
        self.assertAlmostEqual(v['energy_score'][0],0.)
        rng=np.random.default_rng(4)
        y=rng.normal(size=(4,2048,3));x=rng.normal(size=(4,256,3))
        good=scores(x,y,2.);wide=scores(x*4,y,2.)
        self.assertLess(good['crps'].mean(),wide['crps'].mean())
        self.assertLess(abs(good['coverage90'].mean()-.9),.04)
        self.assertGreater(wide['width90'].mean(),good['width90'].mean())

    def test_objectives_have_finite_gradients_and_sampling(self):
        import json
        from pathlib import Path
        c=json.loads((Path(__file__).parent/'config.json').read_text())
        clean,noise=torch.randn(2,2,8,40);ctx=torch.randn(2,40);lead=torch.tensor([.2,1.])
        for method in c['methods']:
            m=ForecastFlow(40,32,2)
            loss=objective(m,clean,noise,ctx,lead,.4,method,c)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertTrue(torch.isfinite(sample(m,noise,ctx,lead,5)).all())


if __name__=='__main__':
    unittest.main()
