import unittest
import json
from pathlib import Path
import numpy as np
import torch
from core import ForecastNet, energy_loss, objective, sample, scores, shuffle_members, variogram
from legacy_core import riesz_loss
from original_riesz import riesz_loss as original


class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small CPU convolution checks should not depend on oneDNN's threaded
        # kernel startup on a shared GPU node. Production is a separate process.
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled=False
        cls.config=json.loads((Path(__file__).parent/'config.json').read_text())

    def test_exact_shared_architecture_initialization_and_gradients(self):
        states=[];counts=[]
        clean,noise=torch.randn(2,2,8,40);ctx,ctrl=torch.randn(2,2,40);lead=torch.tensor([.2,1.])
        for method in self.config['methods']:
            torch.manual_seed(17);m=ForecastNet(16,2)
            states.append({k:v.clone() for k,v in m.state_dict().items()})
            counts.append(sum(p.numel() for p in m.parameters()))
            loss=objective(m,clean,noise,ctx,ctrl,lead,.4,method,self.config)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(float(m.output.weight.grad.norm()),0)
            self.assertTrue(torch.isfinite(sample(m,noise,ctx,ctrl,lead,method,3)).all())
        self.assertEqual(len(set(counts)),1)
        for state in states[1:]:
            self.assertEqual(state.keys(),states[0].keys())
            for key in state:torch.testing.assert_close(state[key],states[0][key],rtol=0,atol=0)

    def test_cyclic_equivariance_and_independent_contexts(self):
        torch.manual_seed(8);m=ForecastNet(16,2)
        torch.nn.init.normal_(m.output.weight,std=.1)
        x=torch.randn(2,5,40);ctx,ctrl=torch.randn(2,2,40);lead=torch.tensor([.2,1.])
        y=m(x,ctx,ctrl,lead,.3)
        rolled=m(x.roll(7,-1),ctx.roll(7,-1),ctrl.roll(7,-1),lead,.3)
        torch.testing.assert_close(rolled,y.roll(7,-1),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(y[:1],m(x[:1],ctx[:1],ctrl[:1],lead[:1],.3),atol=1e-6,rtol=1e-5)

    def test_energy_unbiased_normalization_and_gradient(self):
        x=torch.tensor([[[0.,0.],[2.,2.]]],requires_grad=True)
        y=torch.tensor([[[.9,.9],[1.1,1.1]]])
        self.assertAlmostEqual(float(energy_loss(x,y)),0.,places=6)
        torch.manual_seed(3)
        x=torch.randn(2,5,4,dtype=torch.double,requires_grad=True);y=torch.randn(2,7,4,dtype=torch.double)
        explicit=sum(torch.linalg.vector_norm(x[b,i]-y[b,j]) for b in range(2) for i in range(5) for j in range(7))/70
        explicit-=sum(torch.linalg.vector_norm(x[b,i]-x[b,j]) for b in range(2) for i in range(5) for j in range(5) if i!=j)/80
        actual=energy_loss(x,y)
        torch.testing.assert_close(actual,explicit)
        a,=torch.autograd.grad(actual,x,retain_graph=True);b,=torch.autograd.grad(explicit,x)
        torch.testing.assert_close(a,b)

    def test_riesz_matches_production_field_and_gradient(self):
        torch.manual_seed(7);q=torch.randn(2,11,40,requires_grad=True);p=torch.randn(2,11,40)
        a,field=riesz_loss(q,p)
        parts=[original(q[i],p[i]) for i in range(2)]
        b=torch.stack([v[0] for v in parts]).mean()
        expected=torch.stack([v[1]['frozen_velocity'] for v in parts])
        torch.testing.assert_close(field,expected,atol=2e-5,rtol=2e-4)
        ag,=torch.autograd.grad(a,q,retain_graph=True);bg,=torch.autograd.grad(b,q)
        torch.testing.assert_close(ag,bg,atol=2e-5,rtol=2e-4)

    def test_shuffle_preserves_marginals_but_changes_dependence(self):
        rng=np.random.default_rng(9)
        x=np.repeat(rng.normal(size=(2,256,1)),4,axis=2)
        y=np.repeat(rng.normal(size=(2,512,1)),4,axis=2)
        shuffled=shuffle_members(x)
        np.testing.assert_array_equal(np.sort(shuffled,axis=1),np.sort(x,axis=1))
        self.assertLess(variogram(x,y).mean(),variogram(shuffled,y).mean())
        a,b=scores(x,y,2.),scores(shuffled,y,2.)
        np.testing.assert_allclose(a['crps'],b['crps'],atol=1e-12)
        np.testing.assert_allclose(sum(a[f'rank_bin_{i}'] for i in range(10)),1.)

    def test_known_gaussian_translation_can_be_learned(self):
        torch.manual_seed(55);m=ForecastNet(8,1)
        opt=torch.optim.Adam(m.parameters(),lr=.003)
        ctx=torch.zeros(2,40);lead=torch.full((2,),.2)
        for step in range(400):
            z=torch.randn(2,8,40);t=float(torch.rand(()))
            opt.zero_grad();loss=objective(m,z+.4,z,ctx,ctx,lead,t,'flow_matching',self.config)
            loss.backward();opt.step()
        z=torch.randn(2,128,40)
        draws=sample(m,z,ctx,ctx,lead,'flow_matching',10)
        self.assertLess(float((draws-z-.4).square().mean()),.01)


if __name__=='__main__':unittest.main()
