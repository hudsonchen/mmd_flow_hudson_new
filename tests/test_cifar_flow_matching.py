"""Flow mathematics, integration accuracy, distributed batching and FID moments."""
import math
import unittest
import numpy as np
import torch

from cifar_flow_matching import DDPMFlow, FeatureMoments, edm_time_grid, flow_target, sample_heun, skewed_times
from scripts.lsf.run_cifar_fm_ddp import rank_indices


class FlowMatchingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_affine_path_and_target_are_consistent(self):
        clean,noise=torch.randn(5,3,4,4),torch.randn(5,3,4,4)
        t=torch.linspace(0,1,5)
        x,v=flow_target(clean,noise,t)
        xp,_=flow_target(clean,noise,t+.001)
        torch.testing.assert_close((xp-x)/.001,v,atol=.0004,rtol=.001)
        torch.testing.assert_close(x[0],noise[0])
        torch.testing.assert_close(x[-1],clean[-1])

    def test_time_sampling_policy(self):
        t=skewed_times(100000,'cpu',torch.Generator().manual_seed(0))
        self.assertTrue(((t>=.0001)&(t<=1)).all())
        self.assertAlmostEqual(float(torch.logit(t).mean()),1.2,delta=.02)
        self.assertAlmostEqual(float(torch.logit(t).std()),1.2,delta=.02)

    def test_heun_accuracy_endpoints_nfe_and_mode(self):
        class Exponential(torch.nn.Module):
            def __init__(self):
                super().__init__();self.calls=0;self.times=[]
            def forward(self,x,t):
                self.calls+=1;self.times.append(float(t));return x
        model=Exponential().train()
        x=torch.ones(2,3,2,2)
        coarse=sample_heun(model,x,25)
        self.assertTrue(model.training)
        self.assertEqual(model.calls,50)
        self.assertEqual(min(model.times),0.)
        self.assertEqual(max(model.times),1.)
        fine=sample_heun(model,x,100)
        self.assertLess(float((fine-math.e).abs().max()),float((coarse-math.e).abs().max())/5)
        self.assertLess(float((fine-math.e).abs().max()),.001)
        grid=edm_time_grid(50)
        self.assertEqual(len(grid),51)
        self.assertTrue((grid[1:]>grid[:-1]).all())

    def test_four_rank_batches_are_disjoint_reproducible_and_shuffle(self):
        first=[rank_indices(50000,256,4,r,1,42) for r in range(4)]
        self.assertEqual(len(torch.cat(first).unique()),256)
        self.assertTrue(all(len(v)==64 for v in first))
        torch.testing.assert_close(first[2],rank_indices(50000,256,4,2,1,42))
        self.assertFalse(torch.equal(first[0],rank_indices(50000,256,4,0,196,42)))

    def test_microbatch_gradients_match_full_mean(self):
        torch.manual_seed(1)
        model=torch.nn.Linear(7,4)
        x,y=torch.randn(256,7),torch.randn(256,4)
        ((model(x)-y)**2).mean().backward()
        expected=[p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        # DDP averages four rank gradients; each rank accumulates two microbatches.
        for xx,yy in zip(x.split(32),y.split(32)):
            (((model(xx)-yy)**2).mean()/8).backward()
        for p,g in zip(model.parameters(),expected):
            torch.testing.assert_close(p.grad,g,atol=1e-7,rtol=1e-6)

    def test_distributed_moments_match_numpy_covariance(self):
        features=np.random.default_rng(7).normal(size=(103,11))
        full=FeatureMoments(11)
        for shard in np.array_split(features,4):
            partial=FeatureMoments(11);partial.add(shard)
            full.count+=partial.count;full.sum+=partial.sum;full.outer+=partial.outer
        mean,cov,count=full.statistics()
        np.testing.assert_allclose(mean,features.mean(0),atol=1e-14)
        np.testing.assert_allclose(cov,np.cov(features,rowvar=False),atol=1e-14)
        self.assertEqual(count,103)

    def test_exact_backbone_and_time_adapter(self):
        model=DDPMFlow()
        self.assertEqual(sum(p.numel() for p in model.parameters()),61804419)
        self.assertTrue(model.backbone.time_conditioned)
        self.assertEqual(model.backbone.base_channels,128)
        self.assertEqual([len(level) for level in model.backbone.encoders],[4]*4)
        class Capture(torch.nn.Module):
            def forward(self,x,t):
                self.t=t;return x
        model.backbone=Capture()
        model(torch.zeros(3,3,32,32),torch.tensor([0.,.5,1.]))
        torch.testing.assert_close(model.backbone.t,torch.tensor([999.,499.5,0.]))


if __name__=='__main__':unittest.main()
