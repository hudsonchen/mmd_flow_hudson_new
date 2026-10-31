"""Velocity backbone, reference-time distribution, and distributed FID statistics."""
import unittest
import numpy as np
import torch
from velocity_model import DDPMFlow, skewed_times
from fid import FeatureMoments


class VelocityModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_time_sampling_policy(self):
        t=skewed_times(100000,'cpu',torch.Generator().manual_seed(0))
        self.assertTrue(((t>=.0001)&(t<=1)).all())
        self.assertAlmostEqual(float(torch.logit(t).mean()),1.2,delta=.02)
        self.assertAlmostEqual(float(torch.logit(t).std()),1.2,delta=.02)


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

