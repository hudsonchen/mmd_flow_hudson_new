"""Verify global Riesz gradients survive partitioning and direct velocity sampling."""
import copy
import math
import unittest
import torch
from cifar_riesz_velocity import global_particle_loss, training_time, sample_velocity
from riesz_loss import riesz_loss


class RieszVelocityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(51)

    def test_four_rank_gradient_matches_unpartitioned_coupled_loss(self):
        class Velocity(torch.nn.Linear):
            def forward(self, x, t):return super().forward(x)
        model=Velocity(6,6)
        clean,noise=torch.randn(16,6),torch.randn(16,6)
        # Include a shortened endpoint interval and both weighting choices.
        for time in [.1,.77,.999]:
            for time_power in [0.,.5,1.]:
                model.zero_grad()
                h=min(.01,1-time)
                source=(1-time)*noise+time*clean
                target=(1-time-h)*noise+(time+h)*clean
                expected=riesz_loss(source+h*model(source,time),target,
                    taus=(.2,.05,.02),distance_powers=(.8,1.,1.5))[0]*(1-time)**time_power/h
                expected.backward()
                reference=[p.grad.clone() for p in model.parameters()]
                h=min(.01,1-time)
                source=(1-time)*noise+time*clean
                target=(1-time-h)*noise+(time+h)*clean
                with torch.no_grad():predictions=list((source+h*model(source,time)).chunk(4))
                gradients=[]
                for rank in range(4):
                    replica=copy.deepcopy(model);replica.zero_grad()
                    local=source.chunk(4)[rank]
                    prediction=local+h*replica(local,time)
                    loss,_=global_particle_loss(prediction,predictions,list(target.chunk(4)),rank,time,.01,time_power=time_power)
                    torch.testing.assert_close(loss,expected,rtol=2e-5,atol=1e-5)
                    (4*loss).backward()
                    gradients.append([p.grad.clone() for p in replica.parameters()])
                for index,wanted in enumerate(reference):
                    actual=sum(g[index] for g in gradients)/4
                    torch.testing.assert_close(actual,wanted,rtol=3e-5,atol=2e-6)

    def test_skewed_time_distribution_and_horizon_are_safe(self):
        rng=torch.Generator().manual_seed(71)
        samples=torch.stack([training_time('cpu','skewed',rng) for _ in range(12000)])
        self.assertTrue(((samples>=.0001)&(samples<=.999)).all())
        self.assertAlmostEqual(float(samples.median()),float(torch.sigmoid(torch.tensor(1.2))),delta=.015)
        self.assertTrue((1-samples>0).all())

    def test_direct_velocity_euler_has_no_endpoint_division(self):
        class ConstantVelocity(torch.nn.Module):
            def forward(self,x,t):return torch.full_like(x,2.)
        model=ConstantVelocity().train()
        x=torch.randn(4,3,2,2)
        torch.testing.assert_close(model(x,torch.tensor(1.)),torch.full_like(x,2.))
        torch.testing.assert_close(sample_velocity(model,x,100),x+2,atol=1e-5,rtol=1e-5)
        self.assertTrue(model.training)

    def test_heun_accuracy_and_call_count(self):
        class Exponential(torch.nn.Module):
            def __init__(self):super().__init__();self.times=[]
            def forward(self,x,t):self.times.append(float(t));return x
        model=Exponential().train()
        noise=torch.ones(2,3,2,2)
        actual=sample_velocity(model,noise,50,solver='heun')
        self.assertEqual(len(model.times),100)
        self.assertEqual((min(model.times),max(model.times)),(0.,1.))
        self.assertLess(float((actual-math.e).abs().max()),.0002)
        self.assertTrue(model.training)
        euler=sample_velocity(model,noise,100)
        self.assertLess(float((actual-math.e).abs().max()),float((euler-math.e).abs().max())/10)

    def test_evaluation_only_resume_rejects_training_changes(self):
        from scripts.lsf.run_cifar_riesz_ddp import validate_resume_config
        saved=dict(output='old',sample_steps=100,sampling='old',reference_time_power=1.,world_size=4)
        new=dict(saved,output='new',sample_steps=50,sampling='new',solver='heun',required_gpu='RTX 2080 Ti')
        self.assertTrue(validate_resume_config(saved,new,True))
        with self.assertRaises(ValueError):validate_resume_config(saved,new)
        for change in [dict(reference_time_power=.5),dict(world_size=1),dict(lr=.001)]:
            with self.assertRaises(ValueError):validate_resume_config(saved,dict(new,**change),True)


if __name__=='__main__':unittest.main()
