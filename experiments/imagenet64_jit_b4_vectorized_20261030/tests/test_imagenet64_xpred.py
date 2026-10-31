"""X-prediction conversion, four-rank Riesz gradients, and checkpointed JiT."""
import copy
import unittest
import torch
from imagenet64_jit import JiTVelocity, clean_to_velocity, sample_velocity, sampling_nfe
from cifar_riesz_velocity import global_particle_loss
from riesz_loss import riesz_loss
from vendor.jit.model_jit import JiT


class XPredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_conversion_and_gradient_at_endpoints(self):
        state = torch.randn(4,3,4,4)
        clean = torch.randn_like(state).requires_grad_()
        times = torch.tensor([0.,.5,.999,1.])
        denom = torch.tensor([1.,.5,.05,.05])[:,None,None,None]
        actual = clean_to_velocity(clean,state,times)
        torch.testing.assert_close(actual,(clean-state)/denom)
        actual.sum().backward()
        torch.testing.assert_close(clean.grad,denom.reciprocal().expand_as(clean))
        self.assertTrue(actual.isfinite().all())

    def test_four_rank_gradient_matches_global_xpred_riesz(self):
        model = torch.nn.Linear(6,6)
        clean, noise = torch.randn(12,6), torch.randn(12,6)
        for t in [.001,.77,.999]:
            h=min(.01,1-t)
            state=(1-t)*noise+t*clean
            target=(1-t-h)*noise+(t+h)*clean
            model.zero_grad()
            prediction=state+h*clean_to_velocity(model(state),state,t)
            expected=riesz_loss(prediction,target,taus=(.2,.05,.02),distance_powers=(.8,1.,1.5))[0]*(1-t)/h
            expected.backward()
            reference=[p.grad.clone() for p in model.parameters()]
            predictions=list(prediction.detach().chunk(4))
            gradients=[]
            for rank in range(4):
                replica=copy.deepcopy(model); replica.zero_grad()
                local=state.chunk(4)[rank]
                query=local+h*clean_to_velocity(replica(local),local,t)
                loss,_=global_particle_loss(query,predictions,list(target.chunk(4)),rank,t,.01)
                (loss*4).backward()
                gradients.append([p.grad.clone() for p in replica.parameters()])
            for index,wanted in enumerate(reference):
                torch.testing.assert_close(sum(g[index] for g in gradients)/4,wanted,rtol=3e-5,atol=2e-6)

    def test_b4_geometry_and_clean_output_conversion(self):
        model=JiTVelocity(4,'x',gradient_checkpointing=True)
        self.assertEqual(model.backbone.x_embedder.num_patches,256)
        self.assertEqual(len(model.backbone.blocks),12)
        state=torch.randn(1,3,64,64)
        # adaLN-Zero starts with exactly zero clean-image output.
        output=model(state,torch.tensor(.5))
        torch.testing.assert_close(output,-2*state)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                            for p in model.parameters() if p.requires_grad))

    def test_block_checkpointing_preserves_outputs_and_gradients(self):
        ordinary=JiT(input_size=8,patch_size=4,hidden_size=64,depth=2,num_heads=4,
                     bottleneck_dim=16,in_context_len=2,in_context_start=1,num_classes=10)
        torch.nn.init.normal_(ordinary.final_layer.linear.weight,std=.01)
        for block in ordinary.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.01)
        checkpointed=copy.deepcopy(ordinary)
        checkpointed.gradient_checkpointing=True
        state=torch.randn(2,3,8,8); time=torch.tensor([.2,.8]); labels=torch.tensor([2,3])
        a=ordinary(state,time,labels); b=checkpointed(state,time,labels)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().sum().backward(); b.square().sum().backward()
        for x,y in zip(ordinary.parameters(),checkpointed.parameters()):
            if x.requires_grad:torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)

    def test_xpred_sampling_uses_final_euler_and_stays_finite(self):
        class FixedClean(torch.nn.Module):
            prediction='x'
            def __init__(self):super().__init__(); self.times=[]
            def forward(self,x,t,labels):
                self.times.append(float(t))
                return clean_to_velocity(torch.zeros_like(x),x,t)
        model=FixedClean()
        result=sample_velocity(model,torch.randn(2,3,4,4),50,False,'heun',cfg=1)
        self.assertEqual(len(model.times),sampling_nfe(50,'heun','x'))
        self.assertLess(max(model.times),1.)
        self.assertTrue(result.isfinite().all())


if __name__=='__main__':unittest.main()
