import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training_objectives import training_loss
from cifar_riesz_velocity import global_particle_loss
from imagenet64_jit import JiTVelocity, sample_velocity
from imagenet64_data import RandomImageBatches


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_riesz_exact_loss_and_gradient_including_endpoint(self):
        torch.manual_seed(3)
        clean,noise = torch.randn(2,5,3,4,4)
        for t in [.001,.77,.999]:
            h=min(.01,1-t)
            source=(1-t)*noise+t*clean
            velocity=torch.randn_like(clean,requires_grad=True)
            actual=training_loss(velocity,source,clean,noise,t,.01,'riesz')
            pred=source+h*velocity
            target=(1-t-h)*noise+(t+h)*clean
            expected,_=global_particle_loss(pred,[pred.detach()],[target],0,t,.01)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            a,=torch.autograd.grad(actual,velocity,retain_graph=True)
            b,=torch.autograd.grad(expected,velocity)
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_flow_matching_target_and_gradient(self):
        clean,noise=torch.randn(2,4,3,4,4)
        target=clean-noise
        v=target.clone().requires_grad_()
        self.assertEqual(float(training_loss(v,noise,clean,noise,.2,.01,'flow_matching')),0.)
        v=(target+1).detach().requires_grad_()
        loss=training_loss(v,noise,clean,noise,.2,.01,'flow_matching')
        loss.backward()
        torch.testing.assert_close(v.grad,torch.full_like(v,2/v.numel()))

    def test_class_batches_reproducible_and_unconditional(self):
        class Fake:
            records=[(str(i),i//2) for i in range(400)]
            images=np.broadcast_to(np.arange(400,dtype=np.uint16)[:,None,None,None],(400,3,2,2))
            def __len__(self):return 400
        data=Fake()
        conditional=RandomImageBatches(data,32,7,dropout=0.)
        for attempt in [0,10,503]:
            a,b=conditional[attempt],conditional[attempt]
            self.assertEqual(a['label'],b['label'])
            self.assertTrue(torch.equal(a['images'],b['images']))
            self.assertTrue(bool((a['images'].long()//2==a['label']).all()))
        mixed=RandomImageBatches(data,32,7,dropout=1.)[0]
        self.assertEqual(mixed['label'],200)
        self.assertGreater(len(mixed['images'][:,0,0,0].unique()),1)

    def test_model_dimensions_backward_and_sampler(self):
        torch.manual_seed(0)
        model=JiTVelocity(4,'x',.05,True)
        self.assertEqual(model.backbone.hidden_size,384)
        self.assertEqual(len(model.backbone.blocks),12)
        self.assertEqual(model.backbone.num_heads,6)
        self.assertEqual(model.backbone.y_embedder.embedding_table.num_embeddings,201)
        self.assertEqual(model.backbone.pos_embed.shape[1],256)
        clean,noise=torch.randn(2,2,3,64,64)
        source=.7*noise+.3*clean
        labels=torch.tensor([199,200])
        for objective in ['riesz','flow_matching']:
            model.zero_grad(set_to_none=True)
            velocity=model(source,torch.tensor(.3),labels)
            self.assertEqual(velocity.shape,source.shape)
            loss=training_loss(velocity,source,clean,noise,.3,.01,objective)
            loss.backward()
            grad=model.backbone.final_layer.linear.weight.grad
            self.assertTrue(bool(grad.isfinite().all()))
            self.assertGreater(float(grad.norm()),0.)
        samples=sample_velocity(model,noise,steps=2,labels=labels,cfg=1.,amp=False)
        self.assertTrue(bool(samples.isfinite().all()))
        print('parameter_count',sum(p.numel() for p in model.parameters()))

    def test_resume_rejects_objective_change(self):
        spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/cs_cluster/run_imagenet64_jit.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.validate_resume_config({'objective':'riesz'},{'objective':'flow_matching'})


if __name__=='__main__':
    unittest.main()
