"""Checks for transport integration and the new reference-guided objectives."""
import os
import copy
os.environ['DRIFT_COMPILE'] = '0'
import unittest
import tempfile
from pathlib import Path
import torch
from small_flow import ToyVelocity, PotentialVelocity, ImageVelocity, DenoiserVelocity, sample, marginal_loss, coupled_reference_loss, riesz_loss, validate_riesz_parent, image_riesz_loss, reference_centered_loss, pixel_moment_features, verify_reconstruction_encoder, local_reference_hybrid_loss
from small_flow import pixel_gradient_features, spatial_riesz_loss
from small_flow import restore_image_checkpoint,vgg_endpoint_riesz_loss


class SmallFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_self_normalization_cancels_identical_empirical_populations(self):
        torch.manual_seed(104)
        target=torch.randn(9,6,requires_grad=True)
        for smoothing in [0.,.05]:
            for normalization in ['none','temperature','global']:
                prediction=target.detach().clone().requires_grad_()
                value,info=riesz_loss(prediction,target,taus=[.2,.05,.02],affinity='row-column',
                    normalization=normalization,include_self_negative=True,distance_smoothing=smoothing)
                value.backward()
                torch.testing.assert_close(info['frozen_velocity'],torch.zeros_like(prediction),rtol=0,atol=1e-7)
                torch.testing.assert_close(prediction.grad,torch.zeros_like(prediction),rtol=0,atol=1e-7)
                self.assertIsNone(target.grad)
        _,legacy=riesz_loss(target.detach(),target,taus=[.2,.05,.02],affinity='row-column')
        self.assertGreater(float(legacy['frozen_velocity'].norm()),1e-4)

    def test_smooth_self_normalized_force_recovers_small_translation(self):
        # A shared translation must be corrected, while forces vanish at a match.
        torch.manual_seed(105)
        target=torch.randn(12,7)
        shift=torch.full_like(target,.001)
        prediction=(target+shift).requires_grad_()
        value,info=riesz_loss(prediction,target,taus=[.2,.05],affinity='row-column',
            include_self_negative=True,distance_smoothing=.05,normalization='none')
        gradient=torch.autograd.grad(value,prediction)[0]
        self.assertGreater(float((gradient*shift).sum()),0.)
        self.assertTrue(torch.isfinite(gradient).all())
        _,nearer=riesz_loss(target+shift/10,target,taus=[.2,.05],affinity='row-column',
            include_self_negative=True,distance_smoothing=.05,normalization='none')
        self.assertLess(float(nearer['frozen_velocity'].norm()),float(info['frozen_velocity'].norm())/5)
        perm=torch.randperm(len(target))
        _,reordered=riesz_loss(prediction[perm],target.flip(0),taus=[.2,.05],affinity='row-column',
            include_self_negative=True,distance_smoothing=.05,normalization='none')
        torch.testing.assert_close(reordered['frozen_velocity'],info['frozen_velocity'][perm],rtol=1e-4,atol=1e-6)

    def test_self_normalization_rejects_inconsistent_masks(self):
        x=torch.randn(4,3)
        with self.assertRaisesRegex(ValueError,'positive diagonal'):
            riesz_loss(x,x,affinity='row-column',include_self_negative=True,exclude_target_diagonal=True)
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            riesz_loss(x,x,affinity='row-column',distance_smoothing=-1.)

    def test_reconstruction_parent_restores_generator_and_verified_encoder(self):
        model=ImageVelocity(8)
        encoder=copy.deepcopy(model).requires_grad_(False)
        parent_model=ImageVelocity(8)
        parent=dict(config=dict(objective='reconstruction',fork=None),model=parent_model.state_dict(),
                    ema=parent_model.state_dict(),update=20000,lineage_has_flow_matching=False)
        restore_image_checkpoint(model,parent,encoder)
        for key,value in parent_model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
            torch.testing.assert_close(encoder.state_dict()[key],value,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as folder:
            origin=Path(folder)/'parent.pt';torch.save(parent,origin)
            checkpoint=dict(config=dict(objective='path-endpoint',fork=str(origin)),encoder=encoder.state_dict())
            proof=verify_reconstruction_encoder(checkpoint)
            self.assertEqual(proof['priming_updates'],20000)
            self.assertEqual(proof['origin'],str(origin))
            with torch.no_grad():next(encoder.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_reconstruction_encoder(checkpoint)
        parent['lineage_has_flow_matching']=True
        with self.assertRaisesRegex(ValueError,'flow-matching'):
            restore_image_checkpoint(model,parent,encoder)

    def test_vgg_endpoint_field_keeps_input_gradient_and_detaches_targets(self):
        class FrozenFeatures(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale=torch.nn.Parameter(torch.ones(1),requires_grad=False)
                self.seen=[]
            def multiscale_features(self,images,pool_size=8):
                self.seen.append(images.detach().clone())
                x=images.reshape(-1,3,32,32)*self.scale
                return dict(spatial=torch.nn.functional.adaptive_avg_pool2d(x,2).flatten(1),
                    global_mean=x.mean((-2,-1)))
        encoder=FrozenFeatures()
        prediction=(2*torch.randn(4,3072)).requires_grad_()
        target=torch.randn(6,3072,requires_grad=True)
        loss=vgg_endpoint_riesz_loss(prediction,target,encoder,affinity='row-column',normalization='temperature')
        loss.backward()
        self.assertTrue(prediction.grad.isfinite().all() and prediction.grad.abs().sum()>0)
        self.assertIsNone(target.grad)
        self.assertIsNone(encoder.scale.grad)
        torch.testing.assert_close(encoder.seen[0],(prediction.detach()+1)/2)
        self.assertTrue((encoder.seen[0]>1).any() and (encoder.seen[0]<0).any())
        encoder.scale.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'frozen'):
            vgg_endpoint_riesz_loss(prediction,target,encoder)

    def test_rgb_rollout_and_spatial_loss_backpropagate_all_channels(self):
        torch.manual_seed(91)
        model=DenoiserVelocity(ImageVelocity(8,deep=True,channels=3,image_size=32))
        encoder=copy.deepcopy(model).requires_grad_(False)
        noise=torch.randn(3,3072,requires_grad=True)
        target=torch.randn(5,3072,requires_grad=True)
        prediction=sample(model,noise,3,'euler',checkpoint_steps=True)
        self.assertEqual(prediction.shape,(3,3072))
        loss=image_riesz_loss(prediction,target,encoder=encoder,statistics_weight=1.,
            spatial_feature_weight=1.,spatial_feature_locations=3,image_shape=(3,32,32),
            taus=[.2,.05],normalization='temperature',affinity='row-column')
        loss.backward()
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertTrue((model.base.output.weight.grad.flatten(1).norm(dim=1)>0).all())
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_rgb_pixel_moments_preserve_color_and_spatial_variance(self):
        x=torch.tensor([1.,2.,3.])[None,:,None,None].expand(2,3,32,32).clone()
        x[1,0]=torch.arange(32)[None,:].remainder(2).expand(32,32)
        features=pixel_moment_features(x.flatten(1),(3,32,32))
        for means,std in zip(features[::2],features[1::2]):
            means,std=means.reshape(2,3,-1),std.reshape(2,3,-1)
            torch.testing.assert_close(means[0],torch.tensor([1.,2.,3.])[:,None].expand_as(means[0]))
            torch.testing.assert_close(std[0],torch.full_like(std[0],1e-4))
            torch.testing.assert_close(means[1,0],torch.full_like(means[1,0],.5))
            torch.testing.assert_close(std[1,0],torch.full_like(std[1,0],.5))

    def test_cifar_fid_preserves_rgb_and_uses_full_reference(self):
        from unittest.mock import patch
        from argparse import Namespace
        from scripts.lsf.run_small_image_flow import rgb32,ImageEvaluation
        real=torch.zeros(17,3,32,32,dtype=torch.uint8);real[:,0]=255
        pixels=rgb32(real.flatten(1),(3,32,32))
        self.assertEqual(pixels.shape,(17,3,32,32))
        self.assertTrue(torch.equal(pixels[:,0],torch.ones_like(pixels[:,0])))
        self.assertEqual(float(pixels[:,1:].sum()),0.)
        with patch('scripts.lsf.run_small_image_flow.CIFAR10FID') as fid:
            ImageEvaluation(real,real,Namespace(dataset='cifar10',eval_samples=5))
            self.assertEqual(len(fid.call_args.args[0]),17)
            ImageEvaluation(torch.zeros(17,784,dtype=torch.uint8),real,Namespace(dataset='mnist',eval_samples=5))
            self.assertEqual(len(fid.call_args.args[0]),5)

    def test_spatial_riesz_matches_single_location_force_and_gradient(self):
        torch.manual_seed(80)
        a=torch.randn(7,5,requires_grad=True)
        b=torch.randn(11,5,requires_grad=True)
        for normalization in ['none','temperature','global']:
            kwargs=dict(taus=[.2,.05],normalization=normalization,affinity='row-column')
            original=riesz_loss(a,b,**kwargs)[0]
            batched=spatial_riesz_loss(a[None],b[None],**kwargs)
            torch.testing.assert_close(batched,original,atol=2e-5,rtol=2e-5)
            ga=torch.autograd.grad(original,a)[0]
            gb=torch.autograd.grad(batched,a)[0]
            torch.testing.assert_close(gb,ga,atol=2e-5,rtol=2e-5)
        self.assertIsNone(b.grad)

    def test_spatial_feature_loss_reaches_images_without_updating_encoder(self):
        torch.manual_seed(81)
        encoder=DenoiserVelocity(ImageVelocity(8,deep=True)).requires_grad_(False)
        a=torch.randn(4,784,requires_grad=True)
        b=torch.randn(6,784,requires_grad=True)
        loss=image_riesz_loss(a,b,encoder=encoder,spatial_feature_weight=1.,spatial_feature_locations=3,
            taus=[.2,.05],normalization='temperature',affinity='row-column')
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.norm()),0.)
        self.assertIsNone(b.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_spatial_riesz_near_collisions(self):
        torch.manual_seed(83)
        a=(1000+1e-4*torch.randn(7,5)).requires_grad_()
        b=1000+1e-4*torch.randn(11,5)
        kwargs=dict(taus=[.2,.05],normalization='temperature',affinity='row-column')
        original=riesz_loss(a,b,**kwargs)[0]
        batched=spatial_riesz_loss(a[None],b[None],**kwargs)
        torch.testing.assert_close(batched,original)
        torch.testing.assert_close(torch.autograd.grad(batched,a)[0],torch.autograd.grad(original,a)[0])

    def test_spatial_riesz_particle_permutation(self):
        torch.manual_seed(82)
        a=torch.randn(3,7,5,requires_grad=True)
        b=torch.randn(3,11,5,requires_grad=True)
        kwargs=dict(taus=[.2,.05],normalization='temperature')
        loss=spatial_riesz_loss(a,b,**kwargs)
        permuted=spatial_riesz_loss(a[:,[4,3,0,1,5,6,2]],b.flip(1),**kwargs)
        torch.testing.assert_close(loss,permuted,atol=2e-5,rtol=2e-5)
        loss.backward()
        self.assertIsNone(b.grad)
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_reference_time_weight_scales_denoiser_gradient(self):
        torch.manual_seed(84)
        model=DenoiserVelocity(ToyVelocity(16))
        clean,noise,target,target_noise=[torch.randn(8,2) for _ in range(4)]
        t=.99
        kwargs=dict(affinity='row-column',taus=[.2,.05],normalization='temperature')
        original=marginal_loss(model,clean,noise,target,target_noise,t,.1,**kwargs)
        weighted=marginal_loss(model,clean,noise,target,target_noise,t,.1,time_weight_power=1.,**kwargs)
        torch.testing.assert_close(weighted,(1-t)*original)
        ga=torch.autograd.grad(original,tuple(model.parameters()))
        gb=torch.autograd.grad(weighted,tuple(model.parameters()))
        for a,b in zip(ga,gb):torch.testing.assert_close(b,(1-t)*a)

    def test_restore_generator_keeps_fresh_reconstruction_encoder(self):
        torch.manual_seed(85)
        parent_model=ToyVelocity(16)
        old_encoder=ToyVelocity(16)
        checkpoint=dict(config=dict(objective='path-endpoint',fork=None),ema=parent_model.state_dict(),encoder=old_encoder.state_dict())
        model=ToyVelocity(16)
        fresh_encoder=ToyVelocity(16).requires_grad_(False)
        before=copy.deepcopy(fresh_encoder.state_dict())
        restore_image_checkpoint(model,checkpoint,fresh_encoder,keep_encoder=True)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,checkpoint['ema'][k]))
        for k,v in fresh_encoder.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(any(not torch.equal(v,checkpoint['encoder'][k]) for k,v in fresh_encoder.state_dict().items()))
        restore_image_checkpoint(model,checkpoint,fresh_encoder,keep_encoder=False)
        for k,v in fresh_encoder.state_dict().items():self.assertTrue(torch.equal(v,checkpoint['encoder'][k]))

    def test_restore_rejects_fm_before_changing_weights(self):
        model=ToyVelocity(16)
        before=copy.deepcopy(model.state_dict())
        fm=dict(config=dict(objective='fm',fork=None),ema=ToyVelocity(16).state_dict())
        with self.assertRaisesRegex(ValueError,'flow-matching'):
            restore_image_checkpoint(model,fm)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_query_tail_weighting_preserves_force_direction_and_bounds(self):
        torch.manual_seed(86)
        target=.1*torch.randn(16,2)
        query=torch.cat([target[:7]+.001,torch.tensor([[3.,3.]])]).requires_grad_()
        kwargs=dict(affinity='row-column',taus=[.2],normalization='none')
        base,binfo=riesz_loss(query,target,**kwargs)
        weighted,winfo=riesz_loss(query,target,query_tail_scale=.05,query_tail_max_weight=4.,**kwargs)
        weights=winfo['query_tail_weights']
        self.assertTrue(((weights>=1)&(weights<=4)).all())
        self.assertGreater(float(weights[-1]),float(weights[:-1].mean()))
        torch.testing.assert_close(winfo['frozen_velocity'],binfo['frozen_velocity']*weights)
        ga=torch.autograd.grad(base,query)[0];gb=torch.autograd.grad(weighted,query)[0]
        torch.testing.assert_close(gb,ga*weights,atol=1e-6,rtol=1e-5)

    def test_query_tail_weighting_keeps_matched_support_equilibrium(self):
        torch.manual_seed(87)
        query=torch.randn(8,2,requires_grad=True)
        target=torch.randn(16,2,requires_grad=True)
        loss,_=riesz_loss(query,target,negative=target,affinity='row-column',taus=[.2,.05],
            normalization='temperature',query_tail_scale=.05)
        loss.backward()
        torch.testing.assert_close(query.grad,torch.zeros_like(query),atol=1e-7,rtol=0)
        self.assertIsNone(target.grad)

    def test_sampler_and_full_rollout_gradient(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rate = torch.nn.Parameter(torch.tensor(.3))
            def forward(self, x, t):
                return self.rate*x
        model = Linear()
        x = torch.ones(3,2)
        result = sample(model,x,20,'heun')
        torch.testing.assert_close(result,torch.exp(model.rate)*x,rtol=2e-5,atol=2e-5)
        result.sum().backward()
        self.assertGreater(float(model.rate.grad),0)
        expected_grad=model.rate.grad.clone()
        model.zero_grad()
        checked=sample(model,x,20,'heun',checkpoint_steps=True)
        checked.sum().backward()
        torch.testing.assert_close(checked,result)
        torch.testing.assert_close(model.rate.grad,expected_grad)

    def test_denoiser_euler_endpoint_cancels_residual_noise(self):
        class ConstantClean(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clean=torch.nn.Parameter(torch.tensor([.3,-.7]))
            def forward(self,x,t): return self.clean.expand_as(x)
        model=DenoiserVelocity(ConstantClean())
        noise=torch.randn(8,2)
        generated=sample(model,noise,10,'euler')
        torch.testing.assert_close(generated,model.base.clean.expand_as(noise),atol=2e-6,rtol=0)
        generated.sum().backward()
        torch.testing.assert_close(model.base.clean.grad,torch.full((2,),8.))

    def test_marginal_target_is_unpaired_and_detached(self):
        torch.manual_seed(1)
        model = ToyVelocity(16)
        clean, noise = torch.randn(8,2), torch.randn(8,2)
        target = torch.randn(12,2,requires_grad=True)
        target_noise = torch.randn_like(target)
        value = marginal_loss(model,clean,noise,target,target_noise,.4,.1)
        expected = riesz_loss(.6*noise+.4*clean+.1*model(.6*noise+.4*clean,.4),
                             .5*target_noise+.5*target)[0]/.1
        torch.testing.assert_close(value,expected)
        permuted = marginal_loss(model,clean,noise,target.flip(0),target_noise.flip(0),.4,.1)
        torch.testing.assert_close(value,permuted)
        value.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))

    def test_native_pixel_scale_preserves_local_loss_and_gradient(self):
        class Shift(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.velocity=torch.nn.Parameter(torch.randn(16)*.1)
            def forward(self,x,t):return self.velocity.expand_as(x)
        torch.manual_seed(91)
        model=Shift()
        clean,noise,target,target_noise=[torch.randn(8,16) for _ in range(4)]
        options=dict(affinity='row-column',taus=[.2,.05],normalization='temperature')
        plain=marginal_loss(model,clean,noise,target,target_noise,.4,.1,**options)
        native=marginal_loss(model,clean,noise,target,target_noise,.4,.1,
                            pixel_scales=[4],image_shape=(1,4,4),**options)
        torch.testing.assert_close(plain,native)
        torch.testing.assert_close(torch.autograd.grad(plain,model.velocity)[0],
                                   torch.autograd.grad(native,model.velocity)[0])

    def test_multiscale_local_pixels_learn_translation_with_one_model_call(self):
        class Translation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.velocity=torch.nn.Parameter(torch.tensor(0.))
                self.calls=0
            def forward(self,x,t):
                self.calls+=1
                return self.velocity.expand_as(x)
        model=Translation()
        noise=torch.tensor([-10.,-5.,5.,10.])[:,None].expand(-1,16)
        clean=(noise+.7).clone().requires_grad_()
        loss=coupled_reference_loss(model,clean,noise,.4,.1,pixel_scales=[4,2,1],
                                   image_shape=(1,4,4),affinity='row-column',
                                   taus=[.02],normalization='temperature')
        loss.backward()
        self.assertEqual(model.calls,1)
        self.assertTrue(torch.isfinite(model.velocity.grad))
        self.assertLess(float(model.velocity.grad),0.)

    def test_image_time_conditioning_and_shape(self):
        model = ImageVelocity(8)
        x = torch.randn(2,784)
        a,b = model(x,.1),model(x,.9)
        self.assertEqual(a.shape,x.shape)
        self.assertFalse(torch.equal(a,b))
        a.mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_coupled_reference_field_learns_known_translation(self):
        class Translation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.velocity=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x,t): return self.velocity.expand_as(x)
        model=Translation()
        noise=torch.tensor([[-10.],[-5.],[5.],[10.]])
        clean=noise+.7
        before=(model.velocity.detach()-.7).abs()
        loss=coupled_reference_loss(model,clean,noise,.4,.1,
            affinity='row-column',taus=[.02],normalization='temperature')
        loss.backward()
        self.assertTrue(torch.isfinite(model.velocity.grad))
        self.assertLess(float(model.velocity.grad),0.)
        with torch.no_grad(): model.velocity.sub_(.01*model.velocity.grad)
        self.assertLess(float((model.velocity-.7).abs().detach()),float(before))

    def test_local_hybrid_one_call_and_known_clean_fixed_point(self):
        class ConstantClean(torch.nn.Module):
            def __init__(self,value):
                super().__init__();self.value=torch.nn.Parameter(value.clone());self.calls=0
            def forward(self,x,t):
                self.calls+=1
                return self.value.expand_as(x)
        truth=torch.tensor([.3,-.2])
        clean=truth.expand(8,2).clone().requires_grad_()
        noise=torch.randn_like(clean,requires_grad=True)
        for offset in [torch.zeros(2),torch.tensor([.4,-.3])]:
            base=ConstantClean(truth+offset);model=DenoiserVelocity(base)
            loss,parts=local_reference_hybrid_loss(model,clean,noise,clean,noise,.5,.5,
                affinity='row-column',taus=[.2,.05,.02],normalization='temperature',exclude_target_diagonal=True)
            loss.backward()
            self.assertEqual(base.calls,1)
            self.assertIsNone(clean.grad);self.assertIsNone(noise.grad)
            self.assertTrue(torch.isfinite(base.value.grad).all())
            if offset.abs().sum()==0:
                torch.testing.assert_close(loss,torch.tensor(0.))
                torch.testing.assert_close(base.value.grad,torch.zeros(2))
            else:
                self.assertGreater(float((base.value.grad*offset).sum()),0.)
                self.assertGreater(float(parts['weighted_reference_clean_mse']),0.)
                self.assertGreater(float(parts['weighted_local_riesz']),0.)

    def test_local_hybrid_cannot_silently_disable_either_objective(self):
        model=DenoiserVelocity(ToyVelocity(16))
        clean,noise=torch.randn(8,2),torch.randn(8,2)
        for options in [dict(reference_weight=0.),dict(riesz_weight=0.),dict(reference_weight=float('nan'))]:
            with self.assertRaises(ValueError):
                local_reference_hybrid_loss(model,clean,noise,clean,noise,.4,.1,**options)

    def test_independent_negative_control_cancels_identical_populations(self):
        x=torch.randn(8,2,requires_grad=True)
        target=torch.randn(16,2)
        value,_=riesz_loss(x,target,negative=target)
        value.backward()
        torch.testing.assert_close(x.grad,torch.zeros_like(x))

    def test_no_flow_matching_initialization(self):
        for parent in [dict(config=dict(objective='fm')),dict(config=dict(objective='endpoint'),lineage_has_flow_matching=True)]:
            with self.assertRaises(ValueError): validate_riesz_parent(parent)
        validate_riesz_parent(dict(config=dict(objective='marginal',fork=None)))
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'parent.pt'
            torch.save(dict(config=dict(objective='fm',fork=None)),parent)
            with self.assertRaises(ValueError):
                validate_riesz_parent(dict(config=dict(objective='endpoint',fork=parent),lineage_has_flow_matching=False))

    def test_frozen_encoder_matches_reconstruction_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=dict(weight=torch.tensor([1.,2.]))
            torch.save(dict(config=dict(objective='reconstruction',fork=None),model=original,update=3),Path(tmp)/'reconstruction.pt')
            checkpoint=dict(config=dict(objective='path-endpoint',prime_steps=3,output=tmp),encoder=original)
            self.assertTrue(verify_reconstruction_encoder(checkpoint)['unchanged'])
            checkpoint['encoder']=dict(weight=torch.tensor([1.,3.]))
            with self.assertRaises(ValueError): verify_reconstruction_encoder(checkpoint)

    def test_row_column_riesz_symmetry_detachment_and_collisions(self):
        torch.manual_seed(19)
        x=torch.randn(8,3,requires_grad=True)
        y=torch.randn(12,3,requires_grad=True)
        n=torch.randn(12,3,requires_grad=True)
        options=dict(affinity='row-column',taus=[.2,.05],normalization='temperature')
        loss,_=riesz_loss(x,y,negative=n,**options)
        permuted,_=riesz_loss(x.flip(0),y.flip(0),negative=n.flip(0),**options)
        torch.testing.assert_close(loss,permuted)
        permuted_gradient=torch.autograd.grad(permuted,x)[0]
        loss.backward()
        torch.testing.assert_close(x.grad,permuted_gradient)
        self.assertTrue(x.grad.isfinite().all() and x.grad.abs().sum()>0)
        self.assertIsNone(y.grad)
        self.assertIsNone(n.grad)
        x.grad=None
        cancelled,_=riesz_loss(x,y,negative=y,**options)
        cancelled.backward()
        torch.testing.assert_close(x.grad,torch.zeros_like(x),atol=1e-6,rtol=0)
        duplicate=torch.zeros(4,3,requires_grad=True)
        collision,_=riesz_loss(duplicate,torch.zeros(6,3),**options)
        collision.backward()
        self.assertTrue(duplicate.grad.isfinite().all())

    def test_paired_leave_one_out_field_cancels_stationary_population(self):
        torch.manual_seed(53)
        x=torch.randn(8,4,requires_grad=True)
        target=x.detach().clone()
        options=dict(affinity='row-column',taus=[.2,.05,.02],normalization='temperature',exclude_target_diagonal=True)
        value,_=riesz_loss(x,target,**options)
        value.backward()
        torch.testing.assert_close(x.grad,torch.zeros_like(x),atol=1e-7,rtol=0)
        shifted=target+.1
        first,_=riesz_loss(x,shifted,**options)
        permuted,_=riesz_loss(x.flip(0),shifted.flip(0),**options)
        torch.testing.assert_close(torch.autograd.grad(first,x)[0],torch.autograd.grad(permuted,x)[0])
        with self.assertRaises(ValueError): riesz_loss(x,shifted[:-1],**options)

    def test_potential_velocity_has_symmetric_jacobian_and_trains(self):
        model=PotentialVelocity(16,spatial_fourier=True)
        x=torch.randn(1,2,requires_grad=True)
        jac=torch.autograd.functional.jacobian(lambda v:model(v,.3),x).reshape(2,2)
        torch.testing.assert_close(jac,jac.T)
        sample(model,x,2).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        with torch.no_grad():
            self.assertFalse(sample(model,x.detach(),2).requires_grad)

    def test_reconstruction_features_preserve_generator_gradient(self):
        encoder=ImageVelocity(8,spatial_fourier=True).requires_grad_(False)
        x=torch.randn(3,784,requires_grad=True)
        target=torch.randn(4,784,requires_grad=True)
        image_riesz_loss(x,target,encoder=encoder,statistics_weight=1.,gradient_weight=1.,normalization='temperature').backward()
        self.assertTrue(x.grad.isfinite().all() and x.grad.abs().sum()>0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_frozen_reconstruction_decoder_preserves_transport_gradient(self):
        encoder=ImageVelocity(8).requires_grad_(False).eval()
        x=torch.randn(3,784,requires_grad=True)
        target=torch.randn(4,784)
        decoded=encoder(x,1.)
        image_riesz_loss(decoded,target,encoder=encoder,normalization='temperature',affinity='row-column').backward()
        self.assertTrue(x.grad.isfinite().all() and x.grad.abs().sum()>0)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_local_variance_field_contracts_background_noise(self):
        torch.manual_seed(31)
        x=(.1*torch.randn(8,784)).requires_grad_()
        target=torch.zeros(12,784)
        a,b=pixel_moment_features(x),pixel_moment_features(target)
        loss=sum(riesz_loss(a[i],b[i],affinity='row-column',normalization='temperature')[0] for i in [1,3])
        loss.backward()
        self.assertTrue(x.grad.isfinite().all())
        after=pixel_moment_features(x.detach()-.01*x.grad)
        self.assertLess(float(after[1].mean()+after[3].mean()),float(a[1].detach().mean()+a[3].detach().mean()))
        zero=torch.zeros(8,784,requires_grad=True)
        image_riesz_loss(zero,target,statistics_weight=1.,normalization='temperature').backward()
        self.assertTrue(zero.grad.isfinite().all())

    def test_gradient_features_restore_missing_structure(self):
        stripe=torch.zeros(1,1,28,28)
        stripe[:,:,:,::2]=1.
        target=stripe.expand(6,-1,-1,-1).flatten(1)
        x=(.2*stripe.expand(4,-1,-1,-1)).flatten(1).clone().requires_grad_()
        features=pixel_gradient_features(x)
        shifted=pixel_gradient_features(x+.3)
        for a,b in zip(features,shifted): torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        loss=sum(riesz_loss(a,b,affinity='row-column',normalization='temperature')[0]
            for a,b in zip(features,pixel_gradient_features(target)))
        loss.backward()
        self.assertTrue(x.grad.isfinite().all())
        template=target[0]-.5
        before=(x.detach()*template).mean()
        after=((x.detach()-.01*x.grad)*template).mean()
        self.assertGreater(float(after),float(before))

    def test_deep_image_model_shape_and_rollout_gradient(self):
        model=ImageVelocity(8,spatial_fourier=True,deep=True)
        x=torch.randn(2,784)
        out=sample(model,x,2,'euler',checkpoint_steps=True)
        self.assertEqual(out.shape,x.shape)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        self.assertEqual(len(model.features(x)),3)

    def test_reference_control_variate_cancels_stationary_sampling_noise(self):
        model=ToyVelocity(16)
        torch.nn.init.zeros_(model.net[-1].weight)
        torch.nn.init.zeros_(model.net[-1].bias)
        x,y,z=torch.randn(8,2),torch.randn(12,2),torch.randn(12,2)
        loss=reference_centered_loss(model,x,x,y,y,z,z,.3,.1)
        torch.testing.assert_close(loss,torch.tensor(0.),atol=1e-9,rtol=0)
        loss.backward()
        self.assertTrue(all(p.grad.abs().max()<1e-6 for p in model.parameters()))

    def test_reference_control_variate_gradient_and_support_permutation(self):
        torch.manual_seed(8)
        model=ToyVelocity(16)
        x,z=torch.randn(8,2),torch.randn(8,2)
        y,ny=torch.randn(12,2,requires_grad=True),torch.randn(12,2)
        neg,nn=torch.randn(12,2),torch.randn(12,2)
        a=reference_centered_loss(model,x,z,y,ny,neg,nn,.4,.05)
        b=reference_centered_loss(model,x,z,y.flip(0),ny.flip(0),neg.flip(0),nn.flip(0),.4,.05)
        torch.testing.assert_close(a,b)
        a.backward()
        self.assertIsNone(y.grad)
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
