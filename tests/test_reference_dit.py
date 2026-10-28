"""Reconstruction-preserving DiT conversion and no-FM ancestry checks."""
import os
os.environ['DRIFT_COMPILE']='0'
import tempfile
from pathlib import Path
import unittest
import torch
from backbone import CIFAR10DiTParticleGenerator, CIFAR10UNetParticleGenerator
from reference_dit import ReferenceDiT,ReferenceDDPMUNet,image_base_from_config,initialize_from_reconstruction,verified_reconstruction_checkpoint
from small_flow import DenoiserVelocity,sample,validate_riesz_parent


class ReferenceDiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_diffusion_unet_adapter_preserves_backbone_and_time_conditioning(self):
        diffusion=CIFAR10UNetParticleGenerator(16,1,time_conditioned=True).eval()
        with torch.no_grad():
            diffusion.output[2].weight.normal_(0,.02)
        flow=ReferenceDDPMUNet(16,1).eval()
        flow.load_state_dict(diffusion.state_dict(),strict=True)
        x=torch.randn(2,3072)
        for t in [0.,.5,.999,torch.tensor([.2,.8])]:
            time=torch.as_tensor(t).expand(2)
            expected=diffusion(x,(1-time)*999).flatten(1)
            torch.testing.assert_close(flow(x,t),expected,rtol=0,atol=0)
        prediction=sample(DenoiserVelocity(flow),x,3,'euler')
        self.assertEqual(prediction.shape,x.shape)
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters()))
        self.assertGreater(float(flow.time_mlp[0].weight.grad.abs().sum()),0.)

    def test_diffusion_unet_factory_matches_full_diffusion_parameterization(self):
        with torch.device('meta'):
            flow=image_base_from_config(dict(architecture='ddpm-unet',dataset='cifar10'))
            diffusion=CIFAR10UNetParticleGenerator(128,4,time_conditioned=True)
        self.assertEqual(sum(p.numel() for p in flow.parameters()),61804419)
        self.assertEqual({k:tuple(v.shape) for k,v in flow.state_dict().items()},
                         {k:tuple(v.shape) for k,v in diffusion.state_dict().items()})
        with self.assertRaises(ValueError):
            image_base_from_config(dict(architecture='ddpm-unet',dataset='mnist'))

    def checkpoint(self, path):
        torch.manual_seed(19)
        source=CIFAR10DiTParticleGenerator(32,2,4,4).eval()
        with torch.no_grad():
            for block in source.blocks:block.modulation.normal_(0,.2)
            source.head.weight.normal_(0,.1)
        checkpoint=dict(config=dict(training_stage='reconstruction',dataset='cifar10',
            objective='softmax-riesz',training_rule='paired_denoising_mse_plus_lpips',
            init_checkpoint=None,fork_checkpoint=None,resume=None),
            ema_model=source.state_dict(),update_count=20000,iteration=80)
        torch.save(checkpoint,path)
        return source,checkpoint

    def test_conversion_preserves_primed_clean_prediction_at_every_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prime.pt';source,_=self.checkpoint(path)
            converted=ReferenceDiT(32,2,4,4,centered_output=True).eval()
            proof=initialize_from_reconstruction(converted,path)
            self.assertTrue(proof['transferred_tensors_exact'])
            self.assertTrue(proof['new_time_modulations_zero'])
            x=torch.randn(3,3072)
            for time in [0.,.2,.8,1.,torch.tensor([.1,.4,.9])]:
                torch.testing.assert_close(converted(x,time),2*source(x)-1,rtol=1e-6,atol=1e-6)
            prediction=sample(DenoiserVelocity(converted),x,3,'euler',checkpoint_steps=True)
            self.assertTrue((prediction.abs()<=1.000001).all())
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in converted.parameters()))
            self.assertGreater(float(converted.time_modulations[0][-1].weight.grad.abs().sum()),0.)

    def test_fm_or_unverified_ancestry_and_checksum_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prime.pt';_,checkpoint=self.checkpoint(path)
            _,digest=verified_reconstruction_checkpoint(path)
            child=dict(config=dict(objective='path-endpoint',dit_prime_checkpoint=str(path),dit_priming_sha256=digest))
            validate_riesz_parent(child)
            for field,value in [('objective','flow-matching'),('training_stage','distributional'),('init_checkpoint','fm.pt')]:
                altered={**checkpoint,'config':{**checkpoint['config'],field:value}}
                torch.save(altered,path)
                with self.assertRaises(ValueError):verified_reconstruction_checkpoint(path)
                with self.assertRaisesRegex(ValueError,'checksum'):validate_riesz_parent(child)
            torch.save({**checkpoint,'lineage_has_flow_matching':True},path)
            with self.assertRaisesRegex(ValueError,'never flow matching'):verified_reconstruction_checkpoint(path)


if __name__=='__main__':unittest.main()
