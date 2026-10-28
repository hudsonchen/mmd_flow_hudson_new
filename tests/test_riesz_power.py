"""Power-kernel force checks independent of the detached surrogate gradient."""
import os
os.environ['DRIFT_COMPILE']='0'
import unittest
import torch
from small_flow import riesz_loss


class PowerRieszTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_field_matches_frozen_affinity_potential_derivative(self):
        # Differentiate the scalar potential independently, with affinity weights
        # frozen as required by the surrogate. This catches missing p factors,
        # wrong exponents, signs, and scale factors, including smoothed kernels.
        torch.manual_seed(710)
        original=torch.randn(5,4)
        target=torch.randn(7,4)
        negative=torch.randn(6,4)
        support=torch.cat([negative,target])
        dist=torch.cdist(original,support,compute_mode='donot_use_mm_for_euclid_dist')
        scale=dist.mean().clamp_min(1e-3)
        for powers in [(1.,),(.8,),(1.5,),(.8,1.,1.5)]:
            for smoothing in [0.,.05]:
                _,info=riesz_loss(original,target,negative=negative,taus=[.2,.7],
                    normalization='none',affinity='row-column',distance_powers=powers,
                    distance_smoothing=smoothing)
                q=original.double().clone().requires_grad_()
                radii=((q[:,None]-support.double()[None]).square().sum(-1)+(smoothing*scale.double()).square()).sqrt()
                energy=q.sum()*0
                for power in powers:
                    frozen=((dist.square()+(smoothing*scale).square()).pow(power/2)-(smoothing*scale).pow(power))/scale.pow(power)
                    potential=(radii.pow(power)-(smoothing*scale.double()).pow(power))/scale.double().pow(power)
                    for tau in [.2,.7]:
                        logits=-frozen/tau
                        weights=(logits.softmax(-1)*logits.softmax(-2)).sqrt()
                        neg,pos=weights[:,:len(negative)],weights[:,len(negative):]
                        pos_coeff=pos*neg.sum(-1,keepdim=True)
                        neg_coeff=neg*pos.sum(-1,keepdim=True)
                        energy=energy+scale.double()/tau*((pos_coeff.double()*potential[:,len(negative):]).sum()-(neg_coeff.double()*potential[:,:len(negative)]).sum())
                expected=-torch.autograd.grad(energy,q)[0]
                torch.testing.assert_close(info['frozen_velocity'].double(),expected,rtol=3e-5,atol=2e-6)

    def test_mixture_adds_fields_and_gradients_without_rms_normalization(self):
        torch.manual_seed(711)
        query=torch.randn(8,6,requires_grad=True)
        target=(query.detach()+.1*torch.randn_like(query)).requires_grad_()
        options=dict(taus=[.2,.05,.02],affinity='row-column',normalization='none',include_self_negative=True)
        value,info=riesz_loss(query,target,distance_powers=[.8,1.,1.5],**options)
        singles=[riesz_loss(query,target,distance_powers=[p],**options) for p in [.8,1.,1.5]]
        torch.testing.assert_close(info['frozen_velocity'],sum(x[1]['frozen_velocity'] for x in singles),rtol=2e-6,atol=1e-6)
        actual=torch.autograd.grad(value,query,retain_graph=True)[0]
        expected=sum(torch.autograd.grad(x[0],query,retain_graph=True)[0] for x in singles)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-6)
        self.assertIsNone(target.grad)
        self.assertFalse(info['frozen_velocity'].requires_grad)

    def test_collision_stability_and_exact_empirical_cancellation(self):
        target=torch.tensor([[0.,0.],[0.,0.],[1.,2.],[-1.,2.]])
        for power in [.8,1.,1.5]:
            query=target.clone().requires_grad_()
            value,info=riesz_loss(query,target,affinity='row-column',taus=[.2,.05,.02],
                normalization='none',include_self_negative=True,distance_powers=[power])
            value.backward()
            torch.testing.assert_close(info['frozen_velocity'],torch.zeros_like(query),rtol=0,atol=1e-7)
            self.assertTrue(torch.isfinite(query.grad).all())
        query=(target+1e-8).requires_grad_()
        value,info=riesz_loss(query,target,affinity='row-column',taus=[.2,.05,.02],
            normalization='none',include_self_negative=True,distance_powers=[.8,1.,1.5])
        value.backward()
        self.assertTrue(torch.isfinite(info['frozen_velocity']).all())
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_invalid_powers_and_unsupported_affinity(self):
        x=torch.randn(4,3)
        for powers in [[],[0.],[-1.],[float('nan')],[float('inf')],[2.1],[1.,1.]]:
            with self.assertRaisesRegex(ValueError,'Distance powers'):
                riesz_loss(x,x,affinity='row-column',distance_powers=powers)
        with self.assertRaisesRegex(ValueError,'row-column'):
            riesz_loss(x,x,affinity='row',distance_powers=[.8,1.,1.5])


if __name__=='__main__':
    unittest.main()
