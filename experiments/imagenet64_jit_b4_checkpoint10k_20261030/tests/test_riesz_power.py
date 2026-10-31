"""Power-kernel force checks independent of the detached surrogate gradient."""
import unittest
import torch
from riesz_loss import riesz_loss


class PowerRieszTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_field_matches_frozen_affinity_potential_derivative(self):
        # Differentiate the scalar potential independently, with affinity weights
        # frozen as required by the surrogate. This catches missing p factors,
        # wrong exponents, signs, and scale factors, including fractional distance powers.
        torch.manual_seed(710)
        original=torch.randn(5,4)
        target=torch.randn(7,4)
        negative=original.detach()
        support=torch.cat([negative,target])
        dist=torch.cdist(original,support,compute_mode='donot_use_mm_for_euclid_dist')
        scale=dist.mean().clamp_min(1e-3)
        # The diagonal has zero displacement. Mask its singular radius before
        # differentiating the independent potential, giving exactly zero force.
        self_mask=torch.zeros_like(dist,dtype=torch.bool)
        self_mask[:,:len(original)]=torch.eye(len(original),dtype=torch.bool)
        for powers in [(1.,),(.8,),(1.5,),(.8,1.,1.5)]:
            _,info=riesz_loss(original,target,taus=[.2,.7],distance_powers=powers)
            q=original.double().clone().requires_grad_()
            squared=(q[:,None]-support.double()[None]).square().sum(-1)
            radii=squared.masked_fill(self_mask,1.).sqrt()
            energy=q.sum()*0
            for power in powers:
                frozen=dist.pow(power)/scale.pow(power)
                potential=radii.pow(power)/scale.double().pow(power)
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
        options=dict(taus=[.2,.05,.02])
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
            value,info=riesz_loss(query,target,taus=[.2,.05,.02],
                distance_powers=[power])
            value.backward()
            torch.testing.assert_close(info['frozen_velocity'],torch.zeros_like(query),rtol=0,atol=1e-7)
            self.assertTrue(torch.isfinite(query.grad).all())
        query=(target+1e-8).requires_grad_()
        value,info=riesz_loss(query,target,taus=[.2,.05,.02],
            distance_powers=[.8,1.,1.5])
        value.backward()
        self.assertTrue(torch.isfinite(info['frozen_velocity']).all())
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_invalid_powers_and_temperatures(self):
        x=torch.randn(4,3)
        for powers in [[],[0.],[-1.],[float('nan')],[float('inf')],[2.1],[1.,1.]]:
            with self.assertRaisesRegex(ValueError,'Distance powers'):
                riesz_loss(x,x,distance_powers=powers)
        for taus in [[],[0.],[-1.],[float('nan')],[float('inf')]]:
            with self.assertRaisesRegex(ValueError,'Temperatures'):
                riesz_loss(x,x,taus=taus)


if __name__=='__main__':
    unittest.main()
