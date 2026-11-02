import math
import sys
from pathlib import Path
import unittest
import torch
from transport import integrate,CleanPredictor
from kernel import marginal_loss,objective


class ScientificChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(1)

    def test_exact_constant_predictor_and_bound(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__();self.c=torch.nn.Parameter(torch.tensor(.3,dtype=torch.double))
            def forward(self,x,t):return self.c.expand_as(x)
        model=Constant();z=torch.randn(4,1,28,28,dtype=torch.double)
        for t in (.2,.8,1.):
            ratio=(1-t+.001)/1.001
            x=integrate(model,z,end=t,steps=5)
            torch.testing.assert_close(x,ratio*z+(1-ratio)*model.c)
            derivative=torch.autograd.grad(x.mean(),model.c)[0]
            torch.testing.assert_close(derivative,derivative.new_tensor(1-ratio))
        neural=CleanPredictor(4).double()
        x=integrate(neural,z,steps=3)
        self.assertTrue(bool((x.abs() <= .001/1.001*z.abs()+1/1.001+1e-12).all()))

    def test_second_order_and_checkpoint_gradient(self):
        class Linear(torch.nn.Module):
            def __init__(self):super().__init__();self.a=torch.nn.Parameter(torch.tensor(.2,dtype=torch.double))
            def forward(self,x,t):return self.a*x
        model=Linear();z=torch.randn(5,2,dtype=torch.double)
        end=.8;epsilon=.01;s=math.log((1+epsilon)/(1-end+epsilon))
        exact=z*torch.exp((model.a-1)*s)
        errors=[float((integrate(model,z,end,steps,epsilon)-exact).abs().max().detach()) for steps in (32,64)]
        self.assertLess(errors[1],errors[0]/3.8)
        regular=integrate(model,z,end,8,epsilon)
        checked=integrate(model,z,end,8,epsilon,True)
        torch.testing.assert_close(regular,checked)
        torch.testing.assert_close(torch.autograd.grad(regular.sum(),model.a)[0],torch.autograd.grad(checked.sum(),model.a)[0])

    def test_kernel_gradient_equivalence(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trajectory_mmd_20261102'))
        from core import pixel_marginal_mmd
        x=torch.randn(5,1,4,4,dtype=torch.double,requires_grad=True)
        y=torch.randn(7,1,4,4,dtype=torch.double)
        full=marginal_loss(x,y,.7,include_constant=True)
        prior=pixel_marginal_mmd(x,y,.7,scales=(1,))
        torch.testing.assert_close(full,prior)
        partial=marginal_loss(x,y,.7)
        torch.testing.assert_close(torch.autograd.grad(partial,x)[0],torch.autograd.grad(full,x)[0])
        self.assertTrue(torch.autograd.gradcheck(lambda a:marginal_loss(a,y,.7),(x,)))

    def test_independent_permutations_with_patches(self):
        x=torch.randn(5,1,28,28,dtype=torch.double,requires_grad=True)
        y=torch.randn(7,1,28,28,dtype=torch.double)
        loss=objective(x,y,.6,torch.Generator().manual_seed(2))
        changed=objective(x[torch.randperm(5)],y[torch.randperm(7)],.6,torch.Generator().manual_seed(2))
        torch.testing.assert_close(loss,changed)
        torch.testing.assert_close(torch.autograd.grad(loss,x)[0],torch.autograd.grad(changed,x)[0])


if __name__=='__main__':unittest.main()
