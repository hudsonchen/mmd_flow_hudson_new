"""Compare against the frozen pre-optimization implementation, including gradients."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from riesz_loss import riesz_loss

spec = importlib.util.spec_from_file_location('frozen_riesz', ROOT/'tests/riesz_loss_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def compare(query, target, powers=(.8, 1., 1.5), taus=(.2, .05, .02)):
    results = []
    for implementation in [reference.riesz_loss, riesz_loss]:
        q = query.detach().clone().requires_grad_()
        loss, info = implementation(q, target, taus, powers)
        gradient, = torch.autograd.grad(loss, q)
        results.append((loss.detach(), info['frozen_velocity'], gradient))
    errors = {}
    for name, old, new in zip(['loss', 'field', 'gradient'], *results):
        assert bool(torch.isfinite(new).all()), name
        relative = float((new-old).double().norm()/old.double().norm().clamp_min(1e-20))
        absolute = float((new-old).abs().max())
        errors[name] = dict(relative_l2=relative, max_absolute=absolute)
        # Norm-based comparison avoids misleading relative errors at zero force
        # coordinates; absolute bounds cover exactly coincident populations.
        if relative > 2e-4 and absolute > 2e-6:
            raise AssertionError((name, errors[name]))
    return errors


class VectorizedRieszChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_paired_endpoint_fields_loss_and_gradient(self):
        torch.manual_seed(41)
        clean, noise, velocity = torch.randn(3, 17, 96)
        for t in [.001, .2, .77, .999]:
            h = min(.01, 1-t)
            source = (1-t)*noise+t*clean
            compare(source+h*velocity, (1-t-h)*noise+(t+h)*clean)

    def test_unequal_populations_and_kernel_options(self):
        torch.manual_seed(17)
        for n, m in [(2, 1), (7, 11), (33, 19), (65, 65)]:
            q, p = torch.randn(n, 48), torch.randn(m, 48)
            compare(q, p)
            compare(q, p, powers=(1., 2.), taus=(.3,))

    def test_duplicate_and_coincident_particles(self):
        torch.manual_seed(21)
        q = torch.randn(9, 64)
        q[1] = q[0]
        compare(q, q.clone())
        compare(torch.ones(9, 64), torch.ones(7, 64))
        p = q.clone()
        p[3] += .001
        compare(q, p)

    def test_common_translation(self):
        torch.manual_seed(28)
        q, p = torch.randn(2, 13, 64)+1000
        compare(q, p)


if __name__ == '__main__':
    unittest.main()
