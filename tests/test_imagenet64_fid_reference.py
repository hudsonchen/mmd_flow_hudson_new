"""Benchmark cache integrity and evaluation-only checkpoint continuation."""
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from imagenet64_fid_reference import load_benchmark_reference


ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT/'scripts/cs_cluster/run_imagenet64_jit.py').read_text())
scope = {}
resume_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'validate_resume_config')
exec(compile(ast.Module(body=[resume_function], type_ignores=[]), '<resume-validator>', 'exec'), scope)
validate_resume_config = scope['validate_resume_config']


class BenchmarkReferenceTests(unittest.TestCase):
    def test_saved_statistics_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'reference.npz'
            np.savez(path, mu=np.zeros(2048), sigma=np.eye(2048))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            reference, metadata = load_benchmark_reference(path, digest)
            self.assertEqual(reference[0].shape, (2048,))
            self.assertIsNone(reference[2])
            self.assertEqual(metadata['reference_images_processed'], 0)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_benchmark_reference(path, '0'*64)
            np.savez(path, mu=np.full(2048, np.nan), sigma=np.eye(2048))
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                load_benchmark_reference(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_reference_change_requires_opt_in_and_preserves_training(self):
        old = dict(output='old', solver='heun', required_gpu='H100', lr=.0002, fid='train')
        new = dict(old, output='new', fid='benchmark', fid_reference='/stats.npz',
                   fid_reference_sha256='a'*64)
        with self.assertRaises(ValueError):
            validate_resume_config(old, new)
        self.assertTrue(validate_resume_config(old, new, True))
        with self.assertRaisesRegex(ValueError, 'lr'):
            validate_resume_config(old, dict(new, lr=.001), True)
        self.assertFalse(validate_resume_config(old, dict(old, fid_reference=None,
                                                        fid_reference_sha256=None)))

    def test_external_reference_never_calls_reference_image_extraction(self):
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        ensure = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'ensure_fid')
        factory = ast.parse('def factory():\n inception = object()\n reference = (1, 2, None)\n').body[0]
        factory.body += [ensure, ast.parse('return ensure_fid').body[0]]
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        namespace = {}
        exec(compile(module, '<ensure-reference>', 'exec'), namespace)
        # No dataset, distributed group, or image extractor exists in this scope.
        namespace['factory']()()


if __name__ == '__main__':
    unittest.main()
