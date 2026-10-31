"""Validated, immutable ImageNet64 benchmark FID statistics."""
import hashlib
from pathlib import Path

import numpy as np


def load_benchmark_reference(path, expected_sha256):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected_sha256 or digest != expected_sha256:
        raise ValueError('ImageNet64 benchmark statistics checksum mismatch')
    with np.load(path, allow_pickle=False) as saved:
        mean = np.array(saved['mu'], dtype=np.float64)
        covariance = np.array(saved['sigma'], dtype=np.float64)
    if mean.shape != (2048,) or covariance.shape != (2048, 2048):
        raise ValueError('Expected 2048-dimensional ImageNet64 FID statistics')
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Nonfinite ImageNet64 benchmark statistics')
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10):
        raise ValueError('Asymmetric ImageNet64 benchmark covariance')
    metadata = dict(path=str(path), sha256=digest, dataset='ImageNet64',
        reference_split='benchmark', reference_count=None,
        reference_count_note='The published statistics archive does not encode a sample count.',
        source_archive='VIRTUAL_imagenet64_labeled.npz',
        reference_images_processed=0, statistics_recomputed=False,
        feature_extractor='pytorch-fid TensorFlow-compatible InceptionV3 pool3',
        generated_preprocessing='unquantized float [0,1]')
    return (mean, covariance, None), metadata
