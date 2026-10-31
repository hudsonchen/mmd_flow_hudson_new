#!/usr/bin/env python3
"""CIFAR-10 FID using the TensorFlow-compatible 2048-dimensional Inception.

Feature extraction, distributed sufficient statistics, and Frechet distance.
The velocity trainer owns sampling, reference caching, and evaluation output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

ASSETS = Path(__file__).resolve().parent / "assets"
WEIGHTS = ASSETS / "pt_inception-2015-12-05-6726825d.pth"


def load_inception(device):
    from _fid_inception import InceptionV3
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing FID weights: {WEIGHTS}")
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    expected = (ASSETS / "inception.sha256").read_text().split()[0]
    if digest != expected:
        raise RuntimeError("FID Inception weight checksum mismatch")
    return InceptionV3([3]).eval().to(device)


def feature_statistics(batches, inception, device):
    """Accumulate float64 mean/covariance without storing all image features."""
    count, mean, scatter = 0, None, None
    with torch.inference_mode():
        for images in batches:
            if images.ndim == 2 and images.shape[1] == 3072:
                images = images.reshape(-1, 3, 32, 32)
            if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
                raise ValueError("Expected CIFAR-10 NCHW images or flattened RGB rows")
            byte_pixels = images.dtype == torch.uint8
            images = images.to(device=device, dtype=torch.float32)
            if byte_pixels:
                images = images / 255
            if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
                raise ValueError("FID pixels must be finite and in [0, 1]")
            features = inception(images)[0].flatten(1).cpu().numpy().astype(np.float64)
            n = len(features)
            if not n:
                continue
            batch_mean = features.mean(axis=0)
            centered = features - batch_mean
            batch_scatter = centered.T @ centered
            if count == 0:
                mean, scatter = batch_mean, batch_scatter
            else:
                delta = batch_mean - mean
                scatter += batch_scatter + np.outer(delta, delta) * (count * n / (count + n))
                mean += delta * n / (count + n)
            count += n
    if count < 2:
        raise ValueError("FID requires at least two images per distribution")
    return mean, scatter / (count - 1), count


def frechet_distance(mean1, cov1, mean2, cov2):
    """Gaussian Wasserstein distance; symmetric PSD form handles singular covariances."""
    values, vectors = linalg.eigh(cov1)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    middle = root @ cov2 @ root
    eigenvalues = linalg.eigvalsh((middle + middle.T) / 2)
    value = float(np.sum((mean1 - mean2) ** 2) + np.trace(cov1) + np.trace(cov2)
                  - 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum())
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite FID")
    return max(value, 0.0)


class FeatureMoments:
    """Mergeable float64 sufficient statistics for distributed FID."""
    def __init__(self, dimension=2048):
        self.count = 0
        self.sum = np.zeros(dimension, dtype=np.float64)
        self.outer = np.zeros((dimension, dimension), dtype=np.float64)

    def add(self, features):
        values = np.asarray(features, dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite Inception features')
        self.count += len(values)
        self.sum += values.sum(0)
        self.outer += values.T @ values

    def statistics(self):
        if self.count < 2:
            raise ValueError('At least two feature vectors are required')
        mean = self.sum / self.count
        cov = (self.outer - self.count * np.outer(mean, mean)) / (self.count - 1)
        return mean, (cov + cov.T) / 2, self.count
