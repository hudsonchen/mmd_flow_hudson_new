"""Checksum-verified CIFAR-10 data loading."""
from pathlib import Path
import hashlib
import pickle
import tarfile
import urllib.request
import numpy as np
import torch

CIFAR10_FILENAME = 'cifar-10-binary.tar.gz'



CIFAR10_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'



CIFAR10_ARCHIVES = {CIFAR10_FILENAME: CIFAR10_MD5,
                    'cifar-10-python.tar.gz': 'c58f30108f718f92721af3b95e74349a'}



CIFAR10_URL = 'https://cave.cs.toronto.edu/kriz/' + CIFAR10_FILENAME



CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']



def download_cifar10(root: Path) -> Path:
    archive = root / 'CIFAR10' / 'raw' / CIFAR10_FILENAME
    candidates = [archive, archive.with_name('cifar-10-python.tar.gz'),
                  root / 'cifar-10-python.tar.gz']
    for candidate in candidates:
        if candidate.exists() and hashlib.md5(candidate.read_bytes(), usedforsecurity=False).hexdigest() == CIFAR10_ARCHIVES[candidate.name]:
            return candidate
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.gz.part')
    print(f'Downloading CIFAR-10 to {archive}', flush=True)
    try:
        urllib.request.urlretrieve(CIFAR10_URL, temporary)
        if hashlib.md5(temporary.read_bytes(), usedforsecurity=False).hexdigest() != CIFAR10_MD5:
            raise RuntimeError('CIFAR-10 checksum mismatch')
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive



def decode_binary_batch(data: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    if not data or len(data) % 3073:
        raise RuntimeError('Invalid CIFAR-10 binary record length')
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3073)
    if np.any(records[:, 0] > 9):
        raise RuntimeError('Invalid CIFAR-10 class label')
    # Pixels are stored in channel-first R, G, B order.
    return torch.from_numpy(records[:, 1:].copy()), torch.from_numpy(records[:, 0].copy())



def load_cifar10_split(archive: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest() != CIFAR10_ARCHIVES.get(archive.name):
        raise RuntimeError('CIFAR-10 checksum mismatch')
    binary = archive.name == CIFAR10_FILENAME
    suffix = '.bin' if binary else ''
    prefix = 'cifar-10-batches-bin/' if binary else 'cifar-10-batches-py/'
    names = ([f'data_batch_{i}{suffix}' for i in range(1, 6)] if train else [f'test_batch{suffix}'])
    images, labels = [], []
    with tarfile.open(archive, 'r:gz') as stream:
        for name in names:
            member = stream.extractfile(prefix + name)
            if member is None:
                raise RuntimeError(f'Missing CIFAR-10 batch: {name}')
            if binary:
                x, y = decode_binary_batch(member.read())
            else:
                # Only unpickle after verifying the official archive's published checksum.
                record = pickle.load(member, encoding='bytes')
                data = record[b'data']
                labels_array = np.asarray(record[b'labels'], dtype=np.int64)
                if data.shape != (10000, 3072) or data.dtype != np.uint8:
                    raise RuntimeError(f'Invalid CIFAR-10 pixel data: {name}')
                if labels_array.shape != (10000,) or np.any((labels_array < 0) | (labels_array > 9)):
                    raise RuntimeError(f'Invalid CIFAR-10 labels: {name}')
                x, y = torch.from_numpy(data.copy()), torch.from_numpy(labels_array.copy())
            if len(x) != 10000:
                raise RuntimeError(f'Invalid CIFAR-10 batch size: {name}')
            images.append(x)
            labels.append(y)
    return torch.cat(images), torch.cat(labels)
