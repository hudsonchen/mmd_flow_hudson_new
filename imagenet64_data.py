"""ImageNet training images, JiT/ADM center crop to 64px, no learned encoder."""
import hashlib
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor
from vendor.jit.crop import center_crop_arr


def build_index(root, output):
    root = Path(root).resolve()
    images = ImageFolder(root)
    if len(images.classes) != 1000 or len(images) != 1281167:
        raise ValueError(f'Expected full ImageNet train: got {len(images.classes)} classes, {len(images)} images')
    records = [(str(Path(path).relative_to(root)), label) for path,label in images.samples]
    encoded = json.dumps(records, separators=(',',':')).encode()
    destination = Path(output)/'dataset_index.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    identity = dict(root=str(root), count=len(records), classes=images.classes,
        index_sha256=hashlib.sha256(encoded).hexdigest(),
        preprocessing='Official JiT/ADM center crop to 64, RGB uint8; training random horizontal flip; normalize to [-1,1]',
        split='train', conditioning='class-matched global interaction batches; 10% unconditional mixed-class batches',
        spot_sha256={records[i][0]:hashlib.sha256((root/records[i][0]).read_bytes()).hexdigest()
                     for i in range(0,len(records),max(1,len(records)//16))})
    (Path(output)/'dataset_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


class ImageNet64(Dataset):
    def __init__(self, root, index):
        self.root = Path(root)
        self.records = json.loads(Path(index).read_text())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, _ = self.records[int(index)]
        with Image.open(self.root/path) as image:
            return pil_to_tensor(center_crop_arr(image.convert('RGB'),64))


class RandomImageBatches(Dataset):
    """Deterministic per-attempt replacement samples, independent of worker prefetch.

    Resume starts at saved attempt count; the global torch RNG is reserved for
    noise/time/flip draws and is checkpointed separately.
    """
    def __init__(self, dataset, batch, seed, rank=0, dropout=.1):
        self.dataset, self.batch, self.seed = dataset, batch, seed
        self.rank, self.dropout = rank, dropout
        self.class_indices = [[] for _ in range(1000)]
        if hasattr(dataset,'records'):
            for index, (_, label) in enumerate(dataset.records):
                self.class_indices[label].append(index)

    def __len__(self):
        return 2**31

    def __getitem__(self, attempt):
        group_rng = torch.Generator().manual_seed(self.seed+int(attempt)*1000003)
        dropped = bool(torch.rand((),generator=group_rng)<self.dropout)
        label = 1000 if dropped else int(torch.randint(1000,(),generator=group_rng))
        rng = torch.Generator().manual_seed(self.seed+int(attempt)*1000003+1009*(self.rank+1))
        candidates = self.class_indices[label] if label<1000 else []
        if label<1000 and hasattr(self.dataset,'records') and not candidates:
            raise RuntimeError('Missing ImageNet class')
        ids = torch.randint(len(candidates) if candidates else len(self.dataset),
                            (self.batch,),generator=rng)
        if candidates:
            ids = [candidates[int(i)] for i in ids]
        return dict(images=torch.stack([self.dataset[i] for i in ids]),label=label)
