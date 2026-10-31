"""Verified official Tiny ImageNet, native 64px RGB, train and held-out validation."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ARCHIVE_MD5 = '90528d7ca1a48142e341f4ef8d21d0de'


def prepare(root):
    root = Path(root).resolve()
    archive = root.parent/'tiny-imagenet-200.zip'
    if hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5:
        raise ValueError('Official Tiny ImageNet archive checksum mismatch')
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    root.mkdir(parents=True,exist_ok=True)
    bundle = zipfile.ZipFile(archive)
    prefix = 'tiny-imagenet-200/'
    classes = sorted(bundle.read(prefix+'wnids.txt').decode().split())
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError('Expected 200 distinct Tiny ImageNet classes')
    labels = {name:i for i,name in enumerate(classes)}
    train = []
    archive_names = bundle.namelist()
    for name in classes:
        files = sorted(p for p in archive_names
                       if p.startswith(prefix+f'train/{name}/images/') and p.endswith('.JPEG'))
        if len(files) != 500:
            raise ValueError(f'Expected 500 training images for {name}')
        train.extend((p.removeprefix(prefix),labels[name]) for p in files)
    val = sorted((f'val/images/{line.split()[0]}',labels[line.split()[1]])
                 for line in bundle.read(prefix+'val/val_annotations.txt').decode().splitlines())
    if len(val) != 10000 or len({p for p,_ in val}) != 10000:
        raise ValueError('Expected 10000 distinct validation images')
    for split, records in [('train',train),('val',val)]:
        if any(sum(label == i for _,label in records) != (500 if split=='train' else 50)
               for i in range(200)):
            raise ValueError('Unbalanced class counts')
        cache = root/f'{split}_rgb64.npy'
        temporary = cache.with_suffix('.npy.tmp')
        images = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                          shape=(len(records),3,64,64))
        for i,(path,_) in enumerate(records):
            with Image.open(io.BytesIO(bundle.read(prefix+path))) as im:
                if im.size != (64,64):
                    raise ValueError(f'Unexpected image size: {path}')
                images[i] = np.asarray(im.convert('RGB')).transpose(2,0,1)
        images.flush()
        del images
        temporary.replace(cache)
        encoded = json.dumps(records,separators=(',',':')).encode()
        (root/f'{split}_index.json').write_bytes(encoded)
        identity = dict(root=str(root),split=split,count=len(records),classes=classes,
            index_sha256=hashlib.sha256(encoded).hexdigest(),
            archive_sha256=archive_sha,archive_md5=ARCHIVE_MD5,
            cache_sha256=hashlib.file_digest(cache.open('rb'),'sha256').hexdigest(),
            preprocessing='native 64x64 RGB uint8; train-only random horizontal flip; training normalized to [-1,1]',
            source_url='https://cs231n.stanford.edu/tiny-imagenet-200.zip')
        (root/f'{split}_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        print(json.dumps(identity),flush=True)
    bundle.close()


def build_index(root, output):
    root, output = Path(root).resolve(), Path(output)
    identity = json.loads((root/'train_identity.json').read_text())
    encoded = (root/'train_index.json').read_bytes()
    if identity['count'] != 100000 or len(identity['classes']) != 200:
        raise ValueError('Wrong Tiny ImageNet training identity')
    if hashlib.sha256(encoded).hexdigest() != identity['index_sha256']:
        raise ValueError('Training index checksum mismatch')
    (output/'dataset_index.json').write_bytes(encoded)
    (output/'dataset_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity


class ImageNet64(Dataset):
    def __init__(self, root, index=None, split='train'):
        self.root, self.split = Path(root), split
        self.records = json.loads(Path(index or self.root/f'{split}_index.json').read_text())
        self.images = np.load(self.root/f'{split}_rgb64.npy',mmap_mode='r')
        if self.images.shape != (len(self.records),3,64,64) or self.images.dtype != np.uint8:
            raise ValueError('Invalid image cache')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.from_numpy(self.images[int(index)].copy())


class RandomImageBatches(Dataset):
    """Preserve the original class-matched replacement-sampling algorithm."""
    def __init__(self,dataset,batch,seed,rank=0,dropout=.1):
        self.dataset,self.batch,self.seed = dataset,batch,seed
        self.rank,self.dropout = rank,dropout
        self.class_indices = [[] for _ in range(200)]
        for i,(_,label) in enumerate(dataset.records):
            self.class_indices[label].append(i)

    def __len__(self):
        return 2**31

    def __getitem__(self,attempt):
        group_rng = torch.Generator().manual_seed(self.seed+int(attempt)*1000003)
        dropped = bool(torch.rand((),generator=group_rng)<self.dropout)
        label = 200 if dropped else int(torch.randint(200,(),generator=group_rng))
        rng = torch.Generator().manual_seed(self.seed+int(attempt)*1000003+1009*(self.rank+1))
        candidates = self.class_indices[label] if label<200 else []
        if label<200 and not candidates:
            raise RuntimeError('Missing Tiny ImageNet class')
        ids = torch.randint(len(candidates) if candidates else len(self.dataset),
                            (self.batch,),generator=rng)
        if candidates:
            ids = np.asarray(candidates)[ids.numpy()]
        else:
            ids = ids.numpy()
        return dict(images=torch.from_numpy(self.dataset.images[ids].copy()),label=label)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    prepare(parser.parse_args().root)
