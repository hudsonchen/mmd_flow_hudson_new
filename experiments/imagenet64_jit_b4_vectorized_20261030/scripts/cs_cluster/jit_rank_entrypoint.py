"""Isolate one allocated GPU per worker before importing PyTorch.

Seymour uses exclusive CUDA contexts. NCCL asynchronous threads must not try
to initialize another worker's default device while all allocated GPUs are
visible. Global RANK/WORLD_SIZE stay unchanged; each process uses local CUDA 0.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    slot = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if len(devices)!=world or not 0<=slot<world:
        raise RuntimeError('Allocation must match the requested worker count')
    os.environ['MMD_ALLOCATION_LOCAL_RANK']=str(slot)
    os.environ['MMD_ALLOCATED_GPU']=devices[slot]
    os.environ['CUDA_VISIBLE_DEVICES']=devices[slot]
    os.environ['LOCAL_RANK']='0'
    target=Path(__file__).with_name('run_imagenet64_jit.py')
    sys.argv[0]=str(target)
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':
    main()
