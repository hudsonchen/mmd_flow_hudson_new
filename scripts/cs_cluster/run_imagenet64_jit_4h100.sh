#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N in64_b4_xpred
#$ -cwd
#$ -j y
#$ -q gpu.q@seymour3.local
#$ -P aihub_ucl
#$ -pe gpu 4
#$ -l gpu=true,gpu_type=h100,tmem=24G,h_rt=720:00:00
set -euo pipefail
repo="${SOURCE_DIR:-$(pwd -P)}"
study="${STUDY:?Set STUDY to a fresh absolute SAN result directory}"
python_bin="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec "$python_bin" -u "$repo/scripts/cs_cluster/launch_imagenet64_jit.py" --study "$study" --gpus 4 --patch-size 4 --prediction x --gradient-checkpointing
