#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N c10_prime_normoff
#$ -cwd
#$ -j y
#$ -q gpu.q
#$ -P aihub_ucl
#$ -pe gpu 4
#$ -l gpu=true,gpu_type=h100,tmem=16G,h_rt=720:00:00
set -euo pipefail
repo="${SOURCE_DIR:-$(pwd -P)}"
study="${STUDY:?Set STUDY to a fresh absolute result directory}"
python_bin="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export DATA_ROOT="${DATA_ROOT:-/SAN/intelsys/imagenet_mmd_flow/data}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1 DRIFT_COMPILE=0
exec "$python_bin" -u "$repo/scripts/cs_cluster/run_cifar_priming.py" --study "$study" "$@"
