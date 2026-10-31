#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N tiny_jit_s4
#$ -cwd
#$ -j y
#$ -q gpu.q
#$ -P aihub_ucl
#$ -pe gpu 1
#$ -l gpu=true,gpu_type=h100,tmem=32G,h_rt=168:00:00
set -euo pipefail
repo="${SOURCE_DIR:?Set SOURCE_DIR to the frozen experiment source}"
study="${STUDY:?Set STUDY to a fresh absolute SAN output directory}"
objective="${OBJECTIVE:?Set OBJECTIVE to riesz or flow_matching}"
data_root="${DATA_ROOT:?Set DATA_ROOT to prepared Tiny ImageNet}"
python_bin="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec "$python_bin" -u "$repo/scripts/cs_cluster/launch_tinyimagenet.py" \
  --study "$study" --data-root "$data_root" --objective "$objective"
