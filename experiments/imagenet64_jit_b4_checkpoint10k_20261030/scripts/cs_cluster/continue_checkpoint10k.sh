#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N in64_b4_ck10k
#$ -cwd
#$ -j y
#$ -q gpu.q@seymour3.local
#$ -P aihub_ucl
#$ -pe gpu 4
#$ -l gpu=true,gpu_type=h100,tmem=24G,h_rt=720:00:00
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=WARN
exec /home/zongchen/miniconda3/envs/mmd_flow/bin/python -u \
  "${SOURCE_DIR:?}/scripts/cs_cluster/continue_checkpoint10k.py"
