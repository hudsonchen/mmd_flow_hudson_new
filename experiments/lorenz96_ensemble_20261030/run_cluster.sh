#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N l96_ensemble
#$ -cwd
#$ -j y
#$ -q gpu.q
#$ -P aihub_ucl
#$ -pe gpu 1
#$ -l gpu=true,tmem=16G,h_rt=12:00:00
set -euo pipefail
source_dir="${SOURCE_DIR:?Set SOURCE_DIR}"
study_dir="${STUDY_DIR:?Set STUDY_DIR}"
python_bin="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$source_dir"
"$python_bin" -m unittest -v test_science
"$python_bin" -u study.py --output "$study_dir/preflight" --device cuda --smoke
exec "$python_bin" -u study.py --output "$study_dir/production" --device cuda
