#!/usr/bin/env bash
# qsub scripts/cs_cluster/run_cifar_cfg_h100.sh
# Three independent CFG baselines, one H100 per task.
#$ -S /bin/bash
#$ -N cifar_cfg_b
#$ -P aihub_ucl
#$ -cwd
#$ -t 1-3
#$ -pe gpu 1
#$ -l tmem=32G,h_rt=48:00:00,gpu=true,gpu_type=h100
#$ -j y
#$ -o /home/zongchen/
set -euo pipefail
export CFG=1
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
exec bash "$PROJECT_ROOT/scripts/cs_cluster/run_cifar_official_b_h100.sh" "$@"
