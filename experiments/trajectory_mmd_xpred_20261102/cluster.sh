#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N trajectory_xpred
#$ -P aihub_ucl
#$ -cwd
#$ -t 1-2
#$ -pe gpu 1
#$ -l tmem=40G,h_rt=12:00:00,gpu=true,gpu_type=h100
#$ -j y
set -euo pipefail
PYTHON=/home/zongchen/miniconda3/envs/mmd_flow/bin/python
DATA=/SAN/intelsys/imagenet_mmd_flow/trajectory_mmd_assets
task="${SGE_TASK_ID:-1}"
if [ "$task" = 1 ]; then probability=0; variant=integral; else probability=.5; variant=endpoint; fi
OUT="results/trajectory_mmd_xpred_20261102/${JOB_ID}/${variant}/seed_0"
mkdir -p "$(dirname "$OUT")"
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "{\"state\":\"failed\",\"exit_code\":%s}\n" "$code" > "$(dirname "$OUT")/launcher_failure.json"; fi' EXIT
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(),flush=True)'
"$PYTHON" experiments/trajectory_mmd_xpred_20261102/train.py --output "${OUT}_preflight" \
  --updates 2 --eval-every 2 --eval-n 32 --eval-steps 4 --endpoint-probability "$probability" \
  --data-root "$DATA" --classifier "$DATA/evaluation/classifier.pt" --amp
"$PYTHON" experiments/trajectory_mmd_xpred_20261102/train.py --output "$OUT" \
  --endpoint-probability "$probability" --data-root "$DATA" \
  --classifier "$DATA/evaluation/classifier.pt" --amp
