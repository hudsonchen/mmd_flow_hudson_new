#!/usr/bin/env bash
# Submit from the isolated Git checkout with qsub -v STUDY_ROOT=...,DATA_ROOT=...,CLASSIFIER=... this_file
#$ -S /bin/bash
#$ -N trajectory_mmd
#$ -P aihub_ucl
#$ -cwd
#$ -pe gpu 1
#$ -l tmem=32G,h_rt=12:00:00,gpu=true,gpu_type=h100
#$ -j y
set -euo pipefail
PYTHON="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
: "${STUDY_ROOT:?Set a fresh output directory}"
: "${DATA_ROOT:?Set the MNIST data root}"
: "${CLASSIFIER:?Set the evaluation-only classifier checkpoint}"
mkdir -p "$STUDY_ROOT"
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "{\"state\":\"failed\",\"exit_code\":%s}\n" "$code" > "$STUDY_ROOT/launcher_status.json"; fi' EXIT
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)'
for seed in 0 1 2; do
  "$PYTHON" experiments/trajectory_mmd_20261102/run.py --dataset mog --seed "$seed" \
    --updates 4000 --batch 256 --reference-batch 512 --train-steps 16 \
    --eval-steps 64 --eval-n 2048 --output "$STUDY_ROOT"
done
"$PYTHON" experiments/trajectory_mmd_20261102/run.py --dataset mnist --seed 0 \
  --updates 6000 --width 512 --lr .0002 --batch 256 --reference-batch 512 \
  --train-steps 16 --eval-steps 64 --eval-every 1000 --eval-n 2048 \
  --data-root "$DATA_ROOT" --classifier "$CLASSIFIER" --output "$STUDY_ROOT"
printf '{"state":"complete"}\n' > "$STUDY_ROOT/launcher_status.json"
