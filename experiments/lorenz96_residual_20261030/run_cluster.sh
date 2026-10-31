#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N l96_residual
#$ -cwd
#$ -j y
#$ -q gpu.q
#$ -P aihub_ucl
#$ -pe gpu 1
#$ -l gpu=true,tmem=16G,h_rt=12:00:00
set -euo pipefail
source_dir="${SOURCE_DIR:?Set SOURCE_DIR}"
study_dir="${STUDY_DIR:?Set STUDY_DIR}"
dataset="${DATASET:?Set DATASET}"
python_bin="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$source_dir"
on_error() {
    failure_code=$?
    "$python_bin" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); (p/"launch_status.json").write_text(json.dumps({"stage":"failed","exit_code":int(sys.argv[2]),"reason":"Launcher or preflight failed; inspect scheduler.log"}))' "$study_dir" "$failure_code"
    exit "$failure_code"
}
trap on_error ERR
timeout 180 "$python_bin" -u -c 'import faulthandler, unittest; faulthandler.dump_traceback_later(45, repeat=True); unittest.main(module="test_science", verbosity=2)'
"$python_bin" -u study.py --output "$study_dir/preflight" --data "$dataset" --device cuda --smoke
exec "$python_bin" -u study.py --output "$study_dir/production" --data "$dataset" --device cuda
