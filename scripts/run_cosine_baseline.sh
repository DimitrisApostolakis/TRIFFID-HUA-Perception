#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_dir/ugv_models/cosine_runtime:$project_dir/ugv_models/devit_runtime:$project_dir/src/triffid_ugv_perception${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$project_dir/ugv_models/ultralytics_config"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${COSINE_PYTHON:-/home/oracle/miniforge3/envs/fewshot-dev/bin/python}" "$project_dir/scripts/run_cosine_baseline.py" "$@"
