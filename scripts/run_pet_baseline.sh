#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_dir/ugv_models/pet_runtime:$project_dir/ugv_models/cosine_runtime:$project_dir/ugv_models/devit_runtime:$project_dir/ugv_models/pet_dino:$project_dir/src/triffid_ugv_perception"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR=/tmp/triffid_pet_ultralytics
exec /home/oracle/miniforge3/envs/fewshot-dev/bin/python "$project_dir/scripts/run_fewshot_baseline.py" --backend pet_dino "$@"
