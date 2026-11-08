#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_dir/ugv_models/opus/third_party/mmdetection:$project_dir/ugv_models/opus:$project_dir/src/triffid_ugv_perception"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=/tmp/triffid_opus_matplotlib
exec "$project_dir/ugv_models/opus_env/bin/python" "$project_dir/scripts/run_fewshot_baseline.py" --backend opus "$@"
