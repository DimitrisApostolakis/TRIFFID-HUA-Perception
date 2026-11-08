#!/usr/bin/env bash
set -euo pipefail
task_backend="${1:?Use foris, insid3, frost, rsrm or fssdino for the active evaluation suite}"
task_root="$(cd "$(dirname "$0")/.." && pwd)"
case "$task_backend" in
  foris|frost) task_python=/home/oracle/miniforge3/envs/frost/bin/python ;;
  insid3) task_python=/home/oracle/miniforge3/envs/frost/bin/python ;;
  rsrm|fssdino) task_python=/home/oracle/miniforge3/envs/frost/bin/python ;;
  sansa) task_python=/home/oracle/miniforge3/envs/sansa/bin/python ;;
  fss-sam3|fss-sam3-text) task_python=/home/oracle/miniforge3/envs/fss-sam3/bin/python ;;
  *) exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$task_root/uav_models/runtime/dino:$task_root/src/triffid_uav_perception${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=/tmp/triffid_uav_holdout_mpl
shift
exec "$task_python" "$task_root/scripts/benchmark_uav_holdout.py" --backend "$task_backend" "$@"
