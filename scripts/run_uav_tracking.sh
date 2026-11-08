#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=/tmp/triffid_uav_masa_mpl
export PYTHONPATH="$task_root/src/triffid_uav_perception:$task_root/src/triffid_ugv_perception:$task_root/ugv_models/opus/third_party/mmdetection${PYTHONPATH:+:$PYTHONPATH}"
exec "$task_root/ugv_models/opus_env/bin/python" -m triffid_uav_perception.uav_node track-frames "$@"
