#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# WSL exposes this NTFS workspace through two case aliases; use the writable
# spelling configured for this session while retaining an explicit override.
if [ "$project_dir" = /mnt/d/Projects/TRIFFID-HUA-Perception ]; then
  project_dir=/mnt/d/projects/triffid-hua-perception
fi
project_dir="${TRIFFID_PROJECT_ROOT:-$project_dir}"
export TRIFFID_PROJECT_ROOT="$project_dir"
model_name="${1:?Usage: model_python.sh FSOD-VFM|DINOv|YOLOE SCRIPT [ARGS]}"
shift
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false YOLO_AUTOINSTALL=false
export MPLCONFIGDIR="$project_dir/ugv_models/runtime_config/matplotlib"
export YOLO_CONFIG_DIR="$project_dir/ugv_models/runtime_config/yoloe"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
case "$model_name" in
  FSOD-VFM)
    model_python="$project_dir/ugv_models/fsod_vfm_env/bin/python"
    export PYTHONPATH="$project_dir/ugv_models/fsod_vfm:$project_dir/ugv_models/msda_bridge:$project_dir/src/triffid_ugv_perception"
    ;;
  DINOv)
    model_python="$project_dir/ugv_models/dinov_env/bin/python"
    export PYTHONPATH="$project_dir/ugv_models/dinov:$project_dir/ugv_models/msda_bridge:$project_dir/ugv_models/pet_runtime:$project_dir/ugv_models/devit_runtime:$project_dir/src/triffid_ugv_perception"
    ;;
  YOLOE)
    model_python="$project_dir/ugv_models/yoloe_env/bin/python"
    export PYTHONPATH="$project_dir/ugv_models/yoloe:$project_dir/src/triffid_ugv_perception"
    ;;
  *) echo "Unknown model: $model_name" >&2; exit 2 ;;
esac
if [ "$#" -eq 0 ]; then
  set -- "$project_dir/scripts/check_detector_setup.py" "$model_name"
fi
exec "$model_python" "$@"
