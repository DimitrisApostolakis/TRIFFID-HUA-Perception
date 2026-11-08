#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/mnt/d/projects/triffid-hua-perception/src/triffid_uav_perception
export MPLCONFIGDIR=/mnt/d/projects/triffid-hua-perception/ugv_models/runtime_config/matplotlib
exec /home/oracle/miniforge3/envs/sansa/bin/python /mnt/d/projects/triffid-hua-perception/scripts/run_uav_sansa_test.py
