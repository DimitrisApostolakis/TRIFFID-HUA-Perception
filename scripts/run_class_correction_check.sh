#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/oracle/miniforge3/envs/fewshot-dev/bin/python /mnt/d/projects/triffid-hua-perception/scripts/prepare_class_correction_cache.py
