#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/oracle/miniforge3/envs/fewshot-dev/bin/python /mnt/d/projects/triffid-hua-perception/scripts/run_pui_comparison.py "$@"
