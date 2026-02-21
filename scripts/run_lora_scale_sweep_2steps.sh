#!/usr/bin/env bash
set -euo pipefail

# Runs LoRA scale sweep sequentially for step_5000 then step_10000 in background.
# It activates the DitControlnet conda env and uses nohup, writing PID to logs/.
#
# Usage:
#   bash scripts/run_lora_scale_sweep_2steps.sh [MANIFEST] [OUT_ROOT]
#
# Defaults:
#   MANIFEST = data/manifests/eval_manifest_200.jsonl
#   OUT_ROOT = outputs/pixart_lora_eval_10k_12000steps

MANIFEST="${1:-data/manifests/eval_manifest_200.jsonl}"
OUT_ROOT="${2:-outputs/pixart_lora_eval_10k_12000steps}"

LORA_5000="outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_5000"
LORA_10000="outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_10000"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/scale_sweep_5000_then_10000.log"
PID_FILE="${LOG_DIR}/scale_sweep_5000_then_10000.pid"

CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"

CMD="source '${CONDA_SH}' && conda activate DitControlnet && \
bash scripts/run_lora_scale_sweep.sh '${LORA_5000}' '${MANIFEST}' '${OUT_ROOT}/scale_sweep_step_5000' && \
bash scripts/run_lora_scale_sweep.sh '${LORA_10000}' '${MANIFEST}' '${OUT_ROOT}/scale_sweep_step_10000'"

nohup bash -lc "${CMD}" > "${LOG_FILE}" 2>&1 & echo $! | tee "${PID_FILE}"
