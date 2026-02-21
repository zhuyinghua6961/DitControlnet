#!/usr/bin/env bash
set -euo pipefail

# Evaluate style-loss runs on fixed 100-sample manifest.
#
# Settings:
#   Setting-1: CFG=2.5, schedule=true, target_scale=0.5
#   Setting-2: CFG=4.5, schedule=true, target_scale=0.5
#
# Usage:
#   bash scripts/run_eval_styleloss_100.sh [RUN_ROOT] [MANIFEST] [OUT_ROOT]
#
# Defaults:
#   RUN_ROOT = outputs/pixart_lora_baseline_styleloss
#   MANIFEST = data/manifests/eval_manifest_100.jsonl
#   OUT_ROOT = outputs/pixart_lora_eval_10k_12000steps/styleloss_eval_100

RUN_ROOT="${1:-outputs/pixart_lora_baseline_styleloss}"
MANIFEST="${2:-data/manifests/eval_manifest_100.jsonl}"
OUT_ROOT="${3:-outputs/pixart_lora_eval_10k_12000steps/styleloss_eval_100}"

if [ -e "${OUT_ROOT}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_ROOT="${OUT_ROOT}_run${TS}"
fi

# Expect run directories from style-loss sweep
LORA_RUNS=(
  "${RUN_ROOT}/run0_baseline"
  "${RUN_ROOT}/run1_styleloss_001"
  "${RUN_ROOT}/run2_styleloss_005"
)

CFG_LIST=(2.5 4.5)
TARGET=0.5
STEPS=30
H=512
W=512
MAX_SAMPLES=100

for lora_dir in "${LORA_RUNS[@]}"; do
  if [ ! -d "${lora_dir}" ]; then
    echo "[warn] missing lora dir: ${lora_dir} (skip)"
    continue
  fi
  # Prefer best/ if present, otherwise use the run directory directly.
  LORA_PATH="${lora_dir}"
  if [ -d "${lora_dir}/best" ]; then
    LORA_PATH="${lora_dir}/best"
  fi
  tag="$(basename "${lora_dir}")"
  for cfg in "${CFG_LIST[@]}"; do
    out_dir="${OUT_ROOT}/${tag}/cfg_${cfg}"
    mkdir -p "${out_dir}"
    echo "[run] lora=${LORA_PATH} cfg=${cfg} test=${MANIFEST} out=${out_dir}"
    python scripts/evaluate_pixart_lora_baseline.py \
      --config configs/pixart_lora_eval_10k.json \
      --test "${MANIFEST}" \
      --lora "${LORA_PATH}" \
      --lora-scale "${TARGET}" \
      --lora-schedule \
      --lora-schedule-target "${TARGET}" \
      --guidance-scale "${cfg}" \
      --num-steps "${STEPS}" \
      --height "${H}" \
      --width "${W}" \
      --max-samples "${MAX_SAMPLES}" \
      --output-jsonl "${out_dir}/eval.jsonl" \
      --summary-json "${out_dir}/summary.json" \
      --image-dir "${out_dir}/images"
  done
done
