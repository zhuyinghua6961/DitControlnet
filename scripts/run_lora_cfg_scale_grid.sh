#!/usr/bin/env bash
set -euo pipefail

# Grid: fixed CFG=2.5, LoRA scale in {0.0,0.3,0.5}
# Uses fixed eval subset (100 samples) for quick trend checks.
#
# Usage:
#   bash scripts/run_lora_cfg_scale_grid.sh [LORA_PATH] [MANIFEST] [OUT_ROOT] [MAX_SAMPLES]
#
# Defaults:
#   LORA_PATH   = outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_7000
#   MANIFEST    = data/manifests/eval_manifest_100.jsonl
#   OUT_ROOT    = outputs/pixart_lora_eval_10k_12000steps/cfg25_scale_sweep_100
#   MAX_SAMPLES = 100

LORA_PATH="${1:-outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_7000}"
MANIFEST="${2:-data/manifests/eval_manifest_100.jsonl}"
OUT_ROOT="${3:-outputs/pixart_lora_eval_10k_12000steps/cfg25_scale_sweep_100}"
MAX_SAMPLES="${4:-100}"

CFG_LIST=(2.5)
SCALE_LIST=(0.0 0.3 0.5)

for cfg in "${CFG_LIST[@]}"; do
  for s in "${SCALE_LIST[@]}"; do
    out_dir="${OUT_ROOT}/cfg_${cfg}_scale_${s}"
    mkdir -p "${out_dir}"
    echo "[run] cfg=${cfg} scale=${s} lora=${LORA_PATH} test=${MANIFEST} out=${out_dir}"
    python scripts/evaluate_pixart_lora_baseline.py \
      --config configs/pixart_lora_eval_10k.json \
      --test "${MANIFEST}" \
      --lora "${LORA_PATH}" \
      --lora-scale "${s}" \
      --guidance-scale "${cfg}" \
      --max-samples "${MAX_SAMPLES}" \
      --output-jsonl "${out_dir}/eval.jsonl" \
      --summary-json "${out_dir}/summary.json" \
      --image-dir "${out_dir}/images"
  done
done
