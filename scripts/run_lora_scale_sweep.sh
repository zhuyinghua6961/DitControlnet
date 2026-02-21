#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_lora_scale_sweep.sh [LORA_PATH] [MANIFEST] [OUT_ROOT] [MAX_SAMPLES]
#
# Defaults:
#   LORA_PATH  = outputs/pixart_lora_baseline_10k_12000steps/final
#   MANIFEST   = data/manifests/eval_manifest_200.jsonl
#   OUT_ROOT   = outputs/pixart_lora_eval_10k_12000steps/scale_sweep
#   MAX_SAMPLES= 200

LORA_PATH="${1:-outputs/pixart_lora_baseline_10k_12000steps/final}"
MANIFEST="${2:-data/manifests/eval_manifest_200.jsonl}"
OUT_ROOT="${3:-outputs/pixart_lora_eval_10k_12000steps/scale_sweep}"
MAX_SAMPLES="${4:-200}"

SCALES=(0.0 0.2 0.5 1.0)

for s in "${SCALES[@]}"; do
  out_dir="${OUT_ROOT}/scale_${s}"
  mkdir -p "${out_dir}"
  echo "[run] scale=${s} lora=${LORA_PATH} test=${MANIFEST} out=${out_dir}"
  python scripts/evaluate_pixart_lora_baseline.py \
    --config configs/pixart_lora_eval_10k.json \
    --test "${MANIFEST}" \
    --lora "${LORA_PATH}" \
    --lora-scale "${s}" \
    --max-samples "${MAX_SAMPLES}" \
    --output-jsonl "${out_dir}/eval.jsonl" \
    --summary-json "${out_dir}/summary.json" \
    --image-dir "${out_dir}/images"
done
