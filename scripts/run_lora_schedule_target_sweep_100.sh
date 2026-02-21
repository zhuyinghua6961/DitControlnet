#!/usr/bin/env bash
set -euo pipefail

# Target scale sweep with schedule enabled.
# Fixed: CFG=2.5, num_steps=30, eval 100 samples.
#
# Usage:
#   bash scripts/run_lora_schedule_target_sweep_100.sh [LORA_PATH] [MANIFEST] [OUT_ROOT]
#
# Defaults:
#   LORA_PATH = outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_7000
#   MANIFEST  = data/manifests/eval_manifest_100.jsonl
#   OUT_ROOT  = outputs/pixart_lora_eval_10k_12000steps/target_sweep_100_step7000

LORA_PATH="${1:-outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_7000}"
MANIFEST="${2:-data/manifests/eval_manifest_100.jsonl}"
OUT_ROOT="${3:-outputs/pixart_lora_eval_10k_12000steps/target_sweep_100_step7000}"

if [ -e "${OUT_ROOT}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_ROOT="${OUT_ROOT}_run${TS}"
fi

CFG=2.5
STEPS=30
H=512
W=512
MAX_SAMPLES=100
TARGETS=(0.3 0.5 0.8 1.0)

for t in "${TARGETS[@]}"; do
  out_dir="${OUT_ROOT}/target_${t}"
  mkdir -p "${out_dir}"
  echo "[run] target=${t} lora=${LORA_PATH} test=${MANIFEST} out=${out_dir}"
  python scripts/evaluate_pixart_lora_baseline.py \
    --config configs/pixart_lora_eval_10k.json \
    --test "${MANIFEST}" \
    --lora "${LORA_PATH}" \
    --lora-scale "${t}" \
    --lora-schedule \
    --lora-schedule-target "${t}" \
    --guidance-scale "${CFG}" \
    --num-steps "${STEPS}" \
    --height "${H}" \
    --width "${W}" \
    --max-samples "${MAX_SAMPLES}" \
    --output-jsonl "${out_dir}/eval.jsonl" \
    --summary-json "${out_dir}/summary.json" \
    --image-dir "${out_dir}/images"
done
