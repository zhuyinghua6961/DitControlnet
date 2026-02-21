#!/usr/bin/env bash
set -euo pipefail

# A/B/C schedule sanity (20 samples, fixed cfg/steps).
# A: LoRA off (scale=0.0)
# B: LoRA on (scale=0.5)
# C: LoRA schedule (target=0.5; 40/40/20)
#
# Usage:
#   bash scripts/run_lora_schedule_abc_20.sh [LORA_PATH] [MANIFEST] [OUT_ROOT]
#
# Defaults:
#   LORA_PATH = outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_7000
#   MANIFEST  = data/manifests/eval_manifest_200.jsonl
#   OUT_ROOT  = outputs/pixart_lora_eval_10k_12000steps/abc_sched_20_step7000

LORA_PATH="${1:-outputs/pixart_lora_baseline_10k_12000steps/checkpoints/step_7000}"
MANIFEST="${2:-data/manifests/eval_manifest_200.jsonl}"
OUT_ROOT="${3:-outputs/pixart_lora_eval_10k_12000steps/abc_sched_20_step7000}"

if [ -e "${OUT_ROOT}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_ROOT="${OUT_ROOT}_run${TS}"
fi

CFG=2.5
STEPS=30
H=512
W=512
MAX_SAMPLES=20

run_eval () {
  local tag="$1"; shift
  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  echo "[run] ${tag} lora=${LORA_PATH} test=${MANIFEST} out=${out_dir}"
  python scripts/evaluate_pixart_lora_baseline.py \
    --config configs/pixart_lora_eval_10k.json \
    --test "${MANIFEST}" \
    --lora "${LORA_PATH}" \
    --guidance-scale "${CFG}" \
    --num-steps "${STEPS}" \
    --height "${H}" \
    --width "${W}" \
    --max-samples "${MAX_SAMPLES}" \
    --output-jsonl "${out_dir}/eval.jsonl" \
    --summary-json "${out_dir}/summary.json" \
    --image-dir "${out_dir}/images" \
    "$@"
}

# A: LoRA off
run_eval "A_scale_0.0" --lora-scale 0.0

# B: LoRA on (fixed)
run_eval "B_scale_0.5" --lora-scale 0.5

# C: LoRA schedule (target=0.5)
run_eval "C_schedule_0.5" --lora-scale 0.5 --lora-schedule --lora-schedule-target 0.5
