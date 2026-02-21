#!/usr/bin/env bash
set -euo pipefail

# Sequential runs:
# Run-0: baseline (no style loss)
# Run-1: style loss lambda=0.01, every 20 steps, decode size 256
# Run-2: style loss lambda=0.05, every 20 steps, decode size 256
#
# Usage:
#   bash scripts/run_style_loss_sweep.sh [CONFIG] [OUT_ROOT]
#
# Defaults:
#   CONFIG   = configs/pixart_lora_baseline.json
#   OUT_ROOT = outputs/pixart_lora_baseline_styleloss

CONFIG="${1:-configs/pixart_lora_baseline.json}"
OUT_ROOT="${2:-outputs/pixart_lora_baseline_styleloss}"

if [ -e "${OUT_ROOT}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_ROOT="${OUT_ROOT}_run${TS}"
fi

mkdir -p "${OUT_ROOT}"

run_train () {
  local tag="$1"; shift
  local out_dir="${OUT_ROOT}/${tag}"
  local tb_dir="${out_dir}/tb"
  mkdir -p "${out_dir}"
  echo "[run] ${tag} out=${out_dir}"
  python scripts/train_pixart_lora_baseline.py \
    --config "${CONFIG}" \
    --output-dir "${out_dir}" \
    --tensorboard-dir "${tb_dir}" \
    "$@"
}

# Run-0: baseline (lambda=0)
run_train "run0_baseline" --style-loss-weight 0.0

# Run-1: lambda=0.01, K=20, decode size 256
run_train "run1_styleloss_001" \
  --style-loss \
  --style-loss-weight 0.01 \
  --style-loss-every 20 \
  --style-loss-size 256

# Run-2: lambda=0.05, K=20, decode size 256
run_train "run2_styleloss_005" \
  --style-loss \
  --style-loss-weight 0.05 \
  --style-loss-every 20 \
  --style-loss-size 256
