# save as scripts/compare_cfg_metrics.py or run inline
import json

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        rec = json.loads(next(f))
    return rec["metrics"]

paths = {
    "cfg1.0": "outputs/pixart_lora_eval_10k_12000steps/cfg_1.0/eval.jsonl",
    "cfg2.5": "outputs/pixart_lora_eval_10k_12000steps/cfg_2.5/eval.jsonl",
    "cfg4.5": "outputs/pixart_lora_eval_10k_12000steps/cfg_4.5/eval.jsonl",
}

m = {k: load_metrics(v) for k,v in paths.items()}

keys = ["clip_c","dino_c","lpips_c","clip_gt","dino_gt","lpips_gt","clip_style","clip_t"]

print("metrics:")
for k in keys:
    print(k, {cfg: round(m[cfg].get(k, 0.0), 6) for cfg in m})

print("\nΔ vs cfg1.0:")
for k in keys:
    base = m["cfg1.0"].get(k, 0.0)
    print(k, {cfg: round(m[cfg].get(k, 0.0) - base, 6) for cfg in m if cfg != "cfg1.0"})
