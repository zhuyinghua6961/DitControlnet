#!/usr/bin/env python3
"""
Build train/val manifests from quality_records.jsonl and pairs.jsonl.
Keeps only accepted samples by default.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/teacher_out/full_res/quality_records.jsonl")
    ap.add_argument("--pairs", default="data/pairs/pseudo_pairs_train.jsonl")
    ap.add_argument("--output-train", default="data/manifests/train_manifest_9k.jsonl")
    ap.add_argument("--output-val", default="data/manifests/val_manifest_500.jsonl")
    ap.add_argument("--output-test", default="data/manifests/test_manifest_900.jsonl")
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--stratify-grades", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accepted-only", action="store_true", default=True)
    ap.add_argument("--require-image", action="store_true", default=True)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--prompt-fallback", default="a photo")
    args = ap.parse_args()

    pairs_map = {}
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pairs_map[rec["id"]] = rec

    # keep last accepted attempt per id
    chosen = {}
    with Path(args.records).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.accepted_only and not rec.get("accepted", False):
                continue
            rid = rec.get("id")
            attempt = rec.get("gen_params", {}).get("attempt", 0)
            prev = chosen.get(rid)
            if prev is None or attempt >= prev.get("gen_params", {}).get("attempt", -1):
                chosen[rid] = rec

    samples = []
    for rid, rec in chosen.items():
        pair = pairs_map.get(rid, {})
        prompt = pair.get("prompt") or rec.get("prompt") or args.prompt_fallback
        teacher_path = rec.get("image")
        if args.require_image and teacher_path and not Path(teacher_path).exists():
            continue
        sample = {
            "id": rid,
            "content_path": rec.get("content_path") or pair.get("content_path"),
            "style_path": rec.get("style_path") or pair.get("style_path"),
            "prompt": prompt,
            "teacher_path": teacher_path,
            "accepted": True,
            "grade": rec.get("grade"),
            "weight": args.weight,
        }
        samples.append(sample)

    rng = random.Random(args.seed)
    if args.stratify_grades:
        by_grade = {}
        for s in samples:
            by_grade.setdefault(s.get("grade", "NA"), []).append(s)
        for g in by_grade:
            rng.shuffle(by_grade[g])
        def take_split(items, n_val, n_test):
            val = items[:n_val]
            test = items[n_val:n_val + n_test]
            train = items[n_val + n_test:]
            return train, val, test

        train, val, test = [], [], []
        total = len(samples)
        for g, items in by_grade.items():
            n = len(items)
            n_val = int(round(n * args.val_ratio))
            n_test = int(round(n * args.test_ratio))
            tr, va, te = take_split(items, n_val, n_test)
            train.extend(tr)
            val.extend(va)
            test.extend(te)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
    else:
        rng.shuffle(samples)
        total = len(samples)
        n_val = int(round(total * args.val_ratio))
        n_test = int(round(total * args.test_ratio))
        val = samples[:n_val]
        test = samples[n_val:n_val + n_test]
        train = samples[n_val + n_test:]

    out_train = Path(args.output_train)
    out_val = Path(args.output_val)
    out_test = Path(args.output_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    with out_train.open("w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with out_val.open("w", encoding="utf-8") as f:
        for rec in val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with out_test.open("w", encoding="utf-8") as f:
        for rec in test:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"train={len(train)} val={len(val)} test={len(test)} total={len(samples)}")


if __name__ == "__main__":
    main()
