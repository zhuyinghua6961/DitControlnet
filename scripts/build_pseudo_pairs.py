#!/usr/bin/env python3
"""
Build pseudo pair list for teacher generation.
Creates JSONL with (content, style, prompt) and fixed teacher params.
"""
import argparse
import json
import random
from pathlib import Path


def _parse_seeds(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _load_coco_items(jsonl_path, limit=0):
    items = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(items) >= limit:
                break
            rec = json.loads(line)
            items.append(
                {
                    "id": rec.get("id"),
                    "image": rec.get("image"),
                    "text": rec.get("text", ""),
                }
            )
    return items


def _load_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-jsonl", default="data/cocostuff/cocostuff_train_seg.jsonl")
    ap.add_argument("--wikiart-split", default="data/wikiart/splits/train.txt")
    ap.add_argument("--output", default="data/pairs/pseudo_pairs_train.jsonl")
    ap.add_argument("--num-pairs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--sampler", default="ddim")
    ap.add_argument("--coco-root", default="dataset_cocostuff")
    ap.add_argument("--wikiart-root", default="data/wikiart")
    ap.add_argument("--style-cycle", action="store_true", help="Cycle through shuffled WikiArt list for more uniform usage")
    ap.add_argument("--out-root", default="data/teacher_out")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = _parse_seeds(args.seeds)

    coco_items = _load_coco_items(args.coco_jsonl)
    if not coco_items:
        raise SystemExit("No COCO items loaded")

    wikiart_items = _load_lines(args.wikiart_split)
    if not wikiart_items:
        raise SystemExit("No WikiArt items loaded")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.style_cycle:
        rng.shuffle(wikiart_items)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_pairs):
            c = rng.choice(coco_items)
            if args.style_cycle:
                s = wikiart_items[i % len(wikiart_items)]
            else:
                s = rng.choice(wikiart_items)

            coco_id = c.get("id")
            wiki_id = Path(s).stem
            pair_id = f"coco{coco_id}_wiki{wiki_id}"

            for sd in seeds:
                rec = {
                    "id": f"{pair_id}_seed{sd}",
                    "pair_id": pair_id,
                    "content_path": str(Path(args.coco_root) / c["image"]),
                    "style_path": str(Path(args.wikiart_root) / s),
                    "prompt": c.get("text", ""),
                    "seed": sd,
                    "teacher_target_path": str(
                        Path(args.out_root) / f"{coco_id}_{wiki_id}_seed{sd}.png"
                    ),
                    "meta": {
                        "coco_id": coco_id,
                        "wiki_id": wiki_id,
                        "seed": sd,
                        "strength": args.strength,
                        "steps": args.steps,
                        "cfg": args.cfg,
                        "sampler": args.sampler,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
