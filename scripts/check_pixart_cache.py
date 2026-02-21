#!/usr/bin/env python3
"""
Check PixArt model cache completeness:
  - model_index.json exists in snapshot
  - no *.incomplete blobs
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PixArt-alpha/PixArt-XL-2-512x512")
    ap.add_argument("--cache-dir", default="/mnt/fast18/models_cache")
    args = ap.parse_args()

    root = Path(args.cache_dir) / f"models--{args.model_id.replace('/', '--')}"
    if not root.exists():
        raise SystemExit(f"cache missing: {root}")

    blobs = root / "blobs"
    incompletes = list(blobs.glob("*.incomplete")) if blobs.exists() else []
    if incompletes:
        raise SystemExit(f"incomplete blobs: {len(incompletes)}")

    snap_dir = root / "snapshots"
    if not snap_dir.exists():
        raise SystemExit(f"snapshots missing: {snap_dir}")

    model_index = None
    for s in snap_dir.iterdir():
        if (s / "model_index.json").exists():
            model_index = s / "model_index.json"
            break
    if not model_index:
        raise SystemExit("model_index.json not found")

    print("OK")
    print(f"snapshot: {model_index.parent}")
    print(f"model_index: {model_index}")


if __name__ == "__main__":
    main()
