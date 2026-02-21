#!/usr/bin/env python3
"""
Download PixArt base model into local cache with resume support.
Defaults to PixArt-XL-2-512x512.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PixArt-alpha/PixArt-XL-2-512x512")
    ap.add_argument("--cache-dir", default="/mnt/fast18/models_cache")
    ap.add_argument("--include", default="", help="Comma-separated allow patterns (empty = full repo)")
    ap.add_argument("--exclude", default="", help="Comma-separated ignore patterns")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--require-model-index", action="store_true", default=True, help="Fail if model_index.json missing")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow = [p.strip() for p in args.include.split(",") if p.strip()] or None
    ignore = [p.strip() for p in args.exclude.split(",") if p.strip()] or None

    print(f"Downloading {args.model_id} -> {cache_dir}")
    snapshot_dir = snapshot_download(
        repo_id=args.model_id,
        cache_dir=str(cache_dir),
        allow_patterns=allow,
        ignore_patterns=ignore,
        revision=args.revision,
        local_files_only=False,
        resume_download=True,
    )
    snapshot_dir = Path(snapshot_dir)
    if args.require_model_index:
        model_index = snapshot_dir / "model_index.json"
        if not model_index.exists():
            raise SystemExit(f"Download incomplete: missing {model_index}")

    # check for .incomplete blobs
    repo_dir = cache_dir / f"models--{args.model_id.replace('/', '--')}" / "blobs"
    if repo_dir.exists():
        incomplete = list(repo_dir.glob("*.incomplete"))
        if incomplete:
            raise SystemExit(f"Download incomplete: found {len(incomplete)} *.incomplete blobs")

    print("done")


if __name__ == "__main__":
    main()
