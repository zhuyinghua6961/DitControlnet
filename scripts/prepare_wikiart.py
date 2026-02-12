#!/usr/bin/env python3
"""
Prepare WikiArt dataset with mirror-friendly Hugging Face download.
- Downloads/caches the HF dataset (huggan/wikiart)
- Builds data/wikiart/images with symlinks or copies
- Writes meta.csv and split files (train/val/test)
"""
import argparse
import csv
import os
import shutil
import sys
import zlib
from pathlib import Path


def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "|".join(str(i) for i in x)
    return str(x)


def _assign_split(artist, file_id, seed, train_ratio, val_ratio):
    key = f"{artist}::{file_id}::{seed}".encode("utf-8")
    r = (zlib.crc32(key) % 10000) / 10000.0
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def _get_image_path(img_field):
    # datasets.Image(decode=False) yields dict with "path"
    if isinstance(img_field, dict) and "path" in img_field:
        p = img_field.get("path")
        return Path(p) if p else None
    # sometimes it could be a path string
    if isinstance(img_field, str):
        return Path(img_field)
    # fallback: try attribute
    path = getattr(img_field, "path", None)
    if path:
        return Path(path)
    return None


def _infer_ext(img_path, img_field):
    if img_path and img_path.suffix:
        return img_path.suffix
    fmt = getattr(img_field, "format", None)
    if fmt:
        fmt = fmt.lower()
        if fmt == "jpeg":
            return ".jpg"
        return f".{fmt}"
    return ".jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/wikiart", help="Output root for wikiart")
    ap.add_argument("--cache-dir", default="data/wikiart/raw", help="HF cache dir (kept in workspace)")
    ap.add_argument("--hf-endpoint", default="https://hf-mirror.com", help="HF mirror endpoint")
    ap.add_argument("--split-ratios", default="0.8,0.1,0.1", help="train,val,test ratios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Limit samples (0 for all)")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from existing meta/splits/images")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Configure HF mirror + cache inside workspace
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    if args.cache_dir:
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))

    try:
        from datasets import load_dataset, Image
    except Exception as e:
        print(f"Failed to import datasets: {e}", file=sys.stderr)
        sys.exit(1)

    # Load dataset (default split; fallback if no train split)
    try:
        ds = load_dataset("huggan/wikiart", split="train", cache_dir=args.cache_dir)
    except Exception:
        ds_dict = load_dataset("huggan/wikiart", cache_dir=args.cache_dir)
        split_name = list(ds_dict.keys())[0]
        ds = ds_dict[split_name]

    ds = ds.cast_column("image", Image(decode=False))

    ratios = [float(x) for x in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        print("split ratios must sum to 1.0", file=sys.stderr)
        sys.exit(2)
    train_ratio, val_ratio, _ = ratios

    meta_path = output_dir / "meta.csv"
    processed_ids = set()

    if args.resume and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                if row.get("id") is not None:
                    processed_ids.add(str(row["id"]))
        meta_f = meta_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            meta_f,
            fieldnames=["id", "path", "artist", "style", "genre"],
        )
    else:
        meta_f = meta_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            meta_f,
            fieldnames=["id", "path", "artist", "style", "genre"],
        )
        writer.writeheader()

    split_files = {}
    for split_name in ("train", "val", "test"):
        split_path = splits_dir / f"{split_name}.txt"
        mode = "a" if args.resume and split_path.exists() else "w"
        split_files[split_name] = split_path.open(mode, encoding="utf-8")

    count = 0
    for idx, rec in enumerate(ds):
        if args.limit and count >= args.limit:
            break
        img_field = rec.get("image")
        img_path = _get_image_path(img_field)

        file_id = rec.get("id")
        if file_id is None:
            file_id = img_path.stem if img_path else str(idx)
        file_id = str(file_id)

        if file_id in processed_ids:
            continue

        ext = _infer_ext(img_path, img_field)
        rel_path = Path("images") / f"{file_id}{ext}"
        dst = output_dir / rel_path

        if dst.exists() and args.overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()

        if not dst.exists():
            if img_path and args.mode == "symlink":
                try:
                    os.symlink(img_path, dst)
                except FileExistsError:
                    pass
            elif img_path:
                shutil.copy2(img_path, dst)
            else:
                # no path in cache (bytes embedded). write bytes or save PIL.
                if isinstance(img_field, dict) and img_field.get("bytes") is not None:
                    with dst.open("wb") as out_f:
                        out_f.write(img_field["bytes"])
                elif hasattr(img_field, "save"):
                    img_field.save(dst)
                else:
                    continue

        artist = _safe_str(rec.get("artist"))
        style = _safe_str(rec.get("style"))
        genre = _safe_str(rec.get("genre"))

        writer.writerow(
            {
                "id": file_id,
                "path": str(rel_path),
                "artist": artist,
                "style": style,
                "genre": genre,
            }
        )

        split = _assign_split(artist, file_id, args.seed, train_ratio, val_ratio)
        split_files[split].write(str(rel_path) + "\n")

        processed_ids.add(file_id)
        count += 1

    meta_f.close()
    for f in split_files.values():
        f.close()

    print(f"Prepared {count} samples -> {output_dir}")


if __name__ == "__main__":
    main()
