#!/usr/bin/env python3
"""
Bucket teacher outputs by grade/accepted from quality_records.jsonl.
Creates symlinks (default) or writes a manifest only.
"""
import argparse
import json
import os
from pathlib import Path


def _bucket_name(grade, accepted):
    if not accepted:
        return "reject"
    if grade == "A+":
        return "Aplus"
    if grade == "A":
        return "A"
    if grade == "B":
        return "B"
    return "reject"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/teacher_out/dpmpp_karras_v1/quality_records.jsonl")
    ap.add_argument("--out-dir", default="data/teacher_out/dpmpp_karras_v1/bucket")
    ap.add_argument("--mode", choices=["symlink", "copy", "manifest"], default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    records_path = Path(args.records)
    if not records_path.exists():
        raise FileNotFoundError(f"records not found: {records_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "bucket_manifest.jsonl"

    counts = {"Aplus": 0, "A": 0, "B": 0, "reject": 0}

    with records_path.open("r", encoding="utf-8") as f, manifest_path.open("w", encoding="utf-8") as mf:
        for line in f:
            rec = json.loads(line)
            grade = rec.get("grade")
            accepted = bool(rec.get("accepted", False))
            bucket = _bucket_name(grade, accepted)
            counts[bucket] += 1

            img = rec.get("image")
            if img:
                src = Path(img)
            else:
                src = None

            dst_dir = out_dir / bucket
            dst_dir.mkdir(parents=True, exist_ok=True)

            if args.mode in ("symlink", "copy") and src is not None:
                dst = dst_dir / src.name
                if dst.exists():
                    if args.overwrite:
                        dst.unlink()
                    else:
                        pass
                else:
                    if args.mode == "symlink":
                        os.symlink(src.resolve(), dst)
                    else:
                        dst.write_bytes(src.read_bytes())

            rec_out = dict(rec)
            rec_out["bucket"] = bucket
            mf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    summary = out_dir / "bucket_summary.json"
    with summary.open("w", encoding="utf-8") as sf:
        json.dump(counts, sf, ensure_ascii=False, indent=2)

    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
