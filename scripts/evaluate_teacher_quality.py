#!/usr/bin/env python3
"""
Evaluate teacher outputs with detailed multi-metric scoring.
Metrics:
  - CLIP content similarity (out vs content)
  - DINOv2 content similarity (out vs content)
  - LPIPS perceptual distance (out vs content)
  - Multi-scale edge similarity (64/128/256)
  - FFT peakiness + high-frequency energy ratio
  - CLIP style similarity (out vs style)
  - Color stats distance (RGB or Lab)
Outputs per-image JSONL + summary JSON.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path

from PIL import Image


def _set_cache_env(cache_dir: str):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))


def _resize_center_crop(img, size):
    if size is None:
        return img
    w, h = img.size
    if w == size and h == size:
        return img
    if w < h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _edge_similarity(img_a, img_b, size=64):
    import numpy as np
    a = np.array(img_a.convert("L").resize((size, size))).astype("float32")
    b = np.array(img_b.convert("L").resize((size, size))).astype("float32")

    def sobel_mag(x):
        x = np.pad(x, 1, mode="edge")
        gx = (
            -1 * x[:-2, :-2] + 1 * x[:-2, 2:]
            -2 * x[1:-1, :-2] + 2 * x[1:-1, 2:]
            -1 * x[2:, :-2] + 1 * x[2:, 2:]
        )
        gy = (
            -1 * x[:-2, :-2] - 2 * x[:-2, 1:-1] - 1 * x[:-2, 2:]
            +1 * x[2:, :-2] + 2 * x[2:, 1:-1] + 1 * x[2:, 2:]
        )
        return np.sqrt(gx * gx + gy * gy)

    ea = sobel_mag(a).reshape(-1)
    eb = sobel_mag(b).reshape(-1)
    na = (ea * ea).sum() ** 0.5
    nb = (eb * eb).sum() ** 0.5
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float((ea @ eb) / (na * nb))


def _fft_metrics(img, size=256, low_freq_radius=0.05):
    import numpy as np
    g = np.array(img.convert("L").resize((size, size))).astype("float32")
    g = g - g.mean()
    fft = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(fft).astype("float32")
    P = np.log1p(mag)
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    yy = yy - c
    xx = xx - c
    rr = np.sqrt(xx * xx + yy * yy) / (size / 2)
    P = P.copy()
    P[rr < low_freq_radius] = 0.0
    vals = P[rr >= low_freq_radius].reshape(-1)
    if vals.size == 0:
        peak_ratio = 0.0
    else:
        p99 = float(np.percentile(vals, 99))
        p50 = float(np.percentile(vals, 50)) + 1e-6
        peak_ratio = p99 / p50

    # high frequency energy ratio
    hf_mask = rr >= 0.5
    power = (mag ** 2).astype("float32")
    hf_energy = float(power[hf_mask].sum() / (power.sum() + 1e-6))
    return peak_ratio, hf_energy


def _color_stats(img, color_space="rgb"):
    import numpy as np
    if color_space == "lab":
        try:
            from skimage.color import rgb2lab
        except Exception:
            color_space = "rgb"
    arr = np.array(img).astype("float32")
    if color_space == "lab":
        arr = rgb2lab(arr / 255.0)
    mean = arr.reshape(-1, 3).mean(axis=0)
    std = arr.reshape(-1, 3).std(axis=0)
    return mean, std


def _norm_hi(x, ok, good):
    if x is None:
        return None
    if x <= ok:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - ok) / (good - ok))


def _norm_lo(x, good, bad):
    if x is None:
        return None
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return float(1.0 - (x - good) / (bad - good))


def _norm_band(x, good_lo, good_hi, bad_lo, bad_hi):
    if x is None:
        return None
    if good_lo <= x <= good_hi:
        return 1.0
    if x < good_lo:
        if x <= bad_lo:
            return 0.0
        return float((x - bad_lo) / (good_lo - bad_lo))
    if x > good_hi:
        if x >= bad_hi:
            return 0.0
        return float(1.0 - (x - good_hi) / (bad_hi - good_hi))
    return 0.0


def _safe_avg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _grade_from_score(score):
    if score is None:
        return "NA"
    if score >= 0.90:
        return "A+"
    if score >= 0.80:
        return "A"
    if score >= 0.65:
        return "B"
    if score >= 0.50:
        return "C"
    return "D"


def _filename_to_id(stem: str):
    parts = stem.split("_")
    if len(parts) >= 3:
        return f"coco{parts[0]}_wiki{parts[1]}_{parts[2]}"
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/teacher_out/ddim_v3_50_clip")
    ap.add_argument("--pairs", default="data/pairs/pseudo_pairs_train.jsonl")
    ap.add_argument("--output", default="data/teacher_out/quality_grades_detailed.jsonl")
    ap.add_argument("--summary", default="data/teacher_out/quality_summary_detailed.json")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--edge-sizes", default="64,128,256")
    ap.add_argument("--color-space", choices=["rgb", "lab"], default="rgb")
    ap.add_argument("--model-cache", default="/mnt/fast18/models_cache")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--gc-interval", type=int, default=20)
    ap.add_argument("--empty-cache-interval", type=int, default=20)

    # CLIP
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--clip-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-clip", action="store_true")

    # DINOv2
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--dino-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-dino", action="store_true")

    # LPIPS
    ap.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex", "squeeze"])
    ap.add_argument("--lpips-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-lpips", action="store_true")

    # scoring thresholds (higher is better)
    ap.add_argument("--clip-ok", type=float, default=0.30)
    ap.add_argument("--clip-good", type=float, default=0.34)
    ap.add_argument("--dino-ok", type=float, default=0.24)
    ap.add_argument("--dino-good", type=float, default=0.28)
    ap.add_argument("--edge-ok", type=float, default=0.45)
    ap.add_argument("--edge-good", type=float, default=0.55)
    ap.add_argument("--style-ok", type=float, default=0.18)
    ap.add_argument("--style-good", type=float, default=0.25)

    # scoring thresholds (lower is better)
    ap.add_argument("--lpips-good", type=float, default=0.40)
    ap.add_argument("--lpips-bad", type=float, default=0.60)
    ap.add_argument("--fft-peak-good", type=float, default=1.40)
    ap.add_argument("--fft-peak-bad", type=float, default=2.20)
    ap.add_argument("--fft-auto", action="store_true", default=True, help="Auto-calibrate fft_peak thresholds by percentiles")
    ap.add_argument("--fft-good-quantile", type=float, default=0.80)
    ap.add_argument("--fft-ok-quantile", type=float, default=0.90)
    ap.add_argument("--fft-bad-quantile", type=float, default=0.97)
    ap.add_argument("--fft-low-radius", type=float, default=0.05)
    ap.add_argument("--hf-good", type=float, default=0.25)
    ap.add_argument("--hf-bad", type=float, default=0.30)
    ap.add_argument("--color-good-low", type=float, default=40.0)
    ap.add_argument("--color-good-high", type=float, default=120.0)
    ap.add_argument("--color-bad-low", type=float, default=20.0)
    ap.add_argument("--color-bad-high", type=float, default=160.0)

    # weighting
    ap.add_argument("--w-content", type=float, default=0.4)
    ap.add_argument("--w-structure", type=float, default=0.2)
    ap.add_argument("--w-style", type=float, default=0.2)
    ap.add_argument("--w-artifact", type=float, default=0.2)
    args = ap.parse_args()

    _set_cache_env(args.model_cache)
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import numpy as np
    import torch

    # load pairs index
    pairs = {}
    with Path(args.pairs).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pairs[rec["id"]] = rec

    # load models
    clip_model = clip_processor = None
    if not args.no_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(
            args.clip_model,
            local_files_only=args.offline,
            cache_dir=args.model_cache,
        ).to(args.clip_device)
        clip_processor = CLIPProcessor.from_pretrained(
            args.clip_model,
            local_files_only=args.offline,
            cache_dir=args.model_cache,
        )
        clip_model.eval()

    dino_model = dino_processor = None
    if not args.no_dino:
        from transformers import AutoImageProcessor, AutoModel
        dino_model = AutoModel.from_pretrained(
            args.dino_model,
            local_files_only=args.offline,
            cache_dir=args.model_cache,
        ).to(args.dino_device)
        dino_processor = AutoImageProcessor.from_pretrained(
            args.dino_model,
            local_files_only=args.offline,
            cache_dir=args.model_cache,
        )
        dino_model.eval()

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips as lpips_lib
            lpips_model = lpips_lib.LPIPS(net=args.lpips_net)
            lpips_model = lpips_model.to(args.lpips_device)
            lpips_model.eval()
        except Exception as e:
            print(f"[warn] LPIPS unavailable: {e}")
            lpips_model = None

    edge_sizes = [int(x) for x in args.edge_sizes.split(",") if x.strip()]

    def clip_embed(img):
        if clip_model is None:
            return None
        inputs = clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(args.clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
        # Some transformer versions return a model output wrapper
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "image_embeds"):
            feats = feats.image_embeds
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def dino_embed(img):
        if dino_model is None:
            return None
        inputs = dino_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(args.dino_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = dino_model(**inputs)
        # CLS token
        feats = out.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def lpips_dist(img_a, img_b):
        if lpips_model is None:
            return None
        def to_tensor(img):
            arr = np.array(img).astype("float32") / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = t * 2.0 - 1.0
            return t
        ta = to_tensor(img_a).to(args.lpips_device)
        tb = to_tensor(img_b).to(args.lpips_device)
        with torch.no_grad():
            d = lpips_model(ta, tb)
        return float(d.item())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    images = sorted(Path(args.images).glob("*.png"))
    if args.max_samples:
        images = images[: args.max_samples]

    # optional auto-calibration for fft_peak thresholds
    fft_good = args.fft_peak_good
    fft_ok = None
    fft_bad = args.fft_peak_bad
    if args.fft_auto and images:
        fft_vals = []
        for p in images:
            out_img = Image.open(p).convert("RGB")
            out_eval = _resize_center_crop(out_img, args.eval_size)
            peak, _ = _fft_metrics(out_eval, size=args.eval_size, low_freq_radius=args.fft_low_radius)
            fft_vals.append(peak)
            try:
                out_img.close()
            except Exception:
                pass
        if fft_vals:
            fft_good = float(np.quantile(fft_vals, args.fft_good_quantile))
            fft_ok = float(np.quantile(fft_vals, args.fft_ok_quantile))
            fft_bad = float(np.quantile(fft_vals, args.fft_bad_quantile))

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        sample_id = _filename_to_id(stem)
        rec = pairs.get(sample_id)
        if rec is None:
            continue

        content_path = rec["content_path"]
        style_path = rec["style_path"]
        out_img = Image.open(img_path).convert("RGB")
        content_img = Image.open(content_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")

        out_eval = _resize_center_crop(out_img, args.eval_size)
        content_eval = _resize_center_crop(content_img, args.eval_size)
        style_eval = _resize_center_crop(style_img, args.eval_size)

        # CLIP
        clip_c = clip_s = None
        if clip_model is not None:
            cf = clip_embed(content_eval)
            of = clip_embed(out_eval)
            sf = clip_embed(style_eval)
            if cf is not None and of is not None:
                clip_c = float((cf @ of.T).item())
            if sf is not None and of is not None:
                clip_s = float((sf @ of.T).item())

        # DINOv2
        dino_c = None
        if dino_model is not None:
            dfc = dino_embed(content_eval)
            dfo = dino_embed(out_eval)
            if dfc is not None and dfo is not None:
                dino_c = float((dfc @ dfo.T).item())

        # LPIPS
        lpips_c = lpips_dist(out_eval, content_eval)

        # Edge multi-scale
        edge_vals = []
        for s in edge_sizes:
            edge_vals.append(_edge_similarity(out_eval, content_eval, size=s))
        edge_ms = float(sum(edge_vals) / len(edge_vals)) if edge_vals else None

        # FFT artifacts
        fft_peak, hf_energy = _fft_metrics(out_eval, size=args.eval_size, low_freq_radius=args.fft_low_radius)

        # Color stats
        mu_o, std_o = _color_stats(out_eval, args.color_space)
        mu_s, std_s = _color_stats(style_eval, args.color_space)
        color_dist = float(np.linalg.norm(mu_o - mu_s) + np.linalg.norm(std_o - std_s))

        # Normalized metric scores
        score_clip = _norm_hi(clip_c, args.clip_ok, args.clip_good)
        score_dino = _norm_hi(dino_c, args.dino_ok, args.dino_good)
        score_edge = _norm_hi(edge_ms, args.edge_ok, args.edge_good)
        score_style = _norm_hi(clip_s, args.style_ok, args.style_good)

        score_lpips = _norm_lo(lpips_c, args.lpips_good, args.lpips_bad)
        score_fft = _norm_lo(fft_peak, fft_good, fft_bad)
        score_hf = _norm_lo(hf_energy, args.hf_good, args.hf_bad)
        score_color = _norm_band(
            color_dist,
            args.color_good_low,
            args.color_good_high,
            args.color_bad_low,
            args.color_bad_high,
        )

        content_score = _safe_avg([score_clip, score_dino, score_lpips])
        structure_score = _safe_avg([score_edge])
        style_score = _safe_avg([score_style, score_color])
        artifact_score = _safe_avg([score_fft, score_hf])

        scores = [content_score, structure_score, style_score, artifact_score]
        weights = [args.w_content, args.w_structure, args.w_style, args.w_artifact]
        valid = [(s, w) for s, w in zip(scores, weights) if s is not None]
        if valid:
            num = sum(s * w for s, w in valid)
            den = sum(w for _, w in valid) + 1e-6
            overall = float(num / den)
        else:
            overall = None

        grade = _grade_from_score(overall)

        flags = []
        if score_clip is not None and score_clip < 0.5:
            flags.append("clip_c_low")
        if score_dino is not None and score_dino < 0.5:
            flags.append("dino_low")
        if score_lpips is not None and score_lpips < 0.5:
            flags.append("lpips_high")
        if score_edge is not None and score_edge < 0.5:
            flags.append("edge_low")
        if score_fft is not None and score_fft < 0.5:
            flags.append("fft_peak_high")
        if score_hf is not None and score_hf < 0.5:
            flags.append("hf_high")
        if score_style is not None and score_style < 0.5:
            flags.append("style_clip_low")
        if score_color is not None and score_color < 0.5:
            flags.append("color_dist_high")

        rec_out = {
            "id": sample_id,
            "image": str(img_path),
            "metrics": {
                "clip_c": clip_c,
                "dino_c": dino_c,
                "lpips_c": lpips_c,
                "edge_ms": edge_ms,
                "edge_vals": edge_vals,
                "fft_peak": fft_peak,
                "hf_energy": hf_energy,
                "clip_s": clip_s,
                "color_dist": color_dist,
            },
            "scores": {
                "content": content_score,
                "structure": structure_score,
                "style": style_score,
                "artifact": artifact_score,
                "overall": overall,
            },
            "grade": grade,
            "flags": flags,
        }
        results.append(rec_out)

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

        # cleanup
        try:
            out_img.close()
            content_img.close()
            style_img.close()
        except Exception:
            pass
        del out_img, content_img, style_img, out_eval, content_eval, style_eval
        if args.gc_interval and (idx % args.gc_interval == 0):
            gc.collect()
        if args.empty_cache_interval and (idx % args.empty_cache_interval == 0):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # summary
    from collections import Counter, defaultdict
    grades = Counter(r["grade"] for r in results)
    agg = defaultdict(list)
    for r in results:
        for k, v in r["metrics"].items():
            if isinstance(v, list):
                continue
            if v is not None:
                agg[f"metrics.{k}"].append(v)
        for k, v in r["scores"].items():
            if v is not None:
                agg[f"scores.{k}"].append(v)

    def _summary(vals):
        if not vals:
            return None
        return {
            "min": float(min(vals)),
            "mean": float(sum(vals) / len(vals)),
            "max": float(max(vals)),
        }

    summary = {
        "count": len(results),
        "grades": dict(grades),
        "fft_thresholds": {
            "good": fft_good,
            "ok": fft_ok,
            "bad": fft_bad,
        },
        "stats": {k: _summary(v) for k, v in agg.items()},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
