#!/usr/bin/env python3
"""
Generate teacher targets with SDXL base + IP-Adapter.
Reads pseudo pair JSONL and writes outputs to teacher_target_path.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

from PIL import Image


def _parse_int_list(s):
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _resize_center_crop(img, size):
    if size is None:
        return img
    w, h = img.size
    if w == size and h == size:
        return img
    # scale short side to size
    if w < h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    # center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _resize_long_side(img, size):
    if size is None:
        return img
    w, h = img.size
    if max(w, h) == size:
        return img
    if w >= h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def _desaturate_hsv(img, factor=0.6):
    if factor >= 1.0:
        return img
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    s = s.point(lambda p: int(p * factor))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def _set_cache_env(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    # Use cache_dir directly to match existing Hugging Face cache layout
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache_dir / "diffusers"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/pseudo_pairs_train.jsonl")
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--ip-adapter-repo", default="h94/IP-Adapter")
    ap.add_argument("--ip-adapter-subfolder", default="sdxl_models")
    ap.add_argument("--ip-adapter-weight", default="ip-adapter_sdxl.bin")
    ap.add_argument("--ip-adapter-scale", type=float, default=0.22)
    ap.add_argument("--output-root", default=None, help="Optional override for outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--save-size", type=int, default=512, help="Optional output resize before saving (0 disables)")
    ap.add_argument("--square-crop", action="store_true", help="Force center-crop to square instead of keeping aspect")
    ap.add_argument("--seed-filter", default="0")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--empty-cache-interval", type=int, default=20)
    ap.add_argument("--gc-interval", type=int, default=20, help="Run gc.collect() every N samples to free RAM")
    ap.add_argument("--model-cache", default="/mnt/fast18/models_cache")
    ap.add_argument("--enable-xformers", action="store_true")
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--attention-slicing", action="store_true", help="Enable attention slicing (may conflict with IP-Adapter)")
    ap.add_argument(
        "--scheduler",
        choices=["ddim", "euler", "dpmpp_2m_karras"],
        default="dpmpp_2m_karras",
        help="Sampler for generation (default: dpmpp_2m_karras for clean outputs)",
    )
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--force-args", action="store_true", help="Ignore meta values and use CLI args")
    ap.add_argument("--use-vae-fp16-fix", action="store_true", help="Use vae_fp16_fix subfolder to avoid NaNs")
    ap.add_argument("--upcast-vae", action="store_true", help="Upcast VAE to fp32 to avoid NaNs (uses more VRAM)")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--strength", type=float, default=0.30)
    ap.add_argument("--prompt-prefix", default="high quality, clean, sharp, detailed, natural lighting, coherent")
    ap.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted, deformed, artifacts, oversaturated, washed out, text, watermark, logo",
    )
    ap.add_argument("--quality-filter", action="store_true", help="Filter bad images and retry")
    ap.add_argument("--max-retries", type=int, default=1, help="Max attempts per sample when quality filter is enabled")
    ap.add_argument("--retry-scale-decay", type=float, default=0.10, help="Reduce ip_adapter_scale per retry")
    ap.add_argument("--retry-strength-decay", type=float, default=0.05, help="Reduce strength per retry")
    ap.add_argument("--retry-cfg-decay", type=float, default=0.30, help="Reduce cfg per retry")
    ap.add_argument("--saturation-thresh", type=float, default=0.45, help="Reject if saturated pixel ratio exceeds this")
    ap.add_argument("--low-std-thresh", type=float, default=4.0, help="Reject if pixel std < threshold (near-constant)")
    ap.add_argument("--edge-sim-thresh", type=float, default=0.20, help="Reject if edge similarity below threshold")
    ap.add_argument("--edge-sim-hi", type=float, default=0.25, help="Edge sim high threshold at 128x128")
    ap.add_argument("--edge-sim-lo", type=float, default=0.25, help="Edge sim high threshold at 64x64")
    ap.add_argument("--edge-sim-min-hi", type=float, default=0.20, help="Edge sim minimum at 128x128")
    ap.add_argument("--edge-sim-min-lo", type=float, default=0.20, help="Edge sim minimum at 64x64")
    ap.add_argument("--edge-sim-both", action="store_true", default=True, help="Require both edge scales to pass")
    ap.add_argument("--edge-sim-any", action="store_true", help="Allow mixed edge sim (disable both-scale requirement)")
    ap.add_argument("--edge-sim-weight", type=float, default=0.5, help="Weight for 128x128 in mixed edge sim")
    ap.add_argument("--hf-ratio-max", type=float, default=2.5, help="Reject if high-frequency ratio exceeds this")
    ap.add_argument("--quality-records", default=None, help="Write per-attempt evaluation records to JSONL")
    ap.add_argument("--reject-dir", default=None, help="Optional directory to save rejected images")
    ap.add_argument("--color-over-high", type=float, default=220.0, help="Trigger color rescue if color_dist exceeds this")
    ap.add_argument("--color-over-mid", type=float, default=180.0, help="Trigger color rescue if color_dist exceeds this and clip is low")
    ap.add_argument("--color-over-clip", type=float, default=0.60, help="Clip sim cutoff for mid color rescue trigger")
    ap.add_argument("--color-rescue-strength", type=float, default=0.24)
    ap.add_argument("--color-rescue-ip-scale", type=float, default=0.16)
    ap.add_argument("--color-rescue-sat", type=float, default=0.60, help="Saturation scale for style image on rescue")
    ap.add_argument("--no-online-eval", action="store_true", help="Disable detailed online evaluation + grading")
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--eval-edge-sizes", default="64,128,256")
    ap.add_argument("--eval-clip-model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--eval-clip-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eval-dino-model", default="facebook/dinov2-base")
    ap.add_argument("--eval-dino-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--eval-lpips-net", default="vgg", choices=["vgg", "alex", "squeeze"])
    ap.add_argument("--eval-lpips-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-eval-dino", action="store_true")
    ap.add_argument("--no-eval-lpips", action="store_true")
    # evaluation thresholds (match evaluate_teacher_quality defaults)
    ap.add_argument("--eval-clip-ok", type=float, default=0.30)
    ap.add_argument("--eval-clip-good", type=float, default=0.34)
    ap.add_argument("--eval-dino-ok", type=float, default=0.24)
    ap.add_argument("--eval-dino-good", type=float, default=0.28)
    ap.add_argument("--eval-edge-ok", type=float, default=0.45)
    ap.add_argument("--eval-edge-good", type=float, default=0.55)
    ap.add_argument("--eval-style-ok", type=float, default=0.18)
    ap.add_argument("--eval-style-good", type=float, default=0.25)
    ap.add_argument("--eval-lpips-good", type=float, default=0.40)
    ap.add_argument("--eval-lpips-bad", type=float, default=0.60)
    ap.add_argument("--eval-fft-auto", action="store_true", default=True)
    ap.add_argument("--eval-fft-good", type=float, default=1.40)
    ap.add_argument("--eval-fft-bad", type=float, default=2.20)
    ap.add_argument("--eval-fft-good-q", type=float, default=0.80)
    ap.add_argument("--eval-fft-ok-q", type=float, default=0.90)
    ap.add_argument("--eval-fft-bad-q", type=float, default=0.97)
    ap.add_argument("--eval-fft-low-radius", type=float, default=0.05)
    ap.add_argument("--eval-hf-good", type=float, default=0.25)
    ap.add_argument("--eval-hf-bad", type=float, default=0.30)
    ap.add_argument("--eval-color-good-low", type=float, default=40.0)
    ap.add_argument("--eval-color-good-high", type=float, default=120.0)
    ap.add_argument("--eval-color-bad-low", type=float, default=20.0)
    ap.add_argument("--eval-color-bad-high", type=float, default=160.0)
    ap.add_argument("--w-content", type=float, default=0.4)
    ap.add_argument("--w-structure", type=float, default=0.2)
    ap.add_argument("--w-style", type=float, default=0.2)
    ap.add_argument("--w-artifact", type=float, default=0.2)
    ap.add_argument("--weight-floor", type=float, default=0.01)
    ap.add_argument("--content-filter", action="store_true", help="Enable CLIP content similarity filtering")
    ap.add_argument("--content-encoder", default="openai/clip-vit-base-patch16")
    ap.add_argument("--content-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--content-sim-thresh", type=float, default=0.30, help="Reject if CLIP sim < threshold (0 disables)")
    ap.add_argument("--content-sim-high", type=float, default=0.34, help="A-grade threshold for CLIP sim")
    ap.add_argument("--b-weight", type=float, default=0.4, help="Sample weight for B-grade outputs")
    ap.add_argument("--offline", action="store_true", help="Do not download; use local cache only")
    args = ap.parse_args()

    _set_cache_env(args.model_cache)
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch
    from diffusers import (
        StableDiffusionXLImg2ImgPipeline,
        AutoencoderKL,
        DDIMScheduler,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
    )

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    vae = None
    if args.use_vae_fp16_fix:
        try:
            vae = AutoencoderKL.from_pretrained(
                args.model_id,
                subfolder="vae_fp16_fix",
                torch_dtype=dtype,
                cache_dir=args.model_cache,
                local_files_only=args.offline,
            )
        except Exception:
            vae = None

    pipe_kwargs = dict(
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
        cache_dir=args.model_cache,
        local_files_only=args.offline,
    )
    if vae is not None:
        pipe_kwargs["vae"] = vae

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.model_id,
        **pipe_kwargs,
    )
    if args.upcast_vae:
        try:
            pipe.upcast_vae()
        except Exception:
            try:
                pipe.vae.to(dtype=torch.float32)
            except Exception:
                pass
    pipe = pipe.to(args.device)

    # memory controls
    pipe.enable_vae_tiling()
    if args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if args.attention_slicing:
        pipe.enable_attention_slicing()
    if args.low_vram:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

    # load ip-adapter weights
    pipe.load_ip_adapter(
        args.ip_adapter_repo,
        subfolder=args.ip_adapter_subfolder,
        weight_name=args.ip_adapter_weight,
        cache_dir=args.model_cache,
        local_files_only=args.offline,
    )
    try:
        pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    except Exception:
        pass

    seed_filter = _parse_int_list(args.seed_filter)

    out_count = 0
    processed = 0

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}", file=sys.stderr)
        sys.exit(1)

    def _edge_similarity(img_a, img_b, size=64):
        import numpy as np
        a = np.array(img_a.convert("L").resize((size, size)))
        b = np.array(img_b.convert("L").resize((size, size)))
        a = a.astype("float32")
        b = b.astype("float32")

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

    clip_model = None
    clip_processor = None
    need_clip = (args.content_filter and args.content_sim_thresh > 0) or (not args.no_online_eval)
    if need_clip:
        try:
            from transformers import CLIPModel, CLIPProcessor

            clip_model = CLIPModel.from_pretrained(
                args.eval_clip_model,
                local_files_only=args.offline,
                cache_dir=args.model_cache,
            )
            clip_processor = CLIPProcessor.from_pretrained(
                args.eval_clip_model,
                local_files_only=args.offline,
                cache_dir=args.model_cache,
            )
            clip_model = clip_model.to(args.eval_clip_device)
            clip_model.eval()
        except Exception:
            clip_model = None
            clip_processor = None

    def _clip_embed(img):
        if clip_model is None or clip_processor is None:
            return None
        inputs = clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(args.eval_clip_device) for k, v in inputs.items()}
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

    def _hf_ratio(img, content_img):
        import numpy as np
        a = np.array(img.convert("L").resize((128, 128))).astype("float32")
        b = np.array(content_img.convert("L").resize((128, 128))).astype("float32")

        def sobel_energy(x):
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
            return float((gx * gx + gy * gy).mean())

        ea = sobel_energy(a)
        eb = sobel_energy(b)
        return ea / (eb + 1e-6)

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
        hf_mask = rr >= 0.5
        power = (mag ** 2).astype("float32")
        hf_energy = float(power[hf_mask].sum() / (power.sum() + 1e-6))
        return peak_ratio, hf_energy

    def _edge_ms(img, content_img, sizes):
        vals = []
        for s in sizes:
            vals.append(_edge_similarity(img, content_img, size=s))
        return float(sum(vals) / len(vals)) if vals else None

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

    dino_model = dino_processor = None
    if not args.no_online_eval and not args.no_eval_dino:
        try:
            from transformers import AutoImageProcessor, AutoModel
            dino_model = AutoModel.from_pretrained(
                args.eval_dino_model,
                local_files_only=args.offline,
                cache_dir=args.model_cache,
            ).to(args.eval_dino_device)
            dino_processor = AutoImageProcessor.from_pretrained(
                args.eval_dino_model,
                local_files_only=args.offline,
                cache_dir=args.model_cache,
            )
            dino_model.eval()
        except Exception:
            dino_model = None
            dino_processor = None

    lpips_model = None
    if not args.no_online_eval and not args.no_eval_lpips:
        try:
            import lpips as lpips_lib
            lpips_model = lpips_lib.LPIPS(net=args.eval_lpips_net).to(args.eval_lpips_device)
            lpips_model.eval()
        except Exception:
            lpips_model = None

    def _dino_embed(img):
        if dino_model is None or dino_processor is None:
            return None
        inputs = dino_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(args.eval_dino_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = dino_model(**inputs)
        feats = out.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _lpips_dist(img_a, img_b):
        if lpips_model is None:
            return None
        import numpy as np
        import torch

        def to_tensor(img):
            arr = np.array(img).astype("float32") / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = t * 2.0 - 1.0
            return t

        ta = to_tensor(img_a).to(args.eval_lpips_device)
        tb = to_tensor(img_b).to(args.eval_lpips_device)
        with torch.no_grad():
            d = lpips_model(ta, tb)
        return float(d.item())

    eval_edge_sizes = [int(x) for x in args.eval_edge_sizes.split(",") if x.strip()]
    fft_vals = []

    def _color_dist(img, style_img):
        import numpy as np
        a = np.array(img).astype("float32")
        b = np.array(style_img).astype("float32")
        mu_a = a.reshape(-1, 3).mean(axis=0)
        mu_b = b.reshape(-1, 3).mean(axis=0)
        std_a = a.reshape(-1, 3).std(axis=0)
        std_b = b.reshape(-1, 3).std(axis=0)
        return float(((mu_a - mu_b) ** 2).sum() ** 0.5 + ((std_a - std_b) ** 2).sum() ** 0.5)

    def _quality_ok(img, content_img, style_img, content_feat=None):
        import numpy as np
        arr = np.array(img)
        if arr.size == 0:
            return False, "empty", None, {}
        if arr.max() == 0:
            return False, "all_black", None, {}
        if arr.std() < args.low_std_thresh:
            return False, "low_std", None, {}
        sat_ratio = ((arr <= 1) | (arr >= 254)).mean()
        if sat_ratio > args.saturation_thresh:
            return False, f"saturated:{sat_ratio:.3f}", None, {}
        edge_sim_64 = _edge_similarity(img, content_img, size=64)
        edge_sim_128 = _edge_similarity(img, content_img, size=128)
        if args.edge_sim_any:
            w = max(0.0, min(1.0, args.edge_sim_weight))
            edge_mix = w * edge_sim_128 + (1.0 - w) * edge_sim_64
            if edge_mix < args.edge_sim_thresh:
                return False, f"edge_sim:{edge_mix:.3f}", None, {}
        else:
            if edge_sim_64 < args.edge_sim_min_lo or edge_sim_128 < args.edge_sim_min_hi:
                return False, f"edge_sim:{edge_sim_64:.3f}/{edge_sim_128:.3f}", None, {}
        hf_ratio = _hf_ratio(img, content_img)
        if hf_ratio > args.hf_ratio_max:
            return False, f"hf_ratio:{hf_ratio:.3f}", None, {}
        if args.content_filter and args.content_sim_thresh > 0:
            if content_feat is None:
                content_feat = _clip_embed(content_img)
            if content_feat is not None:
                out_feat = _clip_embed(img)
                if out_feat is not None:
                    sim = float((content_feat @ out_feat.T).item())
                    if sim < args.content_sim_thresh:
                        return False, f"clip_sim:{sim:.3f}", None, {"clip_sim": sim}
                else:
                    sim = None
            else:
                sim = None
        else:
            sim = None
        color_dist = _color_dist(img, style_img)
        color_over = color_dist > args.color_over_high
        if sim is not None and color_dist > args.color_over_mid and sim < args.color_over_clip:
            color_over = True

        # grading
        if sim is None:
            sim = 1.0
        if sim >= args.content_sim_high and edge_sim_64 >= args.edge_sim_lo and edge_sim_128 >= args.edge_sim_hi:
            grade = "A"
            weight = 1.0
        else:
            grade = "B"
            weight = float(args.b_weight)
        if color_over:
            return False, f"color_over:{color_dist:.1f}", grade, {
                "edge_sim64": edge_sim_64,
                "edge_sim128": edge_sim_128,
                "hf_ratio": hf_ratio,
                "clip_sim": sim,
                "color_dist": color_dist,
                "weight": weight,
            }
        return True, "ok", grade, {
            "edge_sim64": edge_sim_64,
            "edge_sim128": edge_sim_128,
            "hf_ratio": hf_ratio,
            "clip_sim": sim,
            "color_dist": color_dist,
            "weight": weight,
        }

    def _evaluate_detailed(img, content_img, style_img, content_feat, style_feat, dino_content, dino_style):
        # resize for eval
        out_eval = _resize_center_crop(img, args.eval_size)
        content_eval = _resize_center_crop(content_img, args.eval_size)
        style_eval = _resize_center_crop(style_img, args.eval_size)

        # CLIP
        clip_c = clip_s = None
        if clip_model is not None:
            out_feat = _clip_embed(out_eval)
            if content_feat is not None and out_feat is not None:
                clip_c = float((content_feat @ out_feat.T).item())
            if style_feat is not None and out_feat is not None:
                clip_s = float((style_feat @ out_feat.T).item())

        # DINO
        dino_c = None
        if dino_model is not None:
            out_dino = _dino_embed(out_eval)
            if dino_content is not None and out_dino is not None:
                dino_c = float((dino_content @ out_dino.T).item())

        # LPIPS
        lpips_c = _lpips_dist(out_eval, content_eval)

        # Edge + FFT + HF
        edge_ms = _edge_ms(out_eval, content_eval, eval_edge_sizes)
        fft_peak, hf_energy = _fft_metrics(
            out_eval,
            size=args.eval_size,
            low_freq_radius=args.eval_fft_low_radius,
        )

        # update FFT thresholds if auto
        fft_good = args.eval_fft_good
        fft_bad = args.eval_fft_bad
        fft_ok = None
        if args.eval_fft_auto:
            fft_vals.append(fft_peak)
            if len(fft_vals) >= 5:
                import numpy as np
                fft_good = float(np.quantile(fft_vals, args.eval_fft_good_q))
                fft_ok = float(np.quantile(fft_vals, args.eval_fft_ok_q))
                fft_bad = float(np.quantile(fft_vals, args.eval_fft_bad_q))

        # Color distance
        color_dist = _color_dist(out_eval, style_eval)

        # scores
        score_clip = _norm_hi(clip_c, args.eval_clip_ok, args.eval_clip_good)
        score_dino = _norm_hi(dino_c, args.eval_dino_ok, args.eval_dino_good)
        score_edge = _norm_hi(edge_ms, args.eval_edge_ok, args.eval_edge_good)
        score_style = _norm_hi(clip_s, args.eval_style_ok, args.eval_style_good)
        score_lpips = _norm_lo(lpips_c, args.eval_lpips_good, args.eval_lpips_bad)
        score_fft = _norm_lo(fft_peak, fft_good, fft_bad)
        score_hf = _norm_lo(hf_energy, args.eval_hf_good, args.eval_hf_bad)
        score_color = _norm_band(
            color_dist,
            args.eval_color_good_low,
            args.eval_color_good_high,
            args.eval_color_bad_low,
            args.eval_color_bad_high,
        )

        def _safe_avg(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

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

        # flags
        flags = []
        if clip_c is not None and clip_c < args.eval_clip_ok:
            flags.append("clip_c_low")
        if dino_c is not None and dino_c < args.eval_dino_ok:
            flags.append("dino_low")
        if lpips_c is not None and lpips_c > args.eval_lpips_bad:
            flags.append("lpips_high")
        if edge_ms is not None and edge_ms < args.eval_edge_ok:
            flags.append("edge_low")
        if fft_peak > fft_bad:
            flags.append("fft_peak_high")
        if hf_energy > args.eval_hf_bad:
            flags.append("hf_high")
        if clip_s is not None and clip_s < args.eval_style_ok:
            flags.append("style_clip_low")
        if color_dist < args.eval_color_bad_low or color_dist > args.eval_color_bad_high:
            flags.append("color_dist_high")

        # color rescue trigger
        color_over = color_dist > args.color_over_high
        if clip_c is not None and color_dist > args.color_over_mid and clip_c < args.color_over_clip:
            color_over = True

        metrics = {
            "clip_c": clip_c,
            "dino_c": dino_c,
            "lpips_c": lpips_c,
            "edge_ms": edge_ms,
            "fft_peak": fft_peak,
            "hf_energy": hf_energy,
            "clip_s": clip_s,
            "color_dist": color_dist,
        }
        scores_dict = {
            "content": content_score,
            "structure": structure_score,
            "style": style_score,
            "artifact": artifact_score,
            "overall": overall,
        }

        return {
            "grade": grade,
            "metrics": metrics,
            "scores": scores_dict,
            "flags": flags,
            "fft_thresholds": {"good": fft_good, "ok": fft_ok, "bad": fft_bad},
            "color_over": color_over,
        }

    qc_log_path = Path("data/teacher_out/qc_overrides.jsonl")
    grade_log_path = Path("data/teacher_out/quality_grades.jsonl")
    quality_records_path = None

    if args.quality_records:
        quality_records_path = Path(args.quality_records)
    else:
        if args.output_root:
            quality_records_path = Path(args.output_root) / "quality_records.jsonl"
        else:
            quality_records_path = Path("data/teacher_out/quality_records.jsonl")
    try:
        quality_records_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    def _write_record(rec):
        if quality_records_path is None:
            return
        with quality_records_path.open("a", encoding="utf-8") as qf:
            qf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # write run config once
    try:
        run_config_path = quality_records_path.parent / "run_config.json"
        if not run_config_path.exists():
            git_commit = None
            try:
                import subprocess
                git_commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                )
            except Exception:
                git_commit = None
            run_cfg = {
                "git_commit": git_commit,
                "scheduler": args.scheduler,
                "steps": args.steps,
                "cfg": args.cfg,
                "strength": args.strength,
                "ip_scale": args.ip_adapter_scale,
                "image_size": args.image_size,
                "save_size": args.save_size,
                "prompt_prefix": args.prompt_prefix,
                "negative_prompt": args.negative_prompt,
                "retries": args.max_retries,
                "eval": {
                    "clip_ok": args.eval_clip_ok,
                    "clip_good": args.eval_clip_good,
                    "dino_ok": args.eval_dino_ok,
                    "dino_good": args.eval_dino_good,
                    "edge_ok": args.eval_edge_ok,
                    "edge_good": args.eval_edge_good,
                    "style_ok": args.eval_style_ok,
                    "style_good": args.eval_style_good,
                    "lpips_good": args.eval_lpips_good,
                    "lpips_bad": args.eval_lpips_bad,
                    "fft_auto": args.eval_fft_auto,
                    "fft_good": args.eval_fft_good,
                    "fft_bad": args.eval_fft_bad,
                    "fft_good_q": args.eval_fft_good_q,
                    "fft_ok_q": args.eval_fft_ok_q,
                    "fft_bad_q": args.eval_fft_bad_q,
                    "fft_low_radius": args.eval_fft_low_radius,
                    "hf_good": args.eval_hf_good,
                    "hf_bad": args.eval_hf_bad,
                    "color_good_low": args.eval_color_good_low,
                    "color_good_high": args.eval_color_good_high,
                    "color_bad_low": args.eval_color_bad_low,
                    "color_bad_high": args.eval_color_bad_high,
                },
            }
            with run_config_path.open("w", encoding="utf-8") as f:
                json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("meta", {})
            seed = meta.get("seed")
            if seed_filter is not None and seed not in seed_filter:
                continue

            out_path = Path(rec["teacher_target_path"])
            if args.output_root:
                out_path = Path(args.output_root) / out_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_path.exists():
                processed += 1
                continue

            content_path = Path(rec["content_path"])
            style_path = Path(rec["style_path"])
            if not content_path.exists() or not style_path.exists():
                continue

            prompt = rec.get("prompt", "")
            if args.prompt_prefix:
                if prompt:
                    prompt = f"{args.prompt_prefix}, {prompt}"
                else:
                    prompt = args.prompt_prefix

            if args.use_meta and not args.force_args:
                steps = meta.get("steps", args.steps)
                cfg = meta.get("cfg", args.cfg)
                strength = meta.get("strength", args.strength)
            else:
                steps = args.steps
                cfg = args.cfg
                strength = args.strength

            content = Image.open(content_path).convert("RGB")
            style = Image.open(style_path).convert("RGB")
            if args.square_crop:
                content = _resize_center_crop(content, args.image_size)
                style = _resize_center_crop(style, args.image_size)
            else:
                content = _resize_long_side(content, args.image_size)
                style = _resize_long_side(style, args.image_size)

            content_feat = None
            style_feat = None
            dino_content = None
            dino_style = None
            if args.content_filter and args.content_sim_thresh > 0:
                content_feat = _clip_embed(content)
            if not args.no_online_eval:
                content_feat = content_feat or _clip_embed(content)
                style_feat = _clip_embed(style)
                dino_content = _dino_embed(_resize_center_crop(content, args.eval_size))
                dino_style = _dino_embed(_resize_center_crop(style, args.eval_size))

            import contextlib
            autocast_ctx = (
                torch.autocast(args.device, dtype=dtype)
                if not args.upcast_vae
                else contextlib.nullcontext()
            )

            success = False
            color_rescue_next = False
            for attempt in range(max(args.max_retries, 1)):
                eff_seed = int(seed) if seed is not None else 0
                eff_seed = eff_seed + attempt
                eff_ip_scale = max(0.05, args.ip_adapter_scale - attempt * args.retry_scale_decay)
                eff_strength = max(0.05, strength - attempt * args.retry_strength_decay)
                eff_cfg = max(1.0, cfg - attempt * args.retry_cfg_decay)
                use_style = style
                if color_rescue_next:
                    eff_ip_scale = min(eff_ip_scale, args.color_rescue_ip_scale)
                    eff_strength = min(eff_strength, args.color_rescue_strength)
                    use_style = _desaturate_hsv(style, factor=args.color_rescue_sat)
                    color_rescue_next = False

                try:
                    pipe.set_ip_adapter_scale(eff_ip_scale)
                except Exception:
                    pass

                generator = torch.Generator(device=args.device).manual_seed(eff_seed)
                with torch.no_grad(), autocast_ctx:
                    result = pipe(
                        prompt=prompt,
                        image=content,
                        ip_adapter_image=use_style,
                        strength=eff_strength,
                        guidance_scale=eff_cfg,
                        num_inference_steps=steps,
                        generator=generator,
                        output_type="pil",
                        negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    )
                    image = result.images[0]

                grade = None
                metrics = {}
                scores = {}
                flags = []
                accepted = True
                reason = "accepted"
                if args.quality_filter:
                    if not args.no_online_eval:
                        eval_rec = _evaluate_detailed(
                            image,
                            content,
                            use_style,
                            content_feat,
                            style_feat,
                            dino_content,
                            dino_style,
                        )
                        grade = eval_rec["grade"]
                        metrics = eval_rec["metrics"]
                        scores = eval_rec["scores"]
                        flags = eval_rec["flags"]
                        if eval_rec["color_over"]:
                            accepted = False
                            reason = "color_over"
                            color_rescue_next = True
                        else:
                            accepted = grade in ("A+", "A", "B")
                            if not accepted:
                                reason = f"grade_{grade}"
                    else:
                        ok, reason, grade, metrics = _quality_ok(image, content, use_style, content_feat)
                        accepted = ok
                # write quality record per attempt
                gen_params = {
                    "steps": steps,
                    "cfg": eff_cfg,
                    "strength": eff_strength,
                    "ip_scale": eff_ip_scale,
                    "scheduler": args.scheduler,
                    "image_size": args.image_size,
                    "save_size": args.save_size,
                    "seed": eff_seed,
                    "attempt": attempt,
                }
                base_weight = 0.0
                if grade == "A+":
                    base_weight = 1.0
                elif grade == "A":
                    base_weight = 0.6
                elif grade == "B":
                    base_weight = 0.15
                weight = base_weight
                if base_weight > 0:
                    if "color_dist_high" in flags:
                        weight *= 0.6
                    if "lpips_high" in flags:
                        weight *= 0.5
                    if "dino_low" in flags:
                        weight *= 0.5
                    if "fft_peak_high" in flags:
                        weight *= 0.2
                    if sum(
                        1
                        for f in flags
                        if f
                        in ("color_dist_high", "lpips_high", "dino_low", "fft_peak_high")
                    ) >= 2:
                        weight *= 0.5
                    if weight < args.weight_floor:
                        weight = args.weight_floor
                record = {
                    "id": rec.get("id"),
                    "image": str(out_path),
                    "content_path": str(content_path),
                    "style_path": str(style_path),
                    "gen_params": gen_params,
                    "metrics": metrics,
                    "scores": scores,
                    "flags": flags,
                    "grade": grade,
                    "accepted": bool(accepted),
                    "reason": reason,
                    "weight": weight,
                }
                _write_record(record)

                if args.quality_filter and not accepted:
                    if attempt < max(args.max_retries, 1) - 1:
                        try:
                            del result, image
                        except Exception:
                            pass
                        continue
                    # last attempt failed
                    print(f"QC_FAIL {rec.get('id')} reason={reason}")
                    if args.reject_dir:
                        reject_path = Path(args.reject_dir) / out_path.name
                        reject_path.parent.mkdir(parents=True, exist_ok=True)
                        if args.save_size and args.save_size > 0:
                            image = _resize_long_side(image, args.save_size)
                        image.save(reject_path)
                    try:
                        del result, image
                    except Exception:
                        pass
                    break
                # save result
                if args.save_size and args.save_size > 0:
                    image = _resize_long_side(image, args.save_size)
                image.save(out_path)
                if args.quality_filter and attempt > 0:
                    qc_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with qc_log_path.open("a", encoding="utf-8") as qf:
                        qf.write(
                            json.dumps(
                                {
                                    "id": rec.get("id"),
                                    "orig_seed": seed,
                                    "used_seed": eff_seed,
                                    "used_ip_scale": eff_ip_scale,
                                    "used_strength": eff_strength,
                                    "used_cfg": eff_cfg,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                if args.quality_filter and grade is not None:
                    grade_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with grade_log_path.open("a", encoding="utf-8") as gf:
                        gf.write(
                            json.dumps(
                                {
                                    "id": rec.get("id"),
                                    "grade": grade,
                                    "metrics": metrics,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                success = True
                break

            if not success:
                processed += 1
                continue
            out_count += 1
            processed += 1

            # RAM cleanup
            try:
                content.close()
                style.close()
            except Exception:
                pass
            try:
                del result, image, content, style
            except Exception:
                pass
            try:
                del content_feat
            except Exception:
                pass
            if args.gc_interval and (out_count % args.gc_interval == 0):
                gc.collect()

            if args.empty_cache_interval and (out_count % args.empty_cache_interval == 0):
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            if args.max_samples and out_count >= args.max_samples:
                break

    print(f"Generated {out_count} images")


if __name__ == "__main__":
    main()
