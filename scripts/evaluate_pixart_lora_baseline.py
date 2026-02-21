#!/usr/bin/env python3
"""
Evaluate PixArt LoRA baseline on a test manifest with standard metrics:
  - LPIPS(content, gen)   (lower is better)
  - DINO similarity(content, gen)  (higher is better)
  - CLIP image similarity(style, gen) (higher is better)
  - CLIP text-image similarity(prompt, gen) (higher is better)
Generates images from prompts, compares against content/style targets, and writes JSONL + summary.
Config-first (same style as training script).
"""
import argparse
import gc
import json
import os
import random
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


def _load_eval_img(path, size):
    img = Image.open(path).convert("RGB")
    img = _resize_center_crop(img, size)
    return img


def _safe_avg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# NOTE: legacy manual scaling is disabled in favor of PEFT set_adapters(...)
def _set_lora_scale(model, scale: float):
    return 0


def _set_peft_adapter_scale(model, scale: float):
    if model is None:
        return {"method": "none", "updated": 0}
    adapter = getattr(model, "active_adapter", "default") or "default"
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0] if adapter else "default"
    scale = float(scale)
    # Newer PEFT API
    if hasattr(model, "set_adapters"):
        model.set_adapters([adapter], adapter_weights=[scale])
        return {"method": "set_adapters", "updated": 0}
    # Fallback for older PEFT: set active adapter, then set scale on LoRA layers
    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter(adapter)
        except Exception:
            pass
        updated = 0
        for m in model.modules():
            if hasattr(m, "set_scale"):
                try:
                    m.set_scale(adapter, scale)
                    updated += 1
                except Exception:
                    continue
        return {"method": "set_scale", "updated": updated}
    raise RuntimeError("PEFT adapter scaling not available on this model")


def _lora_scale_schedule(step_idx: int, num_steps: int, target: float = 0.5) -> float:
    # 40%-80% ramp schedule: off -> linear ramp -> hold
    a = int(0.4 * num_steps)
    b = int(0.8 * num_steps)
    if step_idx < a:
        return 0.0
    if step_idx < b:
        denom = max(1, (b - a - 1))
        return target * (step_idx - a) / denom
    return target


def main():
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument("--config", default=None)
    cfg_args, _ = mini.parse_known_args()
    cfg = {}
    if cfg_args.config:
        with Path(cfg_args.config).open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    def cget(key, default):
        return cfg.get(key, default)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to JSON config")
    ap.add_argument("--test", default=cget("test", "data/manifests/test_manifest_10.jsonl"))
    ap.add_argument("--model-id", default=cget("model_id", "PixArt-alpha/PixArt-XL-2-512x512"))
    ap.add_argument("--model-cache", default=cget("model_cache", "/mnt/fast18/models_cache"))
    ap.add_argument("--offline", action="store_true", default=cget("offline", True))
    ap.add_argument("--lora", default=cget("lora", None))
    ap.add_argument("--lora-scale", type=float, default=cget("lora_scale", 1.0))
    ap.add_argument("--lora-schedule", action="store_true", default=cget("lora_schedule", False))
    ap.add_argument("--lora-schedule-target", type=float, default=cget("lora_schedule_target", 0.5))
    ap.add_argument("--output-dir", default=cget("output_dir", "outputs/pixart_lora_eval"))
    ap.add_argument("--output-jsonl", default=cget("output_jsonl", "outputs/pixart_lora_eval/eval_test_detailed.jsonl"))
    ap.add_argument("--summary-json", default=cget("summary_json", "outputs/pixart_lora_eval/eval_test_summary.json"))
    ap.add_argument("--save-images", action="store_true", default=cget("save_images", False))
    ap.add_argument("--image-dir", default=cget("image_dir", "outputs/pixart_lora_eval/images"))
    ap.add_argument("--self-check", action="store_true", default=cget("self_check", False))
    ap.add_argument("--self-check-samples", type=int, default=cget("self_check_samples", 1))
    ap.add_argument("--self-check-save", action="store_true", default=cget("self_check_save", False))
    ap.add_argument("--sanity-check", action="store_true", default=cget("sanity_check", False))
    ap.add_argument("--vae-fp32", action="store_true", default=cget("vae_fp32", False))
    ap.add_argument("--force-fp32", action="store_true", default=cget("force_fp32", False))
    ap.add_argument("--generate-only", action="store_true", default=cget("generate_only", False))
    ap.add_argument("--eval-only", action="store_true", default=cget("eval_only", False))
    ap.add_argument("--max-samples", type=int, default=cget("max_samples", 0))

    # generation
    ap.add_argument("--height", type=int, default=cget("height", 512))
    ap.add_argument("--width", type=int, default=cget("width", 512))
    ap.add_argument("--num-steps", type=int, default=cget("num_steps", 30))
    ap.add_argument("--guidance-scale", type=float, default=cget("guidance_scale", 4.5))
    ap.add_argument("--seed", type=int, default=cget("seed", 42))
    ap.add_argument("--per-sample-seed", action="store_true", default=cget("per_sample_seed", True))
    ap.add_argument("--scheduler", default=cget("scheduler", "dpmpp_2m_karras"))
    ap.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=cget("mixed_precision", "fp16"))

    # eval
    ap.add_argument("--eval-size", type=int, default=cget("eval_size", 256))
    ap.add_argument("--clip-model", default=cget("clip_model", "openai/clip-vit-base-patch16"))
    ap.add_argument("--clip-device", default=cget("clip_device", "cuda"))
    ap.add_argument("--dino-model", default=cget("dino_model", "facebook/dinov2-base"))
    ap.add_argument("--dino-device", default=cget("dino_device", "cuda"))
    ap.add_argument("--lpips-net", default=cget("lpips_net", "vgg"))
    ap.add_argument("--lpips-device", default=cget("lpips_device", "cuda"))
    ap.add_argument("--no-clip", action="store_true", default=cget("no_clip", False))
    ap.add_argument("--no-dino", action="store_true", default=cget("no_dino", False))
    ap.add_argument("--no-lpips", action="store_true", default=cget("no_lpips", False))
    ap.add_argument("--prompt-fallback", default=cget("prompt_fallback", "a photo"))

    ap.add_argument("--gc-interval", type=int, default=cget("gc_interval", 20))
    ap.add_argument("--empty-cache-interval", type=int, default=cget("empty_cache_interval", 20))
    ap.add_argument("--no-tqdm", action="store_true", default=cget("no_tqdm", False))

    args = ap.parse_args()
    if args.generate_only and args.eval_only:
        raise ValueError("Only one of --generate-only / --eval-only can be set.")

    _set_cache_env(args.model_cache)
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import (
        PixArtAlphaPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
    )
    from peft import PeftModel
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    if device_type == "cpu":
        dtype = torch.float32
    if args.force_fp32:
        dtype = torch.float32

    pipe = None
    if not args.eval_only:
        print(f"[init] device={device} dtype={dtype} scheduler={args.scheduler} steps={args.num_steps} cfg={args.guidance_scale}")
        pipe = PixArtAlphaPipeline.from_pretrained(
            args.model_id,
            cache_dir=args.model_cache,
            local_files_only=args.offline,
            torch_dtype=dtype,
        )
        if args.scheduler:
            sch = args.scheduler.lower()
            if sch in ("dpmpp_2m_karras", "dpmpp-karras"):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
                )
            elif sch in ("dpmpp_2m", "dpmpp"):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=False, algorithm_type="dpmsolver++"
                )
            elif sch == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            elif sch == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            elif sch in ("euler_a", "euler-ancestral"):
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        if args.vae_fp32:
            pipe.vae.to(device, dtype=torch.float32)
            print("[init] vae set to fp32")
        if args.force_fp32:
            pipe = pipe.to(device, dtype=torch.float32)
            print("[init] full fp32 enabled")

    # load evaluation models (skip if generate-only)
    clip_model = clip_processor = None
    if not args.no_clip and not args.generate_only:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(
            args.clip_model, local_files_only=args.offline, cache_dir=args.model_cache
        ).to(args.clip_device)
        if args.force_fp32:
            clip_model = clip_model.to(args.clip_device, dtype=torch.float32)
        clip_processor = CLIPProcessor.from_pretrained(
            args.clip_model, local_files_only=args.offline, cache_dir=args.model_cache
        )
        clip_model.eval()

    dino_model = dino_processor = None
    if not args.no_dino and not args.generate_only:
        from transformers import AutoImageProcessor, AutoModel
        dino_model = AutoModel.from_pretrained(
            args.dino_model, local_files_only=args.offline, cache_dir=args.model_cache
        ).to(args.dino_device)
        if args.force_fp32:
            dino_model = dino_model.to(args.dino_device, dtype=torch.float32)
        dino_processor = AutoImageProcessor.from_pretrained(
            args.dino_model, local_files_only=args.offline, cache_dir=args.model_cache
        )
        dino_model.eval()

    lpips_model = None
    if not args.no_lpips and not args.generate_only:
        try:
            import lpips as lpips_lib
            lpips_model = lpips_lib.LPIPS(net=args.lpips_net).to(args.lpips_device)
            if args.force_fp32:
                lpips_model = lpips_model.to(args.lpips_device, dtype=torch.float32)
            lpips_model.eval()
        except Exception as e:
            print(f"[warn] LPIPS unavailable: {e}")
            lpips_model = None

    def clip_img_embed(img):
        if clip_model is None:
            return None
        if not isinstance(img, Image.Image) or img.mode != "RGB":
            raise ValueError("CLIP expects PIL RGB image")
        inputs = clip_processor(images=img, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise KeyError(f"CLIP inputs missing pixel_values: {list(inputs.keys())}")
        inputs = {k: v.to(args.clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)  # [B, D] or wrapper
        if hasattr(feats, "image_embeds"):
            feats = feats.image_embeds
        elif hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        return F.normalize(feats, dim=-1)

    def clip_text_embed(text):
        if clip_model is None:
            return None
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(args.clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)  # [B, D] or wrapper
        if hasattr(feats, "text_embeds"):
            feats = feats.text_embeds
        elif hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        return F.normalize(feats, dim=-1)

    def dino_embed(img):
        if dino_model is None:
            return None
        if not isinstance(img, Image.Image) or img.mode != "RGB":
            raise ValueError("DINO expects PIL RGB image")
        inputs = dino_processor(images=img, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise KeyError(f"DINO inputs missing pixel_values: {list(inputs.keys())}")
        inputs = {k: v.to(args.dino_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = dino_model(**inputs)
        hs = out.last_hidden_state  # [B, 1+N, D] for ViT
        if hs.ndim == 3 and hs.shape[1] >= 1:
            feats = hs[:, 0]
        else:
            feats = hs.mean(dim=1)
        return F.normalize(feats, dim=-1)

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

    # load test manifest
    samples = []
    with Path(args.test).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            samples.append(rec)
    if args.max_samples:
        samples = samples[: args.max_samples]

    # self-check: compare base vs LoRA on a few samples
    if args.self_check and args.lora and not args.eval_only:
        check_dir = Path(args.output_dir) / "self_check"
        if args.self_check_save:
            check_dir.mkdir(parents=True, exist_ok=True)
        n_check = max(1, args.self_check_samples)
        check_samples = samples[: min(n_check, len(samples))]
        print(f"[self-check] running on {len(check_samples)} sample(s)")
        base_outputs = []
        for i, rec in enumerate(check_samples):
            prompt = rec.get("prompt") or args.prompt_fallback
            sample_id = rec.get("id", f"sample_{i:06d}")
            seed = args.seed + i if args.per_sample_seed else args.seed
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=(device_type == "cuda" and dtype != torch.float32)):
                base_img = pipe(
                    prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                ).images[0]
            base_outputs.append((sample_id, prompt, seed, base_img))

        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, Path(args.lora), is_trainable=False)
        if args.lora_scale is not None and not args.lora_schedule:
            info = _set_peft_adapter_scale(pipe.transformer, float(args.lora_scale))
            print(f"[init] lora_scale={args.lora_scale} applied via {info['method']} updated={info['updated']}")
        for sample_id, prompt, seed, base_img in base_outputs:
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=(device_type == "cuda" and dtype != torch.float32)):
                lora_img = pipe(
                    prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                ).images[0]
            arr_b = np.array(base_img).astype("float32")
            arr_l = np.array(lora_img).astype("float32")
            mae = float(np.mean(np.abs(arr_b - arr_l)))
            print(f"[self-check] {sample_id} mae={mae:.3f}")
            if args.self_check_save:
                base_img.save(check_dir / f"{sample_id}_base.png")
                lora_img.save(check_dir / f"{sample_id}_lora.png")
        print(f"[self-check] done; LoRA is now loaded from {args.lora}")
    elif args.lora and not args.eval_only:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, Path(args.lora), is_trainable=False)
        if args.lora_scale is not None and not args.lora_schedule:
            info = _set_peft_adapter_scale(pipe.transformer, float(args.lora_scale))
            print(f"[init] lora_scale={args.lora_scale} applied via {info['method']} updated={info['updated']}")
        print(f"[init] loaded lora: {args.lora}")

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        Path(args.image_dir).mkdir(parents=True, exist_ok=True)

    if tqdm is None or args.no_tqdm:
        pbar = None
    else:
        pbar = tqdm(total=len(samples), dynamic_ncols=True, desc="eval")

    metric_sums = {}
    metric_counts = {}
    metric_sums_sq = {}

    def _cos(a, b):
        return float((a * b).sum(dim=-1).item())

    processed = 0
    sanity_results = {}
    for idx, rec in enumerate(samples, 1):
        prompt = rec.get("prompt") or args.prompt_fallback
        teacher_path = rec.get("teacher_path")
        content_path = rec.get("content_path")
        style_path = rec.get("style_path")
        if teacher_path is None:
            continue
        sample_id = rec.get("id", f"sample_{idx:06d}")
        out_img = None
        out_path = None
        if args.eval_only:
            out_path = Path(args.image_dir) / f"{sample_id}.png"
            if not out_path.exists():
                print(f"[warn] missing image for {sample_id}: {out_path}")
                if pbar:
                    pbar.update(1)
                continue
            out_img = Image.open(out_path).convert("RGB")
        else:
            seed = args.seed + idx if args.per_sample_seed else args.seed
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=(device_type == "cuda" and dtype != torch.float32)):
                if args.lora_schedule and args.lora:
                    # ensure starting scale is 0 before schedule ramp
                    _set_peft_adapter_scale(pipe.transformer, 0.0)
                    def _cb(step_idx, timestep, latents):
                        s = _lora_scale_schedule(step_idx, args.num_steps, target=args.lora_schedule_target)
                        _set_peft_adapter_scale(pipe.transformer, s)
                        if step_idx in (0, 5, 10, 12, 15, 20, 24, args.num_steps - 1):
                            print(f"[sched] step={step_idx:02d}/{args.num_steps} scale={s:.3f}")
                    out_img = pipe(
                        prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        generator=gen,
                        callback=_cb,
                        callback_steps=1,
                    ).images[0]
                else:
                    out_img = pipe(
                        prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        generator=gen,
                    ).images[0]
            if args.save_images:
                out_path = Path(args.image_dir) / f"{sample_id}.png"
                out_img.save(out_path)
                out_path = str(out_path)

        if args.generate_only:
            if pbar:
                pbar.update(1)
            if args.gc_interval and idx % args.gc_interval == 0:
                gc.collect()
            if args.empty_cache_interval and idx % args.empty_cache_interval == 0 and device_type == "cuda":
                torch.cuda.empty_cache()
            try:
                if isinstance(out_img, Image.Image):
                    out_img.close()
            except Exception:
                pass
            continue

        teacher_img = _load_eval_img(teacher_path, args.eval_size)
        if content_path:
            try:
                content_img = _load_eval_img(content_path, args.eval_size)
            except Exception:
                content_img = teacher_img
        else:
            content_img = teacher_img
        if style_path:
            try:
                style_img = _load_eval_img(style_path, args.eval_size)
            except Exception:
                style_img = None
        else:
            style_img = None
        out_eval = _resize_center_crop(out_img, args.eval_size)
        teacher_eval = teacher_img
        content_eval = content_img
        style_eval = style_img

        # CLIP image-image + image-text
        clip_c = clip_t = clip_style = clip_gt = None
        teacher_style = None
        if clip_model is not None:
            of = clip_img_embed(out_eval)
            cf = clip_img_embed(content_eval)
            tf = clip_text_embed(prompt)
            sf = clip_img_embed(style_eval) if style_eval is not None else None
            rf = clip_img_embed(teacher_eval)
            if of is not None and cf is not None:
                clip_c = float((of @ cf.T).item())
            if of is not None and sf is not None:
                clip_style = float((of @ sf.T).item())
            if of is not None and tf is not None:
                clip_t = float((of @ tf.T).item())
            if of is not None and rf is not None:
                clip_gt = float((of @ rf.T).item())
            if rf is not None and sf is not None:
                teacher_style = float((rf @ sf.T).item())

        # DINO (content similarity)
        dino_c = None
        dino_gt = None
        teacher_content_dino = None
        if dino_model is not None:
            dfo = dino_embed(out_eval)
            dfc = dino_embed(content_eval)
            if dfo is not None and dfc is not None:
                dino_c = float((dfo @ dfc.T).item())
            dt = dino_embed(teacher_eval)
            if dfo is not None and dt is not None:
                dino_gt = float((dfo @ dt.T).item())
            if dt is not None and dfc is not None:
                teacher_content_dino = float((dt @ dfc.T).item())

        # LPIPS (content similarity)
        lpips_c = lpips_dist(out_eval, content_eval)
        teacher_lpips_c = lpips_dist(teacher_eval, content_eval)
        lpips_gt = lpips_dist(out_eval, teacher_eval)

        rec_out = {
            "id": sample_id,
            "prompt": prompt,
            "teacher_path": teacher_path,
            "content_path": content_path,
            "style_path": style_path,
            "image_path": out_path,
            "metrics": {
                "lpips_c": lpips_c,
                "lpips_gt": lpips_gt,
                "dino_c": dino_c,
                "dino_gt": dino_gt,
                "clip_c": clip_c,
                "clip_gt": clip_gt,
                "clip_t": clip_t,
                "clip_style": clip_style,
                "teacher_style": teacher_style,
            },
            "sanity": {
                "teacher_style": teacher_style,
                "teacher_content_clip": float((rf @ cf.T).item()) if (clip_model is not None and rf is not None and cf is not None) else None,
                "teacher_content_dino": teacher_content_dino,
                "teacher_lpips_c": teacher_lpips_c,
            } if args.sanity_check else None,
        }

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

        # aggregate metrics
        for k, v in rec_out["metrics"].items():
            if v is None:
                continue
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metric_counts[k] = metric_counts.get(k, 0) + 1
            metric_sums_sq[k] = metric_sums_sq.get(k, 0.0) + float(v) * float(v)

        processed += 1
        if pbar:
            pbar.update(1)

        if args.gc_interval and idx % args.gc_interval == 0:
            gc.collect()
        if args.empty_cache_interval and idx % args.empty_cache_interval == 0 and device_type == "cuda":
            torch.cuda.empty_cache()

        try:
            teacher_img.close()
        except Exception:
            pass
        try:
            if content_img is not teacher_img:
                content_img.close()
        except Exception:
            pass
        try:
            if style_img is not None:
                style_img.close()
        except Exception:
            pass

    if pbar:
        pbar.close()

    if args.generate_only:
        print(f"[done] generated images to {args.image_dir}")
        return

    metrics_mean = {k: metric_sums[k] / max(1, metric_counts[k]) for k in metric_sums}
    metrics_std = {}
    for k in metric_sums:
        n = max(1, metric_counts.get(k, 0))
        mean = metrics_mean[k]
        mean_sq = metric_sums_sq.get(k, 0.0) / n
        var = max(0.0, mean_sq - mean * mean)
        metrics_std[k] = var ** 0.5
    summary = {
        "count": processed,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "sanity": sanity_results if sanity_results else None,
        "config": {
            "model_id": args.model_id,
            "lora": args.lora,
            "lora_scale": args.lora_scale,
            "lora_schedule": args.lora_schedule,
            "lora_schedule_target": args.lora_schedule_target,
            "steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "scheduler": args.scheduler,
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {out_jsonl} and {summary_json}")


if __name__ == "__main__":
    main()
