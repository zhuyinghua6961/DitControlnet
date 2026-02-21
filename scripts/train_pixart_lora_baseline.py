#!/usr/bin/env python3
"""
Baseline LoRA fine-tuning for PixArt-XL-2-512 using teacher targets.
Reads train/val manifests (jsonl) with fields:
  - content_path (unused for baseline)
  - prompt
  - teacher_path (target image)
This baseline is text+teacher target only (content image not used).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def _set_cache_env(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))


def _resize_and_crop(img, short_side, crop_size, random_crop=False, rng=None):
    w, h = img.size
    target_short = max(short_side, crop_size)
    scale = target_short / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    if random_crop:
        if rng is None:
            rng = random
        left = rng.randint(0, max(0, new_w - crop_size))
        top = rng.randint(0, max(0, new_h - crop_size))
    else:
        left = max(0, (new_w - crop_size) // 2)
        top = max(0, (new_h - crop_size) // 2)
    return img.crop((left, top, left + crop_size, top + crop_size))


class TeacherDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        resolution=512,
        short_side=576,
        random_crop=False,
        seed=42,
        prompt_fallback="a photo",
    ):
        self.items = []
        self.resolution = resolution
        self.short_side = short_side
        self.random_crop = random_crop
        self.rng = random.Random(seed)
        self.prompt_fallback = prompt_fallback
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.items.append(rec)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        prompt = rec.get("prompt") or self.prompt_fallback
        teacher_path = rec.get("teacher_path")
        style_path = rec.get("style_path")
        if teacher_path is None:
            raise ValueError("teacher_path missing in manifest")
        img = Image.open(teacher_path).convert("RGB")
        img = _resize_and_crop(
            img,
            short_side=self.short_side,
            crop_size=self.resolution,
            random_crop=self.random_crop,
            rng=self.rng,
        )
        return {"image": img, "prompt": prompt, "style_path": style_path}


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
        "style_paths": [b.get("style_path") for b in batch],
    }


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
    ap.add_argument("--train", default=cget("train", "data/manifests/train_manifest_80.jsonl"))
    ap.add_argument("--val", default=cget("val", "data/manifests/val_manifest_10.jsonl"))
    ap.add_argument("--model-id", default=cget("model_id", "PixArt-alpha/PixArt-XL-2-512x512"))
    ap.add_argument("--model-cache", default=cget("model_cache", "/mnt/fast18/models_cache"))
    ap.add_argument("--output-dir", default=cget("output_dir", "outputs/pixart_lora_baseline"))
    ap.add_argument("--tensorboard-dir", default=cget("tensorboard_dir", "outputs/pixart_lora_baseline/tb"))
    ap.add_argument("--offline", action="store_true", default=cget("offline", True))
    ap.add_argument("--resolution", type=int, default=cget("resolution", 512))
    ap.add_argument("--short-side-train", type=int, default=cget("short_side_train", 576))
    ap.add_argument("--short-side-eval", type=int, default=cget("short_side_eval", 576))
    ap.add_argument("--random-crop", action="store_true", default=cget("random_crop", False))
    ap.add_argument("--batch-size", type=int, default=cget("batch_size", 1))
    ap.add_argument("--grad-accum", type=int, default=cget("grad_accum", 8))
    ap.add_argument("--lr", type=float, default=cget("lr", 1e-4))
    ap.add_argument("--max-steps", type=int, default=cget("max_steps", 10000))
    ap.add_argument("--lr-scheduler", choices=["none", "cosine", "linear"], default=cget("lr_scheduler", "cosine"))
    ap.add_argument("--warmup-steps", type=int, default=cget("warmup_steps", 0))
    ap.add_argument("--save-every", type=int, default=cget("save_every", 500))
    ap.add_argument("--val-every", type=int, default=cget("val_every", 500))
    ap.add_argument("--val-max-batches", type=int, default=cget("val_max_batches", 0))
    ap.add_argument("--val-log-every", type=int, default=cget("val_log_every", 50))
    ap.add_argument("--log-every", type=int, default=cget("log_every", 50))
    ap.add_argument("--loss-ema-decay", type=float, default=cget("loss_ema_decay", 0.95))
    ap.add_argument(
        "--tbin-edges",
        default=cget("tbin_edges", "0.33,0.66"),
        help="Comma-separated edges in [0,1] for timestep buckets; empty to disable.",
    )
    ap.add_argument("--tbin-disable", action="store_true", default=cget("tbin_disable", False))
    ap.add_argument("--style-loss", action="store_true", default=cget("style_loss", False))
    ap.add_argument("--style-loss-weight", type=float, default=cget("style_loss_weight", 0.01))
    ap.add_argument("--style-loss-every", type=int, default=cget("style_loss_every", 20))
    ap.add_argument("--style-loss-size", type=int, default=cget("style_loss_size", 256))
    ap.add_argument("--style-loss-clip-model", default=cget("style_loss_clip_model", "openai/clip-vit-base-patch16"))
    ap.add_argument("--style-loss-clip-device", default=cget("style_loss_clip_device", "cuda"))
    ap.add_argument("--seed", type=int, default=cget("seed", 42))
    ap.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=cget("mixed_precision", "fp16"))
    ap.add_argument("--lora-r", type=int, default=cget("lora_r", 8))
    ap.add_argument("--lora-alpha", type=int, default=cget("lora_alpha", 8))
    ap.add_argument("--lora-dropout", type=float, default=cget("lora_dropout", 0.0))
    ap.add_argument("--max-grad-norm", type=float, default=cget("max_grad_norm", 1.0))
    ap.add_argument("--num-workers", type=int, default=cget("num_workers", 4))
    ap.add_argument("--prompt-fallback", default=cget("prompt_fallback", "a photo"))
    ap.add_argument("--resume", default=cget("resume", None), help="Path to checkpoint dir or 'auto'")
    ap.add_argument("--early-stop-patience", type=int, default=cget("early_stop_patience", 5))
    ap.add_argument("--early-stop-min-delta", type=float, default=cget("early_stop_min_delta", 0.0))
    ap.add_argument("--no-tqdm", action="store_true", default=cget("no_tqdm", False))
    args = ap.parse_args()

    _set_cache_env(args.model_cache)
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # Disable torch.compile/Inductor in restricted environments
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    try:
        import torch._dynamo
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    torch.manual_seed(args.seed)

    from diffusers import PixArtAlphaPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model, PeftModel
    from torch.utils.tensorboard import SummaryWriter
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    if device_type == "cpu" and args.mixed_precision != "no":
        print("[warn] CPU detected; forcing fp32 for stability.")
        dtype = torch.float32

    print(f"[init] device={device} mixed_precision={args.mixed_precision} dtype={dtype} offline={args.offline}")
    if device_type == "cuda":
        try:
            print(f"[init] cuda device count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[init] cuda info unavailable: {e}")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache,
        local_files_only=args.offline,
        torch_dtype=dtype,
    )
    print("[init] pipeline loaded")
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    transformer.train()
    vae.eval()
    text_encoder.eval()
    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    clip_model = clip_image_size = None
    if args.style_loss:
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained(
            args.style_loss_clip_model,
            cache_dir=args.model_cache,
            local_files_only=args.offline,
        )
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model = clip_model.to(device, dtype=torch.float32)
        clip_image_size = getattr(clip_model.config.vision_config, "image_size", 224)

    def _clip_preprocess_tensor(img, size):
        # img: [-1, 1], shape [B,3,H,W]
        img = (img + 1.0) * 0.5
        img = img.clamp(0.0, 1.0)
        img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device)[None, :, None, None]
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device)[None, :, None, None]
        return (img - mean) / std

    def _load_style_tensor(path, size):
        if path is None:
            return None
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None
        img = _resize_and_crop(img, short_side=size, crop_size=size, random_crop=False)
        arr = np.array(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        t = t * 2.0 - 1.0
        return t

    train_ds = TeacherDataset(
        args.train,
        resolution=args.resolution,
        short_side=args.short_side_train,
        random_crop=args.random_crop,
        seed=args.seed,
        prompt_fallback=args.prompt_fallback,
    )
    val_ds = TeacherDataset(
        args.val,
        resolution=args.resolution,
        short_side=args.short_side_eval,
        random_crop=False,
        seed=args.seed,
        prompt_fallback=args.prompt_fallback,
    )
    print(f"[data] train={args.train} val={args.val} batch={args.batch_size} workers={args.num_workers}")
    print(f"[data] resolution={args.resolution} short_side_train={args.short_side_train} short_side_eval={args.short_side_eval} random_crop={args.random_crop}")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device_type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device_type == "cuda"),
    )

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device_type == "cuda" and args.mixed_precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = None
    if use_amp and args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    def _parse_edges(v):
        if v is None:
            return []
        if isinstance(v, list):
            edges = [float(x) for x in v]
        else:
            s = str(v).strip()
            if not s:
                return []
            edges = [float(x) for x in s.split(",") if x.strip()]
        edges = [min(max(e, 0.0), 1.0) for e in edges]
        return sorted(edges)

    tbin_edges = [] if args.tbin_disable else _parse_edges(args.tbin_edges)
    tbin_names = None
    tbin_edges_t = None
    if tbin_edges:
        if len(tbin_edges) == 2:
            tbin_names = ["low", "mid", "high"]
        else:
            tbin_names = [f"bin{i}" for i in range(len(tbin_edges) + 1)]
        tbin_edges_t = torch.tensor(tbin_edges, device=device, dtype=torch.float32)

    def _lr_lambda(step):
        if args.lr_scheduler == "none":
            return 1.0
        if args.warmup_steps and step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if args.lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if args.lr_scheduler == "linear":
            return 1.0 - progress
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    def _save_checkpoint(step, is_best=False):
        save_path = ckpt_dir / f"step_{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(save_path)
        state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_state": random.getstate(),
            "torch_state": torch.get_rng_state().tolist(),
            "best_val": best_val,
            "patience": patience_counter,
        }
        torch.save(state, save_path / "training_state.pt")
        if is_best:
            best_path = output_dir / "best"
            best_path.mkdir(parents=True, exist_ok=True)
            transformer.save_pretrained(best_path)
            torch.save(state, best_path / "training_state.pt")

    # resume
    global_step = 0
    best_val = float("inf")
    patience_counter = 0
    if args.resume:
        resume_path = None
        if args.resume == "auto":
            ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else -1)
            if ckpts:
                resume_path = ckpts[-1]
        else:
            resume_path = Path(args.resume)
        if resume_path and resume_path.exists():
            try:
                transformer = PeftModel.from_pretrained(transformer, resume_path, is_trainable=True)
                state = torch.load(resume_path / "training_state.pt", map_location="cpu")
                optimizer.load_state_dict(state.get("optimizer", {}))
                if scaler is not None and state.get("scaler"):
                    scaler.load_state_dict(state.get("scaler"))
                global_step = int(state.get("step", 0))
                best_val = float(state.get("best_val", best_val))
                patience_counter = int(state.get("patience", 0))
                if "rng_state" in state:
                    random.setstate(state["rng_state"])
                if "torch_state" in state:
                    torch.set_rng_state(torch.tensor(state["torch_state"], dtype=torch.uint8))
                transformer = transformer.to(device)
                print(f"Resumed from {resume_path} at step {global_step}")
            except Exception as e:
                print(f"[warn] resume failed: {e}")

    total_steps = args.max_steps
    if tqdm is None or args.no_tqdm:
        pbar = None
    else:
        pbar = tqdm(total=total_steps, initial=global_step, dynamic_ncols=True, desc="train")

    accum_steps = 0
    loss_accum = 0.0
    loss_ema = None
    tbin_sums = [0.0] * (len(tbin_edges) + 1) if tbin_edges else None
    tbin_counts = [0] * (len(tbin_edges) + 1) if tbin_edges else None
    start_time = time.time()
    print(f"[train] max_steps={args.max_steps} grad_accum={args.grad_accum} log_every={args.log_every} val_every={args.val_every} save_every={args.save_every}")
    if args.val_max_batches:
        print(f"[val] limiting to {args.val_max_batches} batches")
    for epoch in range(10**9):
        for batch in train_loader:
            images = batch["images"]
            prompts = batch["prompts"]
            style_paths = batch.get("style_paths", None)
            if global_step == 0 and accum_steps == 0:
                print(f"[train] first batch: images={len(images)} prompts={len(prompts)}")
            # tokenize
            text_inputs = tokenizer(
                prompts,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]
                encoder_hidden_states = encoder_hidden_states.to(dtype)

            # image to latents
            imgs = torch.stack([torch.from_numpy(np.array(im).astype("float32")) for im in images])
            imgs = imgs.permute(0, 3, 1, 2) / 255.0
            imgs = imgs * 2.0 - 1.0
            imgs = imgs.to(device, dtype=dtype)

            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = None
            if getattr(transformer.config, "sample_size", None) == 128:
                # resolution/aspect ratio conditioning for PixArt
                h = torch.tensor([args.resolution], device=device, dtype=encoder_hidden_states.dtype)
                w = torch.tensor([args.resolution], device=device, dtype=encoder_hidden_states.dtype)
                resolution = torch.stack([h, w], dim=1).repeat(latents.shape[0], 1)
                aspect_ratio = (h / w).repeat(latents.shape[0], 1)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                model_output = transformer(
                    noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                )
                pred = model_output.sample if hasattr(model_output, "sample") else model_output[0]
                if pred.shape[1] != noise.shape[1]:
                    if pred.shape[1] == 2 * noise.shape[1]:
                        pred = pred.chunk(2, dim=1)[0]
                    else:
                        raise RuntimeError(
                            f"Unexpected channel mismatch: pred {pred.shape} vs noise {noise.shape}"
                        )
                raw_loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                style_loss = None
                do_style = (
                    args.style_loss
                    and clip_model is not None
                    and clip_image_size is not None
                    and accum_steps == 0
                    and (args.style_loss_every > 0)
                    and ((global_step + 1) % args.style_loss_every == 0)
                )
                if do_style and style_paths:
                    sp = style_paths[0] if len(style_paths) > 0 else None
                    if sp:
                        # pred x0 for sample 0
                        idx0 = 0
                        t0 = timesteps[idx0]
                        alpha_prod = noise_scheduler.alphas_cumprod.to(device)[t0]
                        sqrt_alpha = alpha_prod.sqrt()
                        sqrt_one_minus = (1.0 - alpha_prod).sqrt()
                        pred_x0 = (noisy_latents[idx0:idx0+1] - sqrt_one_minus * pred[idx0:idx0+1]) / sqrt_alpha
                        # downsample latent for cheaper decode (e.g., 256)
                        target_size = int(args.style_loss_size)
                        if target_size > 0 and target_size != args.resolution:
                            vae_scale = int(getattr(vae.config, "scaling_factor", 8))
                            vae_scale = 8 if vae_scale <= 0 else vae_scale
                            lat_h = max(1, target_size // vae_scale)
                            lat_w = max(1, target_size // vae_scale)
                            pred_x0 = F.interpolate(pred_x0, size=(lat_h, lat_w), mode="bilinear", align_corners=False)
                        # decode to image (small size)
                        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=False):
                            dec = vae.decode(pred_x0 / vae.config.scaling_factor).sample
                        dec = torch.clamp(dec, -1.0, 1.0)
                        # CLIP image embeddings
                        img_in = _clip_preprocess_tensor(dec, clip_image_size)
                        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=False):
                            img_feat = clip_model.get_image_features(pixel_values=img_in)
                        img_feat = F.normalize(img_feat, dim=-1)
                        # style image embedding (no grad)
                        style_t = _load_style_tensor(sp, target_size if target_size > 0 else clip_image_size)
                        if style_t is not None:
                            style_t = style_t.to(device)
                            style_in = _clip_preprocess_tensor(style_t, clip_image_size)
                            with torch.no_grad():
                                style_feat = clip_model.get_image_features(pixel_values=style_in)
                            style_feat = F.normalize(style_feat, dim=-1)
                            cos = (img_feat * style_feat).sum(dim=-1).mean()
                            style_loss = 1.0 - cos
                if style_loss is not None:
                    raw_loss = raw_loss + (args.style_loss_weight * style_loss * args.grad_accum)
                loss = raw_loss / args.grad_accum

                # per-timestep bucket stats (per-sample loss)
                if tbin_edges:
                    num_t = max(1, noise_scheduler.config.num_train_timesteps - 1)
                    t_norm = timesteps.float() / float(num_t)
                    per_sample = (pred.float() - noise.float()).pow(2).flatten(1).mean(1)
                    bin_idx = torch.bucketize(t_norm, tbin_edges_t)
                    for b in range(len(tbin_sums)):
                        mask = bin_idx == b
                        if mask.any():
                            tbin_sums[b] += per_sample[mask].sum().item()
                            tbin_counts[b] += int(mask.sum().item())

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += raw_loss.item()
            accum_steps += 1

            if accum_steps % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if pbar:
                    pbar.update(1)

                step_loss = loss_accum / args.grad_accum
                loss_accum = 0.0
                if args.loss_ema_decay is not None:
                    if loss_ema is None:
                        loss_ema = step_loss
                    else:
                        loss_ema = args.loss_ema_decay * loss_ema + (1.0 - args.loss_ema_decay) * step_loss

                if global_step % args.log_every == 0:
                    cur_loss = step_loss
                    elapsed = time.time() - start_time
                    if loss_ema is not None:
                        print(f"step={global_step} loss={cur_loss:.4f} loss_ema={loss_ema:.4f} elapsed={elapsed:.1f}s")
                    else:
                        print(f"step={global_step} loss={cur_loss:.4f} elapsed={elapsed:.1f}s")
                    tb_writer.add_scalar("train/loss", cur_loss, global_step)
                    if loss_ema is not None:
                        tb_writer.add_scalar("train/loss_ema", loss_ema, global_step)
                    if style_loss is not None:
                        tb_writer.add_scalar("train/style_loss", float(style_loss.item()), global_step)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if tbin_edges and tbin_sums is not None and tbin_counts is not None:
                        tbin_parts = []
                        for b, name in enumerate(tbin_names):
                            if tbin_counts[b] > 0:
                                mean_b = tbin_sums[b] / float(tbin_counts[b])
                                tb_writer.add_scalar(f"train/loss_tbin_{name}", mean_b, global_step)
                                tb_writer.add_scalar(f"train/tbin_{name}_n", tbin_counts[b], global_step)
                                tbin_parts.append(f"{name}={mean_b:.4f}(n={tbin_counts[b]})")
                        if tbin_parts:
                            print(f"[tbin] step={global_step} " + " ".join(tbin_parts))
                        # reset bucket accumulators after logging
                        tbin_sums = [0.0] * (len(tbin_edges) + 1)
                        tbin_counts = [0] * (len(tbin_edges) + 1)
                    if device_type == "cuda":
                        try:
                            mem_gb = torch.cuda.memory_allocated() / (1024**3)
                            print(f"[mem] cuda_allocated={mem_gb:.2f} GB")
                        except Exception:
                            pass

                if args.val_every > 0 and global_step % args.val_every == 0:
                    transformer.eval()
                    val_losses = []
                    with torch.no_grad():
                        for i, vb in enumerate(val_loader, start=1):
                            v_images = vb["images"]
                            v_prompts = vb["prompts"]
                            v_text = tokenizer(
                                v_prompts,
                                max_length=tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt",
                            )
                            v_ids = v_text.input_ids.to(device)
                            v_attn = v_text.attention_mask.to(device)
                            v_enc = text_encoder(v_ids, attention_mask=v_attn)[0]
                            v_enc = v_enc.to(dtype)
                            v_imgs = torch.stack([torch.from_numpy(np.array(im).astype("float32")) for im in v_images])
                            v_imgs = v_imgs.permute(0, 3, 1, 2) / 255.0
                            v_imgs = v_imgs * 2.0 - 1.0
                            v_imgs = v_imgs.to(device, dtype=dtype)
                            v_latents = vae.encode(v_imgs).latent_dist.sample()
                            v_latents = v_latents * vae.config.scaling_factor
                            v_noise = torch.randn_like(v_latents)
                            v_ts = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps, (v_latents.shape[0],), device=device
                            ).long()
                            v_noisy = noise_scheduler.add_noise(v_latents, v_noise, v_ts)
                            v_added = None
                            if getattr(transformer.config, "sample_size", None) == 128:
                                h = torch.tensor([args.resolution], device=device, dtype=v_enc.dtype)
                                w = torch.tensor([args.resolution], device=device, dtype=v_enc.dtype)
                                resolution = torch.stack([h, w], dim=1).repeat(v_latents.shape[0], 1)
                                aspect_ratio = (h / w).repeat(v_latents.shape[0], 1)
                                v_added = {"resolution": resolution, "aspect_ratio": aspect_ratio}
                            v_out = transformer(
                                v_noisy, encoder_hidden_states=v_enc, timestep=v_ts, added_cond_kwargs=v_added
                            )
                            v_pred = v_out.sample if hasattr(v_out, "sample") else v_out[0]
                            if v_pred.shape[1] != v_noise.shape[1]:
                                if v_pred.shape[1] == 2 * v_noise.shape[1]:
                                    v_pred = v_pred.chunk(2, dim=1)[0]
                                else:
                                    raise RuntimeError(
                                        f"Unexpected channel mismatch: pred {v_pred.shape} vs noise {v_noise.shape}"
                                    )
                            v_loss = F.mse_loss(v_pred.float(), v_noise.float(), reduction="mean")
                            val_losses.append(v_loss.item())
                            if args.val_log_every > 0 and i % args.val_log_every == 0:
                                print(f"[val] batch {i} loss={v_loss.item():.4f}")
                            if args.val_max_batches and i >= args.val_max_batches:
                                break
                    mean_val = sum(val_losses) / max(1, len(val_losses))
                    print(f"[val] step={global_step} loss={mean_val:.4f}")
                    tb_writer.add_scalar("val/loss", mean_val, global_step)
                    # early stopping / best model
                    if mean_val + args.early_stop_min_delta < best_val:
                        best_val = mean_val
                        patience_counter = 0
                        _save_checkpoint(global_step, is_best=True)
                    else:
                        patience_counter += 1
                    if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                        print(f"Early stopping at step {global_step} (patience={args.early_stop_patience})")
                        if pbar:
                            pbar.close()
                        return
                    transformer.train()

                # step LR schedule after optimizer update
                scheduler.step()

                if global_step % args.save_every == 0:
                    _save_checkpoint(global_step, is_best=False)

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    # final save
    final_path = ckpt_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(final_path)
    print("done")


if __name__ == "__main__":
    main()
