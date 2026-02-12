#!/usr/bin/env python3
"""
Download teacher model weights (SDXL base + IP-Adapter) into a dedicated cache.
This is a pure download step, separate from generation.
"""
import argparse
import os
import sys
from pathlib import Path


def _set_cache_env(cache_dir, hf_endpoint=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache_dir / "diffusers"))
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache", default="/mnt/fast18/models_cache")
    ap.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    ap.add_argument("--sdxl-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--ip-adapter-id", default="h94/IP-Adapter")
    ap.add_argument("--ip-adapter-subfolder", default="sdxl_models")
    ap.add_argument("--ip-adapter-weight", default="ip-adapter_sdxl.bin")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume download")
    ap.add_argument("--full", action="store_true", help="Download full repos (not minimal)")
    args = ap.parse_args()

    _set_cache_env(args.model_cache, args.hf_endpoint)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"huggingface_hub not available: {e}", file=sys.stderr)
        sys.exit(1)

    resume = not args.no_resume

    if args.full:
        # SDXL base (full snapshot)
        snapshot_download(
            repo_id=args.sdxl_id,
            cache_dir=os.environ.get("HF_HOME"),
            resume_download=resume,
        )
        # IP-Adapter (full snapshot)
        snapshot_download(
            repo_id=args.ip_adapter_id,
            cache_dir=os.environ.get("HF_HOME"),
            resume_download=resume,
        )
    else:
        # SDXL base (minimal required for diffusers pipeline)
        sdxl_patterns = [
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/*",
            "vae/*",
            "vae_fp16_fix/*",
            "*.json",
        ]
        # Avoid large fp32/bin weight shards; prefer safetensors in diffusers repos.
        sdxl_ignore = [
            "*.bin",
            "*.ckpt",
            "*.onnx",
            "*.msgpack",
            "*.h5",
            "*.ot",
            "*.pt",
        ]
        snapshot_download(
            repo_id=args.sdxl_id,
            cache_dir=os.environ.get("HF_HOME"),
            resume_download=resume,
            allow_patterns=sdxl_patterns,
            ignore_patterns=sdxl_ignore,
        )

        # IP-Adapter (only required SDXL adapter + image encoder)
        ip_patterns = [
            f"{args.ip_adapter_subfolder}/{args.ip_adapter_weight}",
            f"{args.ip_adapter_subfolder}/image_encoder/*",
            f"{args.ip_adapter_subfolder}/image_encoder/**",
        ]
        snapshot_download(
            repo_id=args.ip_adapter_id,
            cache_dir=os.environ.get("HF_HOME"),
            resume_download=resume,
            allow_patterns=ip_patterns,
        )

    print("Download complete")


if __name__ == "__main__":
    main()
