#!/usr/bin/env python3
"""
Baseline ControlNet Training Script for PixArt-alpha-XL-2
基于 PixArt-alpha-XL-2 (0.6B) 和 ControlNet 的基准实验
使用 Fill50k 数据集验证 RTX 3090Ti (24GB) 环境
"""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    PixArtAlphaPipeline,
    PixArtTransformer2DModel,
    HunyuanDiT2DControlNetModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

# 导入自定义 ControlNet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.controlnet_dit import ControlNetDiT

# 检查 wandb
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

logger = get_logger(__name__)

# 检查 huggingface_hub
try:
    from huggingface_hub import upload_folder
    has_hf_hub = True
except ImportError:
    has_hf_hub = False


def load_config(config_path="./config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4fe830de030734780e22ef0/diffusers/examples/text_to_image/train_text_to_image_snr.py
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4fe830de030734780e22ef0/diffusers/examples/text_to_image/train_text_to_image_snr.py
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Baseline ControlNet Training for PixArt-alpha-XL-2")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.yaml",
        help="Path to the configuration YAML file.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Create a namespace object to mimic argparse args
    class ConfigArgs:
        pass

    config_args = ConfigArgs()

    # Map config values to args attributes
    # Dataset parameters
    data_config = config.get('data', {})
    config_args.dataset_name = data_config.get('data_dir', "./dataset/data")
    config_args.dataset_config_name = None
    config_args.train_data_dir = None
    config_args.image_column = "Target:FILE"
    config_args.conditioning_image_column = "Cond:FILE"
    config_args.caption_column = "Prompt"

    # Model parameters
    baseline_config = config.get('baseline', {})
    config_args.pretrained_model_name_or_path = baseline_config.get('pretrained_model_name_or_path', "PixArt-alpha/PixArt-XL-2-0.6B")
    config_args.controlnet_model_name_or_path = baseline_config.get('controlnet_model_name_or_path')
    config_args.use_controlnet = baseline_config.get('use_controlnet', False)  # Switch for baseline vs ControlNet

    # Training parameters
    training_config = config.get('training', {})
    hardware_config = config.get('hardware', {})
    config_args.output_dir = baseline_config.get('output_dir', "./output/baseline")
    config_args.cache_dir = baseline_config.get('cache_dir')
    config_args.seed = baseline_config.get('seed')
    config_args.resolution = config.get('data', {}).get('resolution', 512)
    config_args.train_batch_size = training_config.get('batch_size', 1)
    config_args.num_train_epochs = training_config.get('num_epochs', 1)
    config_args.max_train_steps = training_config.get('max_train_steps')
    config_args.resume_from_checkpoint = training_config.get('resume_from_checkpoint')
    config_args.gradient_accumulation_steps = baseline_config.get('gradient_accumulation_steps', 4)
    config_args.gradient_checkpointing = hardware_config.get('gradient_checkpointing', True)
    config_args.learning_rate = float(training_config.get('learning_rate', 5e-6))
    config_args.scale_lr = False
    config_args.lr_scheduler = baseline_config.get('lr_scheduler', "constant")
    config_args.lr_warmup_steps = baseline_config.get('lr_warmup_steps', 500)
    config_args.snr_gamma = config.get('loss', {}).get('snr_gamma')
    # Temporarily disable 8-bit Adam due to BF16 compatibility issues
    config_args.use_8bit_adam = False  # hardware_config.get('use_8bit_adam', True)
    config_args.allow_tf32 = hardware_config.get('allow_tf32', True)
    config_args.use_ema = baseline_config.get('use_ema', False)
    config_args.non_ema_revision = baseline_config.get('non_ema_revision')
    config_args.dataloader_num_workers = hardware_config.get('dataloader_num_workers', 0)
    config_args.adam_beta1 = baseline_config.get('adam_beta1', 0.9)
    config_args.adam_beta2 = baseline_config.get('adam_beta2', 0.999)
    config_args.adam_weight_decay = float(baseline_config.get('adam_weight_decay', 1e-2))
    config_args.adam_epsilon = float(baseline_config.get('adam_epsilon', 1e-08))
    config_args.max_grad_norm = float(baseline_config.get('max_grad_norm', 1.0))

    # Monitoring and logging
    config_args.push_to_hub = baseline_config.get('push_to_hub', False)
    config_args.hub_token = baseline_config.get('hub_token')
    config_args.hub_model_id = baseline_config.get('hub_model_id')
    config_args.logging_dir = baseline_config.get('logging_dir', "logs")
    config_args.report_to = baseline_config.get('report_to', "tensorboard")
    config_args.mixed_precision = hardware_config.get('mixed_precision', "bf16")

    # Validation parameters
    config_args.validation_image = baseline_config.get('validation_image')
    config_args.validation_prompt = baseline_config.get('validation_prompt')
    config_args.validation_steps = baseline_config.get('validation_steps', 500)
    config_args.num_validation_images = baseline_config.get('num_validation_images', 4)

    # Other
    config_args.local_rank = -1
    config_args.checkpointing_steps = training_config.get('checkpointing_steps', 500)
    config_args.checkpoints_total_limit = training_config.get('checkpoints_total_limit')
    config_args.enable_xformers_memory_efficient_attention = baseline_config.get('enable_xformers_memory_efficient_attention', True)
    config_args.max_train_samples = baseline_config.get('max_train_samples')

    return config_args


def make_train_dataset(args, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset.column_names
    if args.image_column is None:
        image_column = dataset_columns["train"][0] if dataset_columns.get("train", None) is not None else dataset_columns[0]
    else:
        image_column = args.image_column
        if image_column not in dataset_columns["train"]:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(dataset_columns['train'])}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns["train"][1] if dataset_columns.get("train", None) is not None else dataset_columns[1]
    else:
        caption_column = args.caption_column
        if caption_column not in dataset_columns["train"]:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(dataset_columns['train'])}"
            )
    if args.conditioning_image_column is None:
        conditioning_image_column = dataset_columns["train"][2] if dataset_columns.get("train", None) is not None else dataset_columns[2]
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in dataset_columns["train"]:
            raise ValueError(
                f"--conditioning_image_column' value '{args.conditioning_image_column}' needs to be one of: {', '.join(dataset_columns['train'])}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        from PIL import Image
        import os
        
        # 加载图片文件
        dataset_dir = args.dataset_name
        images = []
        for img_path in examples[image_column]:
            full_path = os.path.join(dataset_dir, img_path)
            image = Image.open(full_path).convert("RGB")
            images.append(image)
        images = [train_transforms(image) for image in images]

        conditioning_images = []
        for img_path in examples[conditioning_image_column]:
            full_path = os.path.join(dataset_dir, img_path)
            image = Image.open(full_path).convert("RGB")
            conditioning_images.append(image)
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 使用本地模型路径而不是HuggingFace模型名称
    local_model_path = "/mnt/fast18/models_cache/hub/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692"
    if os.path.exists(local_model_path):
        logger.info(f"Loading model from local path: {local_model_path}")
        model_path = local_model_path
    else:
        logger.info(f"Loading model from HuggingFace: {args.pretrained_model_name_or_path}")
        model_path = args.pretrained_model_name_or_path

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", revision=args.non_ema_revision, cache_dir=args.cache_dir
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", revision=args.non_ema_revision, cache_dir=args.cache_dir
    )
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=args.non_ema_revision, cache_dir=args.cache_dir)
    transformer = PixArtTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", revision=args.non_ema_revision, cache_dir=args.cache_dir
    )

    if args.use_controlnet:
        # Use actual ControlNet
        controlnet = ControlNetDiT(transformer)
        # Freeze transformer for ControlNet training
        transformer.requires_grad_(False)
        controlnet.train()

        # Monkey patch transformer blocks for residual injection
        def make_injected_forward(original_forward):
            def injected_forward(self, hidden_states, timestep, encoder_hidden_states, added_cond_kwargs=None):
                out = original_forward(hidden_states, timestep, encoder_hidden_states, added_cond_kwargs)
                if hasattr(self, 'residual') and self.residual is not None:
                    out = out + self.residual
                return out
            return injected_forward

        for block in transformer.blocks[:14]:
            original_forward = block.forward
            block.forward = make_injected_forward(original_forward).__get__(block, block.__class__)
            block.residual = None

    else:
        # Baseline: train transformer directly
        controlnet = transformer
        # Keep transformer trainable for baseline
        # transformer.requires_grad_(False)  # Commented out for baseline

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            transformer.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the HuggingFace hub (the dataset will be downloaded automatically from the datasets Hub)
    train_dataset = make_train_dataset(args, tokenizer, accelerator)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_controlnet = EMAModel(controlnet.parameters())

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encoder and vae to gpu and cast to weight_dtype (they are frozen)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # NOTE: For baseline, transformer is trainable (controlnet = transformer), so keep it in float32
    # When implementing actual ControlNet, transformer will be frozen and can use weight_dtype
    transformer.to(accelerator.device, dtype=torch.float32)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("controlnet-pixart", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Convert target to float32 to match transformer output dtype
                target = target.float()

                # ControlNet conditioning
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                if args.use_controlnet:
                    # Get initial patched hidden_states for controlnet
                    patched_latents = transformer.patch_embed(noisy_latents)
                    patched_latents = transformer.pos_embed(patched_latents)
                    control_residuals = controlnet(
                        patched_latents, timesteps, encoder_hidden_states, conditioning_pixel_values=controlnet_image
                    )
                    # Set residuals for injection
                    for i, block in enumerate(transformer.blocks[:14]):
                        block.residual = control_residuals[i]

                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                if args.use_controlnet:
                    # Clear residuals
                    for block in transformer.blocks[:14]:
                        block.residual = None
                
                # PixArt transformer outputs 8 channels (mean + variance), we only need the first 4 for noise prediction
                model_pred = model_output[:, :4, :, :]

                if args.snr_gamma is None:
                    # Don't convert to float() as it breaks the computation graph
                    loss = F.mse_loss(model_pred, target, reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_t, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred, target, reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at most `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_image is not None and args.validation_prompt is not None:
                if global_step % args.validation_steps == 0:
                    logger.info("Running validation... ")
                    # create pipeline
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_controlnet.store(controlnet.parameters())
                        ema_controlnet.copy_to(controlnet.parameters())
                    # The models need unwrapping because for compatibility in distributed training, the models
                    # have been wrapped in DDP. The state_dicts will be unwrapped by
                    # `accelerator.unwrap_model`. Then we need to unwrap the ControlNetModel
                    pipeline = PixArtAlphaPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),
                        controlnet=accelerator.unwrap_model(controlnet),
                        revision=args.non_ema_revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    if args.enable_xformers_memory_efficient_attention:
                        pipeline.enable_xformers_memory_efficient_attention()

                    # run inference
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    images = []
                    for _ in range(args.num_validation_images):
                        image = pipeline(
                            args.validation_prompt,
                            image=args.validation_image,
                            num_inference_steps=20,
                            generator=generator,
                        ).images[0]
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_controlnet.restore(controlnet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of inference.
        if args.validation_prompt is not None:
            pipeline = PixArtAlphaPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=accelerator.unwrap_model(transformer),
                controlnet=controlnet,
                revision=args.non_ema_revision,
            )
            pipeline.save_pretrained(args.output_dir)

            # run inference
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            images = []
            for _ in range(args.num_validation_images):
                image = pipeline(
                    args.validation_prompt,
                    image=args.validation_image,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]
                images.append(image)

            if args.report_to == "tensorboard":
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")

        if args.push_to_hub:
            upload_folder(
                repo_id=args.hub_model_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    main(args)