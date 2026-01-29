#!/usr/bin/env python3
"""
ControlNet-DiT Training Utilities
训练相关的工具函数
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, diffusion_schedule,
                loss_fn, device, epoch, scaler=None, config=None):
    """训练一个epoch"""
    from tqdm import tqdm

    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        # 获取数据（使用新的数据集格式）
        target_images = batch['pixel_values'].to(device)  # 目标图像 [-1, 1]
        conditioning_images = batch['conditioning_pixel_values'].to(device)  # 条件图像 [0, 1]

        # 添加噪声
        noise = torch.randn_like(target_images)
        t = torch.randint(0, len(diffusion_schedule['betas']), (target_images.shape[0],), device=device)

        # 前向扩散过程
        sqrt_alphas_cumprod = diffusion_schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = diffusion_schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)

        noisy_images = sqrt_alphas_cumprod * target_images + sqrt_one_minus_alphas_cumprod * noise

        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predicted_noise = model(noisy_images, conditioning_images, t)
                loss = loss_fn(predicted_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 预测噪声
            predicted_noise = model(noisy_images, conditioning_images, t)
            loss = loss_fn(predicted_noise, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if config and 'gradient_clip_norm' in config['training']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])

            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, is_best=False):
    """保存检查点"""
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if is_best:
        checkpoint_path = checkpoint_dir / 'best_model.pth'
    else:
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config):
    """设置Weights & Biases"""
    if config['logging']['use_wandb'] and WANDB_AVAILABLE:
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=f"controlnet_dit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info("Weights & Biases initialized")
    elif config['logging']['use_wandb'] and not WANDB_AVAILABLE:
        logger.warning("Weights & Biases requested but not available")


def create_optimizer_and_scheduler(model, config):
    """创建优化器和学习率调度器"""
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器（带warmup）
    if 'warmup_steps' in config['training']:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                   total_iters=config['training']['warmup_steps'])
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])

        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                milestones=[config['training']['warmup_steps']])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['num_epochs']
        )

    return optimizer, scheduler


def setup_device_and_precision(config):
    """设置设备和混合精度"""
    # 设置设备
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])

    logger.info(f"Using device: {device}")

    # 设置混合精度
    scaler = None
    if config['hardware']['mixed_precision'] and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using mixed precision training")

    return device, scaler