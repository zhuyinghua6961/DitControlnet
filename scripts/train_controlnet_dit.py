#!/usr/bin/env python3
"""
ControlNet-DiT Training Script for fill50k Dataset
基于 DiT (Diffusion Transformer) 和 ControlNet 的条件图像生成训练脚本
支持空间对齐和异构插值的数据预处理
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import argparse
import logging
import yaml
from tqdm import tqdm
import math
import numpy as np
from datetime import datetime
import shutil
from dataset import create_dataloaders
from models import ControlNetDiT, create_diffusion_schedule, get_loss_function
from utils import (
    train_epoch, save_checkpoint, load_config, setup_wandb,
    create_optimizer_and_scheduler, setup_device_and_precision
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='ControlNet-DiT Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # 创建必要的目录
    for dir_key in ['log_dir', 'checkpoint_dir', 'sample_dir']:
        Path(config['logging'][dir_key]).mkdir(exist_ok=True)

    # 设置设备和混合精度
    device, scaler = setup_device_and_precision(config)

    # 初始化WandB
    setup_wandb(config)

    # 创建数据加载器
    train_jsonl = os.path.join(config['data']['jsonl_dir'], config['data']['train_jsonl'])
    eval_jsonl = os.path.join(config['data']['jsonl_dir'], config['data']['eval_jsonl'])
    data_root = config['data']['data_dir']

    train_loader, eval_loader = create_dataloaders(
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        data_root=data_root,
        batch_size=config['training']['batch_size'],
        resolution=config['data']['resolution'],
        num_workers=config['data']['num_workers'],
        use_bucketing=config['training']['use_bucketing']
    )

    # 创建模型
    model = ControlNetDiT(config).to(device)

    # 编译模型（如果支持）
    if config['hardware']['compile_model'] and hasattr(torch, 'compile'):
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile")

    # 优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # 扩散调度
    diffusion_schedule = create_diffusion_schedule(config)

    # 损失函数
    loss_fn = get_loss_function(config)

    # 恢复检查点
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")

    # 训练循环
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        loss = train_epoch(model, train_loader, optimizer, scheduler, diffusion_schedule,
                          loss_fn, device, epoch, scaler, config)

        # 记录到WandB
        if config['logging']['use_wandb'] and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'loss': loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # 保存检查点
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, loss, config)

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, scheduler, epoch, loss, config, is_best=True)

    logger.info("Training completed!")
    if config['logging']['use_wandb'] and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()
