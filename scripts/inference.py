#!/usr/bin/env python3
"""
ControlNet-DiT Inference Script
使用训练好的模型进行条件图像生成的推理脚本
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import logging
import yaml
import math
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiTBlock(nn.Module):
    """DiT Transformer Block with AdaLN"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        # AdaLN参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(self, x, c):
        """
        Args:
            x: 输入特征 [B, N, D]
            c: 条件嵌入 [B, D]
        """
        # AdaLN调制
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)

        # Self-attention with AdaLN
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa * attn_out

        # MLP with AdaLN
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp * mlp_out

        return x

class ControlNetDiT(nn.Module):
    """ControlNet-DiT Model for inference"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']

        self.img_size = model_config['img_size']
        self.patch_size = model_config['patch_size']
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.dim = model_config['dim']

        # 图像到patch的嵌入
        self.patch_embed = nn.Conv2d(model_config['in_channels'], self.dim,
                                   kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.dim))

        # ControlNet条件嵌入
        self.control_embed = nn.Conv2d(model_config['in_channels'], self.dim,
                                     kernel_size=self.patch_size, stride=self.patch_size)

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.dim, model_config['num_heads'], model_config['mlp_ratio'])
            for _ in range(model_config['depth'])
        ])

        # 输出投影
        self.norm = nn.LayerNorm(self.dim)
        self.final_proj = nn.Linear(self.dim, self.patch_size * self.patch_size * model_config['out_channels'])

    def forward(self, x, cond, t):
        """
        Args:
            x: 噪声图像 [B, C, H, W]
            cond: 条件图像 [B, C, H, W]
            t: 时间步 [B]
        """
        B, C, H, W = x.shape

        # 嵌入patch
        x_patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        cond_patches = self.control_embed(cond).flatten(2).transpose(1, 2)  # [B, N, D]

        # 添加位置嵌入
        x_patches = x_patches + self.pos_embed

        # 时间步嵌入
        t_embed = self.time_embed(self.timestep_embedding(t, self.dim))

        # 注入条件信息
        c_embed = t_embed + cond_patches.mean(dim=1)  # 全局条件

        # 通过DiT blocks
        for block in self.blocks:
            x_patches = block(x_patches, c_embed)

        # 最终输出
        x_patches = self.norm(x_patches)
        x_patches = self.final_proj(x_patches)

        # 重塑回图像
        x_out = x_patches.view(B, H//self.patch_size, W//self.patch_size,
                              self.patch_size, self.patch_size, C)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_out = x_out.view(B, C, H, W)

        return x_out

    def timestep_embedding(self, t, dim):
        """时间步正弦嵌入"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

def create_diffusion_schedule(config):
    """创建扩散过程的时间表"""
    diffusion_config = config['diffusion']
    num_timesteps = diffusion_config['num_timesteps']
    beta_start = diffusion_config['beta_start']
    beta_end = diffusion_config['beta_end']

    if diffusion_config['schedule_type'] == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {diffusion_config['schedule_type']}")

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
        'one_minus_alphas_cumprod': 1 - alphas_cumprod,
    }

@torch.no_grad()
def sample(model, cond, diffusion_schedule, device, num_steps=50):
    """
    使用DDIM采样从噪声生成图像

    Args:
        model: 训练好的模型
        cond: 条件图像 [B, C, H, W]
        diffusion_schedule: 扩散调度
        device: 设备
        num_steps: 采样步数

    Returns:
        生成的图像 [B, C, H, W]
    """
    model.eval()
    B, C, H, W = cond.shape

    # 从纯噪声开始
    x = torch.randn(B, C, H, W, device=device)

    # DDIM采样
    timesteps = torch.linspace(len(diffusion_schedule['betas']) - 1, 0, num_steps, dtype=torch.long, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # 预测噪声
        predicted_noise = model(x, cond, t_batch)

        # DDIM更新步骤
        alpha_t = diffusion_schedule['alphas_cumprod'][t]
        alpha_t_prev = diffusion_schedule['alphas_cumprod'][timesteps[i+1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

        # 预测x0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        x0_pred = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

        # 计算方向
        dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise

        # 更新x
        x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt

    return x

def load_image(image_path, img_size=256):
    """加载和预处理图像"""
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform(image).unsqueeze(0)  # 添加batch维度

def save_image(tensor, save_path):
    """保存张量为图像"""
    # 反归一化
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)

    # 转换为PIL图像
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.squeeze(0).cpu())

    image.save(save_path)
    logger.info(f"Saved generated image to {save_path}")

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='ControlNet-DiT Inference')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--condition_image', type=str, required=True,
                        help='Condition image path')
    parser.add_argument('--output_path', type=str, default='generated_image.png',
                        help='Output image path')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 加载模型
    model = ControlNetDiT(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    logger.info(f"Loaded model from {args.checkpoint}")

    # 加载条件图像
    cond_image = load_image(args.condition_image, config['model']['img_size']).to(device)
    logger.info(f"Loaded condition image from {args.condition_image}")

    # 创建扩散调度
    diffusion_schedule = create_diffusion_schedule(config)

    # 生成图像
    logger.info(f"Generating image with {args.num_steps} sampling steps...")
    generated_image = sample(model, cond_image, diffusion_schedule, device, args.num_steps)

    # 保存结果
    save_image(generated_image, args.output_path)

    logger.info("Inference completed!")

if __name__ == '__main__':
    main()