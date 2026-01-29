import copy
from typing import List

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode conditioning images (e.g. Canny) to transformer-aligned features.

    Input: (B, 3, H, W) e.g. (B,3,512,512)
    Output: (B, hidden_size, H_lat, W_lat) e.g. (B,1152,64,64)
    """

    def __init__(self, in_channels: int = 3, hidden_size: int = 1152):
        super().__init__()
        # 3 downsampling convs (stride=2) reduce 512 -> 256 -> 128 -> 64
        # final conv maps to transformer hidden_size
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, hidden_size, 3, stride=1, padding=1),
        )
        # patchify to match transformer input: 64x64 -> 32x32 patches
        self.patch_embed = nn.Conv2d(hidden_size, hidden_size, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_patches, hidden_size)."""
        x = self.blocks(x)  # (B, 1152, 64, 64)
        x = self.patch_embed(x)  # (B, 1152, 32, 32)
        # Flatten to token format: (B, 1024, 1152)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C) -> (B, 1024, 1152)
        return x


class ControlNetDiT(nn.Module):
    """A lightweight ControlNet wrapper for DiT-style transformer.

    - Clones the first N blocks from a PixArt transformer into a control branch.
    - Builds zero-initialized linear adapters (ZeroLinear) per-block.
    - Produces per-block residuals shaped (B, L, C) to be injected into main transformer.
    """

    def __init__(self, transformer, n_control_blocks: int = 14):
        super().__init__()
        self.config = transformer.config
        hidden_size = self.config.hidden_size

        # condition encoder: map conditioning image -> (B, L, hidden_size)
        self.condition_encoder = ConditionEncoder(in_channels=3, hidden_size=hidden_size)

        # clone early transformer blocks as control blocks
        self.controlnet_blocks = nn.ModuleList([copy.deepcopy(transformer.blocks[i]) for i in range(n_control_blocks)])

        # zero-initialized linear adapters (applied per-token)
        self.zero_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_control_blocks)])
        for linear in self.zero_linears:
            nn.init.zeros_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor = None,
        conditioning_pixel_values: torch.Tensor = None,
        **forward_kwargs,
    ) -> List[torch.Tensor]:
        """Run control branch and return list of residuals for injection.

        Args:
            hidden_states: (B, L, C) main transformer hidden states at current point (patched tokens)
            timestep: current diffusion timestep (passed to cloned blocks)
            encoder_hidden_states: text encoder embeddings (passed through)
            conditioning_pixel_values: (B, 3, H_img, W_img) e.g. Canny images

        Returns:
            List of length `n_control_blocks` with elements shaped (B, L, C)
        """
        if conditioning_pixel_values is None:
            raise ValueError("conditioning_pixel_values must be provided to ControlNetDiT.forward")

        cond_features = self.condition_encoder(conditioning_pixel_values)

        hs = hidden_states
        residuals: List[torch.Tensor] = []

        for i, block in enumerate(self.controlnet_blocks):
            # add conditioning features (token-aligned)
            inp = hs + cond_features

            out = block(inp, timestep=timestep, encoder_hidden_states=encoder_hidden_states, **forward_kwargs)
            # some blocks return (hidden_states, ...) tuples
            if isinstance(out, tuple):
                hs = out[0]
            else:
                hs = out

            # apply zero-linear per-token
            res = self.zero_linears[i](hs)
            residuals.append(res)

        return residuals
#!/usr/bin/env python3
"""
ControlNet-DiT Model Architecture
基于 DiT (Diffusion Transformer) 和 ControlNet 的条件图像生成模型
"""

import torch
import torch.nn as nn
import math


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
    """ControlNet-DiT Model with improved architecture"""

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

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化位置嵌入
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 初始化patch嵌入
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 初始化其他层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
    elif diffusion_config['schedule_type'] == 'cosine':
        # Cosine schedule
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
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


def get_loss_function(config):
    """获取损失函数"""
    loss_config = config['loss']
    if loss_config['type'] == 'mse':
        loss_fn = nn.MSELoss(reduction=loss_config['reduction'])
    elif loss_config['type'] == 'l1':
        loss_fn = nn.L1Loss(reduction=loss_config['reduction'])
    elif loss_config['type'] == 'huber':
        loss_fn = nn.HuberLoss(reduction=loss_config['reduction'])
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")

    return loss_fn