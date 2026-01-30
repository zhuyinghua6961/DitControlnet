# ControlNet-DiT: 条件图像生成

基于 DiT (Diffusion Transformer) 和 ControlNet 的条件图像生成模型，支持 fill50k 数据集的高效训练。

## 网络架构详解

### 整体架构概述

ControlNet-DiT 采用"主干网络 + 控制分支"的双路径架构，结合了 Transformer 的全局建模能力和 ControlNet 的精确控制能力。

```
输入图像 (噪声)          条件图像 (Canny/Depth等)
      ↓                           ↓
 Patch Embed              Condition Encoder
      ↓                           ↓
 Position Embed          Feature Extraction
      ↓                           ↓
      ├─────────────→  Control Blocks  ───→ Zero Convs
      ↓                     ↓                    ↓
 DiT Blocks  ←───────── Residual Injection ─────┘
      ↓
 Final Norm
      ↓
 Output Proj
      ↓
  预测噪声/图像
```

### 核心模块详解

#### 1. **ConditionEncoder (条件编码器)**
**作用**: 将条件图像(如 Canny 边缘、深度图)编码为与 Transformer 对齐的特征表示

**结构**:
```python
输入: (B, 3, 512, 512) RGB 条件图像
  ↓
Conv2d(3→64, stride=2)    # 512 → 256
  ↓ SiLU
Conv2d(64→128, stride=2)  # 256 → 128
  ↓ SiLU
Conv2d(128→256, stride=2) # 128 → 64
  ↓ SiLU
Conv2d(256→1152, stride=1) # 特征映射到 hidden_size
  ↓
Patch Embed(2x2, stride=2) # 64 → 32 (patchify)
  ↓
输出: (B, 1024, 1152) # 1024 个 token，每个维度 1152
```

**关键特性**:
- 3 次下采样卷积将空间分辨率从 512 降至 64
- 最终 patch embed 将特征图转为 token 序列
- 输出维度与主 Transformer 的 hidden_size 对齐

#### 2. **DiTBlock (DiT Transformer 块)**
**作用**: 基于 Adaptive Layer Normalization (AdaLN) 的 Transformer 块，实现条件注入

**结构**:
```python
输入: x (B, N, D), c (B, D) 条件嵌入
  ↓
AdaLN Modulation → (shift_msa, scale_msa, gate_msa, 
                     shift_mlp, scale_mlp, gate_mlp)
  ↓
┌─────────────────────────────────┐
│ Self-Attention 分支:             │
│   LayerNorm(x)                  │
│   → AdaLN(shift, scale)         │
│   → MultiheadAttention          │
│   → Gate(gate_msa)              │
│   → Residual Add                │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ MLP 分支:                        │
│   LayerNorm(x)                  │
│   → AdaLN(shift, scale)         │
│   → Linear→GELU→Linear          │
│   → Gate(gate_mlp)              │
│   → Residual Add                │
└─────────────────────────────────┘
  ↓
输出: (B, N, D)
```

**关键特性**:
- **AdaLN (Adaptive Layer Normalization)**: 通过仿射变换 `scale` 和 `shift` 注入时间步和条件信息
- **门控机制 (Gating)**: 通过 `gate` 参数控制每个分支的贡献
- **MLP 扩展比例**: 默认 4.0，即隐藏层维度是输入的 4 倍

#### 3. **ControlNet 分支**
**作用**: 克隆主 Transformer 的前 N 个块，构建独立的控制路径

**结构**:
```python
n_control_blocks = 14  # 通常使用前 14 层

控制分支:
  条件特征 + 主干隐藏状态
    ↓
  Cloned Block 1 → Zero Linear 1 → Residual 1
    ↓
  Cloned Block 2 → Zero Linear 2 → Residual 2
    ↓
    ...
    ↓
  Cloned Block 14 → Zero Linear 14 → Residual 14
    ↓
  注入到主干对应层
```

**关键特性**:
- **零初始化线性层 (ZeroLinear)**: 
  - 训练初期不影响主网络，保证稳定性
  - 权重和偏置初始化为 0
  - 随训练逐步学习控制信号
  
- **逐块残差注入**: 
  - 每个控制块的输出通过 Zero Linear 后注入主干
  - 保持细粒度的空间控制能力

#### 4. **完整 ControlNetDiT 模型**

**主要组件**:

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| `patch_embed` | (B, C, H, W) | (B, N, D) | 将图像分割为 patch 序列 |
| `pos_embed` | - | (1, N, D) | 可学习的位置编码 |
| `control_embed` | (B, C, H, W) | (B, N, D) | 条件图像的 patch 嵌入 |
| `time_embed` | (B,) | (B, D) | 时间步的正弦位置编码 |
| `blocks` | (B, N, D) | (B, N, D) | 主 Transformer 块序列 |
| `norm` | (B, N, D) | (B, N, D) | 输出前的 LayerNorm |
| `final_proj` | (B, N, D) | (B, N, C×P²) | 投影回图像空间 |

**前向传播流程**:
```python
1. 输入处理:
   x_patches = patch_embed(噪声图像) + pos_embed
   cond_patches = control_embed(条件图像)
   
2. 时间和条件嵌入:
   t_embed = time_embed(timestep)
   c_embed = t_embed + mean(cond_patches)  # 全局条件
   
3. Transformer 处理:
   for block in blocks:
       x_patches = block(x_patches, c_embed)
   
4. 输出投影:
   x_patches = norm(x_patches)
   x_patches = final_proj(x_patches)
   x_out = rearrange(x_patches) → (B, C, H, W)
```

### 扩散过程详解

#### 时间步编码 (Timestep Embedding)
```python
def timestep_embedding(t, dim):
    """正弦位置编码"""
    half_dim = dim // 2
    emb = log(10000) / (half_dim - 1)
    emb = exp(arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = concat([sin(emb), cos(emb)], dim=-1)
    return emb  # (B, dim)
```

**作用**: 将离散时间步 t ∈ [0, 1000] 编码为连续向量表示

#### 扩散调度 (Diffusion Schedule)

支持两种调度类型:

**1. Linear Schedule**:
```python
β_t = linear_interp(β_start, β_end, t/T)
α_t = 1 - β_t
ᾱ_t = ∏(α_s) for s=1 to t
```

**2. Cosine Schedule**:
```python
ᾱ_t = cos²((t/T + s)/(1+s) × π/2)
β_t = 1 - (ᾱ_t / ᾱ_{t-1})
```

**关键参数**:
- `num_timesteps`: 1000 (总扩散步数)
- `beta_start`: 0.0001 (噪声起始强度)
- `beta_end`: 0.02 (噪声结束强度)

### 损失函数

支持多种损失类型:

| 损失类型 | 公式 | 适用场景 |
|---------|------|---------|
| MSE | `L = mean((pred - target)²)` | 标准扩散模型 |
| L1 | `L = mean(|pred - target|)` | 对异常值更鲁棒 |
| Huber | `L = smoothL1(pred, target)` | 结合 L1 和 L2 优势 |

### 训练策略

#### 数据增强
```python
# 空间对齐增强 (条件和目标同步变换)
- 随机水平翻转 (p=0.5)
- 随机垂直翻转 (p=0.5)

# 异构插值
- 目标图像: Bilinear (平滑)
- 条件图像: Nearest (保持边缘锐利)
```

#### 优化器配置
```yaml
optimizer: AdamW
learning_rate: 1e-5  # 微调推荐更小 LR
weight_decay: 0.01
gradient_clip_norm: 1.0
warmup_steps: 1000
```

#### 内存优化
- **梯度检查点**: 减少 ~40% 显存
- **混合精度 (bf16/fp16)**: 加速 ~2x
- **梯度累积**: 模拟大 batch size
- **8-bit AdamW**: 节省优化器显存

## 项目结构

```
controlnetDiT/
├── config/                 # 配置文件
│   ├── config.yaml        # 主要配置文件
│   └── __init__.py
├── dataset/               # 数据集相关
│   ├── data/             # JSONL数据文件
│   ├── dataset_fill50k/  # 原始数据集
│   ├── dataset.py        # 数据集类和数据加载器
│   ├── preprocess_data.py # 数据预处理脚本
│   └── __init__.py
├── models/                # 模型架构
│   ├── controlnet_dit.py # ControlNet-DiT模型
│   └── __init__.py
├── scripts/               # 训练和评估脚本
│   ├── train_controlnet_dit.py # 训练脚本
│   ├── inference.py      # 推理脚本
│   └── __init__.py
├── utils/                 # 工具函数
│   ├── training.py       # 训练工具函数
│   └── __init__.py
├── checkpoints/           # 模型检查点
├── logs/                  # 日志文件
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 实验阶段

### 第一阶段：Baseline 基准验证 ✅
使用 PixArt-alpha-XL-2 (0.6B) 在 Fill50k 数据集上进行基准训练，验证 RTX 3090Ti 环境可行性。

**快速开始**:
```bash
accelerate launch scripts/train_baseline.py \
  --dataset_name="./dataset/data" \
  --output_dir="./output/baseline" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --mixed_precision="bf16"
```

📖 [详细文档](BASELINE_README.md)

### 第二阶段：AdaLN Modulation 创新 (规划中)
将 Element-wise Add 替换为 AdaLN Modulation，观察性能提升。

### 第三阶段：完整评估体系 (✅ 已实现)
实现学术级评估指标，包括生成质量、控制精度、语义对齐和工程效率。

**核心指标**:
- **控制精度**: MSE, SSIM, IoU (边缘重合度)
- **生成质量**: FID, Inception Score
- **语义对齐**: CLIP Score
- **工程效率**: 参数量、显存占用、训练速度

**使用方法**:
```bash
python scripts/evaluation.py \
  --generated_images_dir ./output/generated \
  --condition_images_dir ./data/conditions \
  --real_images_dir ./data/real \
  --output_file evaluation_results.json
```

📖 [完整评估指南](EVALUATION_GUIDE.md)

### 第四阶段：Step-0 数值验证 (✅ 已实现)
在正式训练前自动验证模型初始化，确保零初始化（Zero-Linear）成功且输出正常。

**验证内容**:
- ✅ 检查模型输出是否包含 NaN
- ✅ 验证输出数值范围是否正常
- ✅ 确保 ZeroLinear 初始化生效
- ✅ 适用于 Baseline 和 AdaLN 两种模式

**自动执行**: 训练脚本启动时自动运行，无需手动干预。

## 快速开始

### 环境设置
```bash
pip install -r requirements.txt
```

### 数据准备
```bash
# 运行数据预处理
python dataset/preprocess_data.py

# 验证数据
python -c "from dataset import create_dataloaders; print('数据加载器创建成功')"
```

### 训练模型
```bash
# 开始训练
python scripts/train_controlnet_dit.py --config config/config.yaml

# 从检查点恢复训练
python scripts/train_controlnet_dit.py --config config/config.yaml --resume checkpoints/best_model.pth
```

### 推理
```bash
# 运行推理脚本
python scripts/inference.py --checkpoint checkpoints/best_model.pth --input_image path/to/condition.png
```

## 配置说明

主要配置文件位于 `config/config.yaml`，包含以下部分：

- **model**: 模型架构参数 (维度、头数、深度等)
- **training**: 训练超参数 (学习率、批次大小、epoch数等)
- **data**: 数据路径和预处理参数
- **diffusion**: 扩散过程参数
- **hardware**: 硬件和优化设置
- **logging**: 日志和检查点设置

## 数据集

使用 fill50k 数据集，包含50,000对条件-目标图像对：

- **训练集**: 45,000对
- **验证集**: 5,000对
- **分辨率**: 512x512
- **格式**: PNG图像，JSONL索引

## 硬件要求

- **GPU**: RTX 3090 Ti 或更高 (推荐24GB显存)
- **RAM**: 至少32GB
- **存储**: 至少500GB (数据集 + 检查点)

## 性能优化

- **批次大小**: 2 (适合RTX 3090 Ti)
- **混合精度**: 自动启用
- **梯度累积**: 支持大batch训练
- **多进程加载**: 4个worker进程

## 实验跟踪

支持 Weights & Biases 实验跟踪：

```yaml
logging:
  use_wandb: true
  wandb_project: "controlnet-dit"
```

## 引用

如果您在研究中使用了此代码，请引用：

```
ControlNet-DiT: Conditional Image Generation with Diffusion Transformers
```

## 许可证

MIT License

## 项目概述

本项目实现了 ControlNet-DiT 模型，用于基于条件图像生成目标图像。模型结合了：
- **DiT (Diffusion Transformer)**: 使用 Transformer 架构的扩散模型
- **ControlNet**: 通过条件图像控制生成过程
- **fill50k 数据集**: 50,000 对条件-目标图像对

## 环境要求

- Python 3.8+
- PyTorch 2.1.0+ (CUDA 12.1 推荐)
- RTX 3090 Ti 或类似 GPU
- 16GB+ VRAM

## 安装依赖

```bash
# 激活 Conda 环境
conda activate DitControlnet

# 安装 Python 包
pip install -r requirements.txt

# 可选：安装 flash-attn 以提升性能
pip install flash-attn --no-build-isolation
```

## 数据准备

项目已包含处理好的 fill50k 数据集：

```
fill50k/
├── source/     # 条件图像 (50,000 张)
└── target/     # 目标图像 (50,000 张)
```

如果需要重新处理数据：

```bash
# 下载原始数据集
python download_fill50k.py

# 处理 Parquet 格式数据
python process_fill50k.py
```

## 训练模型

### 基本训练

```bash
# 使用默认配置训练
python train_controlnet_dit.py
```

### 自定义配置

编辑 `config.yaml` 文件调整训练参数，然后运行：

```bash
python train_controlnet_dit.py --config config.yaml
```

### 从检查点恢复训练

```bash
python train_controlnet_dit.py --resume checkpoints/checkpoint_epoch_10.pth
```

## 推理生成

使用训练好的模型生成图像：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --condition_image fill50k/source/000000.png \
    --output_path generated.png \
    --num_steps 50
```

## 配置说明

### 模型配置 (config.yaml)

```yaml
model:
  img_size: 256          # 图像尺寸
  patch_size: 16         # Patch 大小
  dim: 768              # 模型维度
  depth: 12             # Transformer 层数
  num_heads: 12         # 注意力头数
  mlp_ratio: 4.0        # MLP 扩展比例

training:
  batch_size: 4         # 批次大小
  num_epochs: 100       # 训练轮数
  learning_rate: 1e-4   # 学习率
  weight_decay: 0.01    # 权重衰减
  gradient_clip_norm: 1.0  # 梯度裁剪

diffusion:
  num_timesteps: 1000   # 扩散步数
  beta_start: 0.0001    # Beta 起始值
  beta_end: 0.02        # Beta 结束值
  schedule_type: "linear"  # 调度类型
```

## 项目结构

```
controlnetDiT/
├── train_controlnet_dit.py    # 训练脚本
├── inference.py              # 推理脚本
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖列表
├── verify_enviroment.py     # 环境验证
├── download_fill50k.py      # 数据下载
├── process_fill50k.py       # 数据处理
├── fill50k/                # 处理后的数据集
│   ├── source/            # 条件图像
│   └── target/            # 目标图像
├── checkpoints/            # 模型检查点
├── logs/                  # 训练日志
└── samples/               # 生成样本
```

## 训练监控

项目支持 Weights & Biases 进行训练监控：

1. 确保 `config.yaml` 中 `logging.use_wandb: true`
2. 登录 W&B: `wandb login`
3. 训练时会自动记录损失、学习率等指标

## 性能优化

### 内存优化
- 使用 `xformers` 进行高效注意力计算
- 支持混合精度训练 (`mixed_precision: true`)
- 梯度累积以处理大批量

### 速度优化
- 使用 `torch.compile` 加速推理 (PyTorch 2.0+)
- 多进程数据加载 (`num_workers: 4`)
- 优化的扩散调度

## 故障排除

### CUDA 内存不足
- 减小 `batch_size`
- 启用梯度检查点
- 使用 `bitsandbytes` 进行量化

### 训练不稳定
- 启用梯度裁剪 (`gradient_clip_norm: 1.0`)
- 调整学习率
- 检查数据预处理

### 推理质量不佳
- 增加采样步数 (`num_steps`)
- 使用更好的检查点
- 调整温度参数

## 引用

如果使用本项目，请考虑引用相关论文：

```
@article{Peebles2023DiT,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2023}
}

@article{Zhang2023ControlNet,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}
```

## 许可证

本项目采用 Apache License 2.0 许可证。