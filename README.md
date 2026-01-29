# ControlNet-DiT: 条件图像生成

基于 DiT (Diffusion Transformer) 和 ControlNet 的条件图像生成模型，支持 fill50k 数据集的高效训练。

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
- **学习率调度**: Warmup + Cosine Annealing
- **WandB集成**: 实验跟踪

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