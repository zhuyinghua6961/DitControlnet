# ControlNet-DiT Baseline 实验指南

## 实验概述

本实验使用 PixArt-alpha-XL-2 (0.6B) 作为基座模型，在 Fill50k 数据集上进行 ControlNet 基准训练，验证 RTX 3090Ti (24GB) 环境下的训练可行性。

## 核心配置

### 硬件优化 (针对 24GB 显存)
- **混合精度**: BF16
- **优化器**: 8-bit AdamW (bitsandbytes)
- **梯度检查点**: 启用
- **Batch Size**: 1 (配合梯度累积使用)
- **梯度累积**: 4 步

### 模型架构
- **基座模型**: PixArt-alpha-XL-2 (0.6B)
- **ControlNet**: 锁定基座，复制分支
- **注入方式**: Element-wise Add
- **初始化**: 零卷积

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 安装 flash-attn (可选，提升性能)
pip install flash-attn --no-build-isolation
```

### 2. 数据准备
确保 Fill50k 数据集已处理完成，位于 `dataset_fill50k/` 目录下。

### 3. 启动训练
```bash
# 简化的启动方式 (推荐)
python scripts/train_baseline.py

# 或者使用 accelerate 进行分布式训练
accelerate launch scripts/train_baseline.py

# 如需自定义配置，可以修改 config/config.yaml 或使用命令行覆盖
python scripts/train_baseline.py --config path/to/custom/config.yaml
```

### 4. 配置文件说明
所有训练参数现在集中在 `config/config.yaml` 中：

```yaml
# 训练配置示例
training:
  batch_size: 2          # RTX 3090Ti 优化批大小
  num_epochs: 100
  learning_rate: 1e-4
  gradient_clip_norm: 1.0

# 硬件配置
hardware:
  mixed_precision: "bf16"    # BF16 混合精度
  gradient_checkpointing: true
  use_8bit_adam: true        # 8-bit AdamW 优化器
  allow_tf32: true          # TF32 加速

# 数据配置
data:
  data_dir: "./dataset_fill50k"
  resolution: 512
  num_workers: 8
```

### 4. 监控训练
- 使用 TensorBoard: `tensorboard --logdir=output/baseline`
- 或 WandB: 设置 `--report_to="wandb"`

## 成功判定标准

1. **收敛性**: Loss 在 2000 步内无 NaN 发散，呈对数下降
2. **显存占用**: 峰值 < 24GB
3. **推理质量**: 生成图像边缘与轮廓对齐，无明显偏移

## 故障排除

### 显存不足
- 减少 `gradient_accumulation_steps` 到 2
- 启用 `--enable_xformers_memory_efficient_attention`

### 训练不稳定
- 检查数据预处理是否正确
- 降低学习率 `--learning_rate=1e-6`

### 导入错误
- 确保 diffusers >= 0.26.0
- 更新 bitsandbytes: `pip install --upgrade bitsandbytes`

## 下一步

完成 Baseline 实验后，可以进行第二步实验：
- 将 "Element-wise Add" 替换为 "AdaLN Modulation"
- 比较性能指标提升