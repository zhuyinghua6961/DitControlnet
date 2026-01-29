# ControlNet-DiT 评估指标使用指南

## 评估指标体系

根据学术标准，我们实现了完整的评估指标体系：

### 1. 空间控制精度 (Spatial Control Accuracy)
- **MSE (Mean Squared Error)**: 像素级差异，越低越好
- **SSIM (Structural Similarity)**: 结构相似性，0-1，越高越好
- **IoU (Intersection over Union)**: 边缘重合度，0-1，越高越好

### 2. 图像生成质量 (Image Quality & Fidelity)
- **FID (Fréchet Inception Distance)**: 分布相似性，越低越好
- **IS (Inception Score)**: 清晰度和多样性，越高越好

### 3. 语义一致性 (Semantic Alignment)
- **CLIP Score**: 图像-文本相似度，-1~1，越高越好

### 4. 工程效率 (Efficiency Metrics)
- **参数量**: 模型复杂度
- **显存占用**: GPU 内存使用
- **训练速度**: it/s (图像/秒)

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
# 可选：安装 CLIP 用于语义评估
pip install git+https://github.com/openai/CLIP.git
```

### 准备数据
确保有以下目录结构：
```
evaluation_data/
├── generated/     # 生成的图像
├── conditions/    # 条件图像（边缘图）
└── real/         # 真实图像（用于 FID）
```

### 运行评估
```bash
python scripts/evaluation.py \
  --generated_images_dir ./evaluation_data/generated \
  --condition_images_dir ./evaluation_data/conditions \
  --real_images_dir ./evaluation_data/real \
  --output_file results.json \
  --max_images 100
```

### 添加文本提示词（可选）
创建 `prompts.json`:
```json
{
  "image_0.png": "a red circle on white background",
  "image_1.png": "a blue square on white background"
}
```

然后运行：
```bash
python scripts/evaluation.py \
  --generated_images_dir ./evaluation_data/generated \
  --condition_images_dir ./evaluation_data/conditions \
  --prompts_file prompts.json \
  --output_file results.json
```

## 输出结果

评估脚本会生成详细的 JSON 结果：

```json
{
  "timestamp": "2024-01-28T10:30:00",
  "num_images": 100,
  "control_accuracy": {
    "control_mse": 125.34,
    "control_ssim": 0.876,
    "control_iou": 0.742,
    "clip_score": 0.312
  },
  "generation_quality": {
    "fid": 15.67,
    "inception_score_mean": 8.45,
    "inception_score_std": 0.23
  },
  "efficiency": {
    "total_parameters": 650000000,
    "trainable_parameters": 12000000,
    "peak_memory_mb": 8192
  }
}
```

## 成功判定标准

### Baseline 实验标准
- **控制精度**: IoU > 0.7, SSIM > 0.8
- **生成质量**: FID < 20 (与原模型差距 < 10%)
- **显存效率**: < 24GB RTX 3090Ti

### 创新实验对比
- **AdaLN Modulation**: IoU 提升 > 5%, FID 下降 > 10%
- **参数效率**: 参数量减少 > 30%, 性能不降

## 故障排除

### 依赖问题
```bash
# 安装 clean-fid
pip install clean-fid

# 安装 CLIP
pip install git+https://github.com/openai/CLIP.git

# 安装 scikit-image
pip install scikit-image
```

### 内存不足
- 减少 `--max_images` 参数
- 使用 CPU 模式评估部分指标

### 指标异常
- 检查图像格式（PNG, RGB）
- 确认条件图像是正确的边缘图
- 验证文件路径和命名一致性

## 学术引用

实现的指标基于以下论文：
- [ControlNet](https://arxiv.org/abs/2302.05543): 原始控制精度指标
- [FID论文](https://arxiv.org/abs/1706.08500): 分布距离度量
- [CLIP论文](https://arxiv.org/abs/2103.00020): 语义相似度评估