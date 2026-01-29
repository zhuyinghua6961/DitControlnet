#!/bin/bash

# 设置Hugging Face镜像和缓存目录
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/fast18/models_cache
export TRANSFORMERS_CACHE=/mnt/fast18/models_cache
export HF_DATASETS_CACHE=/mnt/fast18/models_cache/datasets
# 不设置离线模式，允许在需要时下载

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DitControlnet

# 运行训练脚本
python scripts/train_baseline.py --config config/config.yaml "$@"
