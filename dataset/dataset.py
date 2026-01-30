#!/usr/bin/env python3
"""
ControlNet-DiT 数据集类
实现空间对齐和异构插值的数据加载
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ControlNetDiTDataset(Dataset):
    """
    ControlNet-DiT 数据集类

    实现空间绝对对齐和异构插值策略
    """

    def __init__(self, jsonl_path, data_root, resolution=512, split='train'):
        """
        初始化数据集

        Args:
            jsonl_path: JSONL 文件路径
            data_root: 数据根目录
            resolution: 目标分辨率
            split: 数据集分割 ('train' 或 'eval')
        """
        self.jsonl_path = jsonl_path
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.split = split

        # 加载数据索引
        self.data = self._load_jsonl(jsonl_path)
        logger.info(f"加载了 {len(self.data)} 个 {split} 样本")

        # 数据增强（仅训练集）
        self.use_augmentation = (split == 'train')

    def _load_jsonl(self, jsonl_path):
        """加载 JSONL 文件"""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过无效的 JSON 行: {e}")
                    continue
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建完整路径
        target_path = self.data_root / item['image']
        conditioning_path = self.data_root / item['conditioning_image']

        # 加载图像
        try:
            target_image = Image.open(target_path).convert('RGB')
            conditioning_image = Image.open(conditioning_path).convert('RGB')
        except Exception as e:
            logger.error(f"无法加载图像 {target_path} 或 {conditioning_path}: {e}")
            # 返回一个空白图像作为替代
            target_image = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))
            conditioning_image = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))

        # 同步空间变换（空间绝对对齐）
        if self.use_augmentation:
            # 随机水平翻转
            if random.random() > 0.5:
                target_image = TF.hflip(target_image)
                conditioning_image = TF.hflip(conditioning_image)

            # 随机垂直翻转
            if random.random() > 0.5:
                target_image = TF.vflip(target_image)
                conditioning_image = TF.vflip(conditioning_image)

        # 异构插值缩放
        # Target (原图): 使用双线性插值保持平滑
        target_image = TF.resize(
            target_image,
            (self.resolution, self.resolution),
            interpolation=TF.InterpolationMode.BILINEAR
        )

        # Conditioning (条件图): 使用最近邻插值保持边缘锐度
        conditioning_image = TF.resize(
            conditioning_image,
            (self.resolution, self.resolution),
            interpolation=TF.InterpolationMode.NEAREST
        )

        # 转换为张量并归一化
        # Target: 归一化到 [-1, 1]
        target_tensor = TF.to_tensor(target_image) * 2.0 - 1.0

        # Conditioning: 归一化到 [0, 1]（或根据需要调整）
        conditioning_tensor = TF.to_tensor(conditioning_image)

        return {
            'pixel_values': target_tensor,                    # 目标图像 [-1, 1]
            'conditioning_pixel_values': conditioning_tensor, # 条件图像 [0, 1]
            'text': item['text'],                             # 文本提示
            'width': item.get('width', self.resolution),      # 原始宽度
            'height': item.get('height', self.resolution),    # 原始高度
        }

class AspectRatioBucketDataset(Dataset):
    """
    支持长宽比分桶的数据集类

    根据图像的长宽比将数据分组，提高训练效率
    """

    def __init__(self, jsonl_path, data_root, resolution=512, split='train',
                 bucket_ratios=None, bucket_batch_sizes=None):
        """
        初始化分桶数据集

        Args:
            jsonl_path: JSONL 文件路径
            data_root: 数据根目录
            resolution: 基础分辨率
            split: 数据集分割
            bucket_ratios: 长宽比分桶配置
            bucket_batch_sizes: 各分桶的批次大小
        """
        self.jsonl_path = jsonl_path
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.split = split

        # 默认分桶配置
        if bucket_ratios is None:
            bucket_ratios = [
                (1.0, 1.0),    # 1:1 正方形
                (4.0/3.0, 1.0), # 4:3 横向
                (3.0/4.0, 1.0), # 3:4 纵向
                (16.0/9.0, 1.0), # 16:9 宽屏
                (9.0/16.0, 1.0), # 9:16 竖屏
            ]

        if bucket_batch_sizes is None:
            bucket_batch_sizes = [4, 4, 4, 2, 2]  # 根据显存调整

        self.bucket_ratios = bucket_ratios
        self.bucket_batch_sizes = bucket_batch_sizes

        # 加载并分组数据
        self.buckets = self._create_buckets(jsonl_path)

        logger.info(f"创建了 {len(self.buckets)} 个分桶")
        for i, (ratio, batch_size, data) in enumerate(zip(bucket_ratios, bucket_batch_sizes, self.buckets)):
            logger.info(f"分桶 {i}: 比例 {ratio[0]:.2f}:{ratio[1]:.2f}, 批次大小 {batch_size}, 样本数 {len(data)}")

    def _create_buckets(self, jsonl_path):
        """根据长宽比创建数据分桶"""
        # 加载所有数据
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    all_data.append(item)
                except json.JSONDecodeError:
                    continue

        # 根据长宽比分组
        buckets = [[] for _ in self.bucket_ratios]

        for item in all_data:
            width = item.get('width', 512)
            height = item.get('height', 512)
            aspect_ratio = width / height

            # 找到最合适的分桶
            min_diff = float('inf')
            best_bucket = 0

            for i, (target_ratio, _) in enumerate(self.bucket_ratios):
                diff = abs(aspect_ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_bucket = i

            buckets[best_bucket].append(item)

        return buckets

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets)

    def __getitem__(self, idx):
        # 找到对应的分桶和索引
        cumulative = 0
        for bucket_idx, bucket in enumerate(self.buckets):
            if idx < cumulative + len(bucket):
                item_idx = idx - cumulative
                item = bucket[item_idx]
                break
            cumulative += len(bucket)
        else:
            raise IndexError("索引超出范围")

        # 构建完整路径
        target_path = self.data_root / item['image']
        conditioning_path = self.data_root / item['conditioning_image']

        # 加载图像（复用 ControlNetDiTDataset 的逻辑）
        try:
            target_image = Image.open(target_path).convert('RGB')
            conditioning_image = Image.open(conditioning_path).convert('RGB')
        except Exception as e:
            logger.error(f"无法加载图像: {e}")
            target_image = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))
            conditioning_image = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))

        # 计算目标分辨率（基于分桶比例）
        bucket_ratio = self.bucket_ratios[bucket_idx]
        if bucket_ratio[0] >= bucket_ratio[1]:  # 宽大于高
            target_width = self.resolution
            target_height = int(self.resolution / bucket_ratio[0])
        else:  # 高大于宽
            target_height = self.resolution
            target_width = int(self.resolution * bucket_ratio[0])

        # 确保最小尺寸
        target_width = max(target_width, 256)
        target_height = max(target_height, 256)

        # 数据增强（仅训练集）
        if self.split == 'train':
            if random.random() > 0.5:
                target_image = TF.hflip(target_image)
                conditioning_image = TF.hflip(conditioning_image)

        # 异构插值缩放
        target_image = TF.resize(
            target_image, (target_height, target_width),
            interpolation=TF.InterpolationMode.BILINEAR
        )
        conditioning_image = TF.resize(
            conditioning_image, (target_height, target_width),
            interpolation=TF.InterpolationMode.NEAREST
        )

        # 转换为张量
        target_tensor = TF.to_tensor(target_image) * 2.0 - 1.0
        conditioning_tensor = TF.to_tensor(conditioning_image)

        return {
            'pixel_values': target_tensor,
            'conditioning_pixel_values': conditioning_tensor,
            'text': item['text'],
            'width': target_width,
            'height': target_height,
            'bucket_idx': bucket_idx,
        }


def create_dataloaders(train_jsonl, eval_jsonl, data_root, batch_size=4,
                      resolution=512, num_workers=4, use_bucketing=False):
    """
    创建训练和验证数据加载器

    Args:
        train_jsonl: 训练数据 JSONL 路径
        eval_jsonl: 验证数据 JSONL 路径
        data_root: 数据根目录
        batch_size: 批次大小
        resolution: 图像分辨率
        num_workers: 数据加载线程数
        use_bucketing: 是否使用分桶

    Returns:
        训练和验证数据加载器
    """
    if use_bucketing:
        logger.info("使用分桶数据集...")
        train_dataset = AspectRatioBucketDataset(
            train_jsonl, data_root, resolution, 'train'
        )
        eval_dataset = AspectRatioBucketDataset(
            eval_jsonl, data_root, resolution, 'eval'
        )
    else:
        logger.info("使用标准数据集...")
        train_dataset = ControlNetDiTDataset(
            train_jsonl, data_root, resolution, 'train'
        )
        eval_dataset = ControlNetDiTDataset(
            eval_jsonl, data_root, resolution, 'eval'
        )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    logger.info(f"训练数据集: {len(train_dataset)} 样本")
    logger.info(f"验证数据集: {len(eval_dataset)} 样本")

    return train_loader, eval_loader


if __name__ == '__main__':
    # 测试数据集类
    import argparse

    parser = argparse.ArgumentParser(description='测试数据集类')
    parser.add_argument('--jsonl_path', type=str, default='data/train.jsonl')
    parser.add_argument('--data_root', type=str, default='dataset_fill50k')
    parser.add_argument('--test_samples', type=int, default=5)

    args = parser.parse_args()

    # 创建数据集
    dataset = ControlNetDiTDataset(args.jsonl_path, args.data_root)

    # 测试前几个样本
    for i in range(min(args.test_samples, len(dataset))):
        sample = dataset[i]
        logger.info(f"样本 {i}: 目标图像形状 {sample['pixel_values'].shape}, "
                   f"条件图像形状 {sample['conditioning_pixel_values'].shape}, "
                   f"文本: {sample['text'][:50]}...")

    logger.info("数据集测试完成！")