#!/usr/bin/env python3
"""
ControlNet-DiT 数据预处理脚本
将 CSV 格式转换为 JSONL 格式，并预计算图像尺寸用于分桶
"""

import pandas as pd
import json
import os
from PIL import Image
from tqdm import tqdm
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_csv_to_jsonl(csv_path, output_path, source_base, target_base):
    """
    将 CSV 文件转换为 JSONL 格式

    Args:
        csv_path: CSV 文件路径
        output_path: 输出 JSONL 文件路径
        source_base: source 图像基础路径
        target_base: target 图像基础路径
    """
    logger.info(f"开始处理 {csv_path} -> {output_path}")

    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    logger.info(f"CSV 文件包含 {len(df)} 行数据")

    processed_count = 0
    skipped_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
            try:
                # 从 CSV 中提取相对路径
                # CSV 格式: Prompt,Cond:FILE,Target:FILE
                # 例如: "red circle",source/0.png,target/0.png

                cond_file = row['Cond:FILE']  # source/0.png
                target_file = row['Target:FILE']  # target/0.png
                prompt = row['Prompt']

                # 拼接完整路径
                target_full_path = os.path.join(target_base, os.path.basename(target_file))
                source_full_path = os.path.join(source_base, os.path.basename(cond_file))

                # 检查文件是否存在
                if not os.path.exists(target_full_path):
                    logger.warning(f"目标文件不存在: {target_full_path}")
                    skipped_count += 1
                    continue

                if not os.path.exists(source_full_path):
                    logger.warning(f"条件文件不存在: {source_full_path}")
                    skipped_count += 1
                    continue

                # 获取图像尺寸（用于分桶）
                try:
                    with Image.open(target_full_path) as img:
                        width, height = img.size
                except Exception as e:
                    logger.warning(f"无法读取图像尺寸 {target_full_path}: {e}")
                    skipped_count += 1
                    continue

                # 构造 JSONL 数据行
                data_line = {
                    "text": prompt,
                    "image": target_file,  # 保持相对路径
                    "conditioning_image": cond_file,  # 保持相对路径
                    "width": width,
                    "height": height
                }

                # 写入 JSONL 文件
                f.write(json.dumps(data_line, ensure_ascii=False) + '\n')
                processed_count += 1

            except Exception as e:
                logger.error(f"处理第 {idx} 行时出错: {e}")
                skipped_count += 1
                continue

    logger.info(f"转换完成: {processed_count} 成功, {skipped_count} 跳过")
    logger.info(f"输出文件: {output_path}")

    return processed_count, skipped_count

def validate_dataset(source_dir, target_dir, jsonl_path):
    """
    验证数据集完整性
    """
    logger.info("开始验证数据集完整性...")

    # 读取 JSONL 文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_samples = len(lines)
    valid_samples = 0
    missing_files = 0

    for line in tqdm(lines, desc="验证文件"):
        try:
            data = json.loads(line.strip())

            target_path = os.path.join(target_dir, os.path.basename(data['image']))
            source_path = os.path.join(source_dir, os.path.basename(data['conditioning_image']))

            if os.path.exists(target_path) and os.path.exists(source_path):
                valid_samples += 1
            else:
                missing_files += 1
                if not os.path.exists(target_path):
                    logger.warning(f"目标文件缺失: {target_path}")
                if not os.path.exists(source_path):
                    logger.warning(f"条件文件缺失: {source_path}")

        except Exception as e:
            logger.error(f"验证时出错: {e}")
            missing_files += 1

    logger.info(f"验证完成: {valid_samples}/{total_samples} 有效, {missing_files} 缺失")
    return valid_samples, missing_files

def main():
    parser = argparse.ArgumentParser(description='ControlNet-DiT 数据预处理')
    parser.add_argument('--data_dir', type=str, default='dataset_fill50k',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='输出目录路径')
    parser.add_argument('--validate', action='store_true',
                        help='是否进行数据验证')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义路径
    train_csv = os.path.join(args.data_dir, 'train.csv')
    eval_csv = os.path.join(args.data_dir, 'eval.csv')
    source_dir = os.path.join(args.data_dir, 'source')
    target_dir = os.path.join(args.data_dir, 'target')

    train_jsonl = os.path.join(args.output_dir, 'train.jsonl')
    eval_jsonl = os.path.join(args.output_dir, 'eval.jsonl')

    # 检查输入文件是否存在
    if not os.path.exists(train_csv):
        logger.error(f"训练 CSV 文件不存在: {train_csv}")
        return

    if not os.path.exists(eval_csv):
        logger.error(f"验证 CSV 文件不存在: {eval_csv}")
        return

    if not os.path.exists(source_dir):
        logger.error(f"Source 目录不存在: {source_dir}")
        return

    if not os.path.exists(target_dir):
        logger.error(f"Target 目录不存在: {target_dir}")
        return

    logger.info("开始数据预处理...")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # 转换训练数据
    logger.info("转换训练数据...")
    train_processed, train_skipped = convert_csv_to_jsonl(
        train_csv, train_jsonl, source_dir, target_dir
    )

    # 转换验证数据
    logger.info("转换验证数据...")
    eval_processed, eval_skipped = convert_csv_to_jsonl(
        eval_csv, eval_jsonl, source_dir, target_dir
    )

    # 输出统计信息
    logger.info("=" * 50)
    logger.info("预处理完成统计:")
    logger.info(f"训练集: {train_processed} 成功, {train_skipped} 跳过")
    logger.info(f"验证集: {eval_processed} 成功, {eval_skipped} 跳过")
    logger.info(f"总计: {train_processed + eval_processed} 成功, {train_skipped + eval_skipped} 跳过")

    # 验证数据集（如果启用）
    if args.validate:
        logger.info("开始验证数据集...")

        logger.info("验证训练集...")
        train_valid, train_missing = validate_dataset(source_dir, target_dir, train_jsonl)

        logger.info("验证验证集...")
        eval_valid, eval_missing = validate_dataset(source_dir, target_dir, eval_jsonl)

        logger.info("=" * 50)
        logger.info("验证结果:")
        logger.info(f"训练集: {train_valid} 有效, {train_missing} 缺失")
        logger.info(f"验证集: {eval_valid} 有效, {eval_missing} 缺失")

        if train_missing == 0 and eval_missing == 0:
            logger.info("✅ 数据集验证通过！所有文件都存在且可访问。")
        else:
            logger.warning("⚠️ 数据集验证发现缺失文件，请检查数据完整性。")

    logger.info("数据预处理完成！")
    logger.info(f"输出文件: {train_jsonl}, {eval_jsonl}")

if __name__ == '__main__':
    main()