#!/usr/bin/env python3
"""
ControlNet-DiT Evaluation Metrics
评估生成图像质量、控制精度和语义对齐的完整指标体系
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import argparse
import logging
import yaml
from tqdm import tqdm
import math
import json
from datetime import datetime

# 可选依赖
try:
    from cleanfid import fid
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from torchvision.models import inception_v3
    from torchvision import transforms
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error as mse
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_config(config_path="./config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class ControlNetEvaluator:
    """ControlNet 评估器"""

    def __init__(self, device='cuda'):
        self.device = device

        # 初始化 CLIP 模型（用于 CLIP Score）
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            logger.info("CLIP model loaded for semantic evaluation")
        else:
            logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

        # 初始化 Inception 模型（用于 IS）
        if INCEPTION_AVAILABLE:
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
            self.inception_model.eval()
            logger.info("Inception model loaded for IS calculation")
        else:
            logger.warning("Inception model not available")

    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """
        使用 Canny 算子提取边缘

        Args:
            image: PIL Image 或 numpy array
            low_threshold: Canny 低阈值
            high_threshold: Canny 高阈值

        Returns:
            边缘图 (numpy array, 0-255)
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))  # 转换为灰度
        elif isinstance(image, torch.Tensor):
            # 假设是 [C, H, W] 或 [B, C, H, W]
            if image.dim() == 4:
                image = image[0]  # 取第一张
            if image.shape[0] == 3:  # RGB to grayscale
                image = image.mean(dim=0).cpu().numpy()
            else:
                image = image.cpu().numpy()
            image = (image * 255).astype(np.uint8)

        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def calculate_mse(self, pred_image, target_image):
        """
        计算均方误差 (MSE)

        Args:
            pred_image: 生成图像
            target_image: 目标图像

        Returns:
            MSE 值
        """
        if isinstance(pred_image, Image.Image):
            pred_image = np.array(pred_image.convert('L'))
        if isinstance(target_image, Image.Image):
            target_image = np.array(target_image.convert('L'))

        if SKIMAGE_AVAILABLE:
            return mse(pred_image.astype(float), target_image.astype(float))
        else:
            # 使用 numpy 计算 MSE
            return np.mean((pred_image.astype(float) - target_image.astype(float)) ** 2)

    def calculate_ssim(self, pred_image, target_image):
        """
        计算结构相似性 (SSIM)

        Args:
            pred_image: 生成图像
            target_image: 目标图像

        Returns:
            SSIM 值
        """
        if isinstance(pred_image, Image.Image):
            pred_image = np.array(pred_image.convert('L'))
        if isinstance(target_image, Image.Image):
            target_image = np.array(target_image.convert('L'))

        if SKIMAGE_AVAILABLE:
            return ssim(pred_image, target_image, data_range=255)
        else:
            # 简化的 SSIM 实现
            logger.warning("scikit-image not available, using simplified SSIM calculation")
            # 简单的相关系数作为替代
            pred_norm = pred_image.astype(float) / 255.0
            target_norm = target_image.astype(float) / 255.0
            return np.corrcoef(pred_norm.flatten(), target_norm.flatten())[0, 1]

    def calculate_iou(self, pred_edges, target_edges, threshold=127):
        """
        计算交并比 (IoU) 用于边缘检测

        Args:
            pred_edges: 预测边缘图
            target_edges: 目标边缘图
            threshold: 二值化阈值

        Returns:
            IoU 值
        """
        if isinstance(pred_edges, Image.Image):
            pred_edges = np.array(pred_edges.convert('L'))
        if isinstance(target_edges, Image.Image):
            target_edges = np.array(target_edges.convert('L'))

        # 二值化
        pred_binary = (pred_edges > threshold).astype(np.uint8)
        target_binary = (target_edges > threshold).astype(np.uint8)

        # 计算 IoU
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()

        if union == 0:
            return 1.0  # 如果都没有边缘，返回1

        return intersection / union

    def calculate_fid(self, generated_images_path, real_images_path):
        """
        计算 Fréchet Inception Distance (FID)

        Args:
            generated_images_path: 生成图像文件夹路径
            real_images_path: 真实图像文件夹路径

        Returns:
            FID 分数
        """
        if not CLEANFID_AVAILABLE:
            logger.error("clean-fid not available. Install with: pip install clean-fid")
            return None

        try:
            score = fid.compute_fid(generated_images_path, real_images_path)
            return score
        except Exception as e:
            logger.error(f"FID calculation failed: {e}")
            return None

    def calculate_clip_score(self, images, prompts):
        """
        计算 CLIP Score (图像-文本相似度)

        Args:
            images: PIL Images 列表
            prompts: 对应的文本提示词列表

        Returns:
            平均 CLIP Score
        """
        if not CLIP_AVAILABLE:
            logger.error("CLIP not available")
            return None

        scores = []
        with torch.no_grad():
            for image, prompt in zip(images, prompts):
                # 预处理图像
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

                # 编码文本
                text_input = clip.tokenize([prompt]).to(self.device)

                # 计算相似度
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)

                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 计算余弦相似度
                similarity = (image_features * text_features).sum(dim=-1).item()
                scores.append(similarity)

        return np.mean(scores)

    def calculate_inception_score(self, images, splits=10):
        """
        计算 Inception Score (IS)

        Args:
            images: PIL Images 列表
            splits: 分割数量

        Returns:
            IS 分数
        """
        if not INCEPTION_AVAILABLE:
            logger.error("Inception model not available")
            return None

        # 预处理图像
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 处理所有图像
        preds = []
        for img in images:
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.inception_model(img_tensor)
                preds.append(F.softmax(pred, dim=1).cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # 计算 IS
        scores = []
        for i in range(splits):
            part = preds[i * len(preds) // splits: (i + 1) * len(preds) // splits]
            kl_div = part * (np.log(part) - np.log(np.mean(part, axis=0)))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            scores.append(np.exp(kl_div))

        return np.mean(scores), np.std(scores)

    def evaluate_control_accuracy(self, generated_images, condition_images, prompts=None):
        """
        评估控制精度指标

        Args:
            generated_images: 生成的图像列表 (PIL Images)
            condition_images: 条件图像列表 (PIL Images)
            prompts: 可选的文本提示词列表

        Returns:
            包含各种指标的字典
        """
        results = {}

        mse_scores = []
        ssim_scores = []
        iou_scores = []

        for gen_img, cond_img in tqdm(zip(generated_images, condition_images),
                                     desc="Evaluating control accuracy"):

            # 提取生成图像的边缘
            gen_edges = self.canny_edge_detection(gen_img)

            # 条件图像就是边缘图，直接使用
            cond_edges = np.array(cond_img.convert('L')) if isinstance(cond_img, Image.Image) else cond_img

            # 计算指标
            mse_score = self.calculate_mse(gen_edges, cond_edges)
            ssim_score = self.calculate_ssim(gen_edges, cond_edges)
            iou_score = self.calculate_iou(gen_edges, cond_edges)

            mse_scores.append(mse_score)
            ssim_scores.append(ssim_score)
            iou_scores.append(iou_score)

        results['control_mse'] = np.mean(mse_scores)
        results['control_ssim'] = np.mean(ssim_scores)
        results['control_iou'] = np.mean(iou_scores)

        # CLIP Score (如果有文本提示词)
        if prompts and CLIP_AVAILABLE:
            clip_score = self.calculate_clip_score(generated_images, prompts)
            results['clip_score'] = clip_score

        return results

    def evaluate_generation_quality(self, generated_images_path, real_images_path=None):
        """
        评估生成质量指标

        Args:
            generated_images_path: 生成图像文件夹路径
            real_images_path: 真实图像文件夹路径 (用于 FID)

        Returns:
            包含质量指标的字典
        """
        results = {}

        # FID (需要真实图像)
        if real_images_path and CLEANFID_AVAILABLE:
            fid_score = self.calculate_fid(generated_images_path, real_images_path)
            if fid_score is not None:
                results['fid'] = fid_score

        # IS (从生成图像计算)
        try:
            from PIL import Image
            import os

            generated_images = []
            for img_file in Path(generated_images_path).glob("*.png"):
                generated_images.append(Image.open(img_file).convert('RGB'))

            if generated_images:
                is_mean, is_std = self.calculate_inception_score(generated_images)
                results['inception_score_mean'] = is_mean
                results['inception_score_std'] = is_std

        except Exception as e:
            logger.error(f"IS calculation failed: {e}")

        return results

    def evaluate_efficiency(self, model, input_shape=(1, 3, 512, 512)):
        """
        评估工程效率指标

        Args:
            model: PyTorch 模型
            input_shape: 输入形状 (B, C, H, W)

        Returns:
            包含效率指标的字典
        """
        results = {}

        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results['total_parameters'] = total_params
        results['trainable_parameters'] = trainable_params

        # 尝试计算显存占用
        if torch.cuda.is_available():
            try:
                # 创建虚拟输入
                dummy_input = torch.randn(*input_shape).to('cuda')
                dummy_cond = torch.randn(*input_shape).to('cuda')
                dummy_t = torch.randint(0, 1000, (input_shape[0],)).to('cuda')

                # 前向传播
                model.to('cuda')
                with torch.no_grad():
                    _ = model(dummy_input, dummy_cond, dummy_t)

                # 获取显存使用情况
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB

                results['peak_memory_mb'] = memory_allocated
                results['reserved_memory_mb'] = memory_reserved

            except Exception as e:
                logger.error(f"Memory evaluation failed: {e}")

        return results


def load_images_from_directory(directory, max_images=None):
    """
    从目录加载图像

    Args:
        directory: 图像目录路径
        max_images: 最大加载数量

    Returns:
        PIL Images 列表
    """
    images = []
    for img_path in Path(directory).glob("*.png"):
        if max_images and len(images) >= max_images:
            break
        try:
            images.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")

    return images


def main():
    parser = argparse.ArgumentParser(description='ControlNet-DiT Evaluation')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                        help='Configuration file path')
    parser.add_argument('--generated_images_dir', type=str, default=None,
                        help='Directory containing generated images (overrides config)')
    parser.add_argument('--condition_images_dir', type=str, default=None,
                        help='Directory containing condition images (overrides config)')
    parser.add_argument('--real_images_dir', type=str, default=None,
                        help='Directory containing real images (overrides config)')
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='JSON file containing prompts (overrides config)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for results (overrides config)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to evaluate (overrides config)')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Model checkpoint path (overrides config)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    eval_config = config.get('evaluation', {})

    # Override config with command line args if provided
    generated_images_dir = args.generated_images_dir or eval_config.get('generated_images_dir', "./output/generated")
    condition_images_dir = args.condition_images_dir or eval_config.get('condition_images_dir', "./data/conditions")
    real_images_dir = args.real_images_dir or eval_config.get('real_images_dir')
    prompts_file = args.prompts_file or eval_config.get('prompts_file')
    output_file = args.output_file or eval_config.get('output_file', 'evaluation_results.json')
    max_images = args.max_images or eval_config.get('max_images', 100)
    model_checkpoint = args.model_checkpoint or eval_config.get('model_checkpoint')

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting ControlNet-DiT evaluation...")

    # 初始化评估器
    evaluator = ControlNetEvaluator()

    # 加载图像
    logger.info("Loading images...")
    generated_images = load_images_from_directory(generated_images_dir, max_images)
    condition_images = load_images_from_directory(condition_images_dir, max_images)

    if len(generated_images) != len(condition_images):
        logger.error("Number of generated and condition images must match")
        return

    logger.info(f"Loaded {len(generated_images)} image pairs")

    # 加载提示词（如果有）
    prompts = None
    if prompts_file:
        try:
            with open(prompts_file, 'r') as f:
                prompts_data = json.load(f)
                prompts = [prompts_data.get(f"image_{i}.png", "") for i in range(len(generated_images))]
        except Exception as e:
            logger.warning(f"Failed to load prompts: {e}")

    # 评估控制精度
    logger.info("Evaluating control accuracy...")
    control_results = evaluator.evaluate_control_accuracy(generated_images, condition_images, prompts)

    # 评估生成质量
    logger.info("Evaluating generation quality...")
    quality_results = evaluator.evaluate_generation_quality(generated_images_dir, real_images_dir)

    # 评估效率（如果有模型）
    efficiency_results = {}
    if model_checkpoint:
        logger.info("Evaluating efficiency...")
        try:
            # 这里需要根据您的模型结构加载
            # efficiency_results = evaluator.evaluate_efficiency(model)
            logger.info("Efficiency evaluation requires model loading implementation")
        except Exception as e:
            logger.error(f"Efficiency evaluation failed: {e}")

    # 汇总结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(generated_images),
        'control_accuracy': control_results,
        'generation_quality': quality_results,
        'efficiency': efficiency_results
    }

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation completed. Results saved to {output_file}")

    # 打印关键指标
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    if 'control_mse' in control_results:
        print(".4f")
    if 'control_ssim' in control_results:
        print(".4f")
    if 'control_iou' in control_results:
        print(".4f")
    if 'clip_score' in control_results:
        print(".4f")
    if 'fid' in quality_results:
        print(".2f")
    if 'inception_score_mean' in quality_results:
        print(".2f")
    if 'total_parameters' in efficiency_results:
        print(f"Total Parameters: {efficiency_results['total_parameters']:,}")
    if 'peak_memory_mb' in efficiency_results:
        print(".1f")

    print("="*50)


if __name__ == '__main__':
    main()