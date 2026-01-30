import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

def create_synthetic_fill50k_dataset():
    """
    创建合成的 fill50k 数据集用于测试
    生成简单的圆形填充图案
    """
    print("创建合成的 fill50k 数据集用于测试...")

    # 创建目录
    os.makedirs("./fill50k/source", exist_ok=True)     # 条件图像（圆圈轮廓）
    os.makedirs("./fill50k/target", exist_ok=True)     # 目标图像（填充后的）

    # 生成参数
    num_samples = 1000
    image_size = (512, 512)

    colors = [
        'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan',
        'magenta', 'lime', 'brown', 'gray', 'navy', 'maroon', 'olive', 'teal'
    ]

    backgrounds = ['white', 'lightgray', 'black', 'lightblue', 'lightgreen']

    for i in range(num_samples):
        # 随机选择颜色
        circle_color = colors[i % len(colors)]
        bg_color = backgrounds[i % len(backgrounds)]

        # 创建条件图像（只有圆圈轮廓）
        condition_img = Image.new('RGB', image_size, bg_color)
        draw_condition = ImageDraw.Draw(condition_img)

        # 绘制圆圈轮廓
        center = (256, 256)
        radius = 100 + (i % 50)  # 稍微变化大小
        draw_condition.ellipse(
            [(center[0] - radius, center[1] - radius),
             (center[0] + radius, center[1] + radius)],
            outline=circle_color,
            width=5
        )

        # 创建目标图像（填充的圆圈）
        target_img = Image.new('RGB', image_size, bg_color)
        draw_target = ImageDraw.Draw(target_img)

        # 绘制填充圆圈
        draw_target.ellipse(
            [(center[0] - radius, center[1] - radius),
             (center[0] + radius, center[1] + radius)],
            fill=circle_color
        )

        # 保存图像
        condition_img.save(f"./fill50k/source/{i:06d}.png")
        target_img.save(f"./fill50k/target/{i:06d}.png")

        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1} 对图像...")

    print("\n✅ 合成数据集创建完成！")
    print(f"📁 保存位置: ./fill50k/")
    print(f"🖼️  条件图像数量: {num_samples} (./fill50k/source/)")
    print(f"🎯 目标图像数量: {num_samples} (./fill50k/target/)")
    print("\n💡 数据集说明:")
    print("   - source/: 圆圈轮廓（ControlNet 条件输入）")
    print("   - target/: 填充圆圈（训练目标）")
    print("   - 每对图像文件名对应（000000.png, 000001.png, ...）")

def download_real_fill50k_dataset():
    """
    尝试下载真实的 fill50k 数据集
    """
    print("尝试下载真实的 fill50k 数据集...")

    try:
        # 使用更稳定的下载方式
        import subprocess
        result = subprocess.run([
            'git', 'lfs', 'clone',
            'https://huggingface.co/datasets/HighCWu/fill50k',
            './fill50k_real'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ 真实数据集下载成功！")
            return True
        else:
            print(f"Git LFS 下载失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"下载失败: {e}")
        return False

def main():
    """
    主函数：优先下载真实数据集，不行则创建合成数据集
    """
    print("=== Fill50k 数据集准备工具 ===\n")

    # 首先尝试下载真实数据集
    if download_real_fill50k_dataset():
        print("使用真实数据集")
    else:
        print("下载失败，使用合成数据集进行测试")
        create_synthetic_fill50k_dataset()

    print("\n🎉 数据集准备完成！可以开始训练 ControlNet 模型了。")

if __name__ == "__main__":
    main()
