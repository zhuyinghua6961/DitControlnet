import os
import pandas as pd
from PIL import Image
import io

def process_parquet_dataset():
    """
    处理下载的 Parquet 格式的 fill50k 数据集
    """
    print("处理 Parquet 格式的 fill50k 数据集...")

    parquet_file = "./fill50k_real/data/train-00000-of-00001-0c22d75a94d68109.parquet"
    
    if not os.path.exists(parquet_file):
        print(f"❌ 未找到 Parquet 文件: {parquet_file}")
        return

    # 创建输出目录
    os.makedirs("./fill50k/source", exist_ok=True)
    os.makedirs("./fill50k/target", exist_ok=True)

    print("加载 Parquet 文件...")
    df = pd.read_parquet(parquet_file)
    print(f"数据集包含 {len(df)} 个样本")

    # 处理全部样本 (50000个)
    max_samples = len(df)  # 使用数据集的实际大小
    processed_count = 0

    for idx, row in df.iterrows():
        if processed_count >= max_samples:
            break

        try:
            # 获取图像数据
            if 'image' in row:
                # image 列包含目标图像数据
                image_data = row['image']
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    # 如果是字节数据
                    img = Image.open(io.BytesIO(image_data['bytes']))
                else:
                    # 如果已经是 PIL 图像
                    img = image_data
                
                # 保存为目标图像
                img.save(f"./fill50k/target/{processed_count:06d}.png")

            # 获取条件图像数据
            if 'guide' in row:
                cond_image_data = row['guide']
                if isinstance(cond_image_data, dict) and 'bytes' in cond_image_data:
                    # 如果是字节数据
                    cond_img = Image.open(io.BytesIO(cond_image_data['bytes']))
                else:
                    # 如果已经是 PIL 图像
                    cond_img = cond_image_data
                
                # 保存为条件图像
                cond_img.save(f"./fill50k/source/{processed_count:06d}.png")

                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 个样本...")

        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            continue

    print(f"\n✅ 数据集处理完成！")
    print(f"📁 保存位置: ./fill50k/")
    print(f"🖼️  条件图像数量: {processed_count} (./fill50k/source/)")
    print(f"🎯 目标图像数量: {processed_count} (./fill50k/target/)")

if __name__ == "__main__":
    process_parquet_dataset()
