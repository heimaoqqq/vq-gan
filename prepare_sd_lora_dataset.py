"""
准备SD+LoRA训练/验证数据集
按照Diffusers官方格式创建metadata.jsonl
支持同时准备训练集和验证集
"""

import json
from pathlib import Path
import argparse
from tqdm import tqdm


def create_metadata_jsonl(
    data_root="/kaggle/input/organized-gait-dataset/Normal_line",
    output_dir="./sd_lora_dataset",
    split_file="./latents_cache/data_split.json",
    resolution=512,
    images_per_user=None,
    dataset_type="train"
):
    """
    创建符合Diffusers格式的数据集
    
    Args:
        data_root: 数据集根目录
        output_dir: 输出目录
        split_file: 数据划分文件
        resolution: 图像分辨率
        images_per_user: 每用户使用的图像数（可选）
        dataset_type: 数据集类型（"train" 或 "val"）
    
    目录结构:
    sd_lora_dataset/
    ├── images/
    │   ├── ID_1_001.jpg
    │   ├── ID_1_002.jpg
    │   └── ...
    └── metadata.jsonl
    
    metadata.jsonl格式：
    {"file_name": "ID_1_001.jpg", "text": "user 0"}
    {"file_name": "ID_1_002.jpg", "text": "user 0"}
    ...
    """
    
    dataset_name = "训练集" if dataset_type == "train" else "验证集"
    print("="*60)
    print(f"准备SD+LoRA{dataset_name}")
    print("="*60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据划分信息
    print(f"\n1. 加载数据划分信息...")
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    
    print(f"   用户数: {split_info['num_users']}")
    print(f"   每用户训练样本: {split_info['images_per_user_train']}")
    print(f"   采样方法: {split_info['sampling_method']}")
    
    # 收集图像并创建metadata
    print(f"\n2. 收集{dataset_name}图像并创建metadata...")
    data_root = Path(data_root)
    metadata_entries = []
    
    # 根据dataset_type选择对应的图像列表
    if dataset_type == "train":
        image_key = 'train_images'
    elif dataset_type == "val":
        image_key = 'val_images'
        # 检查是否有验证集
        if split_info.get('images_per_user_val', 0) == 0:
            print(f"\n错误: data_split.json中没有验证集划分")
            print(f"请使用--images_per_user_val参数重新运行preprocess_latents.py")
            return None
    else:
        raise ValueError(f"不支持的dataset_type: {dataset_type}")
    
    total_images = 0
    for user_key, user_info in tqdm(split_info['users'].items(), desc="处理用户"):
        user_id = user_info['user_id']
        label = user_info['label']  # 0-30
        
        # 检查是否有对应的图像列表
        if image_key not in user_info:
            if dataset_type == "val":
                print(f"   Warning: {user_key} 没有验证集，跳过")
            continue
        
        # 获取图像列表
        images = user_info[image_key]
        
        # 如果指定了images_per_user，只使用前N张
        if images_per_user is not None:
            images = images[:images_per_user]
        
        for img_relative_path in images:
            src_path = data_root / img_relative_path
            
            if not src_path.exists():
                print(f"   Warning: {src_path} 不存在，跳过")
                continue
            
            # 创建新的文件名：ID_{user_id}_{type}_{idx}.jpg
            idx = len([e for e in metadata_entries if e['text'] == f"user {label}"])
            suffix = "_val" if dataset_type == "val" else ""
            new_filename = f"ID_{user_id:02d}{suffix}_{idx:03d}.jpg"
            dst_path = images_dir / new_filename
            
            # 复制图像（或创建符号链接）
            if not dst_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)
            
            # 创建metadata条目
            # 关键：text字段用于条件生成
            # 格式："user {label}"，其中label是0-30
            metadata_entries.append({
                "file_name": new_filename,
                "text": f"user {label}"  # 条件文本
            })
            
            total_images += 1
    
    print(f"   ✓ 收集了 {total_images} 张{dataset_name}图像")
    
    # 保存metadata.jsonl
    print(f"\n3. 保存metadata.jsonl...")
    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"   ✓ 保存到 {metadata_file}")
    
    # 统计信息
    print(f"\n{'='*60}")
    print("数据集准备完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_path}")
    print(f"图像目录: {images_dir}")
    print(f"Metadata: {metadata_file}")
    print(f"总图像数: {total_images}")
    print(f"用户数: {split_info['num_users']}")
    print(f"条件文本格式: 'user 0', 'user 1', ..., 'user 30'")
    
    # 显示一些示例
    print(f"\n示例metadata条目:")
    for i, entry in enumerate(metadata_entries[:5]):
        print(f"  {entry}")
    
    if dataset_type == "train":
        print(f"\n现在可以运行SD+LoRA训练:")
        print(f"  python train_sd_lora.py")
    else:
        print(f"\n现在可以运行SD+LoRA训练（使用验证集）:")
        print(f"  python train_sd_lora.py --val_dataset_path {output_dir}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='准备SD+LoRA训练/验证数据集')
    
    parser.add_argument('--data_root', type=str,
                        default='/kaggle/input/organized-gait-dataset/Normal_line',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str,
                        default='./sd_lora_dataset',
                        help='输出目录')
    parser.add_argument('--split_file', type=str,
                        default='./latents_cache/data_split.json',
                        help='数据划分文件')
    parser.add_argument('--resolution', type=int, default=512,
                        help='图像分辨率（SD默认512）')
    parser.add_argument('--images_per_user', type=int, default=None,
                        help='每用户使用的图像数（可选，用于限制数量）')
    parser.add_argument('--dataset_type', type=str, default='train',
                        choices=['train', 'val', 'both'],
                        help='数据集类型：train（训练集），val（验证集），both（两者都准备）')
    parser.add_argument('--val_output_dir', type=str,
                        default='./sd_lora_val_dataset',
                        help='验证集输出目录（仅当dataset_type为both时使用）')
    
    args = parser.parse_args()
    
    if args.dataset_type == 'both':
        # 准备训练集
        print("\n准备训练集...\n")
        create_metadata_jsonl(
            data_root=args.data_root,
            output_dir=args.output_dir,
            split_file=args.split_file,
            resolution=args.resolution,
            images_per_user=args.images_per_user,
            dataset_type='train'
        )
        
        # 准备验证集
        print("\n" + "="*60)
        print("准备验证集...\n")
        create_metadata_jsonl(
            data_root=args.data_root,
            output_dir=args.val_output_dir,
            split_file=args.split_file,
            resolution=args.resolution,
            images_per_user=args.images_per_user,
            dataset_type='val'
        )
        
        print("\n" + "="*60)
        print("全部完成！")
        print("="*60)
        print(f"训练集: {args.output_dir}")
        print(f"验证集: {args.val_output_dir}")
        print(f"\n现在可以运行训练:")
        print(f"  python train_sd_lora.py --val_dataset_path {args.val_output_dir}")
    else:
        # 只准备一个数据集
        create_metadata_jsonl(
            data_root=args.data_root,
            output_dir=args.output_dir,
            split_file=args.split_file,
            resolution=args.resolution,
            images_per_user=args.images_per_user,
            dataset_type=args.dataset_type
        )


if __name__ == '__main__':
    main()
