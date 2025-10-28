"""
预编码脚本 - 将所有图像编码为潜在表示并缓存
同时保存训练集/测试集划分信息，方便后续分类器实验

输出：
  1. latents_cache/ - 所有图像的潜在表示（训练+测试）
  2. data_split.json - 数据集划分信息
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae.kl_vae import KL_VAE


def preprocess_dataset(vae_path, data_path, output_folder, num_users=31, 
                       images_per_user_train=50, seed=42, 
                       encode_all=True, device='cuda'):
    """
    预编码所有图像到潜在空间，并保存数据集划分信息
    
    Args:
        vae_path: VAE权重路径
        data_path: 数据集路径
        output_folder: 缓存输出文件夹
        num_users: 用户数量
        images_per_user_train: 每用户训练图像数（剩余为测试集）
        seed: 随机种子
        encode_all: 是否编码所有图像（包括测试集）
        device: 设备
    """
    
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载VAE
    print(f"Loading VAE from {vae_path}...")
    checkpoint = torch.load(vae_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        vae_state = checkpoint['model_state_dict']
        embed_dim = checkpoint.get('embed_dim', 4)
        scale_factor = checkpoint.get('scale_factor', 0.18215)
        print(f"  Checkpoint format - embed_dim: {embed_dim}, scale_factor: {scale_factor}")
    else:
        vae_state = checkpoint
        embed_dim = 4
        scale_factor = 0.18215
    
    vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
    vae.load_state_dict(vae_state)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully\n")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # 收集所有图像路径并划分训练/测试集
    data_path = Path(data_path)
    train_samples = []
    test_samples = []
    data_split_info = {
        'seed': seed,
        'num_users': num_users,
        'images_per_user_train': images_per_user_train,
        'users': {}
    }
    
    print(f"Collecting and splitting image paths for {num_users} users...")
    print(f"  Train: {images_per_user_train} images/user")
    print(f"  Test: remaining images/user")
    
    for user_id in range(1, num_users + 1):
        user_folder = data_path / f"ID_{user_id}"
        
        if not user_folder.exists():
            print(f"  Warning: {user_folder} not found, skipping...")
            continue
        
        # 收集该用户的所有jpg图像
        image_paths = sorted(list(user_folder.glob("*.jpg")))
        
        # 为每个用户设置独立但可复现的随机种子
        user_seed = seed + user_id
        rng = np.random.RandomState(user_seed)
        
        # 随机打乱
        indices = rng.permutation(len(image_paths))
        image_paths_shuffled = [image_paths[i] for i in indices]
        
        # 划分训练/测试集
        train_paths = image_paths_shuffled[:images_per_user_train]
        test_paths = image_paths_shuffled[images_per_user_train:]
        
        # 保存划分信息
        data_split_info['users'][f'ID_{user_id}'] = {
            'user_id': user_id,
            'label': user_id - 1,
            'total_images': len(image_paths),
            'train_images': [str(p.relative_to(data_path)) for p in train_paths],
            'test_images': [str(p.relative_to(data_path)) for p in test_paths]
        }
        
        # 添加到样本列表
        for img_path in train_paths:
            train_samples.append((img_path, user_id - 1, 'train'))
        
        if encode_all:
            for img_path in test_paths:
                test_samples.append((img_path, user_id - 1, 'test'))
    
    # 合并训练和测试样本
    all_samples = train_samples + (test_samples if encode_all else [])
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples)} images")
    print(f"  Test: {len(test_samples)} images")
    print(f"  Total to encode: {len(all_samples)} images\n")
    
    # 保存数据集划分信息
    split_file = Path(output_folder) / 'data_split.json'
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(data_split_info, f, indent=2, ensure_ascii=False)
    print(f"Data split info saved to: {split_file}\n")
    
    # 批量编码
    print("Encoding images to latent space...")
    batch_size = 56  # 批量处理，加速编码
    
    encoded_count = 0
    skipped_count = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches"):
            batch_samples = all_samples[i:i+batch_size]
            batch_images = []
            batch_paths = []
            
            for img_path, label, split_type in batch_samples:
                # 检查是否已缓存
                # 使用 userID_filename.pt 格式避免文件名冲突
                cache_filename = f"user_{label:02d}_{img_path.stem}.pt"
                cache_path = output_path / cache_filename
                
                if cache_path.exists():
                    skipped_count += 1
                    continue
                
                # 加载图像
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                    batch_paths.append((cache_path, label))
                except Exception as e:
                    print(f"\nWarning: Failed to load {img_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # 批量编码
            batch_tensor = torch.stack(batch_images).to(device)
            latents = vae.encode_images(batch_tensor)  # [B, 4, 32, 32]
            
            # 保存到缓存
            for (cache_path, label), latent in zip(batch_paths, latents):
                torch.save(latent.cpu(), cache_path)
                encoded_count += 1
    
    print(f"\n{'='*60}")
    print("预编码完成！")
    print(f"{'='*60}")
    print(f"新编码: {encoded_count} 张")
    print(f"已存在: {skipped_count} 张")
    print(f"总计: {encoded_count + skipped_count} 张")
    print(f"\n输出文件:")
    print(f"  潜在表示缓存: {output_path}/")
    print(f"  数据集划分: {output_path}/data_split.json")
    print(f"\n现在可以:")
    print(f"  1. 开始DDPM训练: python train_latent_cfg.py")
    print(f"  2. 进行分类器实验（使用data_split.json获取训练/测试集）")


def main():
    parser = argparse.ArgumentParser(description='预编码训练图像到潜在空间')
    
    # 路径参数
    parser.add_argument('--vae_path', type=str,
                        default='/kaggle/input/kl-vae-best-pt/kl_vae_best.pt',
                        help='VAE权重路径')
    parser.add_argument('--data_path', type=str,
                        default='/kaggle/input/organized-gait-dataset/Normal_line',
                        help='数据集路径')
    parser.add_argument('--output_folder', type=str,
                        default='./latents_cache',
                        help='缓存输出文件夹')
    
    # 数据参数
    parser.add_argument('--num_users', type=int, default=31,
                        help='用户数量')
    parser.add_argument('--images_per_user_train', type=int, default=50,
                        help='每用户训练图像数（剩余为测试集）')
    parser.add_argument('--encode_all', action='store_true', default=True,
                        help='是否编码所有图像（包括测试集）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        vae_path=args.vae_path,
        data_path=args.data_path,
        output_folder=args.output_folder,
        num_users=args.num_users,
        images_per_user_train=args.images_per_user_train,
        encode_all=args.encode_all,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()

