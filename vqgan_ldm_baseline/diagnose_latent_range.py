"""
诊断VQ-GAN潜在表示的值域
检查是否需要归一化
"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import VQVAE

def diagnose_latent_range(vqgan_path, data_path, num_samples=100):
    """
    诊断VQ-GAN潜在表示的值域
    
    Args:
        vqgan_path: VQ-GAN checkpoint路径
        data_path: 数据集路径
        num_samples: 采样数量
    """
    print("="*60)
    print("诊断VQ-GAN潜在表示值域")
    print("="*60)
    
    # 加载VQ-GAN
    print(f"\n加载VQ-GAN: {vqgan_path}")
    checkpoint = torch.load(vqgan_path, map_location='cpu')
    vqgan_config = checkpoint['config']
    
    vqvae = VQVAE(
        in_channels=vqgan_config['in_channels'],
        out_channels=vqgan_config['out_channels'],
        ch=vqgan_config['ch'],
        ch_mult=tuple(vqgan_config['ch_mult']),
        num_res_blocks=vqgan_config['num_res_blocks'],
        attn_resolutions=tuple(vqgan_config['attn_resolutions']),
        dropout=vqgan_config['dropout'],
        z_channels=vqgan_config['z_channels'],
        num_embeddings=vqgan_config['num_embeddings'],
        embedding_dim=vqgan_config['embedding_dim'],
        commitment_cost=vqgan_config['commitment_cost'],
    )
    vqvae.load_state_dict(checkpoint['vqvae'])
    vqvae.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vqvae = vqvae.to(device)
    print(f"✓ VQ-GAN已加载到 {device}")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # 收集潜在表示
    print(f"\n编码 {num_samples} 张图像...")
    latents = []
    data_path = Path(data_path)
    
    image_paths = []
    for user_folder in sorted(data_path.glob("ID_*")):
        image_paths.extend(list(user_folder.glob("*.jpg")))
    
    image_paths = image_paths[:num_samples]
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if i % 20 == 0:
                print(f"  进度: {i}/{len(image_paths)}")
            
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            
            # 编码
            latent = vqvae.encode_images(img)
            latents.append(latent.cpu())
    
    # 合并所有潜在表示
    latents = torch.cat(latents, dim=0)  # [N, C, H, W]
    
    print(f"\n✓ 共编码 {latents.shape[0]} 张图像")
    print(f"  潜在表示形状: {latents.shape}")
    
    # 统计分析
    print("\n" + "="*60)
    print("潜在表示统计信息")
    print("="*60)
    
    min_val = latents.min().item()
    max_val = latents.max().item()
    mean_val = latents.mean().item()
    std_val = latents.std().item()
    
    print(f"Min:  {min_val:>10.6f}")
    print(f"Max:  {max_val:>10.6f}")
    print(f"Mean: {mean_val:>10.6f}")
    print(f"Std:  {std_val:>10.6f}")
    print(f"Range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"Span:  {max_val - min_val:.6f}")
    
    # 检查分布
    print("\n值域分析:")
    if abs(min_val) < 0.1 and abs(max_val) < 0.1:
        print("  ⚠️  值域非常小（接近0）- 可能需要放大")
    elif min_val >= 0 and max_val <= 1:
        print("  ✓  在[0, 1]范围内")
    elif min_val >= -1 and max_val <= 1:
        print("  ✓  在[-1, 1]范围内")
    else:
        print(f"  ⚠️  不在标准范围内 - 需要归一化")
    
    # 建议
    print("\n" + "="*60)
    print("建议")
    print("="*60)
    
    if abs(mean_val) < 1e-3 and std_val < 0.1:
        print("❌ 问题：潜在表示值域太小！")
        print("   扩散模型难以学习如此小的值域")
        print("\n解决方案：")
        print("   1. 在LDM训练前，手动标准化潜在表示")
        print("      z_normalized = (z - mean) / std")
        print(f"      mean = {mean_val:.6f}, std = {std_val:.6f}")
        print("\n   2. 或者缩放到[-1, 1]范围:")
        print(f"      z_scaled = (z - {min_val:.6f}) / {max_val - min_val:.6f} * 2 - 1")
    elif std_val > 0:
        print("建议的归一化方法:")
        print(f"  z_normalized = (z - {mean_val:.6f}) / {std_val:.6f}")
        print("\n  然后在LDM采样后反归一化:")
        print(f"  z_original = z_normalized * {std_val:.6f} + {mean_val:.6f}")
    
    # 检查Codebook使用情况
    print("\n" + "="*60)
    print("Codebook分析")
    print("="*60)
    
    # 检查codebook权重
    codebook_weights = vqvae.quantizer.embedding.weight.data  # [num_embeddings, embedding_dim]
    print(f"Codebook shape: {codebook_weights.shape}")
    print(f"Codebook min:  {codebook_weights.min().item():.6f}")
    print(f"Codebook max:  {codebook_weights.max().item():.6f}")
    print(f"Codebook mean: {codebook_weights.mean().item():.6f}")
    print(f"Codebook std:  {codebook_weights.std().item():.6f}")
    
    print("\n" + "="*60)
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_path', type=str, required=True,
                       help='VQ-GAN checkpoint路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='采样数量')
    
    args = parser.parse_args()
    
    stats = diagnose_latent_range(
        args.vqgan_path,
        args.data_path,
        args.num_samples
    )

