#!/usr/bin/env python
# coding=utf-8
"""
VAE 编解码测试脚本
测试 VAE 的重构质量，判断是否是 VAE 导致生成样本不锐利
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vae.kl_vae import KL_VAE


def load_vae(vae_path):
    """加载 VAE 模型"""
    print(f"加载 VAE: {vae_path}")
    vae = KL_VAE(embed_dim=4, scale_factor=0.18215)
    
    vae_ckpt = torch.load(vae_path, map_location='cpu')
    if isinstance(vae_ckpt, dict) and 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)
    
    vae.eval()
    vae.requires_grad_(False)
    print("✓ VAE 加载完成")
    return vae


def load_images(dataset_path, num_images=10):
    """从数据集加载真实图像"""
    print(f"\n加载图像: {dataset_path}")
    image_files = sorted(list(Path(dataset_path).glob("*.png")) + list(Path(dataset_path).glob("*.jpg")))
    
    if len(image_files) == 0:
        raise ValueError(f"未找到图像文件: {dataset_path}")
    
    # 只加载前 num_images 张
    image_files = image_files[:num_images]
    print(f"找到 {len(image_files)} 张图像，将加载前 {num_images} 张")
    
    # 使用 torchvision transforms，与训练代码一致
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()  # 自动归一化到 [0, 1]
    ])
    
    images = []
    for img_path in tqdm(image_files, desc="加载图像"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)  # [3, 256, 256]，值在 [0, 1]
            img_array = img_tensor.permute(1, 2, 0).numpy()  # [256, 256, 3]
            images.append((img_path.name, img_array))
        except Exception as e:
            print(f"⚠️ 加载失败 {img_path}: {e}")
    
    print(f"✓ 成功加载 {len(images)} 张图像")
    return images


def compute_metrics(original, reconstructed):
    """计算重构质量指标"""
    # MSE (Mean Squared Error)
    mse = np.mean((original - reconstructed) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # SSIM (Structural Similarity Index)
    # 简化版本
    mean_original = np.mean(original)
    mean_reconstructed = np.mean(reconstructed)
    var_original = np.var(original)
    var_reconstructed = np.var(reconstructed)
    cov = np.mean((original - mean_original) * (reconstructed - mean_reconstructed))
    
    c1, c2 = 0.01, 0.03
    ssim = ((2 * mean_original * mean_reconstructed + c1) * (2 * cov + c2)) / \
           ((mean_original ** 2 + mean_reconstructed ** 2 + c1) * (var_original + var_reconstructed + c2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }


def test_vae_reconstruction(vae, images, output_dir="./vae_test_results"):
    """测试 VAE 的编解码质量"""
    print(f"\n{'='*60}")
    print("开始 VAE 编解码测试")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    
    all_metrics = {
        'mse': [],
        'psnr': [],
        'ssim': []
    }
    
    with torch.no_grad():
        for img_name, img_array in tqdm(images, desc="处理图像"):
            # img_array 已经是 [H, W, 3]，值在 [0, 1]
            # 转换为张量 [1, 3, H, W]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # 编码（使用 encode_images，会自动乘以 scale_factor）
            # 这与 train_latent_cfg.py 中的使用方式一致
            latents = vae.encode_images(img_tensor)  # [1, 4, 32, 32]，已乘以 scale_factor
            
            # 解码（使用 decode_latents，会自动除以 scale_factor）
            reconstructed_tensor = vae.decode_latents(latents)  # [1, 3, 256, 256]，值在 [0, 1]
            
            # 转换回 numpy [H, W, 3]
            reconstructed_array = reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            reconstructed_array = np.clip(reconstructed_array, 0, 1)
            
            # 计算指标
            metrics = compute_metrics(img_array, reconstructed_array)
            all_metrics['mse'].append(metrics['mse'])
            all_metrics['psnr'].append(metrics['psnr'])
            all_metrics['ssim'].append(metrics['ssim'])
            
            # 保存对比图像
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(img_array)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            axes[1].imshow(reconstructed_array)
            axes[1].set_title(f'VAE 重构\nMSE={metrics["mse"]:.4f}, PSNR={metrics["psnr"]:.2f}, SSIM={metrics["ssim"]:.4f}')
            axes[1].axis('off')
            
            plt.tight_layout()
            save_path = output_path / f"{img_name.replace('.png', '').replace('.jpg', '')}_comparison.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
    
    # 计算平均指标
    print(f"\n{'='*60}")
    print("VAE 编解码测试结果")
    print(f"{'='*60}")
    print(f"平均 MSE:  {np.mean(all_metrics['mse']):.6f}")
    print(f"平均 PSNR: {np.mean(all_metrics['psnr']):.2f} dB")
    print(f"平均 SSIM: {np.mean(all_metrics['ssim']):.4f}")
    print(f"\nMSE 范围:  [{np.min(all_metrics['mse']):.6f}, {np.max(all_metrics['mse']):.6f}]")
    print(f"PSNR 范围: [{np.min(all_metrics['psnr']):.2f}, {np.max(all_metrics['psnr']):.2f}] dB")
    print(f"SSIM 范围: [{np.min(all_metrics['ssim']):.4f}, {np.max(all_metrics['ssim']):.4f}]")
    print(f"\n✓ 对比图像已保存到: {output_path}")
    
    # 解释结果
    print(f"\n{'='*60}")
    print("结果解释")
    print(f"{'='*60}")
    
    avg_psnr = np.mean(all_metrics['psnr'])
    avg_ssim = np.mean(all_metrics['ssim'])
    
    if avg_psnr > 30 and avg_ssim > 0.9:
        print("✅ VAE 重构质量很好（PSNR > 30, SSIM > 0.9）")
        print("   → 生成样本不锐利 NOT 是 VAE 问题")
        print("   → 问题可能在：采样步数、prediction_type、或训练配置")
    elif avg_psnr > 25 and avg_ssim > 0.85:
        print("⚠️  VAE 重构质量中等（PSNR 25-30, SSIM 0.85-0.9）")
        print("   → 可能有轻微的信息损失")
        print("   → 但不是主要原因，还需要检查其他因素")
    else:
        print("❌ VAE 重构质量较差（PSNR < 25, SSIM < 0.85）")
        print("   → 生成样本不锐利 IS VAE 问题")
        print("   → 需要考虑更换 VAE 架构或增加 embed_dim")
    
    print(f"\n参考值：")
    print(f"  PSNR > 30:  非常好的重构质量")
    print(f"  PSNR 25-30: 良好的重构质量")
    print(f"  PSNR < 25:  明显的质量损失")
    print(f"  SSIM > 0.9: 非常相似")
    print(f"  SSIM 0.8-0.9: 相似")
    print(f"  SSIM < 0.8: 明显差异")
    
    return all_metrics


def main():
    # 配置
    vae_path = "/kaggle/input/kl-vae-best-pt/kl_vae_best.pt"
    dataset_path = "/kaggle/input/organized-gait-dataset/Normal_line/ID_24"
    output_dir = "./vae_test_results"
    num_test_images = 10
    
    # 检查路径
    if not Path(vae_path).exists():
        print(f"❌ VAE 文件不存在: {vae_path}")
        return
    
    if not Path(dataset_path).exists():
        print(f"❌ 数据集目录不存在: {dataset_path}")
        return
    
    # 加载 VAE
    vae = load_vae(vae_path)
    
    # 加载图像
    images = load_images(dataset_path, num_images=num_test_images)
    
    if len(images) == 0:
        print("❌ 没有加载到任何图像")
        return
    
    # 测试
    metrics = test_vae_reconstruction(vae, images, output_dir)


if __name__ == "__main__":
    main()
