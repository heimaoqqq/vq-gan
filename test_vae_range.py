"""
测试VAE编码后的潜在表示数值范围
用于确定DDPM是否需要归一化
"""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, '.')
from vae.kl_vae import KL_VAE


def test_vae_output_range(vae_path, image_paths, device='cuda'):
    """
    测试VAE编码后的数值范围
    
    Args:
        vae_path: VAE权重路径
        image_paths: 测试图像路径列表
        device: 设备
    """
    # 加载VAE
    print(f"Loading VAE from {vae_path}...")
    
    # 加载checkpoint
    checkpoint = torch.load(vae_path, map_location=device)
    
    # 检查checkpoint格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 训练检查点格式
        print("Detected training checkpoint format")
        vae_state = checkpoint['model_state_dict']
        
        # 获取配置信息（如果有）
        embed_dim = checkpoint.get('embed_dim', 4)
        scale_factor = checkpoint.get('scale_factor', 0.18215)
        print(f"  embed_dim: {embed_dim}")
        print(f"  scale_factor: {scale_factor}")
    else:
        # 直接state_dict格式
        print("Detected direct state_dict format")
        vae_state = checkpoint
        embed_dim = 4
        scale_factor = 0.18215
    
    # 创建VAE并加载权重
    vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
    vae.load_state_dict(vae_state)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully\n")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()  # 归一化到[0,1]
    ])
    
    # 收集所有潜在表示
    all_latents = []
    
    print(f"Encoding {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in image_paths[:100]:  # 测试前100张
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            
            # VAE编码（已包含scale_factor）
            latent = vae.encode_images(img)
            all_latents.append(latent.cpu())
    
    # 合并并统计
    all_latents = torch.cat(all_latents, dim=0)  # [N, 4, 64, 64]
    
    print("\n" + "="*60)
    print("VAE潜在表示统计信息：")
    print("="*60)
    print(f"形状: {all_latents.shape}")
    print(f"最小值: {all_latents.min().item():.6f}")
    print(f"最大值: {all_latents.max().item():.6f}")
    print(f"均值: {all_latents.mean().item():.6f}")
    print(f"标准差: {all_latents.std().item():.6f}")
    print(f"中位数: {all_latents.median().item():.6f}")
    print(f"\n各通道统计:")
    for i in range(4):
        ch = all_latents[:, i]
        print(f"  通道{i}: min={ch.min():.4f}, max={ch.max():.4f}, "
              f"mean={ch.mean():.4f}, std={ch.std():.4f}")
    
    print("\n" + "="*60)
    print("判断与建议：")
    print("="*60)
    
    min_val = all_latents.min().item()
    max_val = all_latents.max().item()
    
    if min_val >= 0 and max_val <= 1:
        print("✓ 潜在表示在[0, 1]范围内")
        print("  → 可以使用 auto_normalize=True")
        print("  → DDPM会自动将[0,1]映射到[-1,1]")
    else:
        print("✗ 潜在表示NOT在[0, 1]范围内")
        print(f"  → 实际范围: [{min_val:.4f}, {max_val:.4f}]")
        print("  → 必须使用 auto_normalize=False")
        print("  → DDPM将在原始范围训练")
    
    # 测试重建质量
    print("\n" + "="*60)
    print("测试重建质量：")
    print("="*60)
    
    test_img = Image.open(image_paths[0]).convert('RGB')
    test_tensor = transform(test_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 编码-解码
        latent = vae.encode_images(test_tensor)
        recon = vae.decode_latents(latent)
    
    # 计算重建误差
    mse = ((test_tensor - recon) ** 2).mean().item()
    print(f"重建MSE: {mse:.6f}")
    print(f"原始图像范围: [{test_tensor.min():.4f}, {test_tensor.max():.4f}]")
    print(f"重建图像范围: [{recon.min():.4f}, {recon.max():.4f}]")
    
    if mse < 0.01:
        print("✓ 重建质量良好（MSE < 0.01）")
    elif mse < 0.05:
        print("○ 重建质量可接受（MSE < 0.05）")
    else:
        print("✗ 重建质量较差（MSE >= 0.05）")
        print("  → 检查VAE权重是否正确加载")
    
    return all_latents


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_path', type=str, required=True,
                        help='VAE权重路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径（包含ID_*文件夹）')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 收集图像路径
    data_path = Path(args.data_path)
    image_paths = []
    
    for user_id in range(1, 32):
        user_folder = data_path / f"ID_{user_id}"
        if user_folder.exists():
            image_paths.extend(list(user_folder.glob("*.jpg")))
    
    if len(image_paths) == 0:
        print(f"错误: 在 {data_path} 中没有找到图像")
        return
    
    print(f"找到 {len(image_paths)} 张图像\n")
    
    # 测试VAE输出范围
    test_vae_output_range(args.vae_path, image_paths, args.device)


if __name__ == '__main__':
    main()

