"""
完整诊断LDM训练和生成流程
检查每个环节是否正常
"""
import torch
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from models import VQVAE
from configs import LDMBaselineConfig

print("="*60)
print("LDM流程完整诊断")
print("="*60)

# 1. 加载VQ-GAN
config = LDMBaselineConfig()
print(f"\n1. 加载VQ-GAN: {config.vqgan_path}")

vqgan_checkpoint = torch.load(config.vqgan_path, map_location='cpu')
vqgan_config = vqgan_checkpoint['config']

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
vqvae.load_state_dict(vqgan_checkpoint['vqvae'])
vqvae.eval()
print("✓ VQ-GAN加载成功")

# 2. 测试随机潜在表示解码
print("\n2. 测试随机潜在表示解码")

# 2.1 标准正态分布（归一化后的潜在表示）
z_normalized = torch.randn(1, 256, 32, 32)  # N(0, 1)
print(f"   归一化潜在: mean={z_normalized.mean():.4f}, std={z_normalized.std():.4f}")

# 2.2 反归一化到VQ-GAN空间
z_denorm = z_normalized * config.latent_std + config.latent_mean
print(f"   反归一化后: mean={z_denorm.mean():.4f}, std={z_denorm.std():.4f}")

# 2.3 解码
with torch.no_grad():
    img_from_random = vqvae.decode_latents(z_denorm)

print(f"   解码图像: shape={img_from_random.shape}")
print(f"   像素范围: [{img_from_random.min():.4f}, {img_from_random.max():.4f}]")
print(f"   像素均值: {img_from_random.mean():.4f}")

# 检查是否全是同一个值（会导致"没特征"）
img_std = img_from_random.std().item()
if img_std < 0.01:
    print("   ❌ 警告：解码图像几乎没有变化（std太小）！")
    print("      问题：VQ-GAN解码器可能有问题")
else:
    print(f"   ✓ 解码图像有变化 (std={img_std:.4f})")

# 3. 测试真实图像的编码-解码重建
print("\n3. 测试真实图像编码-解码重建")

from PIL import Image
from torchvision import transforms

data_path = Path(config.data_path)
image_paths = []
for user_folder in sorted(data_path.glob("ID_*"))[:1]:  # 只取第一个用户
    image_paths.extend(list(user_folder.glob("*.jpg"))[:3])  # 取3张图

if image_paths:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    for img_path in image_paths:
        print(f"\n   测试图像: {img_path.name}")
        
        # 读取并预处理
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # 编码
        with torch.no_grad():
            z_encoded = vqvae.encode_images(img_tensor)
        
        print(f"   编码潜在: mean={z_encoded.mean():.4f}, std={z_encoded.std():.4f}")
        
        # 解码（重建）
        with torch.no_grad():
            img_recon = vqvae.decode_latents(z_encoded)
        
        # 计算重建误差
        mse = ((img_tensor - img_recon) ** 2).mean().item()
        print(f"   重建MSE: {mse:.6f}")
        
        if mse < 0.01:
            print(f"   ✓ 重建质量优秀 (MSE<0.01)")
        elif mse < 0.05:
            print(f"   ✓ 重建质量良好 (MSE<0.05)")
        else:
            print(f"   ⚠️  重建质量一般 (MSE={mse:.4f})")
        
        # 归一化潜在表示
        z_norm = (z_encoded - config.latent_mean) / config.latent_std
        print(f"   归一化后: mean={z_norm.mean():.4f}, std={z_norm.std():.4f}")
        
        # 反归一化并重建
        z_denorm = z_norm * config.latent_std + config.latent_mean
        with torch.no_grad():
            img_recon2 = vqvae.decode_latents(z_denorm)
        
        mse2 = ((img_tensor - img_recon2) ** 2).mean().item()
        if abs(mse - mse2) < 1e-6:
            print(f"   ✓ 归一化/反归一化正确 (MSE一致)")
        else:
            print(f"   ❌ 归一化/反归一化有问题！MSE变化: {mse:.6f} -> {mse2:.6f}")
else:
    print("   ⚠️  未找到测试图像")

# 4. 检查LDM checkpoint
print("\n4. 检查LDM训练进度")

results_folder = Path(config.results_folder)
if results_folder.exists():
    checkpoints = list(results_folder.glob("model-*.pt"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))
        ckpt = torch.load(latest, map_location='cpu')
        
        print(f"   最新checkpoint: {latest.name}")
        print(f"   训练步数: {ckpt['step']}")
        
        # 检查是否包含归一化参数
        if 'latent_mean' in ckpt and 'latent_std' in ckpt:
            print(f"   ✓ 包含归一化参数")
            print(f"     mean={ckpt['latent_mean']:.6f}")
            print(f"     std={ckpt['latent_std']:.6f}")
        else:
            print(f"   ❌ 不包含归一化参数！需要重新训练！")

print("\n" + "="*60)
print("诊断完成")
print("="*60)

print("\n建议：")
print("1. 如果随机潜在解码std<0.05，VQ-GAN解码可能有问题")
print("2. 如果真实潜在解码没有明显纹理，检查VQ-GAN训练")
print("3. 如果LDM训练>10000步但随机解码仍是噪声，可能是:")
print("   - 学习率太高")
print("   - 需要更多训练步数")
print("   - VQ-VAE离散空间太难学习")

