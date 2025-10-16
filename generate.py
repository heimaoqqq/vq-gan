"""
Conditional Generation Script for Latent Diffusion Model
=========================================================
从训练好的DDPM模型生成指定用户的微多普勒时频图像

用法:
    # 生成特定用户的图像
    python generate.py --checkpoint model-75.pt --user_id 0 --num_samples 10
    
    # 生成所有用户的图像
    python generate.py --checkpoint model-75.pt --all_users --samples_per_user 5
    
    # 调整CFG强度
    python generate.py --checkpoint model-75.pt --user_id 5 --cond_scale 8.0
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torchvision import utils
import numpy as np
from tqdm import tqdm

# 导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae.kl_vae import KL_VAE
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location=device)
    
    # 从保存的配置恢复模型
    config = data.get('config', {})
    
    # 默认配置（如果checkpoint没有保存config）
    dim = config.get('dim', 48)
    dim_mults = config.get('dim_mults', (1, 2, 4))
    num_users = config.get('num_users', 31)
    latent_channels = config.get('latent_channels', 4)
    latent_size = config.get('latent_size', 64)
    timesteps = config.get('timesteps', 1000)
    sampling_timesteps = config.get('sampling_timesteps', 250)
    objective = config.get('objective', 'pred_v')
    
    # 创建Unet
    model = Unet(
        dim=dim,
        dim_mults=dim_mults,
        num_classes=num_users,
        cond_drop_prob=0.5,
        channels=latent_channels,
        attn_dim_head=32,
        attn_heads=4,
        learned_variance=False
    )
    
    # 创建扩散模型
    diffusion = GaussianDiffusion(
        model,
        image_size=latent_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        beta_schedule='cosine',
        auto_normalize=False  # 潜在空间不归一化
    )
    
    # 加载权重（优先使用EMA权重）
    if 'ema' in data:
        print("Loading EMA weights...")
        ema_state = data['ema']
        # EMA保存的是 'ema_model' 和 'online_model'
        if 'ema_model' in ema_state:
            model_state = ema_state['ema_model']
        else:
            model_state = ema_state
    else:
        print("Loading model weights...")
        model_state = data['model']
    
    diffusion.load_state_dict(model_state)
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    step = data.get('step', 0)
    print(f"Model loaded (trained for {step} steps)")
    
    return diffusion, config


def load_vae(vae_path, device='cuda'):
    """加载VAE解码器"""
    print(f"Loading VAE from {vae_path}...")
    
    # 加载checkpoint
    checkpoint = torch.load(vae_path, map_location=device)
    
    # 检查checkpoint格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 训练检查点格式
        print("  Detected training checkpoint format")
        vae_state = checkpoint['model_state_dict']
        
        # 获取配置信息
        embed_dim = checkpoint.get('embed_dim', 4)
        scale_factor = checkpoint.get('scale_factor', 0.18215)
        print(f"  embed_dim: {embed_dim}, scale_factor: {scale_factor}")
    else:
        # 直接state_dict格式
        vae_state = checkpoint
        embed_dim = 4
        scale_factor = 0.18215
    
    # 创建VAE并加载权重
    vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
    vae.load_state_dict(vae_state)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded")
    return vae


def generate_samples(diffusion, vae, user_ids, cond_scale=6.0, device='cuda'):
    """
    生成样本
    
    Args:
        diffusion: 扩散模型
        vae: VAE解码器
        user_ids: 用户ID列表 (0-30)
        cond_scale: CFG强度 (推荐: 3-8)
        device: 设备
    
    Returns:
        生成的图像 [N, 3, 256, 256]
    """
    user_ids = torch.tensor(user_ids, device=device, dtype=torch.long)
    
    with torch.no_grad():
        # DDPM采样潜在表示
        print(f"Sampling latents for {len(user_ids)} samples (cond_scale={cond_scale})...")
        sampled_latents = diffusion.sample(
            classes=user_ids,
            cond_scale=cond_scale
        )
        
        # VAE解码
        print("Decoding latents to images...")
        sampled_images = vae.decode_latents(sampled_latents)
    
    return sampled_images


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained latent diffusion model')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint (e.g., results/model-75.pt)')
    
    # 生成模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--user_id', type=int, 
                       help='Generate for specific user (0-30)')
    group.add_argument('--all_users', action='store_true',
                       help='Generate for all users')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate (for single user)')
    parser.add_argument('--samples_per_user', type=int, default=5,
                        help='Samples per user (for all users mode)')
    parser.add_argument('--cond_scale', type=float, default=6.0,
                        help='Classifier-free guidance scale (3-8 recommended)')
    
    # 路径参数
    parser.add_argument('--vae_path', type=str,
                        default='/kaggle/input/kl-vae-best-pt/kl_vae_best.pt',
                        help='Path to VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory for generated images')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--save_grid', action='store_true',
                        help='Save as grid instead of individual images')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    diffusion, config = load_model(args.checkpoint, device)
    vae = load_vae(args.vae_path, device)
    
    num_users = config.get('num_users', 31)
    
    # 确定要生成的用户列表
    if args.all_users:
        user_list = list(range(num_users))
        samples_per_user = args.samples_per_user
        print(f"Generating {samples_per_user} samples for each of {num_users} users...")
    else:
        if args.user_id < 0 or args.user_id >= num_users:
            print(f"Error: user_id must be in range [0, {num_users-1}]")
            return
        user_list = [args.user_id]
        samples_per_user = args.num_samples
        print(f"Generating {samples_per_user} samples for user {args.user_id}...")
    
    # 批量生成
    all_generated = []
    all_user_ids = []
    
    for user_id in tqdm(user_list, desc="Generating"):
        # 为该用户生成多个样本
        user_ids = [user_id] * samples_per_user
        
        # 分批生成（避免显存溢出）
        for i in range(0, len(user_ids), args.batch_size):
            batch_user_ids = user_ids[i:i+args.batch_size]
            images = generate_samples(
                diffusion, vae, batch_user_ids,
                cond_scale=args.cond_scale,
                device=device
            )
            all_generated.append(images.cpu())
            all_user_ids.extend(batch_user_ids)
    
    # 合并所有生成的图像
    all_generated = torch.cat(all_generated, dim=0)
    print(f"Generated {len(all_generated)} images")
    
    # 保存图像
    if args.save_grid:
        # 保存为网格
        grid_path = output_dir / f'generated_grid_scale{args.cond_scale}.png'
        utils.save_image(
            all_generated,
            str(grid_path),
            nrow=samples_per_user,
            normalize=False
        )
        print(f"Saved grid to {grid_path}")
    else:
        # 保存为单独文件
        for idx, (img, user_id) in enumerate(zip(all_generated, all_user_ids)):
            save_path = output_dir / f'user_{user_id:02d}_sample_{idx:03d}.png'
            utils.save_image(img, str(save_path), normalize=False)
        print(f"Saved {len(all_generated)} images to {output_dir}")
    
    print("Generation complete!")


if __name__ == '__main__':
    main()

