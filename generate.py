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
from torchvision import utils, transforms
import numpy as np
from tqdm import tqdm

# 导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入，避免触发__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cfg_module", 
    os.path.join(os.path.dirname(__file__), 
                 "denoising_diffusion_pytorch/classifier_free_guidance.py")
)
cfg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg_module)
Unet = cfg_module.Unet
GaussianDiffusion = cfg_module.GaussianDiffusion

from vae.kl_vae import KL_VAE


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location=device)
    
    # 从保存的配置恢复模型
    config = data.get('config', {})
    
    # 默认配置（如果checkpoint没有保存config）
    dim = config.get('dim', 96)  # 与train_latent_cfg.py保持一致
    dim_mults = config.get('dim_mults', (1, 2, 4, 4))  # 与train_latent_cfg.py保持一致
    num_users = config.get('num_users', 31)
    latent_channels = config.get('latent_channels', 4)
    latent_size = config.get('latent_size', 32)  # VAE是8倍下采样，256/8=32
    timesteps = config.get('timesteps', 1000)
    sampling_timesteps = config.get('sampling_timesteps', 100)  # 与train_latent_cfg.py保持一致
    objective = config.get('objective', 'pred_v')
    
    # 创建Unet
    model = Unet(
        dim=dim,
        dim_mults=dim_mults,
        num_classes=num_users,
        cond_drop_prob=config.get('cond_drop_prob', 0.05),  # 与train_latent_cfg.py保持一致
        channels=latent_channels,
        attn_dim_head=config.get('attn_dim_head', 64),
        attn_heads=config.get('attn_heads', 8),  # 与train_latent_cfg.py保持一致
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
        min_snr_loss_weight=config.get('min_snr_loss_weight', True),
        min_snr_gamma=config.get('min_snr_gamma', 5),
        auto_normalize=config.get('auto_normalize', False)  # 从训练配置读取
    )
    
    # 加载权重（优先使用EMA权重）
    print("Analyzing checkpoint structure...")
    
    # 先检查checkpoint的实际结构
    if 'ema' in data:
        print("Found EMA in checkpoint")
        ema_state = data['ema']
        
        # 检查EMA权重是否是扁平结构（带前缀）还是嵌套结构
        sample_keys = list(ema_state.keys())[:5]
        print(f"Sample EMA keys: {sample_keys}")
        
        if any(key.startswith('ema_model.') for key in ema_state.keys()):
            # 扁平结构：所有权重都带ema_model.前缀
            print("Detected flat EMA structure with prefixes")
            fixed_state = {}
            prefix_to_remove = 'ema_model.'
            
            for key, value in ema_state.items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove):]
                    fixed_state[new_key] = value
                    
            model_state = fixed_state
            print(f"Extracted {len(fixed_state)} weights from EMA")
            
        elif 'ema_model' in ema_state:
            # 嵌套结构：ema_model是一个子字典
            print("Detected nested EMA structure")
            model_state = ema_state['ema_model']
        else:
            # 直接使用EMA权重
            print("Using EMA weights directly")
            model_state = ema_state
    else:
        print("Loading model weights...")
        model_state = data['model']
    
    # 加载权重
    try:
        diffusion.load_state_dict(model_state, strict=True)
        print("✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Strict loading failed: {str(e)[:200]}...")
        print("Attempting relaxed loading...")
        
        # 尝试非严格加载
        result = diffusion.load_state_dict(model_state, strict=False)
        if result.missing_keys:
            print(f"Warning: {len(result.missing_keys)} missing keys")
        if result.unexpected_keys:
            print(f"Warning: {len(result.unexpected_keys)} unexpected keys")
        print("✓ Model loaded with warnings")
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


def generate_samples(diffusion, vae, user_ids, cond_scale=7.5, device='cuda'):
    """
    生成样本
    
    Args:
        diffusion: 扩散模型
        vae: VAE解码器
        user_ids: 用户ID列表 (0-30，对应文件夹ID_1到ID_31)
        cond_scale: CFG强度 (推荐: 7.5，与训练配置一致)
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
            cond_scale=cond_scale,
            rescaled_phi=0.7  # CFG++ rescaling，与训练时一致
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
                       help='Generate for specific user (0-30, 对应文件夹ID_1到ID_31)')
    group.add_argument('--all_users', action='store_true',
                       help='Generate for all 31 users (ID_1 to ID_31)')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate (for single user)')
    parser.add_argument('--samples_per_user', type=int, default=50,
                        help='Samples per user (for all users mode, default=50)')
    parser.add_argument('--cond_scale', type=float, default=1.5,
                        help='CFG scale (1.0=无CFG/Baseline, 1.5=优化版默认, 更高=更强条件)')
    
    # 路径参数
    parser.add_argument('--vae_path', type=str,
                        default='/kaggle/input/kl-vae/kl_vae_best.pt',
                        help='Path to VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory for generated images')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--save_grid', action='store_true',
                        help='Save as grid instead of individual images (default: save individual images)')
    
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
    
    # 为每个用户生成并立即保存（实时反馈）
    from PIL import Image
    total_images_generated = 0
    
    print("开始生成图像（每个用户完成后立即保存）...\n")
    
    for user_id in tqdm(user_list, desc="用户进度"):
        # 创建用户文件夹：user 0 -> ID_1, user 1 -> ID_2, ...
        actual_id = user_id + 1  # 0→ID_1, 1→ID_2, ..., 30→ID_31
        user_folder = output_dir / f'ID_{actual_id}'
        user_folder.mkdir(exist_ok=True, parents=True)
        
        # 为该用户生成samples_per_user张图像
        user_images = []
        
        # 分批生成（避免显存溢出）
        for batch_idx in range(0, samples_per_user, args.batch_size):
            n_batch = min(args.batch_size, samples_per_user - batch_idx)
            batch_user_ids = [user_id] * n_batch
            
            # 采样
            images = generate_samples(
                diffusion, vae, batch_user_ids,
                cond_scale=args.cond_scale,
                device=device
            )
            
            # 立即保存该批次图像
            for i in range(n_batch):
                img_idx = batch_idx + i
                save_path = user_folder / f'generated_{img_idx:03d}.jpg'
                
                # 转换为PIL Image并保存为JPG（256×256）
                img_tensor = images[i].clamp(0, 1).cpu()
                img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                img_np = img_np.transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
                
                img_pil = Image.fromarray(img_np, mode='RGB')
                img_pil.save(str(save_path), quality=95)
                
                total_images_generated += 1
            
            user_images.append(images.cpu())
        
        # 该用户完成，输出提示
        tqdm.write(f"  ✓ ID_{actual_id}: {samples_per_user}张已保存 → {user_folder}/")
    
    print(f"\n{'='*60}")
    print("生成完成！")
    print(f"{'='*60}")
    print(f"  总计生成: {total_images_generated}张")
    print(f"  输出目录: {output_dir}/")
    print(f"  输出格式: ID_X/generated_XXX.jpg (256×256)")
    print(f"  CFG设置: cond_scale={args.cond_scale}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

