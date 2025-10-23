"""
使用训练好的SD+LoRA模型生成微多普勒时频图
"""

import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


def generate_samples(
    lora_weights,
    user_id=0,
    num_samples=10,
    guidance_scale=7.5,
    num_inference_steps=50,
    output_dir="./sd_lora_generated",
    device="cuda",
    resize_to_256=True
):
    """
    使用SD+LoRA生成图像
    
    Args:
        lora_weights: LoRA权重路径
        user_id: 用户ID (0-30)
        num_samples: 生成样本数
        guidance_scale: CFG强度
        num_inference_steps: 推理步数
        output_dir: 输出目录
        device: 设备
    """
    
    print("="*60)
    print("SD+LoRA图像生成")
    print("="*60)
    
    # 1. 加载基础SD模型
    print("\n1. 加载Stable Diffusion 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # 关闭安全检查器
    )
    
    # 2. 加载LoRA权重
    print(f"\n2. 加载LoRA权重: {lora_weights}")
    pipe.load_lora_weights(lora_weights)
    
    # 3. 优化设置
    print("\n3. 配置生成参数...")
    # 使用DPM-Solver加速采样
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 启用内存优化
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    
    pipe = pipe.to(device)
    
    print(f"   用户ID: {user_id} (对应文件夹ID_{user_id+1})")
    print(f"   生成数量: {num_samples}")
    print(f"   CFG强度: {guidance_scale}")
    print(f"   推理步数: {num_inference_steps}")
    
    # 4. 生成图像
    print(f"\n4. 生成图像...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 条件文本：与训练时一致
    prompt = f"user {user_id}"
    
    print(f"   Prompt: '{prompt}'")
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    for i in range(num_samples):
        print(f"   生成 {i+1}/{num_samples}...")
        
        # 生成图像（512×512）
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Resize到256×256（如果需要）
        if resize_to_256:
            image = image.resize((256, 256), Image.LANCZOS)
        
        # 保存
        save_path = output_path / f"user_{user_id:02d}_sample_{i:03d}.png"
        image.save(save_path)
    
    print(f"\n✓ 生成完成！")
    print(f"   输出目录: {output_path}")
    print(f"   共生成 {num_samples} 张图像")
    
    # 5. 创建网格预览
    print(f"\n5. 创建网格预览...")
    from torchvision.utils import make_grid
    from torchvision import transforms
    
    # 加载所有生成的图像
    images = []
    for i in range(num_samples):
        img_path = output_path / f"user_{user_id:02d}_sample_{i:03d}.png"
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        images.append(img_tensor)
    
    # 创建网格
    grid = make_grid(images, nrow=5, padding=2)
    grid_img = transforms.ToPILImage()(grid)
    grid_path = output_path / f"user_{user_id:02d}_grid.png"
    grid_img.save(grid_path)
    
    print(f"   ✓ 网格预览: {grid_path}")
    
    print("\n" + "="*60)


def generate_all_users(
    lora_weights,
    samples_per_user=5,
    guidance_scale=7.5,
    num_inference_steps=50,
    output_dir="./sd_lora_generated_all",
    device="cuda",
    resize_to_256=True
):
    """为所有31个用户生成图像"""
    
    print("="*60)
    print("为所有用户生成图像")
    print("="*60)
    
    # 加载模型（只加载一次）
    print("\n加载模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.load_lora_weights(lora_weights)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    
    pipe = pipe.to(device)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 为每个用户生成
    for user_id in range(31):
        print(f"\n生成用户 {user_id} (ID_{user_id+1})...")
        prompt = f"user {user_id}"
        
        user_dir = output_path / f"ID_{user_id+1:02d}"
        user_dir.mkdir(exist_ok=True)
        
        generator = torch.Generator(device=device).manual_seed(42 + user_id)
        
        for i in range(samples_per_user):
            image = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            # Resize到256×256（如果需要）
            if resize_to_256:
                image = image.resize((256, 256), Image.LANCZOS)
            
            save_path = user_dir / f"sample_{i:03d}.png"
            image.save(save_path)
        
        print(f"  ✓ 生成 {samples_per_user} 张图像")
    
    print(f"\n✓ 所有用户生成完成！")
    print(f"   输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='使用SD+LoRA生成微多普勒时频图')
    
    # 必需参数
    parser.add_argument('--lora_weights', type=str, required=True,
                        help='LoRA权重路径 (e.g., sd_lora_output/pytorch_lora_weights.safetensors)')
    
    # 生成模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--user_id', type=int,
                       help='生成特定用户 (0-30)')
    group.add_argument('--all_users', action='store_true',
                       help='生成所有31个用户')
    
    # 生成参数
    parser.add_argument('--num_samples', type=int, default=10,
                        help='每个用户生成的样本数（单用户模式）')
    parser.add_argument('--samples_per_user', type=int, default=5,
                        help='每个用户生成的样本数（全用户模式）')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='CFG强度 (推荐7.5)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='推理步数 (推荐50)')
    parser.add_argument('--output_dir', type=str, default='./sd_lora_generated',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--resize_to_256', action='store_true', default=True,
                        help='生成后resize到256×256（默认开启）')
    parser.add_argument('--keep_512', dest='resize_to_256', action='store_false',
                        help='保持512×512分辨率')
    
    args = parser.parse_args()
    
    if args.all_users:
        generate_all_users(
            lora_weights=args.lora_weights,
            samples_per_user=args.samples_per_user,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            output_dir=args.output_dir,
            device=args.device,
            resize_to_256=args.resize_to_256
        )
    else:
        generate_samples(
            lora_weights=args.lora_weights,
            user_id=args.user_id,
            num_samples=args.num_samples,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            output_dir=args.output_dir,
            device=args.device,
            resize_to_256=args.resize_to_256
        )


if __name__ == '__main__':
    main()
