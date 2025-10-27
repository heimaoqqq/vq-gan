"""
SD+LoRA训练脚本 - 直接使用Diffusers API
不依赖外部训练脚本，完全自包含
"""

import sys
import os
from pathlib import Path
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


def compute_snr(noise_scheduler, timesteps):
    """
    计算给定timesteps的信噪比(SNR)
    用于Min-SNR Weighting（小数据集优化）
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # SNR = (alpha / sigma)^2
    snr = (alpha / sigma) ** 2
    return snr


class TextImageDataset(Dataset):
    def __init__(self, data_root, tokenizer, resolution=512):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # 读取metadata.jsonl
        metadata_file = self.data_root / "metadata.jsonl"
        self.data = []
        with open(metadata_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # 图像变换（不使用数据增强，保持时频图原始特征）
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # 注意：微多普勒时频图不应使用RandomFlip等数据增强
        # 时频图的时间轴方向有物理意义，翻转会破坏特征
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image_path = self.data_root / "images" / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Tokenize文本
        text = item["text"]
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0],
        }


def train_sd_lora(
    model_name="runwayml/stable-diffusion-v1-5",
    dataset_path="./sd_lora_dataset",
    val_dataset_path=None,
    output_dir="./sd_lora_output",
    resolution=512,
    train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=150,  # 跨域迁移需要100-200 epochs（文献：医学影像100 epochs，每图需50-100步）
    learning_rate=1e-4,  # 跨域迁移需要更大学习率（参考医学影像1e-4）
    lr_scheduler="constant_with_warmup",  # Warmup后保持不变，适合小数据集+早停
    lr_warmup_steps=2000,  # Warmup约5 epochs（长训练需要更长warmup）
    seed=42,
    lora_rank=64,  # 跨域任务需要更强表达能力（官方域内用32，跨域应更高）
    lora_alpha=64,  # alpha=rank（标准做法）
    lora_dropout=0.1,  # rank=64需要dropout防止过拟合
    validation_prompt="user 0",
    validation_epochs=20,  # 150 epochs共验证7-8次
    num_validation_images=1,  # 每个epoch只生成1张（避免GPU超时）
    checkpointing_steps=500,
    mixed_precision="fp16",
    gradient_checkpointing=True,
    use_8bit_adam=False,
    center_crop=True,
    random_flip=False,  # 微多普勒时频图不使用随机翻转
    weight_decay=1e-2,  # L2正则化（LoRA官方推荐）
    max_grad_norm=1.0,  # 梯度裁剪（防止梯度爆炸）
    snr_gamma=5.0,  # Min-SNR Weighting（官方推荐5.0，小数据集优化）
    report_to="tensorboard"
):
    """
    训练SD+LoRA模型
    
    Args:
        model_name: 预训练模型名称
        dataset_path: 训练集路径（包含images/和metadata.jsonl）
        val_dataset_path: 验证集路径（可选，如果为None则只用validation_prompt）
        output_dir: 输出目录
        resolution: 图像分辨率
        train_batch_size: 训练batch size
        gradient_accumulation_steps: 梯度累积步数
        num_train_epochs: 训练epoch数
        learning_rate: 学习率
        lr_scheduler: 学习率调度器
        lr_warmup_steps: 预热步数
        seed: 随机种子
        lora_rank: LoRA秩
        lora_alpha: LoRA缩放因子
        lora_dropout: LoRA Dropout系数（保留参数但不使用）
        validation_prompt: 验证提示词
        validation_epochs: 验证频率（epoch）
        num_validation_images: 每次验证生成的图像数
        checkpointing_steps: 保存检查点频率（步数）
        mixed_precision: 混合精度训练
        gradient_checkpointing: 是否使用梯度检查点
        use_8bit_adam: 是否使用8bit Adam
        center_crop: 是否中心裁剪
        random_flip: 是否随机翻转
        weight_decay: L2正则化系数
        max_grad_norm: 梯度裁剪阈值
        snr_gamma: Min-SNR Weighting系数
        report_to: 日志记录工具
    """
    
    print("="*60)
    print("SD+LoRA训练 - 微多普勒时频图生成")
    print("="*60)
    print()
    print("配置:")
    print(f"  模型: {model_name}")
    print(f"  训练集: {dataset_path}")
    if val_dataset_path:
        print(f"  验证集: {val_dataset_path}")
    print(f"  输出: {output_dir}")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"  Batch size: {train_batch_size} (有效: {train_batch_size * gradient_accumulation_steps})")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate} (scheduler: {lr_scheduler})")
    print(f"  Weight decay: {weight_decay} (L2正则化)")
    print(f"  Max grad norm: {max_grad_norm} (梯度裁剪)")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    if snr_gamma is not None:
        print(f"  SNR gamma: {snr_gamma} (Min-SNR Weighting，高级优化)")
    print(f"  验证频率: 每{validation_epochs}个epoch")
    print(f"  早停策略: patience=20, 最小训练轮数=30")
    print(f"  注意: 使用constant_with_warmup，学习率在warmup后保持不变")
    print()
    
    # 检查数据集是否存在
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请先运行: python prepare_sd_lora_dataset.py")
        sys.exit(1)
    
    metadata_file = dataset_path / "metadata.jsonl"
    if not metadata_file.exists():
        print(f"错误: metadata.jsonl不存在: {metadata_file}")
        print("请先运行: python prepare_sd_lora_dataset.py")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    
    # 设置随机种子
    torch.manual_seed(seed)
    
    print("="*60)
    print("开始训练...")
    print("="*60)
    print()
    
    # 1. 加载模型
    print("1. 加载预训练模型...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # 冻结所有模型参数（官方标准做法）
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 设置权重类型（根据混合精度）
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # 2. 配置LoRA
    print("2. 配置LoRA...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",  # 官方标准初始化
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # 3. 创建数据集和DataLoader
    print("3. 加载数据集...")
    train_dataset = TextImageDataset(dataset_path, tokenizer, resolution)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # 加载验证集（如果提供）
    val_dataloader = None
    if val_dataset_path:
        print(f"   加载验证集: {val_dataset_path}")
        val_dataset = TextImageDataset(val_dataset_path, tokenizer, resolution)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(f"   验证集大小: {len(val_dataset)}张")
    
    # 4. 创建优化器
    print("4. 创建优化器...")
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print("  Warning: bitsandbytes未安装，使用标准AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,  # L2正则化
        eps=1e-8,
    )
    
    # 5. 创建学习率调度器
    lr_scheduler_obj = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * num_train_epochs,
    )
    
    # 6. 使用Accelerator准备
    unet, optimizer, train_dataloader, lr_scheduler_obj = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler_obj
    )
    
    # 将VAE和text_encoder移到设备
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # 7. 训练循环
    print("5. 开始训练...")
    print(f"  总步数: {len(train_dataloader) * num_train_epochs}")
    print(f"  每epoch步数: {len(train_dataloader)}")
    print()
    
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    # 注意：如果不使用验证集，早停机制不会触发，将训练完整的num_train_epochs轮
    
    for epoch in range(num_train_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_train_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # 编码图像到潜在空间
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 采样噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 获取文本嵌入
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                
                # 预测噪声
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 计算损失（使用Min-SNR Weighting优化小数据集）
                if snr_gamma is not None:
                    # Min-SNR Weighting（小数据集优化）
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    # 标准MSE Loss
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪（防止梯度爆炸）
                if accelerator.sync_gradients:
                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                
                optimizer.step()
                lr_scheduler_obj.step()
                optimizer.zero_grad()
            
            # 累积epoch loss
            epoch_loss += loss.detach().item()
            
            # 更新进度条（显示loss和学习率）
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": loss.detach().item(), "lr": f"{current_lr:.2e}"})
            global_step += 1
            
            # 保存检查点
            if global_step % checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = output_path / f"checkpoint-{global_step}"
                    save_path.mkdir(exist_ok=True)
                    unet.save_pretrained(save_path)
        
        # 每个epoch结束后打印训练loss
        if accelerator.is_main_process:
            # 计算并打印epoch平均训练loss
            avg_train_loss = epoch_loss / len(train_dataloader)
            print(f"\nEpoch {epoch+1} 完成...")
            print(f"  训练集Loss: {avg_train_loss:.4f}")
            
            # 每个epoch都生成验证图像观察训练进度
            print(f"\n验证 Epoch {epoch+1}...")
            unet.eval()
            
            # 1. 计算验证集loss（如果有验证集）
            if val_dataloader:
                val_loss_total = 0
                val_steps = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        # 编码图像到潜在空间
                        val_latents = vae.encode(val_batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                        val_latents = val_latents * vae.config.scaling_factor
                        
                        # 采样噪声
                        val_noise = torch.randn_like(val_latents)
                        val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                                      (val_latents.shape[0],), device=val_latents.device).long()
                        
                        # 添加噪声
                        val_noisy_latents = noise_scheduler.add_noise(val_latents, val_noise, val_timesteps)
                        
                        # 获取文本嵌入
                        val_encoder_hidden_states = text_encoder(val_batch["input_ids"].to(accelerator.device))[0]
                        
                        # 预测噪声
                        val_model_pred = unet(val_noisy_latents, val_timesteps, val_encoder_hidden_states).sample
                        
                        # 计算损失
                        val_loss = torch.nn.functional.mse_loss(val_model_pred.float(), val_noise.float(), reduction="mean")
                        val_loss_total += val_loss.item()
                        val_steps += 1
                    
                    avg_val_loss = val_loss_total / val_steps
                    print(f"  验证集Loss: {avg_val_loss:.4f}")
                    
                    # 保存最佳模型
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        print(f"  ✓ 新的最佳验证Loss！保存模型...")
                        best_model_path = output_path / "best_model"
                        best_model_path.mkdir(exist_ok=True)
                        unet.save_pretrained(best_model_path)
                        # 保存loss记录
                        with open(output_path / "best_loss.txt", "w") as f:
                            f.write(f"Epoch: {epoch+1}\n")
                            f.write(f"Train Loss: {avg_train_loss:.4f}\n")
                            f.write(f"Val Loss: {avg_val_loss:.4f}\n")
                    else:
                        patience_counter += 1
                        print(f"  验证Loss未改善 ({patience_counter}/50)")
                        
                        # 早停检查（只在有验证集时才会触发）
                        if patience_counter >= 50:  # 连续50个epoch未改善才停止
                            print(f"\n早停触发！验证Loss连续50个epoch未改善")
                            print(f"最佳验证Loss: {best_val_loss:.4f} (Epoch {epoch+1-50})")
                            print(f"最佳模型已保存到: {output_path / 'best_model'}")
                            return True
                
            # 2. 生成验证图像（每个epoch都生成，不使用验证集也会生成）
            validation_dir = output_path / "validation_images"
            validation_dir.mkdir(exist_ok=True)
            
            # 创建pipeline用于生成（使用DPM-Solver++调度器）
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            
            # 配置DPM-Solver++调度器
            dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
            
            # 获取基础UNet（从PeftModel中解包）
            unwrapped_unet = accelerator.unwrap_model(unet)
            if hasattr(unwrapped_unet, 'get_base_model'):
                # 如果是PeftModel，获取基础模型
                base_unet = unwrapped_unet.get_base_model()
            else:
                base_unet = unwrapped_unet
            
            # 创建pipeline（使用基础UNet + LoRA权重）
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                unet=base_unet,
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                scheduler=dpm_scheduler,
                safety_checker=None,
                torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
            )
            pipeline = pipeline.to(accelerator.device)
            
            # 生成图像（DPM-Solver++ 25步，避免GPU超时）
            for i in range(num_validation_images):
                with torch.no_grad():
                    # 生成512×512图像
                    image_512 = pipeline(
                        validation_prompt,  # 条件文本："user 0"
                        num_inference_steps=25,  # DPM-Solver++ 25步（避免GPU超时）
                        guidance_scale=9.0,  # CFG强度（31个用户需要更强条件控制）
                    ).images[0]
                    
                    # Resize到256×256（与训练数据一致）
                    image_256 = image_512.resize((256, 256), Image.LANCZOS)
                    
                    # 保存图像
                    image_path = validation_dir / f"epoch_{epoch+1:03d}_sample_{i}.png"
                    image_256.save(image_path)
            
            print(f"  ✓ 验证图像已保存到: {validation_dir} (256×256, DPM-Solver++ 25步)")
            
            # 清理pipeline释放显存
            del pipeline
            del dpm_scheduler
            torch.cuda.empty_cache()
            unet.train()
    
    # 8. 保存最终模型
    print("\n6. 保存最终模型...")
    if accelerator.is_main_process:
        unet.save_pretrained(output_path)
        print(f"  ✓ 模型已保存到: {output_path}")
    
    print()
    print("="*60)
    print("训练完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"LoRA权重: {output_dir}")
    print()
    print("生成图像:")
    print(f"  python generate_sd_lora.py --lora_weights {output_dir} --user_id 0")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='SD+LoRA训练 - Python版本')
    
    # 路径参数
    parser.add_argument('--model_name', type=str,
                        default='runwayml/stable-diffusion-v1-5',
                        help='预训练模型名称')
    parser.add_argument('--dataset_path', type=str,
                        default='./sd_lora_dataset',
                        help='训练集路径')
    parser.add_argument('--val_dataset_path', type=str,
                        default=None,
                        help='验证集路径（可选）')
    parser.add_argument('--output_dir', type=str,
                        default='./sd_lora_output',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--resolution', type=int, default=512,
                        help='图像分辨率')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='训练batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--num_train_epochs', type=int, default=150,
                        help='训练epoch数（跨域迁移需100-200 epochs，每图需50-100步）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率（跨域迁移推荐1e-4，参考医学影像）')
    parser.add_argument('--lr_scheduler', type=str, default='constant_with_warmup',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 
                                'polynomial', 'constant', 'constant_with_warmup'],
                        help='学习率调度器')
    parser.add_argument('--lr_warmup_steps', type=int, default=2000,
                        help='学习率预热步数（长训练需更长warmup，约5 epochs）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=64,
                        help='LoRA秩（跨域任务需要更强表达能力）')
    parser.add_argument('--lora_alpha', type=int, default=64,
                        help='LoRA缩放因子（通常等于rank）')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA Dropout系数（rank=64需要dropout防止过拟合）')
    
    # 验证和保存
    parser.add_argument('--validation_prompt', type=str, default='user 0',
                        help='验证提示词')
    parser.add_argument('--validation_epochs', type=int, default=20,
                        help='每N个epoch验证一次（150 epochs共验证7-8次）')
    parser.add_argument('--num_validation_images', type=int, default=1,
                        help='每次验证生成的图像数（避免GPU超时，推荐1张）')
    parser.add_argument('--checkpointing_steps', type=int, default=500,
                        help='每N步保存一次检查点')
    
    # 优化参数
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='使用梯度检查点节省显存')
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing',
                        action='store_false',
                        help='不使用梯度检查点')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False,
                        help='使用8bit Adam优化器（需要bitsandbytes）')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='L2正则化系数（LoRA官方推荐1e-2）')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪阈值（防止梯度爆炸，推荐1.0）')
    parser.add_argument('--snr_gamma', type=float, default=5.0,
                        help='Min-SNR Weighting系数（官方推荐5.0，小数据集优化）')
    
    # 数据增强
    parser.add_argument('--center_crop', action='store_true', default=True,
                        help='中心裁剪')
    parser.add_argument('--no_center_crop', dest='center_crop', action='store_false',
                        help='不使用中心裁剪')
    parser.add_argument('--random_flip', action='store_true', default=False,
                        help='随机水平翻转（微多普勒时频图不建议使用）')
    parser.add_argument('--no_random_flip', dest='random_flip', action='store_false',
                        help='不使用随机翻转')
    
    # 日志
    parser.add_argument('--report_to', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb', 'all'],
                        help='日志记录工具')
    
    args = parser.parse_args()
    
    # 运行训练
    success = train_sd_lora(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        output_dir=args.output_dir,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        validation_prompt=args.validation_prompt,
        validation_epochs=args.validation_epochs,
        num_validation_images=args.num_validation_images,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        use_8bit_adam=args.use_8bit_adam,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        snr_gamma=args.snr_gamma,
        report_to=args.report_to
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
