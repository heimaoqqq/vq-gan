#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Stage 1: 预训练通用去噪骨干（Diffusers UNet2DConditionModel）
==========================================================
两阶段训练策略 - 第一阶段

目标：
  - 在1550张训练集上学习通用的去噪特征表示
  - 为Stage 2 LoRA微调提供强大的初始化
  - 使用Diffusers生态，确保与PEFT完美兼容

配置策略：
  - 30000步适度训练（~650 epochs）- 学习通用特征，不过拟合细节
  - 启用所有正则化技术（EMA, Min-SNR, Weight Decay, Warmup）
  - 使用强正则化控制的训练，而非禁用优化来欠拟合

主要修改（基于Diffusers官方train_text_to_image_lora.py）：
  1. 删除text_encoder，使用ClassEmbedding（31类）
  2. 修改UNet配置（dim=96, dim_mults=(1,2,4,4), attn_head=64）
  3. 使用LatentDataset（预处理的VAE latents）
  4. Cosine beta schedule（squaredcos_cap_v2）
  5. v-prediction objective
"""

import argparse
import logging
import math
import os
import random
import warnings
from pathlib import Path
from contextlib import nullcontext

# 过滤pydantic警告（diffusers内部兼容性问题）
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# 导入我们的VAE和数据集
from vae.kl_vae import KL_VAE
from train_latent_cfg import LatentDataset

# 导入EMA
from ema_pytorch import EMA

if is_wandb_available():
    import wandb

# 版本检查：我们使用的是标准功能，0.30.0+即可
check_min_version("0.30.0")

logger = get_logger(__name__, log_level="INFO")


class ClassEmbedding(torch.nn.Module):
    """
    类别嵌入模块 - 替代text_encoder
    将离散的class_id映射到continuous embedding
    
    Args:
        num_classes: 类别数量（31个用户）
        embed_dim: embedding维度（384，对应cross_attention_dim）
    """
    def __init__(self, num_classes=31, embed_dim=384):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_classes, embed_dim)
        # 初始化
        torch.nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, class_ids):
        """
        Args:
            class_ids: [B] 类别ID
        Returns:
            [B, 1, embed_dim] 用于cross-attention的条件向量
        """
        emb = self.embedding(class_ids)  # [B, embed_dim]
        return emb.unsqueeze(1)  # [B, 1, embed_dim]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Diffusers UNet pre-training")
    
    # ========== 路径配置 ==========
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_stage1_diffusers",
        help="输出目录",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=r"D:\Ysj\新建文件夹\VA-VAE\klvae_checkpoints\kl_vae_best.pt",
        help="VAE模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"D:\Ysj\新建文件夹\VA-VAE\dataset\organized_gait_dataset\kaggle\working\organized_gait_dataset\Normal_line",
        help="数据集路径",
    )
    parser.add_argument(
        "--latents_cache_folder",
        type=str,
        default="./latents_cache_gmm",
        help="预处理latents缓存路径（GMM预编码）",
    )
    
    # ========== 数据配置 ==========
    parser.add_argument(
        "--num_users",
        type=int,
        default=31,
        help="用户数量（类别数）",
    )
    parser.add_argument(
        "--images_per_user",
        type=int,
        default=30,
        help="每用户训练图像数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    
    # ========== UNet配置 ==========
    parser.add_argument(
        "--sample_size",
        type=int,
        default=32,
        help="Latent size（32x32）",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        help="Latent channels",
    )
    parser.add_argument(
        "--block_out_channels",
        type=str,
        default="96,192,384,384",
        help="UNet block输出通道（保持train_latent_cfg.py原始配置，控制容量避免过拟合）",
    )
    parser.add_argument(
        "--attention_head_dim",
        type=str,
        default="1,2,4,4",
        help="每层的注意力头数量（num_heads）：head_dim=channels//num_heads=96，适合小数据集",
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=384,
        help="Cross-attention维度（class_emb维度）",
    )
    
    # ========== 训练配置 ==========
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=24,
        help="训练batch size（xformers优化后显存~5GB，无gradient accumulation）",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="训练epochs数",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="最大训练步数（Stage1预训练：~128 epochs，学习通用特征，避免过拟合）",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数（batch=24，无累积，最快训练速度）",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="学习率",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help="学习率调度器（Stage1使用warmup后保持常数）",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=250,
        help="学习率warmup步数（5000步的5%，约125个epoch）",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.99,
        help="Adam beta2",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-4,
        help="Adam weight decay（L2正则化，小数据集防止过拟合）",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam epsilon",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="最大梯度范数（梯度裁剪）",
    )
    
    # ========== Diffusion配置 ==========
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="v_prediction",
        help="Diffusion预测类型",
        choices=["epsilon", "v_prediction", "sample"],
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR weighting gamma（小数据集关键，设为5）",
    )
    
    # ========== EMA配置 ==========
    parser.add_argument(
        "--use_ema",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="使用EMA（Stage1预训练建议启用，提高泛化能力）",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="EMA衰减率（小数据集用0.995）",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        default=10,
        help="EMA更新频率",
    )
    
    # ========== 优化配置 ==========
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练（BF16：速度快50%+，Ada GPU原生支持）",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="启用xFormers加速（默认开启，BF16下速度提升2-3倍）",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="启用gradient checkpointing（默认开启，节省30-40%显存）",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="允许TF32加速（Ampere GPU）",
    )
    
    # ========== 日志和保存 ==========
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="日志目录",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", None],
        help="日志工具（可选，None则不记录）",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="保存checkpoint的步数间隔（每500步保存一次）",
    )
    parser.add_argument(
        "--save_and_sample_every",
        type=int,
        default=500,
        help="可视化采样间隔（每500步生成图像观察训练效果，共10次）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="可视化生成图像数（生成前N个用户各1张）",
    )
    
    # ========== 数据加载 ==========
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="DataLoader工作进程数（Windows建议0，Linux可用4）",
    )
    
    # ========== Resume ==========
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从checkpoint恢复训练",
    )
    
    args = parser.parse_args()
    
    # 解析block_out_channels和attention_head_dim
    args.block_out_channels = tuple(int(x) for x in args.block_out_channels.split(','))
    args.attention_head_dim = tuple(int(x) for x in args.attention_head_dim.split(','))
    
    # 验证配置
    if len(args.attention_head_dim) != len(args.block_out_channels):
        raise ValueError(
            f"attention_head_dim长度({len(args.attention_head_dim)}) "
            f"必须等于block_out_channels长度({len(args.block_out_channels)})"
        )
    
    # 验证配置（xformers要求head_dim是8的倍数）
    for i, (channels, num_heads) in enumerate(zip(args.block_out_channels, args.attention_head_dim)):
        if channels % num_heads != 0:
            raise ValueError(f"Layer {i}: {channels} channels不能被num_heads={num_heads}整除")
        head_dim = channels // num_heads
        if head_dim % 8 != 0:
            raise ValueError(f"Layer {i}: head_dim={head_dim}必须是8的倍数（xformers要求）")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def save_and_sample(
    unet,
    class_emb,
    vae,
    noise_scheduler,
    args,
    accelerator,
    global_step,
    ema_unet=None,
):
    """保存检查点并生成样本（参考train_latent_cfg.py实现）"""
    milestone = global_step // args.save_and_sample_every
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Checkpoint {milestone} (步数: {global_step})")
    logger.info(f"{'='*60}")
    
    # 使用EMA模型或原始模型采样
    if ema_unet is not None:
        sample_unet = ema_unet.ema_model
        logger.info("  Using EMA model for sampling")
    else:
        sample_unet = unet
    
    sample_unet.eval()
    class_emb.eval()
    
    device = accelerator.device
    
    # 生成样本
    try:
        with torch.no_grad():
            # 选择要生成的用户（前num_samples个）
            num_samples = min(args.num_samples, args.num_users)
            user_ids = torch.arange(num_samples, device=device)
            
            # 准备条件
            encoder_hidden_states = class_emb(user_ids)  # [num_samples, 1, 384]
            
            # 初始噪声
            latents = torch.randn(
                num_samples,
                args.in_channels,
                args.sample_size,
                args.sample_size,
                device=device,
                dtype=torch.float32,
            )
            
            # DDIM采样（100步，快速且质量好）
            noise_scheduler.set_timesteps(100, device=device)
            
            for t in noise_scheduler.timesteps:
                # 预测噪声
                latent_model_input = latents
                timestep = t.expand(latents.shape[0])
                
                noise_pred = sample_unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # 更新latents
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # VAE解码（decode_latents内部会处理scale_factor）
            sampled_images = vae.decode_latents(latents)  # [num_samples, 3, 256, 256]
            
            # 保存图像网格（与train_latent_cfg.py保持一致）
            from torchvision import utils
            import math
            save_path = os.path.join(args.output_dir, f'sample-{milestone}.png')
            utils.save_image(
                sampled_images,
                save_path,
                nrow=int(math.sqrt(num_samples))
            )
            logger.info(f"✓ 样本已保存: {save_path}")
            
            # 简单质量检查
            img_min = sampled_images.min().item()
            img_max = sampled_images.max().item()
            img_mean = sampled_images.mean().item()
            
            if img_min < -0.1 or img_max > 1.1:
                logger.warning(f"  ⚠️ 警告: 图像值异常 [{img_min:.3f}, {img_max:.3f}]")
            
            logger.info(f"  图像统计: min={img_min:.3f}, max={img_max:.3f}, mean={img_mean:.3f}")
            
    except Exception as e:
        logger.error(f"  ✗ 生成样本失败: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"{'='*60}\n")
    
    # 恢复训练模式
    if ema_unet is None:
        unet.train()
    class_emb.train()


def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== 加载模型 ====================
    logger.info("Loading models...")
    
    # 1. VAE（使用我们训练的KL-VAE）
    vae = KL_VAE(embed_dim=4, scale_factor=0.18215)
    vae_ckpt = torch.load(args.vae_path, map_location='cpu')
    if isinstance(vae_ckpt, dict) and 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)
    vae.requires_grad_(False)
    vae.eval()
    
    # 2. Noise Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",  # cosine schedule
        prediction_type=args.prediction_type,
    )
    
    # 3. ClassEmbedding（替代text_encoder）
    class_emb = ClassEmbedding(
        num_classes=args.num_users,
        embed_dim=args.cross_attention_dim
    )
    
    # 4. UNet2DConditionModel（Diffusers标准）
    unet = UNet2DConditionModel(
        sample_size=args.sample_size,
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        layers_per_block=2,
        block_out_channels=args.block_out_channels,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        attention_head_dim=args.attention_head_dim,
        cross_attention_dim=args.cross_attention_dim,
    )
    
    logger.info(f"UNet: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M params")
    logger.info(f"ClassEmb: {sum(p.numel() for p in class_emb.parameters()) / 1e6:.2f}M params")
    logger.info(f"Config: channels={args.block_out_channels}, num_heads={args.attention_head_dim}")
    for i, (ch, nh) in enumerate(zip(args.block_out_channels, args.attention_head_dim)):
        logger.info(f"  L{i}: {ch}ch, {nh}heads, head_dim={ch//nh}")
    
    # ==================== 混合精度配置 ====================
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # VAE保持float32（仅推理）
    vae.to(accelerator.device, dtype=torch.float32)
    
    # UNet和ClassEmb可以用mixed precision
    unet.to(accelerator.device)
    class_emb.to(accelerator.device)
    
    # ==================== GPU信息 ====================
    if torch.cuda.is_available() and accelerator.is_main_process:
        device_idx = 0 if isinstance(accelerator.device, str) else (accelerator.device.index or 0)
        gpu_name = torch.cuda.get_device_name(device_idx)
        total_mem = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({total_mem:.0f}GB), 预计显存: 2-3GB")
    
    # ==================== 优化配置 ====================
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers to at least 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers memory-efficient attention enabled")
        else:
            logger.warning(
                "xformers is not available. Training will continue without memory-efficient attention. "
                "This will be slower but will not affect training quality."
            )
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # ==================== 数据集 ====================
    logger.info("Loading dataset...")
    
    train_dataset = LatentDataset(
        vae=vae,
        data_path=args.data_path,
        latents_cache_folder=args.latents_cache_folder,
        num_users=args.num_users,
        images_per_user=args.images_per_user,
        seed=args.seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    
    # ==================== 优化器 ====================
    # 训练UNet和ClassEmb
    trainable_params = list(unet.parameters()) + list(class_emb.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # ==================== 学习率调度器 ====================
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # ==================== EMA初始化 ====================
    ema_unet = None
    if args.use_ema and accelerator.is_main_process:
        ema_unet = EMA(
            unet,
            beta=args.ema_decay,
            update_every=args.ema_update_every
        )
        ema_unet.to(accelerator.device)
        logger.info(f"EMA initialized: decay={args.ema_decay}, update_every={args.ema_update_every}")
    
    # ==================== Accelerator准备 ====================
    unet, class_emb, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, class_emb, optimizer, train_dataloader, lr_scheduler
    )
    
    # 重新计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # ==================== Tracker初始化 ====================
    if accelerator.is_main_process:
        accelerator.init_trackers("stage1-diffusers", config=vars(args))
    
    # ==================== 训练信息 ====================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # ==================== Resume ====================
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # ==================== 训练循环 ====================
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        class_emb.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 解包数据
                latents, labels = batch  # latents: [B, 4, 32, 32], labels: [B]
                
                # 转换为正确的dtype
                latents = latents.to(weight_dtype)
                
                # 获取条件编码
                encoder_hidden_states = class_emb(labels)  # [B, 1, 384]
                
                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # 计算target
                if args.prediction_type == "epsilon":
                    target = noise
                elif args.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                elif args.prediction_type == "sample":
                    target = latents
                else:
                    raise ValueError(f"Unknown prediction type {args.prediction_type}")
                
                # 计算损失（使用Min-SNR weighting）
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
                
                # 累积损失
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # 更新EMA
                if ema_unet is not None and accelerator.sync_gradients:
                    ema_unet.update()
            
            # 检查是否需要更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # 保存checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                # 定期生成可视化样本
                if global_step % args.save_and_sample_every == 0:
                    if accelerator.is_main_process:
                        save_and_sample(
                            unet,
                            class_emb,
                            vae,
                            noise_scheduler,
                            args,
                            accelerator,
                            global_step,
                            ema_unet=ema_unet,
                        )
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
    
    # ==================== 最终保存 ====================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 保存UNet
        unet_save_path = os.path.join(args.output_dir, "unet")
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(unet_save_path)
        
        # 保存EMA UNet（如果启用）
        if ema_unet is not None:
            ema_unet_save_path = os.path.join(args.output_dir, "unet_ema")
            ema_unet.ema_model.save_pretrained(ema_unet_save_path)
            logger.info(f"EMA model saved to {ema_unet_save_path}")
        
        # 保存ClassEmb
        class_emb_save_path = os.path.join(args.output_dir, "class_emb.pt")
        class_emb = accelerator.unwrap_model(class_emb)
        torch.save(class_emb.state_dict(), class_emb_save_path)
        
        # 保存scheduler
        noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))
        
        logger.info(f"Model saved to {args.output_dir}")
        
        # 最终采样
        save_and_sample(
            unet,
            class_emb,
            vae,
            noise_scheduler,
            args,
            accelerator,
            args.max_train_steps,  # 使用最终步数
            ema_unet=ema_unet,
        )
    
    accelerator.end_training()


if __name__ == "__main__":
    main()

  
