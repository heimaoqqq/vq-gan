"""
Latent Diffusion Training with Classifier-Free Guidance
========================================================
微多普勒时频图像生成 - 基于预训练VAE的潜在扩散模型

数据流：
  JPG图像(256×256×3) → VAE编码 → 潜在表示(32×32×4) → DDPM训练

数据特点与挑战：
  ✓ 数据规模：31用户 × 50训练样本 = 1550张（小数据集）
  ✓ 用户内变异：极大（步态周期、角度、状态差异显著）
  ✓ 用户间差异：极小（肉眼难以区分，需要强判别能力）
  ✗ 传统数据增强效果差
  
关键设计决策：
  1. 更大的模型容量（dim=96, 4层）→ 捕捉微妙差异
  2. 更强的条件编码（96维 × 31类）→ 有效表达用户特征
  3. 平衡的CFG设置（drop=0.2, scale=3.5）→ 避免模式崩塌
  4. 稳定的训练策略（低lr, 梯度累积）→ 防止过拟合

Kaggle路径：
  VAE: /kaggle/input/kl-vae-best-pt/kl_vae_best.pt
  数据: /kaggle/input/organized-gait-dataset/Normal_line/ID_*/
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np

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
from accelerate import Accelerator
from ema_pytorch import EMA


# ============================================================
# 配置参数
# ============================================================
class Config:
    """集中管理所有配置参数"""
    
    # === 路径配置 ===
    vae_path = '/kaggle/input/kl-vae-best-pt/kl_vae_best.pt'
    data_path = '/kaggle/input/organized-gait-dataset/Normal_line'
    results_folder = './results'
    latents_cache_folder = './latents_cache'  # 预处理缓存
    
    # === 数据配置 ===
    num_users = 31  # ID_1 到 ID_31
    images_per_user_total = 150  # 每用户总图像数
    images_per_user_train = 50   # 每用户用于DDPM训练的图像数
    # 剩余100张作为测试集，仅用于分类器评估，对DDPM不可见
    image_size = 256
    latent_size = 32  # VAE是8倍下采样（256/8=32）
    latent_channels = 4
    
    # === 模型配置（针对微多普勒数据优化）===
    # 关键考虑：用户间差异极小，需要强大的条件编码和判别能力
    # 实际参数量约44M
    dim = 96  # 基础维度：平衡模型容量与过拟合风险
    dim_mults = (1, 2, 4, 4)  # 4层结构：避免最后一层过大
    attn_dim_head = 64  # 注意力头维度：增加到64以匹配classes_dim=384（64×8=512>384）
    attn_heads = 8  # 增加注意力头数：更好捕捉用户间微妙差异
    cond_drop_prob = 0.0  # ❌ 关闭CFG训练（极小数据集应专注于conditional质量）
    # 分析：
    #   - 我们的目标：条件生成（需要强大的conditional分支）
    #   - CFG训练会削弱conditional（50→35张/类，减少30%）
    #   - 对于50张/类的极小数据集，不应该为CFG牺牲conditional训练
    #   - 结论：放弃CFG，全力训练conditional
    
    # === 扩散配置 ===
    timesteps = 1000
    sampling_timesteps = 150  # DDIM采样步数（100步足够，质量好且快）
    objective = 'pred_v'  # v-prediction
    beta_schedule = 'cosine'
    # === 采样配置 ===
    # cond_drop=0.0时，不用CFG
    cond_scale = 1.0  # 不用CFG（未训练unconditional）
    rescaled_phi = 0.0  # 不用CFG++
    
    # === 训练配置（针对RTX 5880 48GB + 小数据集优化）===
    train_batch_size = 12  # batch size：模型44M参数，48GB显存绰绰有余
    gradient_accumulate_every = 3  # 梯度累积：有效batch=36，保持训练稳定性
    train_lr = 4e-5  # 学习率：标准值
    train_num_steps = 60000  # 训练步数：约1290 epochs
    
    # Learning Rate Schedule
    use_lr_warmup = True  # 使用学习率warmup
    warmup_steps = 1000  # 前1000步warmup
    
    # === 优化配置（防止过拟合 + 增加多样性）===
    use_ema = True  # 是否使用EMA（Baseline设为False）
    ema_decay = 0.995  # EMA平滑：小数据集降低decay，增加多样性（0.999→0.995）
    ema_update_every = 10  # EMA更新频率
    max_grad_norm = 1.0  # 梯度裁剪：防止梯度爆炸
    adam_betas = (0.9, 0.99)  # Adam优化器参数
    weight_decay = 1e-4  # 权重衰减：L2正则化，防止过拟合
    
    # === Min-SNR优化（小数据集关键）===
    # Min-SNR帮助模型更好地学习所有时间步，避免过度关注简单样本
    min_snr_loss_weight = True
    min_snr_gamma = 5  # gamma=5适合小数据集
    
    # === 对比学习配置 ===
    # ⚠️ 关键发现：对比学习导致模式崩溃（所有用户生成相似图像）
    use_contrastive_loss = False  # ❌ 关闭对比学习（实验证明：降低用户特色）
    contrastive_weight = 0.0  # 权重设为0
    # 问题诊断：
    #   - 对比学习过度优化"同类相似"目标
    #   - 导致生成"平均模式"而非真实用户特征
    #   - Baseline（无对比学习）反而保留更多用户特色
    
    contrastive_temperature = 0.07  # SupCon温度参数（标准值）
    
    contrastive_start_step = 5000  # 延迟启动（2000→5000，让扩散先稳定）
    # 调整理由：
    #   - 小数据集需要扩散模型先学会基本生成能力
    #   - 过早加入对比学习可能干扰扩散训练
    #   - 5000步≈107 epochs，扩散模型已基本稳定
    
    # === 归一化配置 ===
    # ⚠️ 重要：运行 test_vae_range.py 确定此参数！
    # 如果VAE潜在表示在[0,1]范围 → auto_normalize=True
    # 如果不在[0,1]范围 → auto_normalize=False
    auto_normalize = False  # 默认False，运行test_vae_range.py后根据结果调整
    
    # === 监控配置 ===
    save_and_sample_every = 1000  # 每1000步保存（小数据集更频繁观察，避免错过最佳点）
    num_samples = 16  # 生成16张检查
    
    # === 其他 ===
    amp = False  # 混合精度（P100不支持Tensor Cores，FP16反而不稳定）
    num_workers = 0  # Windows上设为0避免多进程问题，数据已缓存所以影响不大
    seed = 42
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("训练配置摘要")
        print("="*60)
        
        print(f"\n数据: {self.num_users}用户 × {self.images_per_user_train}张 = {self.num_users * self.images_per_user_train}张")
        
        print(f"\n模型: dim={self.dim}, 层数={len(self.dim_mults)}, 注意力={self.attn_heads}头")
        print(f"      估算参数量 ~{self._estimate_params():.1f}M（实际以训练时显示为准）")
        
        print(f"\nCFG: drop={self.cond_drop_prob}, scale={self.cond_scale}")
        
        print(f"\n训练: batch={self.train_batch_size}×{self.gradient_accumulate_every}, lr={self.train_lr}, steps={self.train_num_steps:,}")
        
        # LR Warmup
        if self.use_lr_warmup:
            print(f"\nLR Warmup: {self.warmup_steps}步")
        
        # 对比学习配置
        if self.use_contrastive_loss:
            print(f"\n对比学习: 启用")
            print(f"      权重={self.contrastive_weight}, 温度={self.contrastive_temperature}")
            print(f"      开始步数={self.contrastive_start_step}")
        else:
            print(f"\n对比学习: 未启用")
        
        print("="*60 + "\n")
    
    def _estimate_params(self):
        """估计模型参数量（百万）"""
        # 更准确的UNet参数估计
        total = 0
        
        # 初始卷积
        total += self.dim * self.latent_channels * 7 * 7
        
        # Encoder/Decoder层（每层2个ResBlock + 注意力）
        dims = [self.dim * m for m in self.dim_mults]
        for i in range(len(dims)):
            d = dims[i]
            # ResBlock参数（2个卷积 + 时间/类别嵌入）
            total += d * d * 3 * 3 * 4  # 4个ResBlock
            # 注意力参数
            total += d * d * 4  # QKV + output projection
        
        # 时间嵌入MLP
        time_dim = self.dim * 4
        total += self.dim * time_dim + time_dim * time_dim
        
        # 类别嵌入MLP
        total += self.dim * time_dim + time_dim * time_dim
        
        # 中间层
        total += dims[-1] * dims[-1] * 3 * 3 * 2
        
        return total / 1e6


# ============================================================
# 数据集类
# ============================================================
class LatentDataset(Dataset):
    """
    加载预处理的潜在表示
    如果缓存不存在，会自动从原始图像编码
    
    数据划分：
    - 每用户150张图像，仅使用前50张训练DDPM
    - 后100张保留作为测试集，用于分类器评估
    """
    def __init__(self, vae, data_path, latents_cache_folder, 
                 num_users=31, images_per_user=50, seed=42):
        super().__init__()
        self.vae = vae
        self.data_path = Path(data_path)
        self.latents_cache_folder = Path(latents_cache_folder)
        self.seed = seed
        
        # 收集所有图像路径和标签
        self.samples = []
        
        # 尝试从data_split.json加载预处理的划分
        split_file = self.latents_cache_folder / 'data_split.json'
        use_precomputed_split = False
        
        if split_file.exists():
            import json
            try:
                with open(split_file, 'r', encoding='utf-8') as f:
                    split_info = json.load(f)
                
                sampling_method = split_info.get('sampling_method', 'unknown')
                print(f"✓ 找到预处理的数据划分: {split_file}")
                print(f"  采样方法: {sampling_method}")
                
                # 使用预处理的训练集划分
                for user_key, user_info in split_info['users'].items():
                    user_id = user_info['user_id']
                    label = user_info['label']
                    
                    # 获取训练集文件路径
                    for rel_path in user_info['train_images']:
                        img_path = self.data_path / rel_path
                        if img_path.exists():
                            self.samples.append((img_path, label))
                
                use_precomputed_split = True
                print(f"✓ 使用预处理的训练集划分 ({sampling_method})")
                
            except Exception as e:
                print(f"Warning: 无法加载data_split.json: {e}")
                print("  将使用随机抽样")
        
        # 如果没有预处理的划分，使用随机抽样（旧方法）
        if not use_precomputed_split:
            print("未找到预处理的数据划分，使用随机抽样")
            
            for user_id in range(1, num_users + 1):
                user_folder = self.data_path / f"ID_{user_id}"
                if not user_folder.exists():
                    print(f"Warning: {user_folder} not found, skipping...")
                    continue
                
                # 收集该用户的所有jpg图像（排序确保一致性）
                image_paths = sorted(list(user_folder.glob("*.jpg")))
                
                # 为每个用户设置独立但可复现的随机种子
                user_seed = seed + user_id
                rng = np.random.RandomState(user_seed)
                
                # 随机打乱（固定种子）
                indices = rng.permutation(len(image_paths))
                image_paths = [image_paths[i] for i in indices]
                
                # 仅使用前images_per_user张
                train_paths = image_paths[:images_per_user]
                
                for img_path in train_paths:
                    self.samples.append((img_path, user_id - 1))  # label: 0-30
        
        print(f"DDPM训练集: {len(self.samples)} 张图像 ({num_users}用户 × {images_per_user}张/用户)")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()  # 自动归一化到[0,1]
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 尝试从缓存加载
        # 使用 userID_filename.pt 格式避免文件名冲突
        cache_filename = f"user_{label:02d}_{img_path.stem}.pt"
        cache_path = self.latents_cache_folder / cache_filename
        
        if cache_path.exists():
            latent = torch.load(cache_path, map_location='cpu', weights_only=True)
        else:
            # 从原始图像编码
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            
            with torch.no_grad():
                img_batch = img.unsqueeze(0).to(next(self.vae.parameters()).device)
                latent = self.vae.encode_images(img_batch)  # [1, 4, 32, 32]
                latent = latent.squeeze(0).cpu()  # [4, 32, 32]
            
            # 保存缓存
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latent, cache_path)
        
        return latent, label


# ============================================================
# 训练器
# ============================================================
class LatentDiffusionTrainer:
    """简化的训练器，集成Accelerator"""
    
    def __init__(self, config):
        self.config = config
        
        # 打印配置摘要
        config.print_config_summary()
        
        # 初始化Accelerator
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if config.amp else 'no'
        )
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 加载VAE
        print("Loading VAE...")
        print(f"  VAE模块路径: {KL_VAE.__module__}")
        import vae.kl_vae
        print(f"  VAE文件路径: {vae.kl_vae.__file__}")
        
        # 加载checkpoint
        checkpoint = torch.load(config.vae_path, map_location='cpu')
        
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
            print("  Detected direct state_dict format")
            vae_state = checkpoint
            embed_dim = 4
            scale_factor = 0.18215
        
        # 创建VAE并加载权重
        self.vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
        self.vae.load_state_dict(vae_state)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(self.accelerator.device)
        print(f"VAE loaded from {config.vae_path}")
        
        # 创建数据集
        print("Creating dataset...")
        train_ds = LatentDataset(
            self.vae, config.data_path, config.latents_cache_folder,
            config.num_users, 
            images_per_user=config.images_per_user_train,
            seed=config.seed
        )
        
        # 创建DataLoader
        self.train_dl = DataLoader(
            train_ds, 
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 创建Unet模型
        print("Creating Unet model...")
        self.model = Unet(
            dim=config.dim,
            dim_mults=config.dim_mults,
            num_classes=config.num_users,
            cond_drop_prob=config.cond_drop_prob,
            channels=config.latent_channels,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
            learned_variance=False
        )
        
        # 计算参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params/1e6:.2f}M")
        
        # 创建扩散模型
        print("Creating diffusion model...")
        self.diffusion = GaussianDiffusion(
            self.model,
            image_size=config.latent_size,
            timesteps=config.timesteps,
            sampling_timesteps=config.sampling_timesteps,
            objective=config.objective,
            beta_schedule=config.beta_schedule,
            min_snr_loss_weight=config.min_snr_loss_weight,
            min_snr_gamma=config.min_snr_gamma,
            auto_normalize=config.auto_normalize  # 从Config读取
        )
        
        print(f"Auto normalize: {config.auto_normalize}")
        if not config.auto_normalize:
            print("  → 潜在空间不做[0,1]→[-1,1]归一化")
        else:
            print("  → 潜在空间会被归一化到[-1,1]")
        
        # 创建优化器（添加weight decay防止过拟合）
        self.opt = torch.optim.Adam(
            self.diffusion.parameters(),
            lr=config.train_lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        # 对比学习损失（如果启用）
        if config.use_contrastive_loss:
            print(f"\n✓ 启用对比学习损失")
            print(f"  权重: {config.contrastive_weight}")
            print(f"  温度: {config.contrastive_temperature}")
            print(f"  开始步数: {config.contrastive_start_step}")
            
            from losses import SupConLoss
            self.contrastive_criterion = SupConLoss(
                temperature=config.contrastive_temperature,
                contrast_mode='all',
                base_temperature=config.contrastive_temperature
            )
        else:
            print("\n✗ 未启用对比学习损失")
            self.contrastive_criterion = None
        
        # 使用Accelerator准备
        self.diffusion, self.opt, self.train_dl = self.accelerator.prepare(
            self.diffusion, self.opt, self.train_dl
        )
        
        # EMA（仅主进程，可选）
        self.use_ema = config.use_ema
        if self.use_ema and self.accelerator.is_main_process:
            self.ema = EMA(
                self.diffusion,
                beta=config.ema_decay,
                update_every=config.ema_update_every
            )
            self.ema.to(self.accelerator.device)
        else:
            self.ema = None
        
        # 创建结果文件夹
        self.results_folder = Path(config.results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)
        
        self.step = 0
        
        # 异常检测
        self.loss_history = []
        self.nan_count = 0
        self.high_loss_count = 0
        
        # 对比学习损失历史（用于监控）
        self.contrastive_loss_history = []
    
    def train(self):
        """训练循环"""
        config = self.config
        
        # 创建无限循环的dataloader
        def cycle(dl):
            while True:
                for data in dl:
                    yield data
        
        dl = cycle(self.train_dl)
        
        with tqdm(
            initial=self.step,
            total=config.train_num_steps,
            disable=not self.accelerator.is_main_process
        ) as pbar:
            while self.step < config.train_num_steps:
                self.diffusion.train()
                total_loss = 0.
                total_diffusion_loss = 0.
                total_contrastive_loss = 0.
                
                # 梯度累积
                for grad_accum_idx in range(config.gradient_accumulate_every):
                    latents, labels = next(dl)
                    latents = latents.to(self.accelerator.device)
                    labels = labels.to(self.accelerator.device)
                    
                    # 调试：第一步打印数据范围
                    if self.step == 0 and grad_accum_idx == 0 and self.accelerator.is_main_process:
                        print(f"\n[调试] 数据范围检查:")
                        print(f"  latents shape: {latents.shape}")
                        print(f"  latents min: {latents.min().item():.4f}")
                        print(f"  latents max: {latents.max().item():.4f}")
                        print(f"  latents mean: {latents.mean().item():.4f}")
                        print(f"  latents std: {latents.std().item():.4f}")
                    
                    with self.accelerator.autocast():
                        # 计算扩散损失（如果启用对比学习，同时提取特征）
                        if (config.use_contrastive_loss and 
                            self.contrastive_criterion is not None and
                            self.step >= config.contrastive_start_step):
                            
                            # 一次前向传播，同时计算损失和提取特征
                            # 需要访问底层模型（去除Accelerator包装）
                            unwrapped_diffusion = self.accelerator.unwrap_model(self.diffusion)
                            
                            # 随机时间步
                            b = latents.shape[0]
                            device = latents.device
                            t = torch.randint(0, unwrapped_diffusion.num_timesteps, (b,), device=device).long()
                            
                            # 归一化（如果需要）
                            img = unwrapped_diffusion.normalize(latents)
                            
                            # 调用p_losses，同时返回特征
                            diffusion_loss, features = unwrapped_diffusion.p_losses(
                                img, t, 
                                classes=labels, 
                                return_features=True
                            )
                            
                            # 准备SupCon输入格式: [B, n_views, C]
                            features = features.unsqueeze(1)  # [B, 1, C]
                            
                            # 计算SupCon损失
                            contrastive_loss = self.contrastive_criterion(features, labels)
                            contrastive_loss = contrastive_loss * config.contrastive_weight
                        else:
                            # 标准扩散损失（不提取特征）
                            diffusion_loss = self.diffusion(latents, classes=labels)
                            contrastive_loss = 0.
                        
                        # 混合损失
                        loss = diffusion_loss + contrastive_loss
                        
                        # 记录各项损失（在除以gradient_accumulate_every之前）
                        total_loss += loss.item()
                        total_diffusion_loss += diffusion_loss.item()
                        if isinstance(contrastive_loss, torch.Tensor):
                            total_contrastive_loss += contrastive_loss.item()
                        
                        # 为梯度累积缩放损失
                        loss = loss / config.gradient_accumulate_every
                    
                    self.accelerator.backward(loss)
                
                # 梯度裁剪
                self.accelerator.clip_grad_norm_(
                    self.diffusion.parameters(),
                    config.max_grad_norm
                )
                
                # 优化器步进
                self.opt.step()
                self.opt.zero_grad()
                
                # Learning Rate Warmup（小数据集稳定训练）
                if config.use_lr_warmup and self.step < config.warmup_steps:
                    # 线性warmup
                    lr_scale = (self.step + 1) / config.warmup_steps
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = config.train_lr * lr_scale
                
                self.accelerator.wait_for_everyone()
                
                # 更新进度（显示详细损失）
                if config.use_contrastive_loss and self.step >= config.contrastive_start_step:
                    pbar.set_description(
                        f'loss: {total_loss:.4f} | '
                        f'diff: {total_diffusion_loss:.4f} | '
                        f'contr: {total_contrastive_loss:.4f}'
                    )
                else:
                    pbar.set_description(f'loss: {total_loss:.4f}')
                
                self.step += 1
                
                # 记录对比学习损失
                if config.use_contrastive_loss and total_contrastive_loss > 0:
                    self.contrastive_loss_history.append(total_contrastive_loss)
                
                # 异常检测
                if self.accelerator.is_main_process:
                    self._check_training_health(total_loss)
                
                # EMA更新
                if self.ema and self.accelerator.is_main_process:
                    self.ema.update()
                
                # 定期保存和采样（不依赖EMA）
                if self.accelerator.is_main_process:
                    if self.step % config.save_and_sample_every == 0:
                        self.save_and_sample(self.step // config.save_and_sample_every)
                
                pbar.update(1)
        
        print('Training complete!')
    
    def _check_training_health(self, loss):
        """检测训练异常"""
        import math
        
        # 记录loss历史（最近100步）
        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        
        # 检查1: NaN或Inf
        if math.isnan(loss) or math.isinf(loss):
            self.nan_count += 1
            print(f"\n⚠️ 警告: Loss is NaN/Inf (第{self.nan_count}次)")
            
            if self.nan_count >= 3:
                print("\n❌ 严重错误: Loss持续NaN/Inf，训练失败！")
                print("   可能原因：学习率过大、梯度爆炸、数据问题")
                print("   建议：降低学习率或检查数据")
                raise RuntimeError("Training diverged - Loss is NaN/Inf")
        
        # 检查2: Loss异常高
        if loss > 1.0 and self.step > 1000:
            self.high_loss_count += 1
            if self.high_loss_count > 50:
                print(f"\n⚠️ 警告: Loss持续异常高 (>{loss:.4f})，训练可能无效")
                print("   检查：数据是否正确加载？VAE是否正确？")
        else:
            self.high_loss_count = 0  # 重置
        
        # 检查3: Loss不下降（每5000步检查）
        if self.step % 5000 == 0 and self.step > 5000:
            if len(self.loss_history) >= 50:
                recent_avg = sum(self.loss_history[-50:]) / 50
                early_avg = sum(self.loss_history[:50]) / 50
                
                if recent_avg >= early_avg * 0.95:  # 几乎没下降
                    print(f"\n⚠️ 注意: Loss下降缓慢 ({early_avg:.4f} → {recent_avg:.4f})")
                    print("   可能原因：学习率过小、已收敛、或训练问题")
        
        # 检查4: Loss过低（可能过拟合）
        if self.step > 50000 and loss < 0.0001:
            print(f"\n⚠️ 警告: Loss非常低 ({loss:.6f})，可能过拟合/记忆训练集")
            print("   建议：检查生成样本是否与训练集完全相同")
    
    def save_and_sample(self, milestone):
        """保存检查点并生成样本"""
        # 使用EMA模型或原始模型采样
        model_for_sample = self.ema.ema_model if self.ema else self.diffusion
        model_for_sample.eval()
        
        print(f"\n{'='*60}")
        print(f"Checkpoint {milestone} (步数: {self.step})")
        print(f"{'='*60}")
        
        # 生成样本（每个用户1张，共num_samples张）
        try:
            with torch.no_grad():
                # 选择要生成的用户
                num_samples = min(self.config.num_samples, self.config.num_users)
                user_ids = torch.arange(num_samples, device=self.accelerator.device)
                
                # DDPM采样（条件扩散 + DDIM + CFG）
                sampled_latents = model_for_sample.sample(
                    classes=user_ids,              # 条件：用户ID
                    cond_scale=self.config.cond_scale,  # CFG强度
                    rescaled_phi=self.config.rescaled_phi  # CFG++ rescaling
                )
                # 采样方式：DDIM 100步（比1000步快10倍，质量相近）
                
                # VAE解码
                sampled_images = self.vae.decode_latents(sampled_latents)
                
                # 保存图像网格
                save_path = self.results_folder / f'sample-{milestone}.png'
                utils.save_image(
                    sampled_images,
                    str(save_path),
                    nrow=int(math.sqrt(num_samples))
                )
                print(f"✓ 样本已保存: {save_path}")
                
                # 简单质量检查
                img_min = sampled_images.min().item()
                img_max = sampled_images.max().item()
                img_mean = sampled_images.mean().item()
                
                if img_min < -0.1 or img_max > 1.1:
                    print(f"  ⚠️ 警告: 图像值异常 [{img_min:.3f}, {img_max:.3f}]")
                
                print(f"  图像统计: min={img_min:.3f}, max={img_max:.3f}, mean={img_mean:.3f}")
                
        except Exception as e:
            print(f"  ✗ 生成样本失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 保存检查点
        try:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.diffusion),
                'opt': self.opt.state_dict(),
                'config': self.config.__dict__,
                'loss_history': self.loss_history[-100:],  # 保存最近100步loss
                'contrastive_loss_history': self.contrastive_loss_history[-100:]  # 保存对比学习损失
            }
            if self.ema:
                data['ema'] = self.ema.state_dict()
            save_path = self.results_folder / f'model-{milestone}.pt'
            torch.save(data, str(save_path))
            print(f"✓ 检查点已保存: {save_path}")
            
            # 打印对比学习损失统计（如果启用）
            if self.config.use_contrastive_loss and len(self.contrastive_loss_history) > 0:
                recent_contr_loss = sum(self.contrastive_loss_history[-100:]) / len(self.contrastive_loss_history[-100:])
                print(f"  对比学习损失（最近100步平均）: {recent_contr_loss:.4f}")
            
            # 同时保存最新检查点（覆盖）
            latest_path = self.results_folder / 'model-latest.pt'
            torch.save(data, str(latest_path))
            print(f"✓ 最新检查点: {latest_path}")
            
            # 保存所有checkpoint（不删除）
            print(f"  💾 保留所有checkpoint以便选择最佳模型")
            
        except Exception as e:
            print(f"  ✗ 保存检查点失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")
    
    def load(self, milestone):
        """加载检查点"""
        load_path = self.results_folder / f'model-{milestone}.pt'
        data = torch.load(str(load_path), map_location=self.accelerator.device)
        
        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(data['model'])
        
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        
        if self.ema and self.accelerator.is_main_process and 'ema' in data:
            self.ema.load_state_dict(data['ema'])
        
        print(f"Loaded checkpoint from {load_path}, step {self.step}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    parser.add_argument('--resume', type=int, default=None, help='Resume from milestone')
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 创建训练器
    trainer = LatentDiffusionTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume is not None:
        trainer.load(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

