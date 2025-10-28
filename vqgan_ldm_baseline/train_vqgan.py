"""
VQ-GAN Training Script
======================
阶段一：训练VQ-GAN自编码器

论文Baseline：VQ-VAE + PatchGAN + LPIPS
不包含任何论文未提及的优化技术
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

# 添加当前目录到路径，支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import VQVAE, PatchGANDiscriminator, LPIPSWithDiscriminator
from configs import VQGANConfig
from accelerate import Accelerator


# ============================================================
# 数据集
# ============================================================
class ImageDataset(Dataset):
    """微多普勒时频图数据集"""
    
    def __init__(
        self, 
        data_path: str, 
        split_file: str = './data_split.json',
        image_size: int = 256
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.image_size = image_size
        
        # ✅ 从统一的划分文件读取训练集
        import json
        split_path = Path(split_file)
        
        if not split_path.exists():
            raise FileNotFoundError(
                f"数据划分文件不存在: {split_file}\n"
                f"请先运行: python vqgan_ldm_baseline/create_data_split.py"
            )
        
        with open(split_path, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        
        print(f"✓ 读取数据划分: {split_file}")
        print(f"  划分方法: {split_info['sampling_method']}")
        
        # 收集训练集图像路径
        self.image_paths = []
        for user_key, user_info in split_info['users'].items():
            for rel_path in user_info['train_images']:
                img_path = self.data_path / rel_path
                if img_path.exists():
                    self.image_paths.append(img_path)
                else:
                    print(f"⚠️  图像不存在: {img_path}")
        
        print(f"VQ-GAN训练集: {len(self.image_paths)} 张图像")
        print(f"  (来自 {split_info['statistics']['total_users']} 用户)")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()  # 自动归一化到[0,1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img


# ============================================================
# 训练器
# ============================================================
class VQGANTrainer:
    """VQ-GAN训练器（严格Baseline，无EMA等优化）"""
    
    def __init__(self, config: VQGANConfig):
        self.config = config
        
        # 打印配置
        config.print_config_summary()
        
        # 初始化Accelerator
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if config.amp else 'no'
        )
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 创建数据集
        print("Creating dataset...")
        train_ds = ImageDataset(
            config.data_path,
            split_file='./data_split.json',
            image_size=config.image_size
        )
        
        self.train_dl = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 创建VQ-VAE模型
        print("Creating VQ-VAE model...")
        self.vqvae = VQVAE(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            z_channels=config.z_channels,
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
        )
        
        # 计算参数量
        num_params = sum(p.numel() for p in self.vqvae.parameters())
        print(f"VQ-VAE parameters: {num_params/1e6:.2f}M")
        
        # 创建判别器
        print("Creating discriminator...")
        self.discriminator = PatchGANDiscriminator(
            input_nc=config.out_channels,
            ndf=config.disc_ndf,
            n_layers=config.disc_n_layers,
        )
        
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Discriminator parameters: {disc_params/1e6:.2f}M")
        
        # 创建损失函数
        print("Creating loss function...")
        self.loss_fn = LPIPSWithDiscriminator(
            disc_start=config.disc_start,
            disc_weight=config.disc_weight,
            perceptual_weight=config.perceptual_weight,
            disc_loss_type=config.disc_loss_type,
            use_adaptive_weight=config.use_adaptive_weight,
        )
        
        # 创建优化器
        self.opt_vqvae = torch.optim.Adam(
            self.vqvae.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.disc_learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        # ⚠️ 关键：loss_fn包含LPIPS(VGG16)，不由prepare管理
        # LPIPS内部的VGG16会自动保持FP32（这是lpips库的限制）
        self.loss_fn.to(self.accelerator.device)
        
        # 使用Accelerator准备（不包含loss_fn）
        (
            self.vqvae,
            self.discriminator,
            self.opt_vqvae,
            self.opt_disc,
            self.train_dl
        ) = self.accelerator.prepare(
            self.vqvae,
            self.discriminator,
            self.opt_vqvae,
            self.opt_disc,
            self.train_dl
        )
        
        # 创建结果文件夹
        self.results_folder = Path(config.results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)
        
        self.step = 0
        
        print("\n" + "="*60)
        print("✓ 初始化完成，开始训练...")
        print("="*60 + "\n")
    
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
            total=config.train_steps,
            disable=not self.accelerator.is_main_process
        ) as pbar:
            while self.step < config.train_steps:
                # ========== 训练VQ-VAE ==========
                self.vqvae.train()
                self.discriminator.train()
                
                # 获取数据
                images = next(dl)
                images = images.to(self.accelerator.device)
                
                # ⚠️ 关键：启用autocast以支持混合精度训练
                with self.accelerator.autocast():
                    # 前向传播
                    reconstructions, vq_loss_dict = self.vqvae(images)
                    
                    # 计算生成器损失
                    g_loss, g_log_dict = self.loss_fn(
                        inputs=images,
                        reconstructions=reconstructions,
                        discriminator=self.discriminator,
                        optimizer_idx=0,  # 训练生成器
                        global_step=self.step,
                        last_layer=self.vqvae.get_last_layer()
                    )
                    
                    # 添加VQ损失
                    total_g_loss = g_loss + vq_loss_dict['vq_loss']
                
                # 反向传播（生成器）
                self.opt_vqvae.zero_grad()
                self.accelerator.backward(total_g_loss)
                # 梯度裁剪（保守值：防止训练不稳定）
                self.accelerator.clip_grad_norm_(self.vqvae.parameters(), max_norm=1.0)
                self.opt_vqvae.step()
                
                # ========== 训练判别器 ==========
                if self.step >= config.disc_start:
                    # 复用之前的重建图像（detach以避免梯度传播到生成器）
                    reconstructions_detached = reconstructions.detach()
                    
                    # ⚠️ 关键：判别器训练也需要autocast
                    with self.accelerator.autocast():
                        # 计算判别器损失
                        d_loss, d_log_dict = self.loss_fn(
                            inputs=images,
                            reconstructions=reconstructions_detached,
                            discriminator=self.discriminator,
                            optimizer_idx=1,  # 训练判别器
                            global_step=self.step,
                            last_layer=None
                        )
                    
                    
                    # 反向传播（判别器）
                    self.opt_disc.zero_grad()
                    self.accelerator.backward(d_loss)
                    # 梯度裁剪（保守值：防止训练不稳定）
                    self.accelerator.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.opt_disc.step()
                else:
                    d_log_dict = {'d_loss': 0.0}
                
                self.accelerator.wait_for_everyone()
                
                # 更新进度条
                if self.step >= config.disc_start:
                    pbar.set_description(
                        f'g_loss: {g_log_dict.get("total_loss", 0):.4f} | '
                        f'd_loss: {d_log_dict.get("d_loss", 0):.4f} | '
                        f'vq: {vq_loss_dict["vq_loss"].item():.4f} | '
                        f'usage: {vq_loss_dict.get("codebook_usage_ratio", 0):.2%}'
                    )
                else:
                    pbar.set_description(
                        f'g_loss: {g_log_dict.get("total_loss", 0):.4f} | '
                        f'vq: {vq_loss_dict["vq_loss"].item():.4f} | '
                        f'usage: {vq_loss_dict.get("codebook_usage_ratio", 0):.2%}'
                    )
                
                self.step += 1
                
                # 定期保存和采样
                if self.accelerator.is_main_process:
                    if self.step % config.save_and_sample_every == 0:
                        self.save_and_sample(self.step // config.save_and_sample_every)
                
                pbar.update(1)
        
        print('VQ-GAN训练完成!')
    
    def save_and_sample(self, milestone):
        """保存检查点并生成重建样本"""
        self.vqvae.eval()
        
        print(f"\n{'='*60}")
        print(f"Checkpoint {milestone} (步数: {self.step})")
        print(f"{'='*60}")
        
        # 生成重建样本
        try:
            with torch.no_grad():
                # 从训练集随机采样
                sample_images = []
                for _ in range(self.config.num_samples):
                    idx = np.random.randint(0, len(self.train_dl.dataset))
                    img = self.train_dl.dataset[idx]
                    sample_images.append(img)
                
                sample_images = torch.stack(sample_images).to(self.accelerator.device)
                
                # 重建
                reconstructions, vq_loss_dict = self.vqvae(sample_images)
                
                # 拼接原图和重建图
                comparison = torch.cat([sample_images, reconstructions], dim=0)
                
                # 保存
                save_path = self.results_folder / f'reconstruction-{milestone}.png'
                utils.save_image(
                    comparison,
                    str(save_path),
                    nrow=self.config.num_samples
                )
                print(f"✓ 重建样本已保存: {save_path}")
                
                # 打印统计
                print(f"  Codebook使用率: {vq_loss_dict.get('codebook_usage_ratio', 0):.2%}")
                
        except Exception as e:
            print(f"  ✗ 生成样本失败: {e}")
        
        # 保存检查点
        if self.accelerator.is_local_main_process:
            try:
                data = {
                    'step': self.step,
                    'vqvae': self.accelerator.get_state_dict(self.vqvae),
                    'discriminator': self.accelerator.get_state_dict(self.discriminator),
                    'opt_vqvae': self.opt_vqvae.state_dict(),
                    'opt_disc': self.opt_disc.state_dict(),
                    'config': self.config.__dict__,
                }
                save_path = self.results_folder / f'vqgan-{milestone}.pt'
                torch.save(data, str(save_path))
                print(f"✓ 检查点已保存: {save_path}")
                
                # 同时保存最新检查点
                latest_path = self.results_folder / 'vqgan_latest.pt'
                torch.save(data, str(latest_path))
                print(f"✓ 最新检查点: {latest_path}")
                
            except Exception as e:
                print(f"  ✗ 保存检查点失败: {e}")
        
        print(f"{'='*60}\n")
    
    def load(self, milestone):
        """加载检查点"""
        load_path = self.results_folder / f'vqgan-{milestone}.pt'
        data = torch.load(str(load_path), map_location=self.accelerator.device)
        
        vqvae = self.accelerator.unwrap_model(self.vqvae)
        vqvae.load_state_dict(data['vqvae'])
        
        disc = self.accelerator.unwrap_model(self.discriminator)
        disc.load_state_dict(data['discriminator'])
        
        self.step = data['step']
        self.opt_vqvae.load_state_dict(data['opt_vqvae'])
        self.opt_disc.load_state_dict(data['opt_disc'])
        
        print(f"Loaded checkpoint from {load_path}, step {self.step}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train VQ-GAN (Baseline)')
    parser.add_argument('--resume', type=int, default=None, help='Resume from milestone')
    args = parser.parse_args()
    
    # 创建配置
    config = VQGANConfig()
    
    # 创建训练器
    trainer = VQGANTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume is not None:
        trainer.load(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

