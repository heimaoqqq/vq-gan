"""
VQ-GAN Configuration
====================
阶段一：VQ-GAN训练配置

严格遵循论文，使用保守参数
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VQGANConfig:
    """
    VQ-GAN训练配置
    
    所有参数都是保守选择，确保训练稳定
    """
    
    # ============================================================
    # 路径配置（⚠️ Windows路径必须使用原始字符串r''或正斜杠/）
    # ============================================================
    data_path: str = '/kaggle/input/organized-gait-dataset/Normal_line'
    results_folder: str = './results/vqgan'
    
    # ============================================================
    # 数据配置
    # ============================================================
    num_users: int = 31
    images_per_user_train: int = 50  # 每用户训练图像数
    total_train_images: int = 31 * 50  # 1550张
    image_size: int = 256
    
    # ============================================================
    # VQ-VAE架构配置（与KL-VAE对齐）
    # ============================================================
    in_channels: int = 3
    out_channels: int = 3
    ch: int = 128  # 基础通道数
    ch_mult: Tuple[int] = (1, 2, 2, 4)  # 4层，8倍下采样
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int] = (16,)
    dropout: float = 0.0  # 论文未提，不使用
    
    # ============================================================
    # VQ配置（VQ-GAN标准配置）
    # ============================================================
    z_channels: int = 256  # 编码器输出通道（VQ-GAN标准，信息容量充足）
    num_embeddings: int = 128  # Codebook大小（基于usage=50%优化）
    embedding_dim: int = 256  # Codebook向量维度（与z_channels一致）
    commitment_cost: float = 0.25  # VQ标准参数
    
    # ============================================================
    # 判别器配置（保守）
    # ============================================================
    disc_type: str = 'PatchGAN'
    disc_ndf: int = 64  # 判别器基础通道数
    disc_n_layers: int = 3  # 判别器层数
    disc_start: int = 10000  # 保守：晚启动判别器
    disc_weight: float = 0.1  # 保守：小权重
    disc_loss_type: str = 'hinge'  # 'hinge' or 'vanilla'
    use_adaptive_weight: bool = False  # ❌ 关闭自适应权重（节省显存，更严格Baseline）
    
    # ============================================================
    # 损失权重（保守）
    # ============================================================
    perceptual_weight: float = 1.0  # LPIPS权重
    
    # ============================================================
    # 训练配置（保守）
    # ============================================================
    batch_size: int = 8  # 保守：小batch
    learning_rate: float = 4.5e-5  # VQ-GAN标准lr
    disc_learning_rate: float = 4.5e-5  # 判别器lr（可以稍高）
    adam_betas: Tuple[float] = (0.5, 0.9)  # GAN训练标准
    weight_decay: float = 0.0  # 保守：不使用
    
    train_steps: int = 30000  # 约150 epochs（防止小数据集过拟合）
    gradient_accumulate_every: int = 1  # 不使用梯度累积（论文未提）
    max_grad_norm: float = 1.0  # 梯度裁剪（保守值，防止训练不稳定）
    
    # ============================================================
    # 不使用的优化（论文未提）
    # ============================================================
    use_ema: bool = False  # ❌ 论文未提
    ema_decay: float = None  # ❌ 不使用
    ema_update_every: int = None  # ❌ 不使用
    
    # ============================================================
    # 监控与保存
    # ============================================================
    save_and_sample_every: int = 1000  # 每1000步保存（约5 epochs），总计30个checkpoint
    num_samples: int = 8  # 保存8张重建样本
    
    # ============================================================
    # 其他
    # ============================================================
    amp: bool = False  # 混合精度（保守：不使用，VQ-GAN训练LPIPS对FP16兼容性差）
    num_workers: int = 0  # Windows兼容
    seed: int = 42
    
    def __post_init__(self):
        """验证配置"""
        assert self.image_size == 256, "仅支持256x256图像"
        assert self.num_embeddings <= self.total_train_images, \
            f"Codebook太大({self.num_embeddings}) > 训练图像数({self.total_train_images})"
        assert self.disc_start >= 0, "disc_start必须非负"
        assert 0 < self.disc_weight <= 1.0, "disc_weight必须在(0, 1]范围"
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("VQ-GAN训练配置（论文Baseline）")
        print("="*60)
        
        print(f"\n数据集:")
        print(f"  用户数: {self.num_users}")
        print(f"  每用户图像: {self.images_per_user_train}")
        print(f"  总训练图像: {self.total_train_images}")
        
        print(f"\nVQ-VAE:")
        print(f"  Codebook: {self.num_embeddings} codes × {self.embedding_dim} dim")
        print(f"  下采样: 8× (256 → 32)")
        print(f"  潜在通道: {self.z_channels}")
        
        print(f"\n判别器:")
        print(f"  类型: {self.disc_type}")
        print(f"  启动步数: {self.disc_start}")
        print(f"  权重: {self.disc_weight}")
        print(f"  损失: {self.disc_loss_type}")
        
        print(f"\n训练:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  训练步数: {self.train_steps:,}")
        print(f"  约等于: {self.train_steps / (self.total_train_images / self.batch_size):.1f} epochs")
        print(f"  梯度裁剪: {self.max_grad_norm} (保守)")
        print(f"  混合精度: {'FP16' if self.amp else 'FP32'}")
        
        print(f"\n不使用的优化（论文未提）:")
        print(f"  ❌ EMA")
        print(f"  ❌ 梯度累积")
        print(f"  ❌ Weight decay")
        print(f"  ❌ 自适应权重")
        
        print("="*60 + "\n")

