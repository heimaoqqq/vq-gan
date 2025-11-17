"""
Baseline配置 - 关闭所有优化技术
train_latent_cfg.py已设置为Baseline参数（batch=8, 无梯度累积, 3000步）
此文件仅关闭优化技术开关
适配新的预编码代码 preprocess_latents_with_gmm.py
"""
from train_latent_cfg import Config

class BaselineConfig(Config):
    """Baseline：关闭所有优化技术（仅保留基础LDM）
    
    参数来源：
    - 训练参数（batch=8, 无梯度累积, 3000步）：继承自train_latent_cfg.py
    - 数据配置（GMM预编码）：继承自train_latent_cfg.py
    - 优化技术开关：本文件关闭所有优化
    """
    
    # === 路径配置 ===
    vae_path = '/kaggle/input/kl-vae-best-pt/kl_vae_best.pt'  # KL-VAE路径
    data_path = '/kaggle/input/organized-gait-dataset/Normal_line'
    
    # === 数据配置（适配新的GMM预编码） ===
    num_users = 31
    images_per_user_train = 30  # 扩散模型只用gen_train的30张
    latents_cache_folder = './latents_cache_gmm'  # 新的预编码缓存路径
    
    # === 关闭所有优化技术 ===
    cond_drop_prob = 0.0  # ❌ 关闭CFG（训练时不dropout条件）
    cond_scale = 1.0  # ❌ 关闭CFG（采样时不用CFG）
    rescaled_phi = 0.0  # ❌ 关闭CFG++
    
    use_contrastive_loss = False  # ❌ 关闭对比学习
    contrastive_weight = 0.0
    
    min_snr_loss_weight = False  # ❌ 关闭Min-SNR
    
    use_ema = False  # ❌ 关闭EMA
    use_lr_warmup = False  # ❌ 关闭学习率warmup
    max_grad_norm = None  # ❌ 关闭梯度裁剪
    
    weight_decay = 0.0  # ❌ 关闭weight decay
    
    # === 修改保存路径避免冲突 ===
    results_folder = './results_baseline_gmm'
    
    # === 其他参数完全继承train_latent_cfg.py ===
    # - train_batch_size = 8
    # - gradient_accumulate_every = 1
    # - num_train_steps = 3000
    # - save_and_sample_every = 300
    # - 学习率、模型配置等


if __name__ == '__main__':
    from train_latent_cfg import LatentDiffusionTrainer
    
    config = BaselineConfig()
    
    print("\n" + "="*60)
    print("LDM Baseline（KL-VAE，关闭所有优化）")
    print("="*60)
    print("\n对比实验设置:")
    print("  优化版(train_latent_cfg.py) vs Baseline(此脚本)")
    print("\n差异（仅优化技术）:")
    print(f"  CFG drop:    {config.cond_drop_prob:<6} vs 0.2")
    print(f"  CFG scale:   {config.cond_scale:<6} vs 7.5")
    print(f"  对比学习:     {str(config.use_contrastive_loss):<6} vs True")
    print(f"  Min-SNR:     {str(config.min_snr_loss_weight):<6} vs True")
    print(f"  EMA:         {str(config.use_ema):<6} vs True")
    print(f"  Weight decay: {config.weight_decay:<6} vs 1e-4")
    print("\n相同（控制变量）:")
    print(f"  Autoencoder: KL-VAE (embed_dim=4)")
    print(f"  UNet: dim={config.dim}, mults={config.dim_mults}")
    print(f"  扩散: {config.objective}, {config.beta_schedule}")
    print(f"  训练: batch={config.train_batch_size}×{config.gradient_accumulate_every}, lr={config.train_lr}")
    print("="*60 + "\n")
    
    # 创建训练器并开始训练
    trainer = LatentDiffusionTrainer(config)
    trainer.train()

