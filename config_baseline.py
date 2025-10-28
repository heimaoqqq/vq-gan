"""
Baseline配置 - 关闭所有优化
复制train_latent_cfg.py的Config类，仅修改优化开关
"""
from train_latent_cfg import Config

class BaselineConfig(Config):
    """Baseline：关闭所有优化技术（仅保留基础LDM）"""
    
    # === 关闭所有优化技术 ===
    cond_drop_prob = 0.0  # ❌ 关闭CFG
    
    use_contrastive_loss = False  # ❌ 关闭对比学习
    contrastive_weight = 0.0
    
    min_snr_loss_weight = False  # ❌ 关闭Min-SNR
    
    use_ema = False  # ❌ 关闭EMA
    
    weight_decay = 0.0  # ❌ 关闭weight decay
    
    # 修改保存路径避免冲突
    results_folder = './results_baseline'
    latents_cache_folder = './latents_cache_baseline'
    
    # 其他参数(VAE, UNet, 扩散配置等)完全继承train_latent_cfg.py


if __name__ == '__main__':
    from train_latent_cfg import train
    
    config = BaselineConfig()
    
    print("\n" + "="*60)
    print("LDM Baseline（KL-VAE，关闭所有优化）")
    print("="*60)
    print("\n对比实验设置:")
    print("  优化版(train_latent_cfg.py) vs Baseline(此脚本)")
    print("\n差异（仅优化技术）:")
    print(f"  CFG:         {config.cond_drop_prob:<6} vs 0.2")
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
    
    train(config)

