"""
Losses for VQ-GAN Training
===========================
组合感知损失（LPIPS）、对抗损失和重建损失

参考：
- LPIPS: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Zhang et al., 2018)
- VQ-GAN: Taming Transformers (Esser et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import lpips  # 需要安装: pip install lpips


class LPIPSWithDiscriminator(nn.Module):
    """
    组合LPIPS感知损失和对抗损失
    
    这是VQ-GAN训练的核心损失函数，包含：
    1. 像素重建损失（L1）
    2. LPIPS感知损失
    3. 对抗损失（GAN）
    
    Args:
        disc_start: 从第几步开始训练判别器
        disc_weight: 判别器损失的最大权重
        perceptual_weight: 感知损失权重
        disc_loss_type: 判别器损失类型 ('hinge' or 'vanilla')
        use_adaptive_weight: 是否使用自适应权重平衡
    """
    
    def __init__(
        self,
        disc_start: int = 10000,
        disc_weight: float = 0.1,
        perceptual_weight: float = 1.0,
        disc_loss_type: str = 'hinge',
        use_adaptive_weight: bool = False,  # ❌ Baseline不使用自适应权重
    ):
        super().__init__()
        
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        self.disc_loss_type = disc_loss_type
        self.use_adaptive_weight = use_adaptive_weight
        
        # LPIPS感知损失（使用VGG网络）
        # net_type='vgg' 使用ImageNet预训练的VGG特征
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
        
        # 冻结LPIPS网络参数
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
        
        print(f"Initialized LPIPS loss with VGG backbone")
        print(f"  Discriminator starts at step {disc_start}")
        print(f"  Disc weight: {disc_weight}, Perceptual weight: {perceptual_weight}")
    
    def calculate_adaptive_weight(
        self, 
        nll_loss: torch.Tensor, 
        g_loss: torch.Tensor, 
        last_layer: torch.Tensor
    ) -> torch.Tensor:
        """
        自适应计算判别器损失权重
        
        平衡重建损失和对抗损失的梯度尺度
        确保判别器不会过度主导训练
        
        Args:
            nll_loss: 重建损失（负对数似然）
            g_loss: 生成器对抗损失
            last_layer: 解码器最后一层权重（用于计算梯度）
            
        Returns:
            adaptive_weight: 自适应权重
        """
        # 计算重建损失对最后一层的梯度
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        
        # 计算对抗损失对最后一层的梯度
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        
        # 自适应权重 = ||∇_nll|| / (||∇_g|| + ε)
        # 确保两个损失的梯度尺度相近
        adaptive_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
        
        return adaptive_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        discriminator: nn.Module,
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失
        
        Args:
            inputs: [B, 3, H, W] 真实图像
            reconstructions: [B, 3, H, W] 重建图像
            discriminator: 判别器模型
            optimizer_idx: 0=生成器, 1=判别器
            global_step: 当前训练步数
            last_layer: 解码器最后一层（用于自适应权重）
            
        Returns:
            loss: 总损失
            log_dict: 损失字典（用于记录）
        """
        # ========== 重建损失 ==========
        # L1损失
        rec_loss = torch.abs(inputs - reconstructions)
        rec_loss = torch.mean(rec_loss)
        
        # ========== 感知损失 ==========
        # LPIPS期望输入在[-1, 1]范围
        # 假设inputs和reconstructions在[0, 1]范围，需要归一化
        inputs_normalized = inputs * 2.0 - 1.0
        reconstructions_normalized = reconstructions * 2.0 - 1.0
        
        # LPIPS输出是每个样本的loss，需要取平均
        p_loss = self.perceptual_loss(
            inputs_normalized.contiguous(), 
            reconstructions_normalized.contiguous()
        )
        p_loss = torch.mean(p_loss)
        
        # ========== 组合重建损失 ==========
        nll_loss = rec_loss + self.perceptual_weight * p_loss
        
        # ========== 对抗损失 ==========
        if optimizer_idx == 0:
            # ===== 训练生成器（VQ-VAE） =====
            if global_step >= self.disc_start:
                # 判别器对重建图像的判断
                logits_fake = discriminator(reconstructions)
                
                # 生成器损失：希望判别器认为重建图像是真的
                if self.disc_loss_type == 'hinge':
                    g_loss = -torch.mean(logits_fake)
                elif self.disc_loss_type == 'vanilla':
                    g_loss = torch.mean(F.softplus(-logits_fake))
                else:
                    raise ValueError(f"Unknown disc_loss_type: {self.disc_loss_type}")
                
                # 自适应权重（如果启用）
                if self.use_adaptive_weight and last_layer is not None:
                    adaptive_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer
                    )
                    disc_weight = adaptive_weight * self.disc_weight
                else:
                    disc_weight = self.disc_weight
                
                # 总损失
                loss = nll_loss + disc_weight * g_loss
                
                log_dict = {
                    'total_loss': loss.item(),
                    'rec_loss': rec_loss.item(),
                    'perceptual_loss': p_loss.item(),
                    'nll_loss': nll_loss.item(),
                    'g_loss': g_loss.item(),
                    'disc_weight': disc_weight if isinstance(disc_weight, float) else disc_weight.item(),
                }
            else:
                # 判别器未启动前，只用重建损失
                loss = nll_loss
                log_dict = {
                    'total_loss': loss.item(),
                    'rec_loss': rec_loss.item(),
                    'perceptual_loss': p_loss.item(),
                    'nll_loss': nll_loss.item(),
                }
        
        elif optimizer_idx == 1:
            # ===== 训练判别器 =====
            if global_step >= self.disc_start:
                # 真实图像的判别
                logits_real = discriminator(inputs.detach())
                
                # 重建图像的判别
                logits_fake = discriminator(reconstructions.detach())
                
                # 判别器损失
                if self.disc_loss_type == 'hinge':
                    d_loss = 0.5 * (
                        torch.mean(F.relu(1.0 - logits_real)) +
                        torch.mean(F.relu(1.0 + logits_fake))
                    )
                elif self.disc_loss_type == 'vanilla':
                    d_loss = 0.5 * (
                        torch.mean(F.softplus(-logits_real)) +
                        torch.mean(F.softplus(logits_fake))
                    )
                else:
                    raise ValueError(f"Unknown disc_loss_type: {self.disc_loss_type}")
                
                loss = d_loss
                
                # 计算判别器准确率（用于监控）
                with torch.no_grad():
                    real_acc = (logits_real > 0).float().mean().item()
                    fake_acc = (logits_fake < 0).float().mean().item()
                    d_acc = (real_acc + fake_acc) / 2
                
                log_dict = {
                    'd_loss': d_loss.item(),
                    'logits_real': logits_real.mean().item(),
                    'logits_fake': logits_fake.mean().item(),
                    'd_acc': d_acc,
                }
            else:
                # 判别器未启动，返回零损失
                loss = torch.tensor(0.0, requires_grad=True, device=inputs.device)
                log_dict = {'d_loss': 0.0}
        
        else:
            raise ValueError(f"Invalid optimizer_idx: {optimizer_idx}")
        
        return loss, log_dict


class SimpleLoss(nn.Module):
    """
    简化的损失函数（不使用判别器）
    
    仅用于调试或快速验证
    包含：L1重建损失 + LPIPS感知损失
    """
    
    def __init__(self, perceptual_weight: float = 1.0):
        super().__init__()
        
        self.perceptual_weight = perceptual_weight
        
        # LPIPS感知损失
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        reconstructions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算简单损失
        
        Args:
            inputs: [B, 3, H, W] 真实图像
            reconstructions: [B, 3, H, W] 重建图像
            
        Returns:
            loss: 总损失
            log_dict: 损失字典
        """
        # L1重建损失
        rec_loss = torch.abs(inputs - reconstructions).mean()
        
        # LPIPS感知损失
        inputs_normalized = inputs * 2.0 - 1.0
        reconstructions_normalized = reconstructions * 2.0 - 1.0
        p_loss = self.perceptual_loss(
            inputs_normalized.contiguous(),
            reconstructions_normalized.contiguous()
        ).mean()
        
        # 总损失
        loss = rec_loss + self.perceptual_weight * p_loss
        
        log_dict = {
            'total_loss': loss.item(),
            'rec_loss': rec_loss.item(),
            'perceptual_loss': p_loss.item(),
        }
        
        return loss, log_dict

