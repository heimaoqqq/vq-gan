"""
PatchGAN Discriminator
======================
用于VQ-GAN的对抗训练

参考：
- pix2pix (Isola et al., 2017)
- Taming Transformers (Esser et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN判别器
    
    对图像的每个patch进行真假判别，而不是整张图像
    这样可以捕捉局部纹理和细节，同时减少参数量
    
    Args:
        input_nc: 输入通道数（RGB图像为3）
        ndf: 第一层卷积的输出通道数
        n_layers: 判别器层数
        use_actnorm: 是否使用ActNorm（默认False，使用BatchNorm）
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        super().__init__()
        
        # 规范化层选择
        if use_actnorm:
            # ActNorm: 自适应归一化，对batch size不敏感
            norm_layer = lambda c: ActNorm(c)
        else:
            # BatchNorm: 标准选择
            norm_layer = lambda c: nn.BatchNorm2d(c)
        
        # 构建判别器
        sequence = [
            # 初始层：不使用归一化
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # 中间层：逐步增加通道数
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # 通道数最多增加到8倍
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # 最后一个卷积层：stride=1，感受野变大但不进一步下采样
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 输出层：输出单通道的真假判别
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.main = nn.Sequential(*sequence)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            out: [B, 1, h, w] 每个patch的真假判别结果
                 (h, w取决于输入大小和网络层数)
        """
        return self.main(x)


class ActNorm(nn.Module):
    """
    Activation Normalization
    
    对batch size不敏感的归一化方法
    第一个batch用于初始化scale和bias
    """
    def __init__(self, num_features, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
    
    def initialize(self, x):
        """用第一个batch初始化"""
        with torch.no_grad():
            # 计算每个通道的均值和标准差
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            
            # 初始化参数
            self.bias.data.copy_(-mean)
            self.weight.data.copy_(1.0 / (std + 1e-6))
            
            self.initialized.fill_(1)
    
    def forward(self, x):
        if self.initialized.item() == 0 and self.training:
            self.initialize(x)
        
        if self.affine:
            return x * self.weight + self.bias
        else:
            return x


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器（可选）
    
    在多个分辨率上进行判别，提高训练稳定性
    论文可能未使用，这里提供以备扩展
    """
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        num_D: int = 2,  # 判别器数量
    ):
        super().__init__()
        
        self.num_D = num_D
        
        # 创建多个判别器
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            self.discriminators.append(
                PatchGANDiscriminator(input_nc, ndf, n_layers)
            )
        
        # 下采样层
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        在多个尺度上判别
        
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            results: List of [B, 1, h, w]，每个尺度的判别结果
        """
        results = []
        for i in range(self.num_D):
            results.append(self.discriminators[i](x))
            if i != self.num_D - 1:
                x = self.downsample(x)
        return results


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Hinge损失（判别器）
    
    Args:
        logits_real: 真实图像的判别结果
        logits_fake: 生成图像的判别结果
        
    Returns:
        loss: 判别器损失
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """
    标准GAN损失（判别器）- BCE损失
    
    Args:
        logits_real: 真实图像的判别结果
        logits_fake: 生成图像的判别结果
        
    Returns:
        loss: 判别器损失
    """
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """
    Hinge损失（生成器）
    
    Args:
        logits_fake: 生成图像的判别结果
        
    Returns:
        loss: 生成器对抗损失
    """
    return -torch.mean(logits_fake)


def vanilla_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """
    标准GAN损失（生成器）
    
    Args:
        logits_fake: 生成图像的判别结果
        
    Returns:
        loss: 生成器对抗损失
    """
    return torch.mean(F.softplus(-logits_fake))

