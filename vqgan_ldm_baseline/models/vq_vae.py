"""
VQ-VAE Model
============
完整的VQ-VAE模型，集成Encoder、Quantizer和Decoder

论文方法：VQ-GAN = VQ-VAE + 对抗训练 + 感知损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoder_decoder import Encoder, Decoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """
    VQ-VAE模型
    
    架构：
        输入图像 -> Encoder -> VQ -> Decoder -> 重建图像
    
    Args:
        in_channels: 输入图像通道数 (3 for RGB)
        out_channels: 输出图像通道数 (3 for RGB)
        ch: 基础通道数
        ch_mult: 通道倍增序列
        num_res_blocks: 每层ResBlock数量
        attn_resolutions: 使用注意力的分辨率
        dropout: Dropout率
        z_channels: 潜在空间通道数
        num_embeddings: Codebook大小
        embedding_dim: Codebook向量维度
        commitment_cost: Commitment loss权重
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int] = (16,),
        dropout: float = 0.0,
        z_channels: int = 256,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        # 保存配置
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            z_channels=z_channels,
        )
        
        # Pre-quantization convolution (if z_channels != embedding_dim)
        if z_channels != embedding_dim:
            self.pre_quant_conv = nn.Conv2d(z_channels, embedding_dim, kernel_size=1)
            self.post_quant_conv = nn.Conv2d(embedding_dim, z_channels, kernel_size=1)
        else:
            self.pre_quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        
        # Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        
        # Decoder
        self.decoder = Decoder(
            out_ch=out_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            z_channels=z_channels,
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        编码并量化
        
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            z_q: [B, z_channels, h, w] 量化后的潜在表示
            indices: [B, h, w] 量化索引
            loss_dict: VQ损失字典
        """
        # 编码
        z = self.encoder(x)  # [B, z_channels, h, w]
        
        # 预量化卷积
        z = self.pre_quant_conv(z)  # [B, embedding_dim, h, w]
        
        # 量化
        z_q, loss_dict, indices = self.quantizer(z)
        
        # 后量化卷积
        z_q = self.post_quant_conv(z_q)  # [B, z_channels, h, w]
        
        return z_q, indices, loss_dict
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        从量化潜在表示解码
        
        Args:
            z_q: [B, z_channels, h, w] 量化后的潜在表示
            
        Returns:
            x_recon: [B, 3, H, W] 重建图像
        """
        x_recon = self.decoder(z_q)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        完整前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像（值域[0, 1]）
            
        Returns:
            x_recon: [B, 3, H, W] 重建图像（值域[0, 1]）
            loss_dict: 损失字典（仅包含VQ损失）
        """
        # 编码 + 量化
        z_q, indices, loss_dict = self.encode(x)
        
        # 解码（Decoder最后有sigmoid，确保输出在[0,1]）
        x_recon = self.decode(z_q)
        
        # 添加codebook使用统计（用于监控）
        with torch.no_grad():
            usage, usage_ratio = self.quantizer.get_codebook_usage(indices)
            loss_dict['codebook_usage_ratio'] = usage_ratio
        
        return x_recon, loss_dict
    
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码为离散索引（用于压缩存储或分析）
        
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            indices: [B, h, w] 量化索引
        """
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        _, _, indices = self.quantizer(z)
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从离散索引解码（用于从压缩表示重建）
        
        Args:
            indices: [B, h, w] 量化索引
            
        Returns:
            x_recon: [B, 3, H, W] 重建图像
        """
        z_q = self.quantizer.get_codebook_entry(indices)
        z_q = self.post_quant_conv(z_q)
        x_recon = self.decode(z_q)
        return x_recon
    
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像到潜在空间（用于LDM训练）
        
        Args:
            x: [B, 3, H, W] 输入图像（值域[0, 1]）
            
        Returns:
            z_q: [B, z_channels, h, w] 量化后的潜在表示
        """
        with torch.no_grad():
            z_q, _, _ = self.encode(x)
        return z_q
    
    def decode_latents(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        从潜在表示解码（用于LDM采样）
        
        Args:
            z_q: [B, z_channels, h, w] 潜在表示
            
        Returns:
            x: [B, 3, H, W] 图像（值域[0, 1]）
        """
        with torch.no_grad():
            x = self.decode(z_q)
            # Decoder已有sigmoid，但为安全起见再clamp一次
            x = torch.clamp(x, 0, 1)
        return x
    
    def get_last_layer(self):
        """
        获取最后一层（用于自适应权重调整）
        """
        return self.decoder.conv_out.weight

