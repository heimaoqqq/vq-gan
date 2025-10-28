"""
Vector Quantizer for VQ-VAE
============================
实现标准的VQ（Vector Quantization）机制

参考：
- Neural Discrete Representation Learning (van den Oord et al., 2017)
- Taming Transformers (Esser et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    向量量化器
    
    将连续的潜在表示映射到离散的codebook空间
    
    Args:
        num_embeddings: Codebook大小（代码本中向量的数量）
        embedding_dim: 每个代码向量的维度
        commitment_cost: Commitment loss的权重系数（beta）
    
    Forward:
        输入: [B, C, H, W] 连续潜在表示
        输出: [B, C, H, W] 量化后的潜在表示, loss_dict
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: [num_embeddings, embedding_dim]
        # 初始化为均匀分布
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, z: torch.Tensor):
        """
        量化过程
        
        Args:
            z: [B, C, H, W] 编码器输出的连续潜在表示
            
        Returns:
            z_q: [B, C, H, W] 量化后的潜在表示
            loss_dict: 包含codebook_loss和commitment_loss
            indices: [B, H, W] 量化索引（用于可视化/分析）
        """
        # 转换形状: [B, C, H, W] -> [B, H, W, C]
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embedding_dim)  # [B*H*W, C]
        
        # 计算距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        # [B*H*W, num_embeddings]
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # 找到最近的codebook向量
        # [B*H*W]
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 获取量化后的向量
        # [B*H*W, C]
        z_q = self.embedding(encoding_indices)
        
        # 恢复形状: [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        z_q = z_q.view(z.shape)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z = rearrange(z, 'b h w c -> b c h w').contiguous()
        
        # 计算损失
        # Codebook loss: 让codebook向编码器输出靠近
        codebook_loss = F.mse_loss(z_q.detach(), z)
        
        # Commitment loss: 让编码器输出向codebook靠近
        commitment_loss = F.mse_loss(z_q, z.detach())
        
        # 总VQ损失
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: 前向使用量化值，反向传播梯度给z
        z_q = z + (z_q - z).detach()
        
        # 保存indices用于分析
        indices = encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])
        
        # ⚠️ 关键：loss_dict必须使用.item()转为python值，避免保留计算图导致显存泄漏
        loss_dict = {
            'vq_loss': vq_loss,  # 保留tensor用于反向传播
            'codebook_loss': codebook_loss.item(),  # 仅用于记录，转为python值
            'commitment_loss': commitment_loss.item(),  # 仅用于记录，转为python值
        }
        
        return z_q, loss_dict, indices
    
    def get_codebook_entry(self, indices: torch.Tensor):
        """
        根据索引获取codebook向量（用于从离散索引重建）
        
        Args:
            indices: [B, H, W] 量化索引
            
        Returns:
            z_q: [B, C, H, W] 量化后的潜在表示
        """
        # [B, H, W] -> [B*H*W]
        indices_flattened = indices.view(-1)
        
        # 从codebook获取向量
        z_q = self.embedding(indices_flattened)  # [B*H*W, C]
        
        # 恢复形状
        z_q = z_q.view(indices.shape[0], indices.shape[1], indices.shape[2], -1)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        return z_q
    
    def get_codebook_usage(self, indices: torch.Tensor):
        """
        统计codebook使用情况（用于监控codebook collapse）
        
        Args:
            indices: [B, H, W] 量化索引
            
        Returns:
            usage: [num_embeddings] 每个code的使用次数
            usage_ratio: float 实际使用的code占比
        """
        indices_flattened = indices.view(-1)
        usage = torch.bincount(indices_flattened, minlength=self.num_embeddings)
        usage_ratio = (usage > 0).float().mean().item()
        
        return usage, usage_ratio

