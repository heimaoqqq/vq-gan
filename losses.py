"""
Supervised Contrastive Learning Loss
=====================================
完整实现SupConLoss，用于条件扩散模型的对比学习

Reference:
    Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    Original implementation: https://github.com/HobbitLong/SupContrast

Author: Yonglong Tian (yonglong@mit.edu)
Modified for diffusion model integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    
    论文: https://arxiv.org/pdf/2004.11362.pdf
    
    核心思想:
        - 拉近同一类别的样本特征
        - 推远不同类别的样本特征
        - 适用于有监督学习场景
    
    Args:
        temperature (float): 温度参数，控制分布的平滑度
            - 较小的值(0.05-0.1)使分布更尖锐，对比更强
            - 较大的值(0.3-0.5)使分布更平滑，对比更弱
            - 默认: 0.07 (论文推荐值)
        
        contrast_mode (str): 对比模式
            - 'all': 使用所有样本作为anchor（推荐）
            - 'one': 只使用第一个view作为anchor
            - 默认: 'all'
        
        base_temperature (float): 基础温度，用于缩放损失
            - 通常与temperature相同
            - 默认: 0.07
    
    Input:
        features: 特征张量，形状为 [batch_size, n_views, feature_dim]
            - batch_size: 批次大小
            - n_views: 每个样本的视角数量（对于扩散模型通常为1）
            - feature_dim: 特征维度
            - 注意: 特征必须经过L2归一化
        
        labels: 标签张量，形状为 [batch_size]
            - 同一类别的样本应该有相同的label
            - 如果为None，退化为SimCLR的无监督对比学习
    
    Output:
        loss: 标量损失值
    
    Example:
        >>> criterion = SupConLoss(temperature=0.07)
        >>> features = torch.randn(32, 1, 128)  # 32个样本，1个view，128维特征
        >>> features = F.normalize(features, dim=2)  # L2归一化
        >>> labels = torch.randint(0, 10, (32,))  # 10类分类
        >>> loss = criterion(features, labels)
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        计算监督对比学习损失
        
        Args:
            features: [batch_size, n_views, feature_dim] 或 [batch_size, feature_dim]
            labels: [batch_size] 类别标签
            mask: [batch_size, batch_size] 可选的对比mask
        
        Returns:
            loss: 标量损失值
        """
        device = torch.device('cuda' if features.is_cuda else 'cpu')
        
        # 检查输入维度
        if len(features.shape) < 3:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...], '
                'at least 3 dimensions are required'
            )
        
        # 如果特征维度>3，展平为[bsz, n_views, -1]
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        
        # 构建对比mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 无监督模式：只有同一样本的不同view是正样本
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 有监督模式：同一类别的样本是正样本
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask[i,j] = 1 if labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # 提取对比特征
        contrast_count = features.shape[1]  # n_views
        # 将所有view连接起来: [bsz*n_views, feature_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # 选择anchor
        if self.contrast_mode == 'one':
            # 只使用第一个view作为anchor
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # 使用所有view作为anchor（推荐）
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # 计算相似度矩阵
        # anchor_dot_contrast: [anchor_count*bsz, contrast_count*bsz]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # 数值稳定性：减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 扩展mask以匹配anchor和contrast的数量
        # mask: [bsz, bsz] -> [anchor_count*bsz, contrast_count*bsz]
        mask = mask.repeat(anchor_count, contrast_count)
        
        # 创建logits_mask，排除自身对比
        # logits_mask[i,i] = 0, 其他为1
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        
        # 应用mask：排除自身
        mask = mask * logits_mask
        
        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算正样本对的平均log-likelihood
        # 处理边界情况：如果某个anchor没有正样本对
        # 例如: features=[4,1,...], labels=[0,1,1,2]
        # 则label=0和label=2的样本没有其他正样本
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # 计算最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (可选的替代方案)
    
    这是对比学习的另一个经典损失函数，与SupConLoss类似但更简单
    适用于无监督场景或作为辅助损失
    
    Args:
        temperature (float): 温度参数，默认0.07
    """
    
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features):
        """
        计算InfoNCE损失
        
        Args:
            features: [batch_size, n_views, feature_dim]
                假设n_views=2，features[:,0,:]和features[:,1,:]是正样本对
        
        Returns:
            loss: 标量损失值
        """
        if features.shape[1] != 2:
            raise ValueError('InfoNCE requires exactly 2 views, got {}'.format(features.shape[1]))
        
        batch_size = features.shape[0]
        device = features.device
        
        # 提取两个view
        z_i = features[:, 0, :]  # [bsz, dim]
        z_j = features[:, 1, :]  # [bsz, dim]
        
        # 拼接
        z = torch.cat([z_i, z_j], dim=0)  # [2*bsz, dim]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2*bsz, 2*bsz]
        
        # 创建正样本mask
        # 对于z_i[k]，正样本是z_j[k]（在位置bsz+k）
        # 对于z_j[k]，正样本是z_i[k]（在位置k）
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2*bsz]
        
        # 排除自身
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


if __name__ == '__main__':
    """测试代码"""
    print("Testing SupConLoss...")
    
    # 测试1: 基本功能
    criterion = SupConLoss(temperature=0.07)
    features = torch.randn(8, 1, 128)  # 8个样本，1个view，128维
    features = F.normalize(features, dim=2)  # L2归一化
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 4类，每类2个样本
    
    loss = criterion(features, labels)
    print(f"Test 1 - Basic: loss = {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive!"
    
    # 测试2: 多个view
    features_multi = torch.randn(8, 2, 128)  # 8个样本，2个view
    features_multi = F.normalize(features_multi, dim=2)
    loss_multi = criterion(features_multi, labels)
    print(f"Test 2 - Multi-view: loss = {loss_multi.item():.4f}")
    
    # 测试3: 边界情况（某些类别只有一个样本）
    labels_edge = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # 每类1个样本
    loss_edge = criterion(features, labels_edge)
    print(f"Test 3 - Edge case: loss = {loss_edge.item():.4f}")
    assert not torch.isnan(loss_edge), "Loss is NaN in edge case!"
    
    # 测试4: 无监督模式（SimCLR）
    loss_unsup = criterion(features)
    print(f"Test 4 - Unsupervised: loss = {loss_unsup.item():.4f}")
    
    print("\n✓ All tests passed!")
    print("\nSupConLoss is ready to use in your diffusion model training!")
