"""
预编码脚本 - 将所有图像编码为潜在表示并缓存
同时保存训练集/测试集划分信息，方便后续分类器实验

输出：
  1. latents_cache/ - 所有图像的潜在表示（训练+测试）
  2. data_split.json - 数据集划分信息

更新日志:
  - 2025-01-21: 集成K-Medoids聚类方法选择最具代表性的训练样本
  - 原因: 随机抽样可能导致训练集缺少某些步态模式，影响生成质量
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    print("Warning: sklearn_extra not found. Install with: pip install scikit-learn-extra")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae.kl_vae import KL_VAE


def stratified_random_sampling(image_paths, n_samples, seed=42):
    """
    分层随机抽样：将时间序列分成N段，每段随机选1个
    适合步态这种时间序列数据，确保时间覆盖均匀
    
    Args:
        image_paths: 所有图像路径列表（已排序）
        n_samples: 要选择的样本数量
        seed: 随机种子
    
    Returns:
        selected_indices: 选中的样本索引（排序后）
    """
    n_total = len(image_paths)
    if n_samples >= n_total:
        return np.arange(n_total)
    
    rng = np.random.RandomState(seed)
    selected = []
    
    # 分成n_samples段，每段随机选1个
    segment_size = n_total / n_samples
    for i in range(n_samples):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        if end > n_total:
            end = n_total
        # 在每段内随机选一个索引
        idx = rng.randint(start, end)
        selected.append(idx)
    
    return np.array(sorted(selected))


def select_representative_samples_kmedoids(image_paths, vae, transform, n_samples, device, seed=42):
    """
    使用K-Medoids聚类选择最具代表性的训练样本
    
    Args:
        image_paths: 所有图像路径列表
        vae: 预加载的VAE模型
        transform: 图像预处理transform
        n_samples: 要选择的样本数量
        device: 设备
        seed: 随机种子
    
    Returns:
        selected_indices: 选中的样本索引
    """
    if not KMEDOIDS_AVAILABLE:
        print("  K-Medoids不可用，使用随机抽样")
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(image_paths))
        return indices[:n_samples]
    
    print(f"  使用K-Medoids从{len(image_paths)}张中选择{n_samples}张代表性样本...")
    
    # 1. 提取VAE特征
    features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), 
                     desc="    提取特征", leave=False):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"\n    Warning: Failed to load {img_path}: {e}")
                    # 使用零向量占位
                    batch_images.append(torch.zeros(3, 256, 256))
            
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(device)
                latents = vae.encode_images(batch_tensor)  # [B, 4, 32, 32]
                
                # 展平为一维特征向量
                for latent in latents:
                    feature = latent.flatten().cpu().numpy()
                    features.append(feature)
    
    features = np.array(features)
    print(f"    特征维度: {features.shape}")
    
    # 2. 特征归一化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 3. K-Medoids聚类
    # 策略：使用validate_cluster_number.py验证的最优簇数
    # 理论依据：步态周期通常有2-4个主要阶段
    # 目标：捕捉核心模式，最小化用户内"伪多样性"
    # 
    # ⚠️ 重要：运行validate_cluster_number.py确定最优k值后修改这里
    # 推荐流程：
    # 1. python validate_cluster_number.py --vae_path ... --data_folder ...
    # 2. 查看综合推荐的k值
    # 3. 修改下面的n_clusters
    n_clusters = 4  # TODO: 根据validate_cluster_number.py的结果修改
    print(f"    执行K-Medoids聚类（k={n_clusters}）...")
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed, method='pam')
    kmedoids.fit(features_scaled)
    
    # 4. 获取medoid索引和簇信息
    medoid_indices = kmedoids.medoid_indices_
    labels = kmedoids.labels_
    cluster_sizes = np.bincount(labels)
    
    print(f"    ✓ 聚类完成 - Inertia: {kmedoids.inertia_:.2f}")
    print(f"    簇大小: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}")
    print(f"    异常值簇(size=1)数量: {(cluster_sizes == 1).sum()}/{n_clusters}")
    
    # 5. 按比例从每个簇采样，忽略异常值簇
    # 策略：大簇（主要模式）多选，小簇（次要模式）少选
    # 比例：40%, 30%, 20%, 10%（按簇大小排序）
    # 只选择size>=2的簇（忽略异常值）
    valid_clusters = [(i, cluster_sizes[i]) for i in range(n_clusters) if cluster_sizes[i] >= 2]
    
    ignored_clusters = (cluster_sizes == 1).sum()
    if ignored_clusters > 0:
        print(f"    ⚠️ 忽略 {ignored_clusters} 个异常值簇（size=1）")
    
    if len(valid_clusters) == 0:
        print("    错误：没有有效的簇")
        return []
    
    # 按簇大小排序（从大到小）
    valid_clusters.sort(key=lambda x: x[1], reverse=True)
    
    # 定义采样比例（根据簇数动态调整）
    if len(valid_clusters) == 4:
        proportions = [0.40, 0.30, 0.20, 0.10]
    elif len(valid_clusters) == 3:
        proportions = [0.50, 0.30, 0.20]
    elif len(valid_clusters) == 2:
        proportions = [0.60, 0.40]
    else:
        # 如果簇数不是2-4，使用平均分配
        proportions = [1.0 / len(valid_clusters)] * len(valid_clusters)
    
    print(f"    有效簇数: {len(valid_clusters)}")
    print(f"    簇大小（从大到小）: {[size for _, size in valid_clusters]}")
    print(f"    采样比例: {[f'{p*100:.0f}%' for p in proportions]}")
    
    rng = np.random.RandomState(seed)
    selected_indices = []
    
    for idx, (cluster_id, cluster_size) in enumerate(valid_clusters):
        # 获取该簇的所有样本
        cluster_members = np.where(labels == cluster_id)[0].tolist()
        
        # 按比例计算该簇应该采样的数量
        n_to_sample = int(n_samples * proportions[idx])
        
        # 不能超过簇大小
        n_to_sample = min(n_to_sample, cluster_size)
        
        # 采样策略：选择最接近medoid的样本（核心样本）
        # 理论依据：Core-set selection (Har-Peled & Mazumdar, 2004)
        # 目标：最小化簇内差异，选择最一致的样本
        if n_to_sample >= cluster_size:
            # 全部选择
            sampled = cluster_members
        else:
            # 选择最接近medoid的k个样本
            medoid_idx = medoid_indices[cluster_id]
            
            if n_to_sample == 1:
                sampled = [medoid_idx]
            else:
                # 计算所有样本到medoid的距离
                medoid_feature = features_scaled[medoid_idx]
                distances = []
                for member_idx in cluster_members:
                    if member_idx == medoid_idx:
                        distances.append((member_idx, 0.0))  # medoid距离为0
                    else:
                        member_feature = features_scaled[member_idx]
                        dist = np.linalg.norm(medoid_feature - member_feature)
                        distances.append((member_idx, dist))
                
                # 按距离排序，选择最近的k个
                distances.sort(key=lambda x: x[1])
                sampled = [idx for idx, _ in distances[:n_to_sample]]
        
        selected_indices.extend(sampled)
        print(f"      簇{cluster_id}(size={cluster_size}): 采样{len(sampled)}个 ({len(sampled)/n_samples*100:.0f}%)")
    
    print(f"    ✓ 按比例采样完成: {len(selected_indices)} 个样本")
    
    # 检查是否达到目标数量（平均采样应该已经足够）
    if len(selected_indices) < n_samples:
        remaining_needed = n_samples - len(selected_indices)
        all_indices = set(range(len(image_paths)))
        used_indices = set(selected_indices)
        remaining = list(all_indices - used_indices)
        
        rng = np.random.RandomState(seed)
        additional = rng.choice(remaining, size=remaining_needed, replace=False)
        selected_indices.extend(additional)
        print(f"    随机补充: {remaining_needed} 个样本")
    
    selected_indices = np.array(selected_indices[:n_samples])
    print(f"    ✓ 最终选择: {len(selected_indices)} 个训练样本")
    
    return selected_indices


def preprocess_dataset(vae_path, data_path, output_folder, num_users=31, 
                       images_per_user_train=50, images_per_user_val=0, seed=42, 
                       encode_all=True, use_kmedoids=False, device='cuda'):
    """
    预编码所有图像到潜在空间，并保存数据集划分信息
    
    Args:
        vae_path: VAE权重路径
        data_path: 数据集路径
        output_folder: 缓存输出文件夹
        num_users: 用户数量
        images_per_user_train: 每用户训练图像数
        images_per_user_val: 每用户验证图像数（0表示无验证集，剩余为测试集）
        seed: 随机种子
        encode_all: 是否编码所有图像（包括验证集和测试集）
        use_kmedoids: 是否使用K-Medoids选择代表性样本（否则分层随机抽样）
        device: 设备
    """
    
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载VAE
    print(f"Loading VAE from {vae_path}...")
    checkpoint = torch.load(vae_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        vae_state = checkpoint['model_state_dict']
        embed_dim = checkpoint.get('embed_dim', 4)
        scale_factor = checkpoint.get('scale_factor', 0.18215)
        print(f"  Checkpoint format - embed_dim: {embed_dim}, scale_factor: {scale_factor}")
    else:
        vae_state = checkpoint
        embed_dim = 4
        scale_factor = 0.18215
    
    vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
    vae.load_state_dict(vae_state)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully\n")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # 收集所有图像路径并划分训练/验证/测试集
    data_path = Path(data_path)
    train_samples = []
    val_samples = []
    test_samples = []
    data_split_info = {
        'seed': seed,
        'num_users': num_users,
        'images_per_user_train': images_per_user_train,
        'images_per_user_val': images_per_user_val,
        'sampling_method': 'kmedoids' if use_kmedoids else 'stratified',
        'users': {}
    }
    
    sampling_method = 'K-Medoids聚类' if use_kmedoids else '分层随机抽样'
    print(f"Collecting and splitting image paths for {num_users} users...")
    print(f"  采样方法: {sampling_method}")
    print(f"  Train: {images_per_user_train} images/user")
    if images_per_user_val > 0:
        print(f"  Val: {images_per_user_val} images/user")
    print(f"  Test: remaining images/user")
    
    if not use_kmedoids:
        print(f"  ✓ 使用分层随机抽样（时间序列均匀覆盖）")
    
    for user_id in range(1, num_users + 1):
        user_folder = data_path / f"ID_{user_id}"
        
        if not user_folder.exists():
            print(f"  Warning: {user_folder} not found, skipping...")
            continue
        
        # 收集该用户的所有jpg图像（排序确保一致性）
        image_paths = sorted(list(user_folder.glob("*.jpg")))
        
        if len(image_paths) < images_per_user_train:
            print(f"  Warning: ID_{user_id} only has {len(image_paths)} images, skipping...")
            continue
        
        # 为每个用户设置独立但可复现的随机种子
        user_seed = seed + user_id
        
        if use_kmedoids:
            # 使用K-Medoids选择代表性样本
            selected_indices = select_representative_samples_kmedoids(
                image_paths, vae, transform, images_per_user_train, device, user_seed
            )
            
            # 创建训练/测试集
            train_indices_set = set(selected_indices)
            train_paths = [image_paths[i] for i in selected_indices]
            test_paths = [image_paths[i] for i in range(len(image_paths)) 
                         if i not in train_indices_set]
        else:
            # 分层随机抽样（推荐方法）
            train_indices = stratified_random_sampling(
                image_paths, images_per_user_train, user_seed
            )
            
            # 创建训练/验证/测试集
            train_indices_set = set(train_indices)
            train_paths = [image_paths[i] for i in train_indices]
            
            # 如果需要验证集，从剩余样本中再次分层抽样
            if images_per_user_val > 0:
                remaining_paths = [image_paths[i] for i in range(len(image_paths)) 
                                  if i not in train_indices_set]
                remaining_indices = [i for i in range(len(image_paths)) 
                                    if i not in train_indices_set]
                
                if len(remaining_paths) >= images_per_user_val:
                    val_indices_in_remaining = stratified_random_sampling(
                        remaining_paths, images_per_user_val, user_seed + 1000
                    )
                    val_indices = [remaining_indices[i] for i in val_indices_in_remaining]
                    val_indices_set = set(val_indices)
                    val_paths = [image_paths[i] for i in val_indices]
                    
                    # 测试集 = 全部 - 训练集 - 验证集
                    test_paths = [image_paths[i] for i in range(len(image_paths)) 
                                 if i not in train_indices_set and i not in val_indices_set]
                else:
                    print(f"  Warning: ID_{user_id} 剩余样本不足，无法划分验证集")
                    val_paths = []
                    test_paths = remaining_paths
            else:
                val_paths = []
                test_paths = [image_paths[i] for i in range(len(image_paths)) 
                             if i not in train_indices_set]
        
        # 保存划分信息
        user_info = {
            'user_id': user_id,
            'label': user_id - 1,
            'total_images': len(image_paths),
            'train_images': [str(p.relative_to(data_path)) for p in train_paths],
            'test_images': [str(p.relative_to(data_path)) for p in test_paths]
        }
        if images_per_user_val > 0:
            user_info['val_images'] = [str(p.relative_to(data_path)) for p in val_paths]
        
        data_split_info['users'][f'ID_{user_id}'] = user_info
        
        # 添加到样本列表
        for img_path in train_paths:
            train_samples.append((img_path, user_id - 1, 'train'))
        
        if images_per_user_val > 0:
            for img_path in val_paths:
                val_samples.append((img_path, user_id - 1, 'val'))
        
        if encode_all:
            for img_path in test_paths:
                test_samples.append((img_path, user_id - 1, 'test'))
    
    # 合并所有样本
    all_samples = train_samples + val_samples + (test_samples if encode_all else [])
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples)} images")
    if images_per_user_val > 0:
        print(f"  Val: {len(val_samples)} images")
    print(f"  Test: {len(test_samples)} images")
    print(f"  Total to encode: {len(all_samples)} images\n")
    
    # 保存数据集划分信息
    split_file = Path(output_folder) / 'data_split.json'
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(data_split_info, f, indent=2, ensure_ascii=False)
    print(f"Data split info saved to: {split_file}\n")
    
    # 批量编码
    print("Encoding images to latent space...")
    batch_size = 32  # 批量处理，加速编码
    
    encoded_count = 0
    skipped_count = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches"):
            batch_samples = all_samples[i:i+batch_size]
            batch_images = []
            batch_paths = []
            
            for img_path, label, split_type in batch_samples:
                # 检查是否已缓存
                # 使用 userID_filename.pt 格式避免文件名冲突
                cache_filename = f"user_{label:02d}_{img_path.stem}.pt"
                cache_path = output_path / cache_filename
                
                if cache_path.exists():
                    skipped_count += 1
                    continue
                
                # 加载图像
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                    batch_paths.append((cache_path, label))
                except Exception as e:
                    print(f"\nWarning: Failed to load {img_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # 批量编码
            batch_tensor = torch.stack(batch_images).to(device)
            latents = vae.encode_images(batch_tensor)  # [B, 4, 32, 32]
            
            # 保存到缓存
            for (cache_path, label), latent in zip(batch_paths, latents):
                torch.save(latent.cpu(), cache_path)
                encoded_count += 1
    
    print(f"\n{'='*60}")
    print("预编码完成！")
    print(f"{'='*60}")
    print(f"新编码: {encoded_count} 张")
    print(f"已存在: {skipped_count} 张")
    print(f"总计: {encoded_count + skipped_count} 张")
    print(f"\n输出文件:")
    print(f"  潜在表示缓存: {output_path}/")
    print(f"  数据集划分: {output_path}/data_split.json")
    print(f"\n现在可以:")
    print(f"  1. 开始DDPM训练: python train_latent_cfg.py")
    print(f"  2. 进行分类器实验（使用data_split.json获取训练/测试集）")


def main():
    parser = argparse.ArgumentParser(description='预编码训练图像到潜在空间')
    
    # 路径参数
    parser.add_argument('--vae_path', type=str,
                        default='/kaggle/input/kl-vae-best-pt/kl_vae_best.pt',
                        help='VAE权重路径')
    parser.add_argument('--data_path', type=str,
                        default='/kaggle/input/organized-gait-dataset/Normal_line',
                        help='数据集路径')
    parser.add_argument('--output_folder', type=str,
                        default='./latents_cache',
                        help='缓存输出文件夹')
    
    # 数据参数
    parser.add_argument('--num_users', type=int, default=31,
                        help='用户数量')
    parser.add_argument('--images_per_user_train', type=int, default=30,
                        help='每用户训练图像数')
    parser.add_argument('--images_per_user_val', type=int, default=20,
                        help='每用户验证图像数（0表示无验证集）')
    parser.add_argument('--encode_all', action='store_true', default=True,
                        help='是否编码所有图像（包括验证集和测试集）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 采样方法
    parser.add_argument('--use_kmedoids', action='store_true', default=False,
                        help='使用K-Medoids选择代表性样本')
    parser.add_argument('--stratified_sampling', dest='use_kmedoids', action='store_false',
                        help='使用分层随机抽样（推荐，默认）')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        vae_path=args.vae_path,
        data_path=args.data_path,
        output_folder=args.output_folder,
        num_users=args.num_users,
        images_per_user_train=args.images_per_user_train,
        images_per_user_val=args.images_per_user_val,
        encode_all=args.encode_all,
        use_kmedoids=args.use_kmedoids,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()

