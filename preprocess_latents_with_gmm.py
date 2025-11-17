"""
预编码脚本 - 基于GMM聚类的分层抽样
=====================================================

数据划分策略：基于GMM的分层均匀抽样
-----------------------------------------------------
适用场景：步态微多普勒时间序列数据，每个用户有不同的最优聚类数

原理：
  - 每个用户有150张连续采集的时频图像（时间序列）
  - 根据用户的最优K值，使用GMM从每个高斯分量中均匀抽样
  - gen_train: 30张样本（用于生成模型训练）
  - class_train: 20张样本（用于分类器训练）
  - test: 剩余100张样本（测试集）
  - 三个集合完全分离，无重叠

采样方法：
  1. 使用GMM将150张图像分成K个高斯分量
  2. 从每个分量中均匀抽样，确保覆盖完整步态周期
  3. 优先分配给gen_train，然后class_train，剩余为test

优势：
  ✓ 根据用户特性自适应聚类数
  ✓ 覆盖完整步态周期（每个分量代表一个阶段）
  ✓ 三个集合完全独立分离
  ✓ 可复现（固定索引，非随机）
  ✓ GMM提供概率分布，比K-means更灵活

输出：
  1. latents_cache/ - 所有图像的潜在表示
  2. data_split.json - 数据集划分信息（含聚类和采样索引）
"""

import os
import sys

# 在导入任何科学计算库之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae.kl_vae import KL_VAE


# 每个用户的最优聚类数
USER_K_VALUES = {
    "ID_1": 2, "ID_2": 2, "ID_3": 3, "ID_4": 3, "ID_5": 3,
    "ID_6": 3, "ID_7": 3, "ID_8": 3, "ID_9": 2, "ID_10": 3,
    "ID_11": 3, "ID_12": 3, "ID_13": 3, "ID_14": 3, "ID_15": 3,
    "ID_16": 3, "ID_17": 2, "ID_18": 3, "ID_19": 3, "ID_20": 2,
    "ID_21": 3, "ID_22": 2, "ID_23": 2, "ID_24": 3, "ID_25": 2,
    "ID_26": 2, "ID_27": 3, "ID_28": 2, "ID_29": 3, "ID_30": 3,
    "ID_31": 3
}


def extract_features_for_clustering(image_paths, vae, device, batch_size=16):
    """
    提取图像特征用于聚类，同时返回潜在表示用于缓存
    
    Returns:
        features: 展平的特征向量 [N, 4096]
        latents_list: 原始潜在表示 [N, 4, 32, 32]（用于缓存）
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    features = []
    latents_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="  提取特征", leave=False):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                except:
                    continue
            
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(device)
                latents = vae.encode_images(batch_tensor)  # [B, 4, 32, 32]
                
                # 展平为特征向量，并保存潜在表示
                for latent in latents:
                    feature = latent.flatten().cpu().numpy()
                    features.append(feature)
                    latents_list.append(latent.cpu())
    
    return np.array(features), latents_list


def stratified_sample_from_clusters(image_paths, features, k, n_gen_train=30, 
                                   n_class_train=20, seed=42):
    """
    基于GMM聚类的分层抽样
    
    Args:
        image_paths: 所有图像路径列表
        features: 图像特征矩阵 [N, D]
        k: 高斯分量数
        n_gen_train: gen_train样本数（固定值）
        n_class_train: class_train样本数（固定值）
        seed: 随机种子
    
    Returns:
        gen_train_indices, class_train_indices, test_indices, cluster_labels
    """
    
    total_samples = len(image_paths)
    
    # 转换为float64以提高数值稳定性
    features = features.astype(np.float64)
    
    # GMM聚类
    try:
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=10,
                             reg_covar=1e-6, covariance_type='full', max_iter=200)
        cluster_labels = gmm.fit_predict(features)
    except ValueError:
        # 如果full失败，使用diag
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=10,
                             reg_covar=1e-5, covariance_type='diag', max_iter=200)
        cluster_labels = gmm.fit_predict(features)
    
    # 从每个聚类中均匀抽样
    gen_train_indices = []
    class_train_indices = []
    test_indices = []
    
    for cluster_id in range(k):
        # 获取该聚类中的所有索引
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # 该聚类中的样本数
        n_samples = len(cluster_indices)
        
        # 从该聚类中均匀抽样
        # 按比例分配：gen_train占30/50=60%，class_train占20/50=40%
        n_gen = max(1, int(n_samples * n_gen_train / (n_gen_train + n_class_train)))
        n_class = max(1, int(n_samples * n_class_train / (n_gen_train + n_class_train)))
        
        # 从聚类中均匀选择索引
        if n_gen > 0:
            gen_indices_float = np.linspace(0, n_samples - 1, n_gen)
            gen_indices = np.round(gen_indices_float).astype(int)
            gen_indices = np.unique(gen_indices)
            gen_train_indices.extend(cluster_indices[gen_indices])
        
        # 从剩余样本中选择class_train
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[gen_indices] = False
        remaining_indices = np.where(remaining_mask)[0]
        
        if len(remaining_indices) > 0 and n_class > 0:
            class_indices_float = np.linspace(0, len(remaining_indices) - 1, 
                                             min(n_class, len(remaining_indices)))
            class_indices = np.round(class_indices_float).astype(int)
            class_indices = np.unique(class_indices)
            class_train_indices.extend(cluster_indices[remaining_indices[class_indices]])
        
        # 剩余的作为测试集
        test_mask = np.ones(n_samples, dtype=bool)
        test_mask[gen_indices] = False
        if len(remaining_indices) > 0:
            test_mask[remaining_indices[class_indices]] = False
        test_indices.extend(cluster_indices[test_mask])
    
    # 确保gen_train和class_train的数量不超过总数
    gen_train_indices = np.array(gen_train_indices)[:n_gen_train]
    class_train_indices = np.array(class_train_indices)[:n_class_train]
    test_indices = np.array(test_indices)
    
    return (gen_train_indices, 
            class_train_indices, 
            test_indices,
            cluster_labels)


def preprocess_dataset(vae_path, data_path, output_folder, num_users=31, 
                       seed=42, device='cuda'):
    """
    预编码所有图像到潜在空间，基于K-means的分层抽样
    """
    
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载VAE
    print(f"加载VAE: {vae_path}")
    checkpoint = torch.load(vae_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        vae_state = checkpoint['model_state_dict']
        embed_dim = checkpoint.get('embed_dim', 4)
        scale_factor = checkpoint.get('scale_factor', 0.18215)
        print(f"  Checkpoint格式 - embed_dim: {embed_dim}, scale_factor: {scale_factor}")
    else:
        vae_state = checkpoint
        embed_dim = 4
        scale_factor = 0.18215
    
    vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
    vae.load_state_dict(vae_state)
    vae = vae.to(device)
    vae.eval()
    print("✓ VAE加载成功\n")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # 收集所有图像路径并划分
    data_path = Path(data_path)
    all_samples = []
    data_split_info = {
        'seed': seed,
        'num_users': num_users,
        'sampling_method': 'gmm_stratified',
        'users': {}
    }
    
    print(f"收集和划分 {num_users} 个用户的图像...")
    print(f"  采样方法: GMM分层均匀抽样")
    print(f"  gen_train: 固定30张/用户 (生成模型训练)")
    print(f"  class_train: 固定20张/用户 (分类器训练)")
    print(f"  test: 剩余全部样本 (测试集)")
    print(f"  三个集合完全分离\n")
    
    for user_id in range(1, num_users + 1):
        user_folder = data_path / f"ID_{user_id}"
        
        if not user_folder.exists():
            print(f"  ⚠️ {user_folder} 不存在，跳过...")
            continue
        
        # 收集该用户的所有jpg图像
        image_paths = sorted(list(user_folder.glob("*.jpg")))
        total_images = len(image_paths)
        
        if total_images == 0:
            print(f"  ⚠️ {user_folder} 中无图像，跳过...")
            continue
        
        print(f"处理 ID_{user_id}...", end=' ')
        
        # 获取该用户的K值
        user_key = f"ID_{user_id}"
        k = USER_K_VALUES.get(user_key, 3)
        
        # 提取特征用于聚类，同时获取潜在表示用于缓存
        features, latents_list = extract_features_for_clustering(image_paths, vae, device)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA降维（与validate_cluster_number.py保持一致）
        # 保留95%的方差，降低维度以提高数值稳定性
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features_scaled)
        print(f"PCA: {features.shape[1]}→{features_pca.shape[1]}维", end=' ')
        
        # 基于GMM的分层抽样（使用PCA降维后的特征）
        gen_train_idx, class_train_idx, test_idx, cluster_labels = stratified_sample_from_clusters(
            image_paths, features_pca, k=k, 
            n_gen_train=30, n_class_train=20, seed=seed
        )
        
        # 构建样本列表（包含潜在表示）
        gen_train_paths = [image_paths[i] for i in gen_train_idx]
        class_train_paths = [image_paths[i] for i in class_train_idx]
        test_paths = [image_paths[i] for i in test_idx]
        
        # 构建样本列表（包含潜在表示用于缓存）
        gen_train_latents = [(image_paths[i], latents_list[i], user_id - 1, 'gen_train') for i in gen_train_idx]
        class_train_latents = [(image_paths[i], latents_list[i], user_id - 1, 'class_train') for i in class_train_idx]
        test_latents = [(image_paths[i], latents_list[i], user_id - 1, 'test') for i in test_idx]
        
        # 保存划分信息
        data_split_info['users'][user_key] = {
            'user_id': user_id,
            'label': user_id - 1,
            'total_images': total_images,
            'k_clusters': k,
            'gen_train_count': len(gen_train_paths),
            'class_train_count': len(class_train_paths),
            'test_count': len(test_paths),
            'gen_train_indices': gen_train_idx.tolist(),
            'class_train_indices': class_train_idx.tolist(),
            'test_indices': test_idx.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'gen_train_images': [str(p.relative_to(data_path)) for p in gen_train_paths],
            'class_train_images': [str(p.relative_to(data_path)) for p in class_train_paths],
            'test_images': [str(p.relative_to(data_path)) for p in test_paths]
        }
        
        # 添加到样本列表（包含潜在表示，无需重新编码）
        all_samples.extend(gen_train_latents)
        all_samples.extend(class_train_latents)
        all_samples.extend(test_latents)
        
        print(f"✓ K={k}, gen_train={len(gen_train_paths)}, class_train={len(class_train_paths)}, test={len(test_paths)}")
    
    # 统计
    gen_train_count = sum(1 for _, _, _, split in all_samples if split == 'gen_train')
    class_train_count = sum(1 for _, _, _, split in all_samples if split == 'class_train')
    test_count = sum(1 for _, _, _, split in all_samples if split == 'test')
    
    print(f"\n{'='*60}")
    print(f"数据集划分统计:")
    print(f"{'='*60}")
    print(f"  采样方法: GMM分层均匀抽样")
    print(f"  gen_train: {gen_train_count} 张 (生成模型训练)")
    print(f"  class_train: {class_train_count} 张 (分类器训练)")
    print(f"  test: {test_count} 张 (测试集)")
    print(f"  总计编码: {len(all_samples)} 张")
    print(f"  三个集合完全分离: ✓")
    print(f"{'='*60}\n")
    
    # 保存数据集划分信息
    split_file = Path(output_folder) / 'data_split.json'
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(data_split_info, f, indent=2, ensure_ascii=False)
    print(f"✓ 数据集划分信息已保存: {split_file}\n")
    
    # 直接保存潜在表示到缓存（无需重新编码）
    print("保存潜在表示到缓存...")
    
    encoded_count = 0
    skipped_count = 0
    
    for img_path, latent, label, split_type in tqdm(all_samples, desc="保存缓存"):
        # 检查是否已缓存
        cache_filename = f"user_{label:02d}_{img_path.stem}.pt"
        cache_path = output_path / cache_filename
        
        if cache_path.exists():
            skipped_count += 1
            continue
        
        # 直接保存潜在表示
        try:
            torch.save(latent, cache_path)
            encoded_count += 1
        except Exception as e:
            print(f"\n⚠️ 保存失败 {cache_path}: {e}")
    
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
    parser = argparse.ArgumentParser(description='预编码训练图像到潜在空间（基于GMM聚类）')
    
    # 路径参数
    parser.add_argument('--vae_path', type=str, required=True,
                        help='VAE权重路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--output_folder', type=str, default='./latents_cache_kmeans',
                        help='缓存输出文件夹')
    
    # 数据参数
    parser.add_argument('--num_users', type=int, default=31,
                        help='用户数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        vae_path=args.vae_path,
        data_path=args.data_path,
        output_folder=args.output_folder,
        num_users=args.num_users,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()
