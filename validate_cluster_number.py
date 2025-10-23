"""
验证最优簇数：使用肘部法则和轮廓系数
目标：为K-Medoids聚类找到最优的k值
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))


def load_vae(vae_path, device='cuda'):
    """加载VAE模型（与preprocess_latents.py保持一致）"""
    print(f"加载VAE: {vae_path}")
    checkpoint = torch.load(vae_path, map_location=device)
    
    # 导入VAE
    from vae.kl_vae import KL_VAE
    
    # 从checkpoint中读取配置
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        vae_state = checkpoint['model_state_dict']
        embed_dim = checkpoint.get('embed_dim', 4)
        scale_factor = checkpoint.get('scale_factor', 0.18215)
        print(f"  Checkpoint格式 - embed_dim: {embed_dim}, scale_factor: {scale_factor}")
    else:
        vae_state = checkpoint
        embed_dim = 4
        scale_factor = 0.18215
        print(f"  使用默认配置 - embed_dim: {embed_dim}, scale_factor: {scale_factor}")
    
    # 创建VAE实例（使用实际配置）
    vae = KL_VAE(embed_dim=embed_dim, scale_factor=scale_factor)
    vae.load_state_dict(vae_state)
    vae = vae.to(device)
    vae.eval()
    print("✓ VAE加载成功\n")
    
    return vae


def encode_images_to_features(image_folder, vae, device='cuda', max_images=None):
    """将图像编码为潜在特征向量"""
    image_folder = Path(image_folder)
    image_paths = sorted(list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg")))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"  加载 {len(image_paths)} 张图像...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    features = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="  编码图像"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                except:
                    continue
            
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(device)
                latents = vae.encode_images(batch_tensor)  # [B, 4, 32, 32]
                
                # 展平为一维特征向量
                for latent in latents:
                    feature = latent.flatten().cpu().numpy()
                    features.append(feature)
    
    return np.array(features)


def compute_elbow_metrics(features_scaled, k_range, seed=42):
    """计算不同k值的评估指标"""
    print(f"\n测试簇数范围: {list(k_range)}")
    
    metrics = {
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    for k in tqdm(k_range, desc="聚类评估"):
        # K-Medoids聚类
        kmedoids = KMedoids(n_clusters=k, random_state=seed, method='pam')
        labels = kmedoids.fit_predict(features_scaled)
        
        # 1. Inertia（簇内距离和，越小越好）
        metrics['inertia'].append(kmedoids.inertia_)
        
        # 2. Silhouette Score（轮廓系数，[-1, 1]，越大越好）
        silhouette = silhouette_score(features_scaled, labels)
        metrics['silhouette'].append(silhouette)
        
        # 3. Davies-Bouldin Index（越小越好）
        db_score = davies_bouldin_score(features_scaled, labels)
        metrics['davies_bouldin'].append(db_score)
        
        # 4. Calinski-Harabasz Index（越大越好）
        ch_score = calinski_harabasz_score(features_scaled, labels)
        metrics['calinski_harabasz'].append(ch_score)
        
        print(f"  k={k}: Inertia={kmedoids.inertia_:.2f}, "
              f"Silhouette={silhouette:.4f}, "
              f"DB={db_score:.4f}, "
              f"CH={ch_score:.2f}")
    
    return metrics


def find_elbow_point(x, y):
    """使用膝点检测算法找到肘部"""
    # 归一化
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # 计算每个点到直线的距离
    # 直线：从第一个点到最后一个点
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    
    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        # 点到直线的距离
        d = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        distances.append(d)
    
    # 最大距离对应的点就是肘部
    elbow_idx = np.argmax(distances)
    return x[elbow_idx]


def plot_metrics(k_range, metrics, output_path='cluster_validation.png'):
    """绘制评估指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    k_list = list(k_range)
    
    # 1. Inertia（肘部法则）
    ax = axes[0, 0]
    ax.plot(k_list, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
    elbow_k = find_elbow_point(np.array(k_list), np.array(metrics['inertia']))
    ax.axvline(x=elbow_k, color='r', linestyle='--', linewidth=2, label=f'Elbow at k={elbow_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (lower is better)', fontsize=12)
    ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 2. Silhouette Score
    ax = axes[0, 1]
    ax.plot(k_list, metrics['silhouette'], 'go-', linewidth=2, markersize=8)
    best_k = k_list[np.argmax(metrics['silhouette'])]
    ax.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best at k={best_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score (higher is better)', fontsize=12)
    ax.set_title('Silhouette Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 3. Davies-Bouldin Index
    ax = axes[1, 0]
    ax.plot(k_list, metrics['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
    best_k = k_list[np.argmin(metrics['davies_bouldin'])]
    ax.axvline(x=best_k, color='g', linestyle='--', linewidth=2, label=f'Best at k={best_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index (lower is better)', fontsize=12)
    ax.set_title('Davies-Bouldin Index', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 4. Calinski-Harabasz Index
    ax = axes[1, 1]
    ax.plot(k_list, metrics['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
    best_k = k_list[np.argmax(metrics['calinski_harabasz'])]
    ax.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best at k={best_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Index (higher is better)', fontsize=12)
    ax.set_title('Calinski-Harabasz Index', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 评估图表已保存: {output_path}")
    plt.close()


def analyze_cluster_sizes(features_scaled, k_range, seed=42):
    """分析不同k值下的簇大小分布"""
    print("\n簇大小分布分析:")
    print("="*60)
    
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, random_state=seed, method='pam')
        labels = kmedoids.fit_predict(features_scaled)
        cluster_sizes = np.bincount(labels)
        
        print(f"\nk={k}:")
        print(f"  簇大小: {sorted(cluster_sizes, reverse=True)}")
        print(f"  最大簇占比: {cluster_sizes.max() / len(labels) * 100:.1f}%")
        print(f"  最小簇占比: {cluster_sizes.min() / len(labels) * 100:.1f}%")
        print(f"  标准差: {cluster_sizes.std():.2f}")
        print(f"  变异系数: {cluster_sizes.std() / cluster_sizes.mean():.2f}")


def main():
    parser = argparse.ArgumentParser(description='验证最优簇数')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='VAE模型路径')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='数据文件夹（如 data/ID_1）')
    parser.add_argument('--k_min', type=int, default=2,
                        help='最小簇数（默认2）')
    parser.add_argument('--k_max', type=int, default=8,
                        help='最大簇数（默认8）')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最多使用的图像数（默认全部）')
    parser.add_argument('--output', type=str, default='cluster_validation.png',
                        help='输出图表路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    print("="*60)
    print("验证最优簇数")
    print("="*60)
    
    # 1. 加载VAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = load_vae(args.vae_path, device)
    
    # 2. 编码图像为特征
    print(f"\n编码图像: {args.data_folder}")
    features = encode_images_to_features(args.data_folder, vae, device, args.max_images)
    print(f"  特征维度: {features.shape}")
    
    # 3. 特征归一化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 4. 测试不同的k值
    k_range = range(args.k_min, args.k_max + 1)
    metrics = compute_elbow_metrics(features_scaled, k_range, args.seed)
    
    # 5. 绘制评估图表
    plot_metrics(k_range, metrics, args.output)
    
    # 6. 分析簇大小分布
    analyze_cluster_sizes(features_scaled, k_range, args.seed)
    
    # 7. 总结推荐
    print("\n" + "="*60)
    print("推荐分析:")
    print("="*60)
    
    k_list = list(k_range)
    
    # 肘部法则
    elbow_k = find_elbow_point(np.array(k_list), np.array(metrics['inertia']))
    print(f"肘部法则推荐: k={elbow_k}")
    
    # 轮廓系数
    best_silhouette_k = k_list[np.argmax(metrics['silhouette'])]
    print(f"轮廓系数推荐: k={best_silhouette_k} (score={max(metrics['silhouette']):.4f})")
    
    # Davies-Bouldin
    best_db_k = k_list[np.argmin(metrics['davies_bouldin'])]
    print(f"Davies-Bouldin推荐: k={best_db_k} (score={min(metrics['davies_bouldin']):.4f})")
    
    # Calinski-Harabasz
    best_ch_k = k_list[np.argmax(metrics['calinski_harabasz'])]
    print(f"Calinski-Harabasz推荐: k={best_ch_k} (score={max(metrics['calinski_harabasz']):.2f})")
    
    # 综合推荐
    recommendations = [elbow_k, best_silhouette_k, best_db_k, best_ch_k]
    from collections import Counter
    most_common = Counter(recommendations).most_common(1)[0][0]
    
    print(f"\n综合推荐: k={most_common}")
    print(f"  (基于{Counter(recommendations)[most_common]}个指标的一致推荐)")
    
    # 步态理论对比
    print(f"\n步态理论推荐: k=4 (基于步态周期的4个主要阶段)")
    
    if most_common == 4:
        print("✓ 综合推荐与步态理论一致！")
    else:
        print(f"⚠️ 综合推荐(k={most_common})与步态理论(k=4)不一致")
        print(f"   建议：优先考虑领域知识，使用k=4")
    
    print("="*60)


if __name__ == '__main__':
    main()
