"""
多种子分类器实验
依次训练种子为 6, 42, 888 的分类器
统计每个种子的测试准确率，计算平均值和标准差
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image
import numpy as np
import random
import argparse
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from load_dataset import MicroDopplerDataset


def set_random_seed(seed=42):
    """设置所有随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 已设置随机种子: {seed}")


def train_classifier(model, train_loader, criterion, optimizer, device, epochs=15, scheduler=None):
    """训练分类器 - 固定epoch数，训练完成后再测试"""
    
    print(f"训练数据信息：{len(train_loader.dataset)} 张图像")
    
    # 检查第一个batch的图像尺寸
    for images, labels in train_loader:
        print(f"图像尺寸: {images.shape} (Batch, Channels, Height, Width)")
        print(f"图像数据类型: {images.dtype}, 值域: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print(f"开始训练 {epochs} epochs（与文献一致）")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(pbar.n+1):.4f}',
                'train_acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.2f}%")
        
        # 学习率调度（基于训练loss）
        if scheduler:
            scheduler.step(avg_train_loss)
    
    print(f"训练完成，共进行 {epochs} epochs（与文献一致）")


def evaluate_classifier(model, test_loader, device, num_classes):
    """评估分类器"""
    model.eval()
    
    correct = 0
    total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 统计每个类别
            for label, pred in zip(labels, predicted):
                per_class_total[label.item()] += 1
                if label == pred:
                    per_class_correct[label.item()] += 1
    
    overall_acc = 100. * correct / total
    
    print(f"\n整体准确率: {overall_acc:.2f}% ({correct}/{total})")
    
    # 每个用户的准确率
    print("\n各用户准确率:")
    for i in range(num_classes):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            print(f"  用户{i:2d}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
    return overall_acc, per_class_correct, per_class_total


class SyntheticDataset(torch.utils.data.Dataset):
    """
    加载生成的合成图像
    支持格式：ID_X/sample_XXX.png 或 ID_X/generated_XXX.jpg
    """
    def __init__(self, synthetic_folder, transform=None):
        self.samples = []
        
        synthetic_path = Path(synthetic_folder)
        
        # 搜索子文件夹中的图像：ID_X/*.png 或 ID_X/*.jpg
        for user_folder in sorted(synthetic_path.glob("ID_*")):
            if user_folder.is_dir():
                # 从文件夹名解析用户ID：ID_1 → label=0, ID_2 → label=1
                user_id = int(user_folder.name.split('_')[1])  # ID_1 → 1
                label = user_id - 1  # ID_1 → label=0
                
                # 加载该用户文件夹下的所有图像（支持png和jpg）
                img_files = list(user_folder.glob("*.png")) + list(user_folder.glob("*.jpg"))
                for img_path in sorted(img_files):
                    self.samples.append((img_path, label))
        
        self.transform = transform
        print(f"✓ 加载合成数据集: {len(self.samples)}张图像，{len(set(l for _, l in self.samples))}个用户")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def run_single_experiment(seed: int, args, device) -> Dict:
    """运行单个种子的实验"""
    
    logger.info("\n" + "="*60)
    logger.info(f"开始实验: Seed = {seed}")
    logger.info("="*60)
    
    # 设置随机种子
    set_random_seed(seed)
    
    # 加载数据集
    logger.info("加载数据集...")
    
    # 训练集（真实图像）
    train_ds = MicroDopplerDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split='train',
        use_latents=False
    )
    
    # 测试集（真实图像）
    test_ds = MicroDopplerDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split='test',
        use_latents=False
    )
    
    # 如果提供了合成数据
    if args.synthetic_folder:
        logger.info("添加合成数据到训练集...")
        
        synthetic_ds = SyntheticDataset(
            args.synthetic_folder,
            transform=train_ds.transform
        )
        
        # 自动检测合成数据的用户数
        synthetic_users = set(label for _, label in synthetic_ds.samples)
        num_synthetic_users = len(synthetic_users)
        max_label = max(synthetic_users)
        
        logger.info(f"检测到合成数据包含 {num_synthetic_users} 个用户（label 0-{max_label}）")
        
        # 如果未指定num_users，自动使用合成数据的用户数
        if args.num_users is None:
            args.num_users = max_label + 1  # label从0开始，所以+1
            logger.info(f"自动设置为使用前 {args.num_users} 个用户")
        
        # 过滤真实数据，只保留前num_users个用户
        logger.info(f"过滤真实数据，只保留前 {args.num_users} 个用户（label 0-{args.num_users-1}）...")
        train_ds.samples = [(path, label) for path, label in train_ds.samples if label < args.num_users]
        test_ds.samples = [(path, label) for path, label in test_ds.samples if label < args.num_users]
        
        logger.info(f"过滤后训练集: {len(train_ds)} 张真实图像")
        logger.info(f"过滤后测试集: {len(test_ds)} 张真实图像")
        
        # 合并数据集
        train_ds = ConcatDataset([train_ds, synthetic_ds])
        logger.info(f"增强后训练集: {len(train_ds)} 张（真实+合成）")
    elif args.num_users is not None:
        # 没有合成数据，但指定了num_users，也过滤
        logger.info(f"\n过滤数据，只使用前 {args.num_users} 个用户（label 0-{args.num_users-1}）...")
        train_ds.samples = [(path, label) for path, label in train_ds.samples if label < args.num_users]
        test_ds.samples = [(path, label) for path, label in test_ds.samples if label < args.num_users]
        logger.info(f"过滤后训练集: {len(train_ds)} 张图像")
        logger.info(f"过滤后测试集: {len(test_ds)} 张图像")
    
    # 自动推断类别数量
    if isinstance(train_ds, ConcatDataset):
        all_labels = [label for _, label in train_ds.datasets[0].samples]
    else:
        all_labels = [label for _, label in train_ds.samples]
    num_classes = len(set(all_labels))
    logger.info(f"检测到 {num_classes} 个用户类别")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # 创建ResNet18分类器
    logger.info(f"创建ResNet18分类器（{num_classes}个类别）...")
    model = resnet18(weights=None, num_classes=num_classes)
    model = model.to(device)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    
    # 训练
    logger.info("开始训练...")
    train_classifier(
        model, train_loader, criterion, optimizer, device, args.epochs, scheduler
    )
    
    # 评估
    logger.info("评估分类器...")
    accuracy, per_class_correct, per_class_total = evaluate_classifier(
        model, test_loader, device, num_classes
    )
    
    logger.info(f"\n✓ Seed {seed} 完成")
    logger.info(f"  测试准确率: {accuracy:.2f}%")
    
    # 打印各用户准确率
    logger.info("  各用户准确率:")
    for i in range(num_classes):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            logger.info(f"    用户{i:2d}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
    return {
        'seed': seed,
        'accuracy': accuracy,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'num_classes': num_classes
    }


def main():
    parser = argparse.ArgumentParser(description='多种子ResNet18分类器实验')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                        default='./latents_cache/data_split.json',
                        help='数据集划分文件')
    parser.add_argument('--synthetic_folder', type=str, default=None,
                        help='合成数据文件夹（可选）')
    parser.add_argument('--num_users', type=int, default=None,
                        help='使用的用户数量')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练batch size')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[6, 42, 888],
                        help='要测试的种子列表')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logger.info("\n" + "="*60)
    logger.info("多种子分类器实验")
    logger.info("="*60)
    logger.info(f"数据根目录: {args.data_root}")
    logger.info(f"要测试的种子: {args.seeds}")
    logger.info(f"训练epochs: {args.epochs}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60)
    
    # 运行多个种子的实验
    results = []
    for seed in args.seeds:
        result = run_single_experiment(seed, args, device)
        results.append(result)
    
    # 统计结果
    logger.info("\n" + "="*60)
    logger.info("实验结果统计")
    logger.info("="*60)
    
    accuracies = [r['accuracy'] for r in results]
    
    logger.info("\n各种子的测试准确率:")
    for result in results:
        logger.info(f"  Seed {result['seed']:3d}: {result['accuracy']:6.2f}%")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    logger.info(f"\n统计信息:")
    logger.info(f"  平均准确率: {mean_acc:.2f}%")
    logger.info(f"  标准差: {std_acc:.2f}%")
    logger.info(f"  最高准确率: {max_acc:.2f}% (Seed {results[np.argmax(accuracies)]['seed']})")
    logger.info(f"  最低准确率: {min_acc:.2f}% (Seed {results[np.argmin(accuracies)]['seed']})")
    logger.info(f"  准确率范围: [{min_acc:.2f}%, {max_acc:.2f}%]")
    
    # 保存结果
    output_dir = Path("./multi_seed_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'seeds': args.seeds,
        'individual_results': [
            {
                'seed': r['seed'],
                'accuracy': float(r['accuracy']),
                'num_classes': r['num_classes']
            }
            for r in results
        ],
        'statistics': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'min_accuracy': float(min_acc),
            'max_accuracy': float(max_acc),
            'accuracy_range': [float(min_acc), float(max_acc)]
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'data_root': args.data_root,
            'synthetic_folder': args.synthetic_folder,
            'num_users': args.num_users
        }
    }
    
    results_file = output_dir / "multi_seed_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n💾 结果已保存至: {results_file}")
    logger.info("="*60)
    
    return results_summary


if __name__ == '__main__':
    results = main()
