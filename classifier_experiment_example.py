"""
分类器实验示例
演示如何使用数据集划分进行ResNet18分类实验
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18
from tqdm import tqdm
import json
from pathlib import Path

from load_dataset import MicroDopplerDataset


def train_classifier(model, train_loader, criterion, optimizer, device, epochs=100):
    """训练分类器"""
    model.train()
    
    for epoch in range(epochs):
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
                'acc': f'{100.*correct/total:.2f}%'
            })


def evaluate_classifier(model, test_loader, device):
    """评估分类器"""
    model.eval()
    
    correct = 0
    total = 0
    per_class_correct = [0] * 31
    per_class_total = [0] * 31
    
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
    for i in range(31):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            print(f"  用户{i:2d}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
    return overall_acc


class SyntheticDataset(Dataset):
    """
    加载生成的合成图像
    """
    def __init__(self, synthetic_folder, transform=None):
        self.samples = []
        
        synthetic_path = Path(synthetic_folder)
        for img_path in sorted(synthetic_path.glob("*.png")):
            # 假设文件名格式: user_XX_sample_YYY.png
            parts = img_path.stem.split('_')
            if len(parts) >= 2 and parts[0] == 'user':
                label = int(parts[1])
                self.samples.append((img_path, label))
        
        self.transform = transform
        print(f"Loaded synthetic dataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def main():
    """
    完整的分类器实验流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ResNet18分类器实验')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                        default='./latents_cache/data_split.json',
                        help='数据集划分文件')
    parser.add_argument('--synthetic_folder', type=str, default=None,
                        help='合成数据文件夹（可选，用于增强实验）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    print("="*60)
    print("加载数据集")
    print("="*60)
    
    # 训练集（真实图像）
    train_ds = MicroDopplerDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split='train',
        use_latents=False
    )
    
    # 如果提供了合成数据，添加到训练集
    if args.synthetic_folder:
        print("\n添加合成数据到训练集...")
        
        # 与真实图像使用相同的transform
        synthetic_ds = SyntheticDataset(
            args.synthetic_folder,
            transform=train_ds.transform
        )
        
        # 合并数据集
        train_ds = ConcatDataset([train_ds, synthetic_ds])
        print(f"增强后训练集: {len(train_ds)} 张（真实+合成）")
    
    # 测试集（真实图像）
    test_ds = MicroDopplerDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split='test',
        use_latents=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建ResNet18分类器
    print("\n创建ResNet18分类器（不使用预训练）...")
    model = resnet18(pretrained=False, num_classes=31)
    model = model.to(device)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练
    print("\n开始训练...")
    train_classifier(model, train_loader, criterion, optimizer, device, args.epochs)
    
    # 评估
    print("\n评估分类器...")
    accuracy = evaluate_classifier(model, test_loader, device)
    
    return accuracy


if __name__ == '__main__':
    from PIL import Image  # 需要在这里导入
    accuracy = main()

