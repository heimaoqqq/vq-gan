"""
数据集加载工具 - 用于分类器实验
根据data_split.json加载训练集/测试集
"""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MicroDopplerDataset(Dataset):
    """
    微多普勒数据集 - 用于ResNet18分类器实验
    
    支持两种模式:
      1. 像素空间: 加载原始JPG图像（用于分类器）
      2. 潜在空间: 加载预编码的潜在表示（可选）
    """
    def __init__(self, data_root, split_file, split='train', 
                 use_latents=False, latents_cache_folder='./latents_cache'):
        """
        Args:
            data_root: 数据集根目录
            split_file: data_split.json路径
            split: 'train' 或 'test'
            use_latents: 是否使用潜在表示（通常分类器用原始图像）
            latents_cache_folder: 潜在表示缓存文件夹
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.use_latents = use_latents
        self.latents_cache_folder = Path(latents_cache_folder)
        
        # 加载数据集划分信息
        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        
        # 收集样本
        self.samples = []
        
        for user_key, user_info in split_info['users'].items():
            label = user_info['label']
            
            if split == 'train':
                image_files = user_info['train_images']
            elif split == 'test':
                image_files = user_info['test_images']
            else:
                raise ValueError(f"split must be 'train' or 'test', got {split}")
            
            for img_file in image_files:
                # Normalize path separators for cross-platform compatibility
                img_file = img_file.replace('\\', '/')
                img_path = self.data_root / img_file
                self.samples.append((img_path, label))
        
        print(f"Loaded {split} set: {len(self.samples)} images, {split_info['num_users']} users")
        
        # 图像预处理（用于ResNet18）
        if not use_latents:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                   std=[0.229, 0.224, 0.225])     # 虽然不用预训练，但标准化有帮助
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        if self.use_latents:
            # 加载潜在表示
            cache_filename = f"user_{label:02d}_{img_path.stem}.pt"
            cache_path = self.latents_cache_folder / cache_filename
            data = torch.load(cache_path, map_location='cpu')
        else:
            # 加载原始图像
            img = Image.open(img_path).convert('RGB')
            data = self.transform(img)
        
        return data, label


def load_split_info(split_file):
    """
    加载数据集划分信息
    
    Returns:
        dict: 包含所有划分信息
    """
    with open(split_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_split_summary(split_file):
    """
    打印数据集划分摘要
    """
    info = load_split_info(split_file)
    
    print("="*60)
    print("数据集划分摘要")
    print("="*60)
    print(f"随机种子: {info['seed']}")
    print(f"用户数量: {info['num_users']}")
    print(f"每用户训练集: {info['images_per_user_train']} 张")
    
    total_train = 0
    total_test = 0
    
    print(f"\n{'用户':<10} {'标签':<6} {'总数':<6} {'训练':<6} {'测试':<6}")
    print("-"*60)
    
    for user_key in sorted(info['users'].keys()):
        user_info = info['users'][user_key]
        user_id = user_info['user_id']
        label = user_info['label']
        total = user_info['total_images']
        n_train = len(user_info['train_images'])
        n_test = len(user_info['test_images'])
        
        print(f"{user_key:<10} {label:<6} {total:<6} {n_train:<6} {n_test:<6}")
        
        total_train += n_train
        total_test += n_test
    
    print("-"*60)
    print(f"{'总计':<10} {'':<6} {total_train+total_test:<6} {total_train:<6} {total_test:<6}")
    print("="*60)


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='查看数据集划分信息')
    parser.add_argument('--split_file', type=str, 
                        default='./latents_cache/data_split.json',
                        help='data_split.json路径')
    parser.add_argument('--test_load', action='store_true',
                        help='测试加载数据集')
    args = parser.parse_args()
    
    # 打印摘要
    print_split_summary(args.split_file)
    
    # 测试加载（如果指定）
    if args.test_load:
        print("\n测试加载数据集...")
        
        data_root = Path(args.split_file).parent.parent / 'Normal_line'
        
        # 加载训练集
        train_ds = MicroDopplerDataset(
            data_root=data_root,
            split_file=args.split_file,
            split='train',
            use_latents=False
        )
        
        # 加载测试集
        test_ds = MicroDopplerDataset(
            data_root=data_root,
            split_file=args.split_file,
            split='test',
            use_latents=False
        )
        
        print(f"\n训练集: {len(train_ds)} 样本")
        print(f"测试集: {len(test_ds)} 样本")
        
        # 测试读取
        img, label = train_ds[0]
        print(f"\n样本形状: {img.shape}")
        print(f"标签: {label}")
        print("\n✓ 数据集加载测试通过！")

