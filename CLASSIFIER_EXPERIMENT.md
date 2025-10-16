# 分类器实验完整流程

## 📋 实验目标

评估DDPM生成的合成数据对分类器性能的提升效果。

**对比实验**：
- 基准分类器：仅使用真实训练集（31×50=1550张）
- 增强分类器：真实训练集 + 合成数据（1550+N张）
- 测试集：真实测试集（31×100=3100张）

---

## 🔧 数据集划分信息

### data_split.json 结构

```json
{
  "seed": 42,
  "num_users": 31,
  "images_per_user_train": 50,
  "users": {
    "ID_1": {
      "user_id": 1,
      "label": 0,
      "total_images": 150,
      "train_images": [
        "ID_1/img_001.jpg",
        "ID_1/img_023.jpg",
        ...  // 50张训练图像路径
      ],
      "test_images": [
        "ID_1/img_045.jpg",
        ...  // 100张测试图像路径
      ]
    },
    "ID_2": { ... },
    ...
    "ID_31": { ... }
  }
}
```

### 使用方式

```python
import json

# 加载划分信息
with open('latents_cache/data_split.json', 'r') as f:
    split_info = json.load(f)

# 获取用户1的训练图像
user1_train = split_info['users']['ID_1']['train_images']
# ['ID_1/img_001.jpg', 'ID_1/img_023.jpg', ...]

# 获取用户1的测试图像  
user1_test = split_info['users']['ID_1']['test_images']
# ['ID_1/img_045.jpg', ...]
```

---

## 🚀 完整实验流程

### 步骤1：预编码所有图像

```bash
# 编码训练集+测试集（~30分钟）
!python preprocess_latents.py \
    --vae_path /kaggle/input/kl-vae-best-pt/kl_vae_best.pt \
    --data_path /kaggle/input/organized-gait-dataset/Normal_line \
    --encode_all  # 编码所有图像（包括测试集）

# 输出:
# Dataset split:
#   Train: 1550 images
#   Test: 3100 images (假设每用户150张)
#   Total to encode: 4650 images
#
# 预编码完成！
# 新编码: 4650 张
# 
# 输出文件:
#   潜在表示缓存: ./latents_cache/
#   数据集划分: ./latents_cache/data_split.json
```

### 步骤2：查看数据集划分

```bash
# 查看划分摘要
!python load_dataset.py --split_file latents_cache/data_split.json

# 输出:
# ============================================================
# 数据集划分摘要
# ============================================================
# 随机种子: 42
# 用户数量: 31
# 每用户训练集: 50 张
#
# 用户       标签   总数   训练   测试  
# ------------------------------------------------------------
# ID_1       0      150    50     100   
# ID_2       1      150    50     100   
# ...
# ID_31      30     150    50     100   
# ------------------------------------------------------------
# 总计              4650   1550   3100  
```

### 步骤3：训练DDPM

```bash
# 训练DDPM（使用1550张训练集）
!python train_latent_cfg.py

# DDPM只使用训练集，测试集对DDPM完全不可见
```

### 步骤4：生成合成数据

```bash
# 为每个用户生成50张合成图像
!python generate.py \
    --checkpoint results/model-75.pt \
    --all_users \
    --samples_per_user 50 \
    --output_dir synthetic_data/checkpoint_75

# 输出: synthetic_data/checkpoint_75/user_XX_sample_YYY.png
```

### 步骤5：基准分类器实验

```bash
# 实验1: 仅使用真实训练集
!python classifier_experiment_example.py \
    --data_root /kaggle/input/organized-gait-dataset/Normal_line \
    --split_file latents_cache/data_split.json \
    --epochs 100

# 输出:
# 整体准确率: XX.XX% (baseline)
```

### 步骤6：增强分类器实验

```bash
# 实验2: 真实训练集 + 合成数据
!python classifier_experiment_example.py \
    --data_root /kaggle/input/organized-gait-dataset/Normal_line \
    --split_file latents_cache/data_split.json \
    --synthetic_folder synthetic_data/checkpoint_75 \
    --epochs 100

# 输出:
# 增强后训练集: 3100 张（真实+合成）
# 整体准确率: YY.YY% (enhanced)
#
# 提升: (YY.YY - XX.XX)%
```

---

## 📊 完整对比实验

### Python脚本示例

```python
"""
完整的对比实验流程
评估多个DDPM检查点
"""

from load_dataset import MicroDopplerDataset
from classifier_experiment_example import train_classifier, evaluate_classifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18

# 配置
data_root = '/kaggle/input/organized-gait-dataset/Normal_line'
split_file = './latents_cache/data_split.json'
checkpoints = [50, 75, 100, 125, 150]  # milestone编号
device = 'cuda'

# 加载真实数据集
print("加载真实数据集...")
real_train_ds = MicroDopplerDataset(
    data_root=data_root,
    split_file=split_file,
    split='train'
)

test_ds = MicroDopplerDataset(
    data_root=data_root,
    split_file=split_file,
    split='test'
)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# === 实验1: 基准分类器 ===
print("\n" + "="*60)
print("实验1: 基准分类器（仅真实数据）")
print("="*60)

model_baseline = resnet18(pretrained=False, num_classes=31).to(device)
optimizer = torch.optim.Adam(model_baseline.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(real_train_ds, batch_size=32, shuffle=True)
train_classifier(model_baseline, train_loader, criterion, optimizer, device, epochs=100)

acc_baseline = evaluate_classifier(model_baseline, test_loader, device)
print(f"\n基准准确率: {acc_baseline:.2f}%")

# === 实验2: 对每个检查点评估 ===
results = {}

for milestone in checkpoints:
    print("\n" + "="*60)
    print(f"实验2.{milestone}: 增强分类器（检查点{milestone}）")
    print("="*60)
    
    # 加载合成数据
    synthetic_folder = f'synthetic_data/checkpoint_{milestone}'
    
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    synthetic_ds = SyntheticDataset(synthetic_folder, transform=transform)
    
    # 合并数据集
    mixed_train_ds = ConcatDataset([real_train_ds, synthetic_ds])
    mixed_train_loader = DataLoader(mixed_train_ds, batch_size=32, shuffle=True)
    
    # 训练增强分类器
    model_enhanced = resnet18(pretrained=False, num_classes=31).to(device)
    optimizer = torch.optim.Adam(model_enhanced.parameters(), lr=1e-3)
    
    train_classifier(model_enhanced, mixed_train_loader, criterion, optimizer, device, epochs=100)
    
    # 评估
    acc_enhanced = evaluate_classifier(model_enhanced, test_loader, device)
    
    improvement = acc_enhanced - acc_baseline
    results[milestone] = {
        'accuracy': acc_enhanced,
        'improvement': improvement
    }
    
    print(f"\n检查点{milestone}准确率: {acc_enhanced:.2f}%")
    print(f"提升: {improvement:+.2f}%")

# === 总结 ===
print("\n" + "="*60)
print("实验总结")
print("="*60)
print(f"基准准确率（仅真实）: {acc_baseline:.2f}%")
print("\n各检查点结果:")

for milestone in checkpoints:
    r = results[milestone]
    print(f"  Checkpoint {milestone:3d}: {r['accuracy']:5.2f}% ({r['improvement']:+5.2f}%)")

# 找到最佳检查点
best_milestone = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n最佳检查点: {best_milestone}")
print(f"最佳准确率: {results[best_milestone]['accuracy']:.2f}%")
print(f"最大提升: {results[best_milestone]['improvement']:+.2f}%")


if __name__ == '__main__':
    main()

