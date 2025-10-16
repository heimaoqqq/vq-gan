# 微多普勒时频图像条件生成 - 完整指南

基于Latent Diffusion和Classifier-Free Guidance的条件DDPM生成系统

---

## 📋 项目概览

**目标**：使用条件DDPM生成微多普勒时频图像，通过合成数据增强ResNet18分类器性能

**技术栈**：
- Latent Diffusion Model (在VAE潜在空间32×32×4训练)
- Classifier-Free Guidance (条件生成31个用户)
- Min-SNR Loss Weighting (小数据集优化)

**数据说明**：
- 31个用户，每用户图像数量不固定（约140-160张）
- DDPM训练：每用户固定取50张（随机打乱后，共1550张）
- 分类器测试：每用户剩余图像（90-110张不等，共约3100张）
- 固定随机种子42，完全可复现

**硬件**：Kaggle P100 16GB

---

## 🚀 快速开始

### 完整工作流（3步）

```bash
# === Kaggle Notebook ===

# 1. 克隆和安装
!git clone https://github.com/heimaoqqq/denoising_diffusion_pytorch.git
%cd denoising_diffusion_pytorch
!pip install -e .

# 2. 预编码（30分钟，包含训练+测试集）
!python preprocess_latents.py

# 3. 训练DDPM（4小时）
!python train_latent_cfg.py
```

---

## 📂 输出文件说明

### 预编码输出

```
latents_cache/
├── data_split.json           ← 数据集划分信息（重要！）
├── user_00_*.pt              ← 用户1的潜在表示
├── user_01_*.pt              ← 用户2的潜在表示
...
└── user_30_*.pt              ← 用户31的潜在表示

总计: 约4600-5000个.pt文件（取决于每用户实际图像数）
  - 训练集: 1550个（31×50，固定）
  - 测试集: 约3000-3400个（剩余图像，数量不固定）
大小: ~150MB
```

### data_split.json 用途

**包含信息**：
- 每个用户的训练集图像路径（50张）
- 每个用户的测试集图像路径（100张）
- 随机种子、用户标签等元数据

**用于**：
1. ✅ 确保DDPM训练和分类器训练使用相同的训练集
2. ✅ 加载测试集进行分类器评估
3. ✅ 可复现实验（记录了确切的划分）
4. ✅ 分析每个用户的数据分布

### DDPM训练输出

```
results/
├── model-*.pt               ← DDPM检查点
└── sample-*.png             ← 生成样本（监控用）
```

### 合成数据输出

```
synthetic_data/
├── checkpoint_50/
│   ├── user_00_sample_000.png
│   ├── user_00_sample_001.png
│   ...
│   └── user_30_sample_049.png  (31用户×50张)
├── checkpoint_75/
...
```

---

## 🔬 分类器实验流程

### 方法1：使用示例脚本（推荐）

```bash
# 基准实验（仅真实数据）
!python classifier_experiment_example.py \
    --data_root /kaggle/input/organized-gait-dataset/Normal_line \
    --split_file latents_cache/data_split.json \
    --epochs 100

# 输出:
# 加载数据集
# Loaded train set: 1550 images, 31 users
# Loaded test set: 3100 images, 31 users
# 整体准确率: XX.XX% (baseline)

# 增强实验（真实+合成）
!python classifier_experiment_example.py \
    --data_root /kaggle/input/organized-gait-dataset/Normal_line \
    --split_file latents_cache/data_split.json \
    --synthetic_folder synthetic_data/checkpoint_75 \
    --epochs 100

# 输出:
# 增强后训练集: 3100 张（真实1550 + 合成1550）
# 整体准确率: YY.YY% (enhanced)
```

### 方法2：自定义脚本

```python
from load_dataset import MicroDopplerDataset
from torch.utils.data import DataLoader

# 1. 加载训练集（使用data_split.json）
train_ds = MicroDopplerDataset(
    data_root='/kaggle/input/organized-gait-dataset/Normal_line',
    split_file='latents_cache/data_split.json',
    split='train',      # 'train' 或 'test'
    use_latents=False   # False=原始图像，True=潜在表示
)

# 2. 加载测试集
test_ds = MicroDopplerDataset(
    data_root='/kaggle/input/organized-gait-dataset/Normal_line',
    split_file='latents_cache/data_split.json',
    split='test'
)

# 3. 创建DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 4. 训练您的ResNet18
from torchvision.models import resnet18
import torch.nn as nn

model = resnet18(pretrained=False, num_classes=31)
# ... 训练代码 ...

# 5. 评估
accuracy = evaluate(model, test_loader)
```

---

## 📊 数据集使用说明

### DDPM训练（train_latent_cfg.py）

```python
使用数据:
  ✓ 训练集: 1550张（data_split.json中的train_images）
  ✗ 测试集: 不使用（对DDPM完全不可见）

数据加载:
  - LatentDataset自动使用与preprocess相同的划分逻辑
  - 确保选择相同的1550张训练图像
  - 直接读取latents_cache/中的.pt文件
```

### 分类器实验（classifier_experiment_example.py）

```python
基准分类器:
  训练: 真实训练集1550张（data_split.json的train_images）
  测试: 真实测试集3100张（data_split.json的test_images）
  
增强分类器:
  训练: 真实训练集1550 + 合成数据N张
  测试: 真实测试集3100张（相同的测试集）
  
关键:
  ✓ 两个实验使用完全相同的训练/测试划分
  ✓ 测试集始终独立，从未用于DDPM训练
  ✓ 可以公平对比性能提升
```

---

## 🔍 验证数据一致性

### 检查DDPM是否使用了正确的训练集

```python
# 方法1: 查看data_split.json
!python load_dataset.py --split_file latents_cache/data_split.json

# 方法2: 检查缓存文件数量
!ls latents_cache/user_*.pt | wc -l
# 应该输出: 4650 (如果encode_all=True)
# 或: 1550 (如果只编码训练集)

# 方法3: 验证特定用户的划分
import json
with open('latents_cache/data_split.json') as f:
    split = json.load(f)

# 查看用户1的划分
print(f"用户1训练集: {len(split['users']['ID_1']['train_images'])} 张")
print(f"用户1测试集: {len(split['users']['ID_1']['test_images'])} 张")
```

---

## ⚙️ 高级选项

### 只编码训练集（节省时间和空间）

```bash
# 如果只需要DDPM训练，不立即做分类器实验
!python preprocess_latents.py --no-encode_all

# 输出:
# Dataset split:
#   Train: 1550 images
#   Test: 3100 images
#   Total to encode: 1550 images (只编码训练集)
```

### 后续补充编码测试集

```python
# 之后需要分类器实验时，可以只编码测试集
# 需要手动实现或重新运行preprocess_latents.py --encode_all
```

---

## 📈 实验结果示例

### 预期结果格式

```
实验报告
============================================================
基准分类器（仅真实1550张）:
  准确率: 75.32%

增强分类器结果:
  Checkpoint 50:  77.45% (+2.13%)
  Checkpoint 75:  79.21% (+3.89%) ← 最佳
  Checkpoint 100: 78.56% (+3.24%)
  Checkpoint 125: 77.89% (+2.57%)
  Checkpoint 150: 77.12% (+1.80%) ← 可能过拟合

结论:
  ✓ 合成数据有效提升分类器性能
  ✓ 最佳检查点: 75 (milestone)
  ✓ 最大提升: +3.89%
  ⚠ 后期检查点性能下降，说明DDPM可能过拟合
```

---

## 📁 完整文件清单

### 核心脚本

| 文件 | 功能 | 何时使用 |
|------|------|---------|
| **preprocess_latents.py** | 预编码+数据划分 | 训练前（一次性） |
| **train_latent_cfg.py** | 训练DDPM | 预编码后 |
| **generate.py** | 生成合成数据 | DDPM训练后 |
| **load_dataset.py** | 加载划分后的数据集 | 分类器实验 |
| **classifier_experiment_example.py** | 分类器实验示例 | 评估阶段 |

### 辅助工具

| 文件 | 功能 |
|------|------|
| **test_vae_range.py** | 测试VAE输出范围 |
| **monitor_training.py** | 监控DDPM训练 |

### 文档

| 文件 | 内容 |
|------|------|
| **START_HERE.md** | 快速开始指南 |
| **CLASSIFIER_EXPERIMENT.md** | 分类器实验详细说明 |
| **ENCODING_COMPARISON.md** | 编码方式对比 |

---

## 🎯 关键要点

### ✅ 数据集划分保证

1. **固定随机种子（42）** - 完全可复现
2. **data_split.json记录** - 明确每张图的用途
3. **DDPM和分类器使用相同划分** - 确保一致性
4. **测试集严格隔离** - 对DDPM不可见

### ✅ 实验流程保证

```
预编码阶段:
  ├─ 划分训练/测试集（固定种子）
  ├─ 编码所有图像到潜在空间
  └─ 保存data_split.json

DDPM训练:
  └─ 仅使用训练集（1550张）

分类器实验:
  ├─ 基准: 训练集1550 → 测试集3100
  └─ 增强: 训练集1550+合成N → 测试集3100
```

---

**现在数据集划分信息会被正确保存，方便后续所有实验！** ✅🎉

