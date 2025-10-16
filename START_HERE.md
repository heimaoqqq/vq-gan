# 🚀 微多普勒图像生成项目 - 启动指南

## 📋 任务概览

**目标**: 使用条件DDPM生成微多普勒时频图像，增强分类器性能

**数据**: 
- 31用户 × 150张/用户
- 训练DDPM: 31×50=1550张
- 测试分类器: 31×100=3100张

**环境**: Kaggle (P100 16GB)

---

## ✅ 预检查清单

在开始之前，确认以下信息：

### 1. 数据路径
```
□ VAE权重: /kaggle/input/kl-vae-best-pt/kl_vae_best.pt
□ 数据集: /kaggle/input/organized-gait-dataset/Normal_line/ID_{1-31}/
□ 每个ID文件夹包含约150张JPG图像
□ 图像规格: 256×256 RGB
```

### 2. 项目文件
```
□ denoising_diffusion_pytorch/classifier_free_guidance.py [已修改]
□ vae/kl_vae.py [VAE模型定义]
□ train_latent_cfg.py [训练脚本]
□ generate.py [生成脚本]
□ test_vae_range.py [VAE测试脚本]
□ monitor_training.py [监控工具]
□ TRAINING_CHECKLIST.md [训练指南]
```

---

## 🔧 启动步骤

### 步骤0: 克隆并安装（Kaggle环境）

```bash
# Kaggle Notebook

# 1. 克隆仓库
!git clone https://github.com/YOUR_USERNAME/denoising-diffusion.git
%cd denoising-diffusion

# 2. 安装依赖
!pip install -e .

# 3. 验证安装
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

### ⚠️ 步骤1: VAE输出范围测试（必须先做！）

**为什么**：确定`auto_normalize`参数设置

```bash
# 运行VAE测试
!python test_vae_range.py \
    --vae_path /kaggle/input/kl-vae-best-pt/kl_vae_best.pt \
    --data_path /kaggle/input/organized-gait-dataset/Normal_line \
    --device cuda
```

**实际输出**（已验证）：
```
VAE潜在表示统计信息：
形状: [100, 4, 32, 32]  ← 注意：32×32，8倍下采样
最小值: -0.911665
最大值: 0.918169
均值: -0.000989
标准差: 0.179068

判断与建议：
✗ 潜在表示NOT在[0, 1]范围内
  → 实际范围: [-0.9117, 0.9182]
  → 必须使用 auto_normalize=False ✓
  → DDPM将在原始范围训练

✓ 重建质量良好（MSE: 0.002646）
```

**关键决策**：
```python
根据输出调整 train_latent_cfg.py 的 Config 类：

# 第89行附近
if 输出显示潜在表示在[0,1]范围:
    auto_normalize = True  # 需要归一化
else:
    auto_normalize = False  # 不需要归一化（默认）
```

**重要**：
- ✅ 如果忘记此步骤，使用错误的auto_normalize会导致训练失败！
- ✅ 记录测试输出，便于后续排查

---

### 步骤2: 小规模测试（可选但推荐）

**目的**：验证pipeline正确性，避免长时间训练后发现问题

```bash
# 修改 train_latent_cfg.py 的 Config:
# train_num_steps = 1000  # 临时改为1000步测试
# save_and_sample_every = 200  # 更频繁保存

# 运行测试训练
!python train_latent_cfg.py

# 预计时间: ~3分钟
# 检查:
# 1. Loss是否下降？
# 2. 是否生成样本（results/sample-*.png）？
# 3. 缓存是否创建（latents_cache/）？
```

**验证通过标志**：
- ✅ 训练正常开始
- ✅ Loss从高到低下降
- ✅ 生成了几个sample图像
- ✅ 没有报错

**如果测试通过**：
```python
# 恢复正常配置
train_num_steps = 150000
save_and_sample_every = 2000
```

---

### 步骤3: 完整训练

```bash
# 确认auto_normalize已根据步骤1调整
# 确认train_num_steps = 150000

# 开始训练
!python train_latent_cfg.py

# 预计时间: ~8小时
# 显存占用: 6-7GB
```

**训练期间监控**（参考TRAINING_CHECKLIST.md）：
```bash
# 在另一个notebook或终端
!python monitor_training.py

# 人工检查（每1-2小时）：
# 1. Loss是否平滑下降？
# 2. 最新的sample图像质量如何？
```

**关键检查点**：
- 50k步（~2.7小时）: 检查条件控制是否有效
- 75k步（~4小时）: 生成样本详细评估
- 100k步（~5.5小时）: 检查是否过拟合
- 150k步（~8小时）: 完成训练

---

### 步骤4: 生成合成数据

```bash
# 为每个用户生成50张合成图像
!python generate.py \
    --checkpoint results/model-75.pt \
    --all_users \
    --samples_per_user 50 \
    --output_dir synthetic_data/checkpoint_75

# 对多个检查点重复
for milestone in 50 75 100; do
    python generate.py \
        --checkpoint results/model-${milestone}.pt \
        --all_users \
        --samples_per_user 50 \
        --output_dir synthetic_data/checkpoint_${milestone}
done
```

---

### 步骤5: 分类器评估

```python
# 您的ResNet18分类实验

# 1. 基准分类器
真实训练集 = 加载31用户×50张（DDPM训练用的那1550张）
测试集 = 加载31用户×100张（保留的测试集）

分类器_baseline = ResNet18(num_classes=31, pretrained=False)
训练(分类器_baseline, 真实训练集, epochs=100)
accuracy_baseline = 评估(分类器_baseline, 测试集)

# 2. 增强分类器（对每个检查点）
for checkpoint in [50, 75, 100]:
    合成训练集 = 加载(f'synthetic_data/checkpoint_{checkpoint}')
    混合训练集 = 真实训练集 + 合成训练集
    
    分类器_enhanced = ResNet18(num_classes=31, pretrained=False)
    训练(分类器_enhanced, 混合训练集, epochs=100)
    accuracy_enhanced = 评估(分类器_enhanced, 测试集)
    
    提升 = accuracy_enhanced - accuracy_baseline
    print(f"Checkpoint {checkpoint}: {提升:.2%} 提升")

# 3. 选择最佳检查点
最佳 = max(结果, key=lambda x: x['accuracy'])
```

---

## 🎯 快速参考

### 核心配置（train_latent_cfg.py）

| 参数 | 值 | 说明 |
|------|---|------|
| **数据** | | |
| num_users | 31 | 用户数 |
| images_per_user_train | 50 | DDPM训练用 |
| latent_size | 32 | 8倍下采样 |
| **模型** | | |
| dim | 48 | ~4M参数 |
| dim_mults | (1,2,4) | 3层 |
| **训练** | | |
| train_batch_size | 8 | P100优化 |
| gradient_accumulate_every | 2 | 有效=16 |
| train_lr | 1e-4 | 标准 |
| train_num_steps | 150000 | ~8小时 |
| **关键优化** | | |
| min_snr_loss_weight | True | ⚠️必须 |
| ema_decay | 0.9995 | 防过拟合 |
| **⚠️ 需确定** | | |
| auto_normalize | ? | 运行步骤1确定 |

### 监控指标

| 阶段 | 步数 | 预期Loss | 图像质量 |
|------|------|---------|---------|
| 早期 | 0-30k | 0.5→0.05 | 噪声→模糊 |
| 中期 | 30-80k | 0.05→0.01 | 模糊→清晰 |
| 后期 | 80-150k | 0.01→0.001 | 清晰→高质量 |

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Loss不下降 | auto_normalize错误 | 重跑步骤1 |
| CUDA OOM | batch太大 | batch=4, grad_accum=4 |
| 模式崩溃 | 条件训练不足 | cond_drop_prob=0.6 |
| 生成模糊 | 训练不足 | 继续训练或↑CFG强度 |

---

## 📊 预期时间线

```
总计: ~10-12小时

步骤1: VAE测试          [5分钟]
步骤2: 小规模测试（可选）  [3分钟]  
步骤3: 完整训练          [8小时]
步骤4: 生成合成数据      [30分钟]
步骤5: 分类器评估        [2-3小时]
```

---

## ✅ 最终检查清单

训练开始前：
```
□ 已运行 test_vae_range.py
□ 已根据结果设置 auto_normalize
□ 已验证数据路径正确
□ 已完成小规模测试（推荐）
□ GPU可用且显存充足
```

训练期间：
```
□ 每1-2小时查看Loss
□ 每2000步检查生成样本
□ 记录异常情况
□ 关键检查点保存
```

训练后：
```
□ 选择2-3个候选检查点
□ 生成合成数据
□ 运行分类器实验
□ 记录结果对比
```

---

## 🆘 获取帮助

遇到问题？按顺序检查：

1. **TRAINING_CHECKLIST.md** - 详细监控指南
2. **monitor_training.py** - 自动诊断
3. **test_vae_range.py** - VAE输出检查
4. **GitHub Issues** - 报告问题

---

## 🎯 成功标志

训练成功的信号：
- ✅ Loss平滑下降到0.001-0.01
- ✅ 生成图像清晰，有微多普勒特征
- ✅ 不同用户生成有明显差异
- ✅ 分类器准确率提升 >0%

**如果分类器准确率提升，说明DDPM学到了有用的特征！** 🎉

---

**下一步：运行步骤1的VAE测试！** 🚀

