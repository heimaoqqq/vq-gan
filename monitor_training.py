"""
训练监控工具
实时分析训练loss和生成样本质量
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


def parse_loss_from_log(log_file):
    """从训练日志中提取loss值"""
    losses = []
    steps = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 假设日志格式: "loss: 0.1234"
            match = re.search(r'loss:\s*([\d.]+)', line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)
                steps.append(len(steps))
    
    return steps, losses


def plot_loss_curve(results_folder='./results'):
    """
    绘制loss曲线并诊断
    
    需要从训练输出中手动记录loss
    或者修改train_latent_cfg.py保存loss历史
    """
    results_path = Path(results_folder)
    
    # 尝试从检查点加载
    checkpoints = sorted(list(results_path.glob('model-*.pt')))
    
    if len(checkpoints) == 0:
        print("未找到检查点文件")
        return
    
    print(f"找到 {len(checkpoints)} 个检查点")
    
    # 简单可视化
    milestones = []
    for cp in checkpoints:
        match = re.search(r'model-(\d+).pt', cp.name)
        if match:
            milestones.append(int(match.group(1)))
    
    print("\n训练进度:")
    print(f"检查点: {milestones}")
    print(f"最新: milestone {max(milestones)}")
    print(f"对应步数: {max(milestones) * 2000} 步")
    print(f"预计完成: {150000 / (max(milestones) * 2000) * 100:.1f}%")


def check_sample_quality(results_folder='./results'):
    """
    检查生成样本质量
    人工检查提示
    """
    results_path = Path(results_folder)
    samples = sorted(list(results_path.glob('sample-*.png')))
    
    if len(samples) == 0:
        print("未找到生成样本")
        return
    
    print(f"\n找到 {len(samples)} 个生成样本")
    print("\n人工检查清单:")
    print("="*60)
    
    for i, sample in enumerate(samples[-5:]):  # 检查最近5个
        milestone = re.search(r'sample-(\d+).png', sample.name).group(1)
        step = int(milestone) * 2000
        
        print(f"\n[{sample.name}] (步数: {step})")
        print("  检查项:")
        print("    □ 图像是否清晰？（模糊→清晰是正常进程）")
        print("    □ 不同用户是否有差异？（应该有明显区别）")
        print("    □ 是否有明显的微多普勒特征？")
        print("    □ 是否所有图像都一样？（模式崩溃警告）")


def training_diagnosis():
    """
    训练诊断指南
    """
    print("\n" + "="*60)
    print("DDPM训练诊断指南")
    print("="*60)
    
    print("\n【正常训练信号】")
    print("  ✓ Loss平滑下降（对数曲线）")
    print("  ✓ 生成样本逐渐清晰")
    print("  ✓ 不同用户生成有明显差异")
    print("  ✓ 同用户多次生成有变化但风格一致")
    
    print("\n【过拟合信号】")
    print("  ✗ Loss非常低（<0.0001）")
    print("  ✗ 生成样本完全复制训练样本")
    print("  ✗ 同用户多次生成几乎完全相同")
    print("  ✗ 缺乏多样性")
    
    print("\n【欠拟合信号】")
    print("  ✗ Loss停止下降但还很高（>0.01）")
    print("  ✗ 生成样本持续模糊")
    print("  ✗ 缺乏细节")
    
    print("\n【模式崩溃信号】")
    print("  ✗ 所有用户生成的图像几乎相同")
    print("  ✗ 生成样本缺乏多样性")
    print("  ✗ 条件控制失效")
    
    print("\n【训练问题排查】")
    print("  1. Loss不下降？")
    print("     → 检查学习率（可能太小）")
    print("     → 检查数据加载（可能有问题）")
    print("     → 检查VAE加载（可能未正确加载）")
    print("  ")
    print("  2. Loss下降后反弹？")
    print("     → 学习率可能太大")
    print("     → 尝试降低学习率或增加梯度裁剪")
    print("  ")
    print("  3. 生成样本不改善？")
    print("     → 检查是否开启了Min-SNR (应该开启)")
    print("     → 检查auto_normalize设置")
    print("     → 尝试增加训练步数")


def suggest_next_steps(results_folder='./results'):
    """
    根据当前进度建议下一步
    """
    results_path = Path(results_folder)
    checkpoints = sorted(list(results_path.glob('model-*.pt')))
    
    if len(checkpoints) == 0:
        print("\n建议: 开始训练！")
        print("  python train_latent_cfg.py")
        return
    
    # 获取最新milestone
    latest = max([int(re.search(r'model-(\d+).pt', cp.name).group(1)) 
                  for cp in checkpoints])
    current_step = latest * 2000
    progress = current_step / 150000 * 100
    
    print(f"\n当前进度: {current_step}/150000 步 ({progress:.1f}%)")
    
    if progress < 30:
        print("\n建议: 继续训练（早期阶段）")
        print("  - 图像可能还比较模糊，这是正常的")
        print("  - 继续训练观察质量提升")
    elif progress < 70:
        print("\n建议: 中期检查点")
        print("  - 可以开始尝试生成样本检查质量")
        print("  - 人工评估用户间差异")
        print(f"  python generate.py --checkpoint results/model-{latest}.pt --all_users --samples_per_user 5 --save_grid")
    else:
        print("\n建议: 后期阶段")
        print("  - 可以开始准备下游分类实验")
        print("  - 生成较多样本用于分类器训练")
        print(f"  python generate.py --checkpoint results/model-{latest}.pt --all_users --samples_per_user 50")
        print("  - 或继续训练到150k步")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--results_folder', type=str, default='./results',
                        help='结果文件夹路径')
    args = parser.parse_args()
    
    print("="*60)
    print("DDPM训练监控")
    print("="*60)
    
    # 检查训练进度
    plot_loss_curve(args.results_folder)
    
    # 检查生成样本
    check_sample_quality(args.results_folder)
    
    # 诊断指南
    training_diagnosis()
    
    # 建议下一步
    suggest_next_steps(args.results_folder)


if __name__ == '__main__':
    main()

