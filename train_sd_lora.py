"""
SD+LoRA训练脚本 - Python版本
直接调用Diffusers官方训练脚本，完整版不简化
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def train_sd_lora(
    model_name="runwayml/stable-diffusion-v1-5",
    dataset_path="./sd_lora_dataset",
    val_dataset_path=None,
    output_dir="./sd_lora_output",
    resolution=512,
    train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=100,
    learning_rate=1e-4,
    lr_scheduler="constant",
    lr_warmup_steps=0,
    seed=42,
    lora_rank=8,
    lora_alpha=8,
    validation_prompt="user 0",
    validation_epochs=10,
    num_validation_images=4,
    checkpointing_steps=500,
    mixed_precision="fp16",
    gradient_checkpointing=True,
    use_8bit_adam=False,
    center_crop=True,
    random_flip=True,
    report_to="tensorboard"
):
    """
    训练SD+LoRA模型
    
    Args:
        model_name: 预训练模型名称
        dataset_path: 训练集路径（包含images/和metadata.jsonl）
        val_dataset_path: 验证集路径（可选，如果为None则只用validation_prompt）
        output_dir: 输出目录
        resolution: 图像分辨率
        train_batch_size: 训练batch size
        gradient_accumulation_steps: 梯度累积步数
        num_train_epochs: 训练epoch数
        learning_rate: 学习率
        lr_scheduler: 学习率调度器
        lr_warmup_steps: 预热步数
        seed: 随机种子
        lora_rank: LoRA秩
        lora_alpha: LoRA缩放因子
        validation_prompt: 验证提示词
        validation_epochs: 验证频率（epoch）
        num_validation_images: 每次验证生成的图像数
        checkpointing_steps: 保存检查点频率（步数）
        mixed_precision: 混合精度训练
        gradient_checkpointing: 是否使用梯度检查点
        use_8bit_adam: 是否使用8bit Adam
        center_crop: 是否中心裁剪
        random_flip: 是否随机翻转
        report_to: 日志记录工具
    """
    
    print("="*60)
    print("SD+LoRA训练 - 微多普勒时频图生成")
    print("="*60)
    print()
    print("配置:")
    print(f"  模型: {model_name}")
    print(f"  训练集: {dataset_path}")
    if val_dataset_path:
        print(f"  验证集: {val_dataset_path}")
    print(f"  输出: {output_dir}")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"  Batch size: {train_batch_size} (有效: {train_batch_size * gradient_accumulation_steps})")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  验证频率: 每{validation_epochs}个epoch")
    print()
    
    # 检查数据集是否存在
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请先运行: python prepare_sd_lora_dataset.py")
        sys.exit(1)
    
    metadata_file = dataset_path / "metadata.jsonl"
    if not metadata_file.exists():
        print(f"错误: metadata.jsonl不存在: {metadata_file}")
        print("请先运行: python prepare_sd_lora_dataset.py")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建训练命令
    train_script = Path("./diffusers/examples/text_to_image/train_text_to_image_lora.py")
    
    if not train_script.exists():
        print(f"错误: 训练脚本不存在: {train_script}")
        print("请确保已克隆Diffusers库到 ./diffusers/")
        sys.exit(1)
    
    # 基础命令
    cmd = [
        "accelerate", "launch",
        f"--mixed_precision={mixed_precision}",
        str(train_script),
        f"--pretrained_model_name_or_path={model_name}",
        f"--train_data_dir={dataset_path}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--num_train_epochs={num_train_epochs}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        f"--lr_warmup_steps={lr_warmup_steps}",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
        f"--validation_prompt={validation_prompt}",
        f"--validation_epochs={validation_epochs}",
        f"--num_validation_images={num_validation_images}",
        f"--checkpointing_steps={checkpointing_steps}",
        f"--rank={lora_rank}",
        f"--report_to={report_to}",
    ]
    
    # 添加LoRA alpha（如果不等于rank）
    if lora_alpha != lora_rank:
        cmd.append(f"--lora_alpha={lora_alpha}")
    
    # 可选参数
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    
    if use_8bit_adam:
        cmd.append("--use_8bit_adam")
    
    if center_crop:
        cmd.append("--center_crop")
    
    if random_flip:
        cmd.append("--random_flip")
    
    # 打印完整命令
    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("="*60)
    print("开始训练...")
    print("="*60)
    print()
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        
        print()
        print("="*60)
        print("训练完成！")
        print("="*60)
        print(f"输出目录: {output_dir}")
        print(f"LoRA权重: {output_dir}/pytorch_lora_weights.safetensors")
        print()
        print("生成图像:")
        print(f"  python generate_sd_lora.py --lora_weights {output_dir}/pytorch_lora_weights.safetensors --user_id 0")
        print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print("训练失败！")
        print("="*60)
        print(f"错误码: {e.returncode}")
        print()
        print("常见问题:")
        print("1. 显存不足 → 减小batch_size或使用gradient_checkpointing")
        print("2. 数据集格式错误 → 检查metadata.jsonl")
        print("3. 依赖缺失 → pip install accelerate peft transformers")
        print()
        return False
    
    except KeyboardInterrupt:
        print()
        print("="*60)
        print("训练被用户中断")
        print("="*60)
        print(f"检查点已保存到: {output_dir}")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(description='SD+LoRA训练 - Python版本')
    
    # 路径参数
    parser.add_argument('--model_name', type=str,
                        default='runwayml/stable-diffusion-v1-5',
                        help='预训练模型名称')
    parser.add_argument('--dataset_path', type=str,
                        default='./sd_lora_dataset',
                        help='训练集路径')
    parser.add_argument('--val_dataset_path', type=str,
                        default=None,
                        help='验证集路径（可选）')
    parser.add_argument('--output_dir', type=str,
                        default='./sd_lora_output',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--resolution', type=int, default=512,
                        help='图像分辨率')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='训练batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--num_train_epochs', type=int, default=100,
                        help='训练epoch数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 
                                'polynomial', 'constant', 'constant_with_warmup'],
                        help='学习率调度器')
    parser.add_argument('--lr_warmup_steps', type=int, default=0,
                        help='学习率预热步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA秩（越大表达能力越强，推荐4-16）')
    parser.add_argument('--lora_alpha', type=int, default=8,
                        help='LoRA缩放因子（通常等于rank）')
    
    # 验证和保存
    parser.add_argument('--validation_prompt', type=str, default='user 0',
                        help='验证提示词')
    parser.add_argument('--validation_epochs', type=int, default=5,
                        help='每N个epoch验证一次（默认5）')
    parser.add_argument('--num_validation_images', type=int, default=4,
                        help='每次验证生成的图像数')
    parser.add_argument('--checkpointing_steps', type=int, default=500,
                        help='每N步保存一次检查点')
    
    # 优化参数
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='使用梯度检查点节省显存')
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing',
                        action='store_false',
                        help='不使用梯度检查点')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False,
                        help='使用8bit Adam优化器（需要bitsandbytes）')
    
    # 数据增强
    parser.add_argument('--center_crop', action='store_true', default=True,
                        help='中心裁剪')
    parser.add_argument('--no_center_crop', dest='center_crop', action='store_false',
                        help='不使用中心裁剪')
    parser.add_argument('--random_flip', action='store_true', default=True,
                        help='随机水平翻转')
    parser.add_argument('--no_random_flip', dest='random_flip', action='store_false',
                        help='不使用随机翻转')
    
    # 日志
    parser.add_argument('--report_to', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb', 'all'],
                        help='日志记录工具')
    
    args = parser.parse_args()
    
    # 运行训练
    success = train_sd_lora(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        output_dir=args.output_dir,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        validation_prompt=args.validation_prompt,
        validation_epochs=args.validation_epochs,
        num_validation_images=args.num_validation_images,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        use_8bit_adam=args.use_8bit_adam,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        report_to=args.report_to
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
