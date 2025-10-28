"""
创建训练集/测试集划分
======================
严格分离训练集和测试集，保存划分信息供后续使用

划分策略：
- 从每个用户的150张图像中，使用等间隔采样选择50张作为训练集
- 剩余100张作为测试集
- 保存划分信息到JSON文件，确保VQ-GAN和LDM使用相同的训练集
"""

import json
from pathlib import Path
import argparse
import numpy as np


def create_data_split(
    data_path: str,
    num_users: int = 31,
    images_per_user_train: int = 50,
    output_file: str = './data_split.json'
):
    """
    创建并保存数据集划分
    
    Args:
        data_path: 数据集根目录
        num_users: 用户数量
        images_per_user_train: 每用户训练图像数
        output_file: 输出JSON文件路径
    """
    data_path = Path(data_path)
    
    split_info = {
        'data_path': str(data_path),
        'num_users': num_users,
        'images_per_user_train': images_per_user_train,
        'sampling_method': 'stratified_uniform',  # 分层等间隔采样
        'users': {}
    }
    
    total_train = 0
    total_test = 0
    
    print("="*60)
    print("创建数据集划分")
    print("="*60)
    
    for user_id in range(1, num_users + 1):
        user_folder = data_path / f"ID_{user_id}"
        
        if not user_folder.exists():
            print(f"⚠️  User {user_id}: 文件夹不存在，跳过")
            continue
        
        # 收集所有jpg图像（排序确保一致性）
        image_paths = sorted(list(user_folder.glob("*.jpg")))
        total_images = len(image_paths)
        
        if total_images == 0:
            print(f"⚠️  User {user_id}: 没有图像，跳过")
            continue
        
        # 分层等间隔采样（确保首尾覆盖）
        if total_images < images_per_user_train:
            print(f"⚠️  User {user_id}: 只有{total_images}张图像，全部用作训练集")
            train_indices = list(range(total_images))
            test_indices = []
        else:
            # ✅ 使用linspace确保均匀分布且包含首尾
            # 例如：151张采样50张 -> [0, 3, 6, ..., 147, 150]
            train_indices = np.linspace(0, total_images - 1, images_per_user_train, dtype=int).tolist()
            
            # 测试集为剩余的所有图像
            train_set = set(train_indices)
            test_indices = [i for i in range(total_images) if i not in train_set]
        
        # 转换为相对路径（便于移植）
        train_images = [f"ID_{user_id}/{image_paths[i].name}" for i in train_indices]
        test_images = [f"ID_{user_id}/{image_paths[i].name}" for i in test_indices]
        
        # 保存用户信息
        split_info['users'][f'user_{user_id}'] = {
            'user_id': user_id,
            'label': user_id - 1,  # 0-30
            'total_images': total_images,
            'train_images': train_images,
            'test_images': test_images,
            'train_count': len(train_images),
            'test_count': len(test_images),
            'train_indices': train_indices,  # 保存索引便于调试
            'test_indices': test_indices
        }
        
        total_train += len(train_images)
        total_test += len(test_images)
        
        print(f"✓ User {user_id:2d}: {total_images:3d}张 → 训练{len(train_images):2d}张 + 测试{len(test_images):3d}张")
    
    # 添加统计信息
    split_info['statistics'] = {
        'total_users': len(split_info['users']),
        'total_train_images': total_train,
        'total_test_images': total_test,
        'total_images': total_train + total_test,
        'train_ratio': total_train / (total_train + total_test) if (total_train + total_test) > 0 else 0
    }
    
    # 保存到JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("✓ 数据集划分完成！")
    print("="*60)
    print(f"总用户数: {split_info['statistics']['total_users']}")
    print(f"训练集: {total_train}张 ({split_info['statistics']['train_ratio']:.1%})")
    print(f"测试集: {total_test}张 ({1-split_info['statistics']['train_ratio']:.1%})")
    print(f"总计: {total_train + total_test}张")
    print(f"\n划分信息已保存到: {output_path}")
    print("="*60)
    
    return split_info


def verify_split(split_file: str):
    """验证数据集划分的正确性"""
    with open(split_file, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    
    print("\n" + "="*60)
    print("验证数据集划分")
    print("="*60)
    
    all_train = []
    all_test = []
    
    for user_key, user_info in split_info['users'].items():
        train_set = set(user_info['train_images'])
        test_set = set(user_info['test_images'])
        
        # 检查1: 训练集和测试集无交集
        intersection = train_set & test_set
        if intersection:
            print(f"❌ {user_key}: 训练集和测试集有重叠！{intersection}")
            return False
        
        # 检查2: 训练集数量正确
        if user_info['train_count'] != len(train_set):
            print(f"❌ {user_key}: 训练集数量不一致！")
            return False
        
        all_train.extend(user_info['train_images'])
        all_test.extend(user_info['test_images'])
    
    # 检查3: 全局无重复
    if len(all_train) != len(set(all_train)):
        print("❌ 训练集存在重复图像！")
        return False
    
    if len(all_test) != len(set(all_test)):
        print("❌ 测试集存在重复图像！")
        return False
    
    # 检查4: 训练集和测试集全局无交集
    global_train = set(all_train)
    global_test = set(all_test)
    global_intersection = global_train & global_test
    if global_intersection:
        print(f"❌ 训练集和测试集有重叠：{len(global_intersection)}张")
        return False
    
    print("✓ 所有检查通过！")
    print(f"  - 训练集：{len(all_train)}张（无重复，无交叉）")
    print(f"  - 测试集：{len(all_test)}张（无重复，无交叉）")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='创建训练集/测试集划分')
    parser.add_argument('--data_path', type=str, 
                       default=r'D:\Ysj\新建文件夹\VA-VAE\dataset\organized_gait_dataset\kaggle\working\organized_gait_dataset\Normal_line',
                       help='数据集根目录')
    parser.add_argument('--num_users', type=int, default=31,
                       help='用户数量')
    parser.add_argument('--images_per_user_train', type=int, default=50,
                       help='每用户训练图像数')
    parser.add_argument('--output', type=str, 
                       default='./data_split.json',
                       help='输出JSON文件')
    parser.add_argument('--verify', action='store_true',
                       help='验证已有的划分文件')
    
    args = parser.parse_args()
    
    if args.verify:
        # 验证模式
        verify_split(args.output)
    else:
        # 创建模式
        split_info = create_data_split(
            args.data_path,
            args.num_users,
            args.images_per_user_train,
            args.output
        )
        
        # 自动验证
        print("\n自动验证划分...")
        verify_split(args.output)


if __name__ == '__main__':
    main()

