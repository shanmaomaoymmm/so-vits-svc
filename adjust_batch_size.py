#!/usr/bin/env python3
"""
Batch Size调整工具
交互式帮助用户调整训练配置以保持训练等效性
"""

import json
import os
import math

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件格式错误 {e}")
        return None

def save_config(config, config_path):
    """保存配置文件"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"错误: 保存配置文件失败 {e}")
        return False

def calculate_adjustment_options(original_bs, target_bs, current_epochs, current_lr):
    """计算调整选项"""
    scale_factor = original_bs / target_bs
    
    options = {
        'gradient_accumulation': {
            'steps': math.ceil(scale_factor),
            'effective_bs': target_bs * math.ceil(scale_factor),
            'description': f"使用梯度累积，保持完全相同的训练效果"
        },
        'epoch_adjustment': {
            'epochs': math.ceil(current_epochs * scale_factor),
            'additional_epochs': math.ceil(current_epochs * scale_factor) - current_epochs,
            'description': f"增加训练轮数来补偿batch_size减小"
        },
        'lr_adjustment': {
            'new_lr': current_lr * (target_bs / original_bs),
            'description': f"调整学习率以适应新的batch_size"
        },
        'hybrid_approach': {
            'accum_steps': max(1, round(math.sqrt(scale_factor))),
            'new_epochs': math.ceil(current_epochs * math.sqrt(scale_factor)),
            'description': f"组合方案：适度增加累积步数和训练轮数"
        }
    }
    
    return options

def interactive_adjustment():
    """交互式调整"""
    print("=== SO-VITS-SVC Batch Size 调整工具 ===\n")
    
    # 获取配置文件路径
    config_path = input("请输入配置文件路径 (默认: configs/config.json): ").strip()
    if not config_path:
        config_path = "configs/config.json"
    
    # 加载配置
    config = load_config(config_path)
    if not config:
        return
    
    train_config = config.get('train', {})
    current_bs = train_config.get('batch_size', 16)
    current_accum = train_config.get('grad_accumulation_steps', 1)
    current_epochs = train_config.get('epochs', 10000)
    current_lr = train_config.get('learning_rate', 0.0001)
    
    effective_bs = current_bs * current_accum
    
    print(f"\n当前配置:")
    print(f"  batch_size: {current_bs}")
    print(f"  grad_accumulation_steps: {current_accum}")
    print(f"  实际有效batch_size: {effective_bs}")
    print(f"  epochs: {current_epochs}")
    print(f"  learning_rate: {current_lr}")
    
    # 获取目标batch_size
    while True:
        try:
            target_bs = int(input(f"\n请输入目标batch_size (当前: {current_bs}): "))
            if target_bs <= 0:
                print("batch_size必须大于0")
                continue
            if target_bs > current_bs:
                print("注意: 增大batch_size可能需要更多显存")
            break
        except ValueError:
            print("请输入有效的数字")
    
    # 计算调整选项
    options = calculate_adjustment_options(effective_bs, target_bs, current_epochs, current_lr)
    
    print(f"\n=== 调整方案 ===")
    print(f"从有效batch_size {effective_bs} 调整到 {target_bs}")
    print(f"缩放因子: {effective_bs/target_bs:.2f}")
    
    print(f"\n方案1 - 纯梯度累积:")
    print(f"  grad_accumulation_steps: {options['gradient_accumulation']['steps']}")
    print(f"  实际有效batch_size: {options['gradient_accumulation']['effective_bs']}")
    print(f"  说明: {options['gradient_accumulation']['description']}")
    
    print(f"\n方案2 - 纯epoch调整:")
    print(f"  新epochs: {options['epoch_adjustment']['epochs']}")
    print(f"  增加epochs: {options['epoch_adjustment']['additional_epochs']}")
    print(f"  说明: {options['epoch_adjustment']['description']}")
    
    print(f"\n方案3 - 学习率调整:")
    print(f"  新learning_rate: {options['lr_adjustment']['new_lr']:.6f}")
    print(f"  说明: {options['lr_adjustment']['description']}")
    
    print(f"\n方案4 - 组合方案:")
    print(f"  grad_accumulation_steps: {options['hybrid_approach']['accum_steps']}")
    print(f"  新epochs: {options['hybrid_approach']['new_epochs']}")
    print(f"  说明: {options['hybrid_approach']['description']}")
    
    # 选择方案
    print(f"\n请选择调整方案:")
    print("1. 方案1 - 纯梯度累积 (推荐)")
    print("2. 方案2 - 纯epoch调整")
    print("3. 方案3 - 学习率调整")
    print("4. 方案4 - 组合方案")
    print("5. 自定义调整")
    print("0. 取消")
    
    while True:
        choice = input("请输入选择 (0-5): ").strip()
        if choice == '0':
            print("取消调整")
            return
        elif choice in ['1', '2', '3', '4']:
            break
        elif choice == '5':
            # 自定义调整
            custom_adjustment(config, target_bs)
            return
        else:
            print("请输入有效选项 (0-5)")
    
    # 应用选择的方案
    if choice == '1':
        apply_gradient_accumulation(config, target_bs, options['gradient_accumulation']['steps'])
    elif choice == '2':
        apply_epoch_adjustment(config, options['epoch_adjustment']['epochs'])
    elif choice == '3':
        apply_lr_adjustment(config, options['lr_adjustment']['new_lr'])
    elif choice == '4':
        apply_hybrid_approach(config, target_bs, 
                            options['hybrid_approach']['accum_steps'],
                            options['hybrid_approach']['new_epochs'])
    
    # 保存配置
    print(f"\n新的配置:")
    new_train_config = config.get('train', {})
    print(f"  batch_size: {new_train_config.get('batch_size')}")
    print(f"  grad_accumulation_steps: {new_train_config.get('grad_accumulation_steps', 1)}")
    print(f"  epochs: {new_train_config.get('epochs')}")
    print(f"  learning_rate: {new_train_config.get('learning_rate')}")
    
    save_choice = input(f"\n是否保存到 {config_path}? (y/n): ").strip().lower()
    if save_choice == 'y':
        if save_config(config, config_path):
            print("配置保存成功!")
        else:
            print("配置保存失败!")

def apply_gradient_accumulation(config, new_bs, accum_steps):
    """应用梯度累积方案"""
    config['train']['batch_size'] = new_bs
    config['train']['grad_accumulation_steps'] = accum_steps

def apply_epoch_adjustment(config, new_epochs):
    """应用epoch调整方案"""
    config['train']['epochs'] = new_epochs

def apply_lr_adjustment(config, new_lr):
    """应用学习率调整方案"""
    config['train']['learning_rate'] = new_lr

def apply_hybrid_approach(config, new_bs, accum_steps, new_epochs):
    """应用组合方案"""
    config['train']['batch_size'] = new_bs
    config['train']['grad_accumulation_steps'] = accum_steps
    config['train']['epochs'] = new_epochs

def custom_adjustment(config, target_bs):
    """自定义调整"""
    print(f"\n自定义调整 (目标batch_size: {target_bs})")
    
    # 设置新的batch_size
    config['train']['batch_size'] = target_bs
    
    # 询问是否调整其他参数
    accum = input("是否设置梯度累积步数? (当前: 1, 回车跳过): ").strip()
    if accum:
        try:
            config['train']['grad_accumulation_steps'] = int(accum)
        except ValueError:
            print("无效输入，保持默认值")
    
    epochs = input("是否调整训练轮数? (当前: 10000, 回车跳过): ").strip()
    if epochs:
        try:
            config['train']['epochs'] = int(epochs)
        except ValueError:
            print("无效输入，保持默认值")
    
    lr = input("是否调整学习率? (当前: 0.0001, 回车跳过): ").strip()
    if lr:
        try:
            config['train']['learning_rate'] = float(lr)
        except ValueError:
            print("无效输入，保持默认值")

def quick_calculate():
    """快速计算模式"""
    print("=== 快速计算模式 ===")
    
    try:
        original_bs = int(input("原始有效batch_size: "))
        target_bs = int(input("目标batch_size: "))
        epochs = int(input("当前epochs: "))
        lr = float(input("当前learning_rate: "))
        
        options = calculate_adjustment_options(original_bs, target_bs, epochs, lr)
        
        print(f"\n计算结果:")
        print(f"梯度累积步数: {options['gradient_accumulation']['steps']}")
        print(f"调整后epochs: {options['epoch_adjustment']['epochs']}")
        print(f"调整后学习率: {options['lr_adjustment']['new_lr']:.6f}")
        
    except ValueError:
        print("输入格式错误")

if __name__ == "__main__":
    print("选择模式:")
    print("1. 交互式调整 (推荐)")
    print("2. 快速计算")
    
    mode = input("请选择模式 (1/2): ").strip()
    
    if mode == '1':
        interactive_adjustment()
    elif mode == '2':
        quick_calculate()
    else:
        print("无效选择")