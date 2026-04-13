"""
诊断工具：检查训练数据是否存在可能导致梯度爆炸的问题
"""
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def check_audio_files(dataset_dir="./dataset/44k"):
    """检查音频文件的统计信息"""
    print("=" * 70)
    print("检查音频文件...")
    print("=" * 70)
    
    wav_files = glob(f"{dataset_dir}/*/*.wav", recursive=True)
    if not wav_files:
        print(f"未在 {dataset_dir} 中找到 WAV 文件")
        return
    
    print(f"找到 {len(wav_files)} 个音频文件\n")
    
    # 采样检查（最多检查100个文件）
    sample_files = wav_files[:min(100, len(wav_files))]
    
    durations = []
    peak_values = []
    rms_values = []
    
    for wav_path in tqdm(sample_files, desc="分析音频"):
        try:
            import librosa
            wav, sr = librosa.load(wav_path, sr=None)
            
            duration = len(wav) / sr
            durations.append(duration)
            
            peak = np.max(np.abs(wav))
            peak_values.append(peak)
            
            rms = np.sqrt(np.mean(wav ** 2))
            rms_values.append(rms)
            
        except Exception as e:
            print(f"处理 {wav_path} 时出错: {e}")
    
    if not durations:
        print("没有成功分析的音频文件")
        return
    
    print(f"\n音频时长统计:")
    print(f"  最短: {min(durations):.2f} 秒")
    print(f"  最长: {max(durations):.2f} 秒")
    print(f"  平均: {np.mean(durations):.2f} 秒")
    
    print(f"\n峰值幅度统计:")
    print(f"  最小: {min(peak_values):.6f}")
    print(f"  最大: {max(peak_values):.6f}")
    print(f"  平均: {np.mean(peak_values):.6f}")
    
    print(f"\nRMS 幅度统计:")
    print(f"  最小: {min(rms_values):.6f}")
    print(f"  最大: {max(rms_values):.6f}")
    print(f"  平均: {np.mean(rms_values):.6f}")
    
    # 检查是否有异常值
    problematic_peaks = [p for p in peak_values if p > 1.0 or p < 0.001]
    if problematic_peaks:
        print(f"\n⚠️  警告: 发现 {len(problematic_peaks)} 个音频文件的峰值幅度异常")
        print(f"   期望范围: 0.001 - 1.0")
        print(f"   实际范围: {min(problematic_peaks):.6f} - {max(problematic_peaks):.6f}")
    else:
        print(f"\n✓ 音频峰值幅度正常")
    
    return peak_values, rms_values


def check_f0_files(dataset_dir="./dataset/44k"):
    """检查 F0 文件是否存在异常值"""
    print("\n" + "=" * 70)
    print("检查 F0 文件...")
    print("=" * 70)
    
    f0_files = glob(f"{dataset_dir}/*/*.f0.npy", recursive=True)
    if not f0_files:
        print(f"未在 {dataset_dir} 中找到 F0 文件")
        print(f"提示: 请先运行 preprocess_hubert_f0.py")
        return
    
    print(f"找到 {len(f0_files)} 个 F0 文件\n")
    
    sample_files = f0_files[:min(100, len(f0_files))]
    
    f0_stats = {
        'zero_count': 0,
        'nan_count': 0,
        'inf_count': 0,
        'max_f0': 0,
        'min_nonzero_f0': float('inf'),
        'avg_nonzero_f0': [],
        'problematic_files': []
    }
    
    for f0_path in tqdm(sample_files, desc="分析F0"):
        try:
            f0_data = np.load(f0_path, allow_pickle=True)
            
            # F0 文件保存的是 object 类型的元组 (f0, uv)
            if f0_data.dtype == object:
                f0, uv = f0_data
            else:
                # 如果不是 object 类型，直接使用
                f0 = f0_data
                uv = None
            
            # 确保 f0 是数值类型
            f0 = np.asarray(f0, dtype=np.float64)
            
            # 检查 NaN 和 Inf
            if np.any(np.isnan(f0)):
                f0_stats['nan_count'] += 1
                f0_stats['problematic_files'].append((f0_path, "NaN"))
            
            if np.any(np.isinf(f0)):
                f0_stats['inf_count'] += 1
                f0_stats['problematic_files'].append((f0_path, "Inf"))
            
            # 统计零值
            zero_count = np.sum(f0 == 0)
            if zero_count == len(f0):
                f0_stats['zero_count'] += 1
            
            # 统计非零 F0 值
            nonzero_f0 = f0[f0 > 0]
            if len(nonzero_f0) > 0:
                f0_stats['max_f0'] = max(f0_stats['max_f0'], np.max(nonzero_f0))
                f0_stats['min_nonzero_f0'] = min(f0_stats['min_nonzero_f0'], np.min(nonzero_f0))
                f0_stats['avg_nonzero_f0'].append(np.mean(nonzero_f0))
                
                # 检查异常大的 F0 值（正常人类语音 F0 不超过 1000Hz）
                if np.max(nonzero_f0) > 1100:
                    f0_stats['problematic_files'].append((f0_path, f"F0>{1100}Hz"))
        
        except Exception as e:
            print(f"处理 {f0_path} 时出错: {e}")
            f0_stats['problematic_files'].append((f0_path, f"Error: {e}"))
    
    print(f"F0 统计信息:")
    print(f"  全零 F0 文件数: {f0_stats['zero_count']}")
    print(f"  含 NaN 的文件数: {f0_stats['nan_count']}")
    print(f"  含 Inf 的文件数: {f0_stats['inf_count']}")
    print(f"  最大 F0 值: {f0_stats['max_f0']:.2f} Hz")
    if f0_stats['min_nonzero_f0'] != float('inf'):
        print(f"  最小非零 F0 值: {f0_stats['min_nonzero_f0']:.2f} Hz")
    if f0_stats['avg_nonzero_f0']:
        print(f"  平均 F0 值: {np.mean(f0_stats['avg_nonzero_f0']):.2f} Hz")
    
    if f0_stats['problematic_files']:
        print(f"\n⚠️  警告: 发现 {len(f0_stats['problematic_files'])} 个问题文件:")
        for path, issue in f0_stats['problematic_files'][:10]:  # 只显示前10个
            print(f"   - {path}: {issue}")
        if len(f0_stats['problematic_files']) > 10:
            print(f"   ... 还有 {len(f0_stats['problematic_files']) - 10} 个")
    else:
        print(f"\n✓ F0 数据正常")
    
    return f0_stats


def check_soft_files(dataset_dir="./dataset/44k"):
    """检查 Hubert 特征文件"""
    print("\n" + "=" * 70)
    print("检查 Hubert 特征文件...")
    print("=" * 70)
    
    soft_files = glob(f"{dataset_dir}/*/*.soft.pt", recursive=True)
    if not soft_files:
        print(f"未在 {dataset_dir} 中找到 .soft.pt 文件")
        print(f"提示: 请先运行 preprocess_hubert_f0.py")
        return
    
    print(f"找到 {len(soft_files)} 个特征文件\n")
    
    sample_files = soft_files[:min(50, len(soft_files))]
    
    stats = {
        'nan_count': 0,
        'inf_count': 0,
        'max_value': 0,
        'min_value': float('inf'),
        'problematic_files': []
    }
    
    for soft_path in tqdm(sample_files, desc="分析特征"):
        try:
            features = torch.load(soft_path, map_location='cpu')
            
            if torch.isnan(features).any():
                stats['nan_count'] += 1
                stats['problematic_files'].append((soft_path, "NaN"))
            
            if torch.isinf(features).any():
                stats['inf_count'] += 1
                stats['problematic_files'].append((soft_path, "Inf"))
            
            max_val = torch.max(torch.abs(features)).item()
            stats['max_value'] = max(stats['max_value'], max_val)
            stats['min_value'] = min(stats['min_value'], torch.min(torch.abs(features)).item())
            
            # 检查异常大的值
            if max_val > 100:
                stats['problematic_files'].append((soft_path, f"Max={max_val:.2f}"))
        
        except Exception as e:
            print(f"处理 {soft_path} 时出错: {e}")
            stats['problematic_files'].append((soft_path, f"Error: {e}"))
    
    print(f"特征统计信息:")
    print(f"  含 NaN 的文件数: {stats['nan_count']}")
    print(f"  含 Inf 的文件数: {stats['inf_count']}")
    print(f"  最大绝对值: {stats['max_value']:.6f}")
    if stats['min_value'] != float('inf'):
        print(f"  最小绝对值: {stats['min_value']:.6f}")
    
    if stats['problematic_files']:
        print(f"\n⚠️  警告: 发现 {len(stats['problematic_files'])} 个问题文件:")
        for path, issue in stats['problematic_files'][:10]:
            print(f"   - {path}: {issue}")
    else:
        print(f"\n✓ Hubert 特征正常")
    
    return stats


def check_config(config_path="configs/config.json"):
    """检查配置文件"""
    print("\n" + "=" * 70)
    print("检查配置文件...")
    print("=" * 70)
    
    import json
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"配置文件: {config_path}\n")
    
    issues = []
    
    # 检查学习率
    lr = config.get('train', {}).get('learning_rate', 0)
    if lr > 0.0002:
        issues.append(f"学习率 {lr} 可能过高，建议 <= 0.0001")
    elif lr < 0.00001:
        issues.append(f"学习率 {lr} 可能过低，建议 >= 0.00001")
    
    # 检查 batch size
    batch_size = config.get('train', {}).get('batch_size', 0)
    if batch_size < 2:
        issues.append(f"Batch size {batch_size} 过小，可能导致训练不稳定")
    
    # 检查梯度累积
    grad_accum = config.get('train', {}).get('grad_accumulation_steps', 1)
    effective_batch = batch_size * grad_accum
    if effective_batch < 8:
        issues.append(f"有效 batch size ({effective_batch}) 较小，建议增加 grad_accumulation_steps")
    
    # 检查混合精度设置
    fp16_run = config.get('train', {}).get('fp16_run', False)
    half_type = config.get('train', {}).get('half_type', 'fp16')
    if fp16_run and half_type == 'fp16':
        issues.append("使用 FP16 可能导致数值不稳定，建议使用 BF16（如果硬件支持）")
    
    # 检查损失权重
    c_mel = config.get('train', {}).get('c_mel', 45)
    c_kl = config.get('train', {}).get('c_kl', 1.0)
    if c_mel > 100:
        issues.append(f"c_mel={c_mel} 可能过高，建议 20-50")
    
    # 检查音量增强
    vol_aug = config.get('train', {}).get('vol_aug', False)
    vol_emb = config.get('model', {}).get('vol_embedding', False)
    if vol_aug or vol_emb:
        issues.append("启用了音量增强，这可能导致数值不稳定，如遇到梯度爆炸可尝试禁用")
    
    if issues:
        print("⚠️  配置问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✓ 配置正常")
    
    print(f"\n关键配置:")
    print(f"  学习率: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  梯度累积: {grad_accum}")
    print(f"  有效 batch size: {effective_batch}")
    print(f"  混合精度: {'FP16' if fp16_run and half_type == 'fp16' else 'BF16' if fp16_run else 'Disabled'}")
    print(f"  音量增强: {'Enabled' if vol_aug else 'Disabled'}")
    
    return config


def generate_report(output_file="diagnosis_report.txt"):
    """生成完整的诊断报告"""
    print("\n" + "=" * 70)
    print("开始诊断...")
    print("=" * 70 + "\n")
    
    report_lines = []
    report_lines.append("So-VITS-SVC 训练数据诊断报告")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # 检查各个部分
    audio_result = check_audio_files()
    f0_result = check_f0_files()
    soft_result = check_soft_files()
    config_result = check_config()
    
    # 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    
    issues_found = False
    
    if audio_result:
        peak_values, rms_values = audio_result
        problematic = [p for p in peak_values if p > 1.0 or p < 0.001]
        if problematic:
            print("❌ 音频数据存在问题")
            issues_found = True
        else:
            print("✓ 音频数据正常")
    
    if f0_result:
        if f0_result['nan_count'] > 0 or f0_result['inf_count'] > 0 or f0_result['problematic_files']:
            print("❌ F0 数据存在问题")
            issues_found = True
        else:
            print("✓ F0 数据正常")
    
    if soft_result:
        if soft_result['nan_count'] > 0 or soft_result['inf_count'] > 0 or soft_result['problematic_files']:
            print("❌ Hubert 特征存在问题")
            issues_found = True
        else:
            print("✓ Hubert 特征正常")
    
    if issues_found:
        print("\n建议:")
        print("  1. 重新运行 resample.py 确保音频正确归一化")
        print("  2. 重新运行 preprocess_hubert_f0.py 生成特征")
        print("  3. 检查并调整 configs/config.json 中的参数")
        print("  4. 考虑禁用 vol_aug 和 vol_embedding")
        print("  5. 降低学习率或增加梯度累积步数")
    else:
        print("\n✓ 所有检查通过，数据看起来正常")
        print("  如果仍然出现梯度爆炸，可能是:")
        print("  - 训练过程中的数值稳定性问题")
        print("  - 模型架构或超参数需要调整")
        print("  - 建议启用梯度裁剪和监控")


if __name__ == "__main__":
    generate_report()
