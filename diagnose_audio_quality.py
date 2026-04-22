"""
训练集音频质量自动诊断工具
快速检测可能导致电子杂音的音频质量问题
"""
import os
import glob
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_audio_quality(audio_dir, sample_rate=44100):
    """
    分析音频目录的质量问题
    """
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not wav_files:
        print(f"❌ 未找到 WAV 文件: {audio_dir}")
        return
    
    print(f"📊 开始分析 {len(wav_files)} 个音频文件...\n")
    
    # 统计信息
    stats = {
        'total': len(wav_files),
        'too_short': [],      # 音频太短 (< 1秒)
        'too_long': [],       # 音频太长 (> 30秒)
        'silent': [],         # 几乎静音
        'clipping': [],       # 削波/爆音
        'low_energy': [],     # 能量过低
        'high_noise': [],     # 噪声过高
        'sample_rate_error': [], # 采样率错误
        'mono_only': [],      # 单声道（应该没问题）
        'valid': []           # 正常文件
    }
    
    duration_list = []
    rms_list = []
    peak_list = []
    
    for wav_file in tqdm(wav_files, desc="分析音频"):
        try:
            # 读取音频
            data, sr = sf.read(wav_file)
            
            # 检查采样率
            if sr != sample_rate:
                stats['sample_rate_error'].append({
                    'file': os.path.basename(wav_file),
                    'sr': sr,
                    'expected': sample_rate
                })
            
            # 如果是立体声，转为单声道
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                stats['mono_only'].append(os.path.basename(wav_file))
            
            # 计算时长
            duration = len(data) / sr
            duration_list.append(duration)
            
            # 检查时长
            if duration < 1.0:
                stats['too_short'].append({
                    'file': os.path.basename(wav_file),
                    'duration': duration
                })
            elif duration > 30.0:
                stats['too_long'].append({
                    'file': os.path.basename(wav_file),
                    'duration': duration
                })
            
            # 计算 RMS (均方根能量)
            rms = np.sqrt(np.mean(data ** 2))
            rms_list.append(rms)
            
            # 计算峰值
            peak = np.max(np.abs(data))
            peak_list.append(peak)
            
            # 检查削波 (峰值接近 1.0)
            if peak > 0.99:
                stats['clipping'].append({
                    'file': os.path.basename(wav_file),
                    'peak': peak
                })
            
            # 检查静音 (RMS 过低)
            if rms < 0.001:
                stats['silent'].append({
                    'file': os.path.basename(wav_file),
                    'rms': rms
                })
            elif rms < 0.01:
                stats['low_energy'].append({
                    'file': os.path.basename(wav_file),
                    'rms': rms
                })
            
            # 检查噪声水平 (通过高频段能量)
            if len(data) > 4410:  # 至少 0.1 秒
                # 计算频谱
                fft_data = np.fft.rfft(data[:4410])  # 前 0.1 秒
                freqs = np.fft.rfftfreq(4410, 1/sr)
                
                # 高频段 (8kHz 以上) 能量占比
                high_freq_mask = freqs > 8000
                if np.any(high_freq_mask):
                    high_freq_energy = np.sum(np.abs(fft_data[high_freq_mask]) ** 2)
                    total_energy = np.sum(np.abs(fft_data) ** 2)
                    
                    if total_energy > 0:
                        high_freq_ratio = high_freq_energy / total_energy
                        
                        # 如果高频能量占比过高，可能有噪声
                        if high_freq_ratio > 0.3:
                            stats['high_noise'].append({
                                'file': os.path.basename(wav_file),
                                'high_freq_ratio': high_freq_ratio
                            })
            
            # 如果没有问题，标记为有效
            if (duration >= 1.0 and duration <= 30.0 and 
                rms >= 0.001 and peak <= 0.99):
                stats['valid'].append(os.path.basename(wav_file))
        
        except Exception as e:
            print(f"\n❌ 处理失败: {os.path.basename(wav_file)} - {e}")
    
    # 打印报告
    print("\n" + "="*80)
    print("📋 音频质量诊断报告")
    print("="*80)
    
    print(f"\n✅ 总文件数: {stats['total']}")
    print(f"✅ 正常文件: {len(stats['valid'])} ({len(stats['valid'])/stats['total']*100:.1f}%)")
    
    print(f"\n⚠️ 发现的问题:")
    
    if stats['too_short']:
        print(f"  • 音频过短 (<1秒): {len(stats['too_short'])} 个")
        for item in stats['too_short'][:5]:
            print(f"    - {item['file']}: {item['duration']:.2f}秒")
        if len(stats['too_short']) > 5:
            print(f"    ... 还有 {len(stats['too_short'])-5} 个")
    
    if stats['too_long']:
        print(f"  • 音频过长 (>30秒): {len(stats['too_long'])} 个")
        for item in stats['too_long'][:5]:
            print(f"    - {item['file']}: {item['duration']:.2f}秒")
        if len(stats['too_long']) > 5:
            print(f"    ... 还有 {len(stats['too_long'])-5} 个")
    
    if stats['silent']:
        print(f"  • 几乎静音: {len(stats['silent'])} 个")
        for item in stats['silent'][:5]:
            print(f"    - {item['file']}: RMS={item['rms']:.6f}")
    
    if stats['clipping']:
        print(f"  • 削波/爆音: {len(stats['clipping'])} 个 ⚠️⚠️⚠️")
        for item in stats['clipping'][:10]:
            print(f"    - {item['file']}: peak={item['peak']:.4f}")
        if len(stats['clipping']) > 10:
            print(f"    ... 还有 {len(stats['clipping'])-10} 个")
        print(f"    💡 建议: 这些文件可能导致周期性杂音！")
    
    if stats['low_energy']:
        print(f"  • 能量过低: {len(stats['low_energy'])} 个")
    
    if stats['high_noise']:
        print(f"  • 高频噪声过高: {len(stats['high_noise'])} 个 ⚠️⚠️")
        for item in stats['high_noise'][:10]:
            print(f"    - {item['file']}: ratio={item['high_freq_ratio']:.2%}")
        if len(stats['high_noise']) > 10:
            print(f"    ... 还有 {len(stats['high_noise'])-10} 个")
        print(f"    💡 建议: 高频噪声会导致电子杂音！")
    
    if stats['sample_rate_error']:
        print(f"  • 采样率错误: {len(stats['sample_rate_error'])} 个")
        for item in stats['sample_rate_error'][:5]:
            print(f"    - {item['file']}: {item['sr']}Hz (期望 {item['expected']}Hz)")
    
    # 统计信息
    if duration_list:
        print(f"\n📊 时长统计:")
        print(f"  • 平均时长: {np.mean(duration_list):.2f}秒")
        print(f"  • 最短: {np.min(duration_list):.2f}秒")
        print(f"  • 最长: {np.max(duration_list):.2f}秒")
        print(f"  • 总时长: {sum(duration_list)/3600:.2f}小时")
    
    if rms_list:
        print(f"\n📊 能量统计:")
        print(f"  • 平均 RMS: {np.mean(rms_list):.4f}")
        print(f"  • 最小 RMS: {np.min(rms_list):.6f}")
        print(f"  • 最大 RMS: {np.max(rms_list):.4f}")
    
    if peak_list:
        print(f"\n📊 峰值统计:")
        print(f"  • 平均峰值: {np.mean(peak_list):.4f}")
        print(f"  • 最大峰值: {np.max(peak_list):.4f}")
    
    # 生成可视化
    print(f"\n📈 生成可视化图表...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 时长分布
        axes[0, 0].hist(duration_list, bins=50, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Audio Duration Distribution')
        axes[0, 0].axvline(x=1.0, color='red', linestyle='--', label='Min threshold')
        axes[0, 0].axvline(x=30.0, color='orange', linestyle='--', label='Max threshold')
        axes[0, 0].legend()
        
        # RMS 分布
        axes[0, 1].hist(rms_list, bins=50, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('RMS Energy')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Audio Energy Distribution')
        axes[0, 1].axvline(x=0.001, color='red', linestyle='--', label='Silent threshold')
        axes[0, 1].legend()
        
        # 峰值分布
        axes[1, 0].hist(peak_list, bins=50, color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('Peak Amplitude')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Peak Amplitude Distribution')
        axes[1, 0].axvline(x=0.99, color='red', linestyle='--', label='Clipping threshold')
        axes[1, 0].legend()
        
        # 问题文件比例
        problem_types = ['Valid', 'Too Short', 'Too Long', 'Silent', 'Clipping', 'High Noise']
        problem_counts = [
            len(stats['valid']),
            len(stats['too_short']),
            len(stats['too_long']),
            len(stats['silent']),
            len(stats['clipping']),
            len(stats['high_noise'])
        ]
        colors = ['green', 'yellow', 'orange', 'gray', 'red', 'darkred']
        axes[1, 1].pie(problem_counts, labels=problem_types, autopct='%1.1f%%', colors=colors)
        axes[1, 1].set_title('File Quality Distribution')
        
        plt.tight_layout()
        plt.savefig('audio_quality_report.png', dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存: audio_quality_report.png")
    except Exception as e:
        print(f"⚠️ 图表生成失败: {e}")
    
    # 保存问题文件列表
    problem_files = set()
    for key in ['too_short', 'too_long', 'silent', 'clipping', 'high_noise']:
        for item in stats[key]:
            if isinstance(item, dict):
                problem_files.add(item['file'])
            else:
                problem_files.add(item)
    
    if problem_files:
        with open('problem_audio_files.txt', 'w', encoding='utf-8') as f:
            f.write("# 有问题的音频文件列表\n")
            f.write("# 建议检查或删除这些文件\n\n")
            for fname in sorted(problem_files):
                f.write(f"{fname}\n")
        print(f"\n✅ 问题文件列表已保存: problem_audio_files.txt")
        print(f"   共 {len(problem_files)} 个文件需要检查")
    
    print("\n" + "="*80)
    print("💡 建议:")
    print("="*80)
    
    if stats['clipping']:
        print("  1. ⚠️ 削波文件会导致严重的周期性杂音！")
        print("     建议: 使用音频编辑软件修复或重新录制")
    
    if stats['high_noise']:
        print("  2. ⚠️ 高频噪声会导致电子杂音！")
        print("     建议: 使用降噪软件处理或移除这些文件")
    
    if stats['too_short']:
        print("  3. 过短的音频可能影响训练效果")
        print("     建议: 合并或删除小于 1 秒的文件")
    
    if len(stats['valid']) / stats['total'] < 0.9:
        print("\n  🔴 警告: 超过 10% 的文件有问题！")
        print("     这很可能是导致电子杂音的主要原因！")
        print("     建议: 清理问题文件后重新预处理和训练")
    else:
        print("\n  ✅ 数据质量总体良好")
        print("     如果仍有电子杂音，可能是 F0 提取或其他配置问题")
    
    print("="*80)

if __name__ == "__main__":
    import sys
    
    # 默认路径
    audio_dir = "dataset_raw/huawu"
    
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
    
    if not os.path.exists(audio_dir):
        print(f"❌ 目录不存在: {audio_dir}")
        sys.exit(1)
    
    analyze_audio_quality(audio_dir)
