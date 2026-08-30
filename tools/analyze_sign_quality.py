"""
sign 数据集音质分析工具
对比 dataset_raw/huawu 中 sign 数据与非 sign 数据的音质差异,
用于确认 sign 数据是否存在音质问题, 决定是否在后续训练中排除。
"""
import os
import sys
import glob
import random
import numpy as np
import soundfile as sf

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join("dataset_raw", "huawu")
FFT_N = 4096
SAMPLE_SIGN_COUNT = 94   # 非 sign 抽样数量, 与 sign 数量相同保证公平

random.seed(42)


def is_sign(filename):
    """根据文件名判断是否为 sign 数据"""
    base = os.path.basename(filename)
    return base.startswith("sign")


def frame_metrics(data, sr):
    """分段 FFT 计算高频噪声占比(>8kHz 能量比例)"""
    hop = FFT_N // 4
    win = np.hanning(FFT_N)
    freqs = np.fft.rfftfreq(FFT_N, 1.0 / sr)
    high_mask = freqs > 8000
    low_mask = freqs < 1000

    total_energy = 0.0
    high_energy = 0.0
    low_energy = 0.0
    frames = 0
    for start in range(0, max(1, len(data) - FFT_N), hop):
        frame = data[start:start + FFT_N] * win
        spec = np.abs(np.fft.rfft(frame)) ** 2
        total_energy += spec.sum()
        high_energy += spec[high_mask].sum()
        low_energy += spec[low_mask].sum()
        frames += 1
        if frames >= 200:  # 抽样足够帧数即可
            break
    if total_energy <= 0 or frames == 0:
        return 0.0, 0.0
    return high_energy / total_energy, low_energy / total_energy


def analyze_file(path):
    """读取音频并计算音质指标"""
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = np.asarray(data, dtype=np.float64)
    duration = len(data) / sr

    # 全局能量 / 峰值
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data)) if len(data) else 0.0
    clip_ratio = float(np.mean(np.abs(data) > 0.99)) if peak > 0.99 else 0.0

    # 分帧估计噪声底: 取 RMS 最低的 5% 帧的平均 RMS 作为背景噪声
    frame_len = 2048
    hop = 1024
    frame_rms = []
    for start in range(0, max(1, len(data) - frame_len), hop):
        frame_rms.append(np.sqrt(np.mean(data[start:start + frame_len] ** 2)))
    if len(frame_rms) > 0:
        frame_rms = np.array(frame_rms)
        noise_floor = float(np.percentile(frame_rms, 5))
    else:
        noise_floor = rms

    # 信噪比估计 (dB), 保护分母
    if noise_floor > 1e-8:
        snr_est = 20.0 * np.log10(max(rms, 1e-9) / noise_floor)
    else:
        snr_est = 60.0

    # 静音占比: RMS 低于噪声阈值 3 倍的时间比例
    silent_ratio = float(np.mean(frame_rms < max(noise_floor * 3, 1e-4))) if len(frame_rms) > 0 else 0.0

    high_freq_ratio, low_freq_ratio = frame_metrics(data, sr)

    return {
        "file": os.path.basename(path),
        "sr": sr,
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "clip_ratio": clip_ratio,
        "high_freq_ratio": high_freq_ratio,
        "low_freq_ratio": low_freq_ratio,
        "noise_floor": noise_floor,
        "snr_est": snr_est,
        "silent_ratio": silent_ratio,
    }


def quality_score(m):
    """综合音质评分 0-100, 越高越好"""
    score = 90.0
    # 时长: 1-20s 最佳
    if m["duration"] < 0.8:
        score -= 30
    elif m["duration"] < 1.5:
        score -= 15
    elif m["duration"] > 30:
        score -= 20

    # 削波
    if m["clip_ratio"] > 0.01:
        score -= 25
    elif m["clip_ratio"] > 0.001:
        score -= 12

    # 能量过低
    if m["rms"] < 0.005:
        score -= 20
    elif m["rms"] < 0.02:
        score -= 8

    # 高频噪声过多
    if m["high_freq_ratio"] > 0.3:
        score -= 25
    elif m["high_freq_ratio"] > 0.15:
        score -= 10

    # 信噪比过低
    if m["snr_est"] < 5:
        score -= 30
    elif m["snr_est"] < 10:
        score -= 15
    elif m["snr_est"] < 15:
        score -= 8

    # 静音占比过高
    if m["silent_ratio"] > 0.5:
        score -= 20
    elif m["silent_ratio"] > 0.3:
        score -= 10

    # 低频占比过高 + 峰值过低 -> 声音发闷/有混响感, 轻微扣分
    if m["low_freq_ratio"] > 0.75 and m["peak"] < 0.25:
        score -= 10

    return float(np.clip(score, 0, 100))


def summary(items, key):
    vals = np.array([it[key] for it in items])
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def print_table(name, sign_items, other_items, key, fmt="{:.3f}", unit=""):
    s, o = summary(sign_items, key), summary(other_items, key)
    print(f"  {name:<18} | sign: mean={fmt.format(s['mean'])}{unit}  med={fmt.format(s['median'])}{unit} "
          f"min={fmt.format(s['min'])}{unit} max={fmt.format(s['max'])}{unit} | "
          f"非sign: mean={fmt.format(o['mean'])}{unit}  med={fmt.format(o['median'])}{unit} "
          f"min={fmt.format(o['min'])}{unit} max={fmt.format(o['max'])}{unit}")


def main():
    wav_files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    sign_files = [f for f in wav_files if is_sign(f)]
    other_files = [f for f in wav_files if not is_sign(f)]

    print("=" * 100)
    print("sign 数据音质分析报告")
    print("=" * 100)
    print(f"数据集目录: {DATA_DIR}")
    print(f"总 wav: {len(wav_files)}, sign: {len(sign_files)}, 非sign: {len(other_files)}")

    if not sign_files:
        print("未找到 sign 数据, 退出")
        return

    # 抽样非 sign 数据
    sample_n = min(SAMPLE_SIGN_COUNT, len(other_files))
    random.shuffle(other_files)
    other_sample = other_files[:sample_n]
    print(f"非 sign 抽样 {len(other_sample)} 个用于对比\n")

    print("分析中 (sign + 非sign抽样)...")
    sign_metrics = [analyze_file(f) for f in sign_files]
    other_metrics = [analyze_file(f) for f in other_sample]

    # 评分
    for m in sign_metrics:
        m["score"] = quality_score(m)
    for m in other_metrics:
        m["score"] = quality_score(m)

    print("\n" + "-" * 100)
    print("指标对比 (sign vs 非sign)")
    print("-" * 100)
    print_table("采样率(Hz)", sign_metrics, other_metrics, "sr", fmt="{:.0f}", unit="Hz")
    print_table("时长(秒)", sign_metrics, other_metrics, "duration", unit="s")
    print_table("RMS 能量", sign_metrics, other_metrics, "rms")
    print_table("峰值", sign_metrics, other_metrics, "peak")
    print_table("削波比例", sign_metrics, other_metrics, "clip_ratio", fmt="{:.5f}")
    print_table("高频噪声比", sign_metrics, other_metrics, "high_freq_ratio")
    print_table("低频能量比", sign_metrics, other_metrics, "low_freq_ratio")
    print_table("噪声底", sign_metrics, other_metrics, "noise_floor", fmt="{:.5f}")
    print_table("SNR估计(dB)", sign_metrics, other_metrics, "snr_est", unit="dB")
    print_table("静音占比", sign_metrics, other_metrics, "silent_ratio")
    print_table("综合评分", sign_metrics, other_metrics, "score", fmt="{:.1f}")

    # 问题统计
    print("\n" + "-" * 100)
    print("问题文件统计")
    print("-" * 100)
    sign_bad = [m for m in sign_metrics if m["score"] < 70]
    other_bad = [m for m in other_metrics if m["score"] < 70]
    print(f"  sign: 评分<70 的问题文件 {len(sign_bad)}/{len(sign_metrics)} ({len(sign_bad)/len(sign_metrics)*100:.1f}%)")
    print(f"  非sign: 评分<70 的问题文件 {len(other_bad)}/{len(other_metrics)} ({len(other_bad)/len(other_metrics)*100:.1f}%)")

    # 详细列出最差的 sign 文件
    sign_sorted = sorted(sign_metrics, key=lambda m: m["score"])
    print("\n  sign 数据中最差的 15 个文件 (按评分):")
    print(f"  {'文件':<28} {'时长':>6} {'RMS':>7} {'峰值':>6} {'削波%':>7} {'高频%':>7} {'SNR':>6} {'评分':>6}")
    for m in sign_sorted[:15]:
        print(f"  {m['file']:<28} {m['duration']:>6.2f} {m['rms']:>7.4f} {m['peak']:>6.3f} "
              f"{m['clip_ratio']*100:>6.2f}% {m['high_freq_ratio']*100:>6.1f}% {m['snr_est']:>6.1f} {m['score']:>6.1f}")

    # 采样率异常检查
    sr_set = set(m["sr"] for m in sign_metrics)
    other_sr_set = set(m["sr"] for m in other_metrics)
    print(f"\n  采样率分布: sign={sorted(sr_set)}, 非sign={sorted(other_sr_set)}")

    # 汇总结论
    print("\n" + "=" * 100)
    print("结论摘要")
    print("=" * 100)
    s_score = np.mean([m["score"] for m in sign_metrics])
    o_score = np.mean([m["score"] for m in other_metrics])
    s_high = np.mean([m["high_freq_ratio"] for m in sign_metrics])
    o_high = np.mean([m["high_freq_ratio"] for m in other_metrics])
    s_snr = np.mean([m["snr_est"] for m in sign_metrics])
    o_snr = np.mean([m["snr_est"] for m in other_metrics])
    print(f"  平均综合评分: sign={s_score:.1f}, 非sign={o_score:.1f} (差 {s_score-o_score:+.1f})")
    print(f"  平均高频噪声比: sign={s_high*100:.1f}%, 非sign={o_high*100:.1f}%")
    print(f"  平均SNR: sign={s_snr:.1f}dB, 非sign={o_snr:.1f}dB")
    print(f"  sign 问题文件比例: {len(sign_bad)}/{len(sign_metrics)}")


if __name__ == "__main__":
    main()
