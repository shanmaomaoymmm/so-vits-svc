"""
全量音质扫描工具
对 dataset_raw/huawu 中所有非 sign 音频进行完整音质分析,
找出静音、削波、噪声大、信噪比低、时长异常等问题文件。
支持多进程并行加速。
"""
import os
import sys
import glob
import time
import numpy as np
import soundfile as sf
from multiprocessing import Pool
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join("dataset_raw", "huawu")
FFT_N = 4096
FFT_FRAMES = 100          # 每文件 FFT 帧数采样上限
N_PROC = max(4, os.cpu_count() // 2)  # 并行进程数

REPORT_PATH = os.path.join("doc", "audio_quality_report.txt")

# 问题判定阈值
TH_DURATION_MIN = 1.0
TH_DURATION_MAX = 30.0
TH_SILENT_RMS = 0.001     # 几乎静音
TH_LOWENERGY_RMS = 0.01   # 能量过低
TH_CLIP_PEAK = 0.99       # 削波
TH_HIGHFREQ_RATIO = 0.15  # 高频噪声
TH_SNR_BAD = 10.0         # 信噪比差
TH_SNR_WARN = 15.0        # 信噪比警告


def is_excluded(filename):
    """被排除的数据(如 sign)不再分析"""
    base = os.path.basename(filename)
    return base.startswith("sign") or base.startswith("excluded")


def analyze_one(path):
    """分析单个音频文件, 返回指标 dict"""
    result = {"file": os.path.basename(path), "path": path, "error": None}
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float64)
        duration = len(data) / sr

        rms = np.sqrt(np.mean(data ** 2))
        peak = np.max(np.abs(data)) if len(data) else 0.0
        clip_ratio = float(np.mean(np.abs(data) > 0.99)) if peak > 0.99 else 0.0

        # 分帧 RMS 估计噪声底 + 静音占比
        frame_len = 2048
        hop = 1024
        frame_rms = []
        for start in range(0, max(1, len(data) - frame_len), hop):
            frame_rms.append(np.sqrt(np.mean(data[start:start + frame_len] ** 2)))
        if len(frame_rms) > 0:
            fr = np.array(frame_rms)
            noise_floor = float(np.percentile(fr, 5))
            silent_ratio = float(np.mean(fr < max(noise_floor * 3, 1e-4)))
        else:
            noise_floor = rms
            silent_ratio = 0.0

        if noise_floor > 1e-8:
            snr_est = 20.0 * np.log10(max(rms, 1e-9) / noise_floor)
        else:
            snr_est = 60.0

        # 分段 FFT 计算高频/低频能量占比
        hop_f = FFT_N // 4
        win = np.hanning(FFT_N)
        freqs = np.fft.rfftfreq(FFT_N, 1.0 / sr)
        high_mask = freqs > 8000
        low_mask = freqs < 1000
        total_e, high_e, low_e, frames = 0.0, 0.0, 0.0, 0
        for start in range(0, max(1, len(data) - FFT_N), hop_f):
            frame = data[start:start + FFT_N] * win
            spec = np.abs(np.fft.rfft(frame)) ** 2
            total_e += spec.sum()
            high_e += spec[high_mask].sum()
            low_e += spec[low_mask].sum()
            frames += 1
            if frames >= FFT_FRAMES:
                break
        if total_e > 0 and frames > 0:
            high_freq_ratio = high_e / total_e
            low_freq_ratio = low_e / total_e
        else:
            high_freq_ratio = low_freq_ratio = 0.0

        result.update({
            "sr": sr, "duration": duration, "rms": rms, "peak": peak,
            "clip_ratio": clip_ratio, "high_freq_ratio": high_freq_ratio,
            "low_freq_ratio": low_freq_ratio, "noise_floor": noise_floor,
            "snr_est": snr_est, "silent_ratio": silent_ratio,
        })
        result["problems"] = detect_problems(result)
    except Exception as e:
        result["error"] = str(e)
    return result


def detect_problems(m):
    """根据指标判定问题标签"""
    probs = []
    if m["sr"] != 44100:
        probs.append(f"采样率异常({m['sr']}Hz)")
    if m["duration"] < TH_DURATION_MIN:
        probs.append(f"过短({m['duration']:.2f}s)")
    elif m["duration"] > TH_DURATION_MAX:
        probs.append(f"过长({m['duration']:.2f}s)")
    if m["rms"] < TH_SILENT_RMS:
        probs.append(f"几乎静音(RMS={m['rms']:.5f})")
    elif m["rms"] < TH_LOWENERGY_RMS:
        probs.append(f"能量过低(RMS={m['rms']:.4f})")
    if m["clip_ratio"] > 0.001:
        probs.append(f"削波({m['clip_ratio']*100:.2f}%)")
    if m["high_freq_ratio"] > TH_HIGHFREQ_RATIO:
        probs.append(f"高频噪声({m['high_freq_ratio']*100:.1f}%)")
    if m["snr_est"] < TH_SNR_BAD:
        probs.append(f"信噪比极差({m['snr_est']:.1f}dB)")
    elif m["snr_est"] < TH_SNR_WARN:
        probs.append(f"信噪比偏低({m['snr_est']:.1f}dB)")
    if m["silent_ratio"] > 0.5:
        probs.append(f"静音占比过高({m['silent_ratio']*100:.0f}%)")
    return probs


def main():
    wav_files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    wav_files = [f for f in wav_files if not is_excluded(f)]
    total = len(wav_files)
    print(f"待扫描文件数: {total} (进程数: {N_PROC})")
    print("扫描中...")
    t0 = time.time()

    results = []
    with Pool(processes=N_PROC) as pool:
        for r in tqdm(pool.imap_unordered(analyze_one, wav_files, chunksize=8),
                      total=total, desc="分析音频"):
            results.append(r)

    elapsed = time.time() - t0
    ok = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    print(f"\n扫描完成: {len(ok)} 成功, {len(failed)} 失败, 用时 {elapsed:.0f}s")

    # 汇总所有问题文件
    problematic = [r for r in ok if r.get("problems")]
    # 无问题但质量评分低于平均较多的 (如能量/峰值异常)
    print(f"发现 {len(problematic)} 个问题文件\n")

    # 统计各类问题
    from collections import Counter
    issue_counter = Counter()
    for r in problematic:
        for p in r["problems"]:
            issue_counter[p.split("(")[0]] += 1

    print("=" * 90)
    print("问题类型统计")
    print("=" * 90)
    for issue, cnt in issue_counter.most_common():
        print(f"  {issue:<12} {cnt}")

    # 写详细报告
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"音质扫描报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目录: {DATA_DIR}, 总文件: {total}, 问题文件: {len(problematic)}, "
                f"失败: {len(failed)}\n")
        f.write("=" * 90 + "\n\n")
        f.write("=== 问题文件详情 ===\n")
        for r in sorted(problematic, key=lambda x: -len(x["problems"])):
            f.write(f"\n[{r['file']}]\n")
            f.write(f"  时长={r['duration']:.2f}s RMS={r['rms']:.4f} 峰值={r['peak']:.3f} "
                    f"高频={r['high_freq_ratio']*100:.1f}% SNR={r['snr_est']:.1f}dB "
                    f"静音={r['silent_ratio']*100:.0f}%\n")
            f.write(f"  问题: {'; '.join(r['problems'])}\n")
        if failed:
            f.write("\n=== 读取失败文件 ===\n")
            for r in failed:
                f.write(f"  {r['file']}: {r['error']}\n")
    print(f"\n完整报告已保存: {REPORT_PATH}")

    # 控制台打印前 40 个问题文件
    print("\n" + "=" * 90)
    print("问题文件列表 (前 40 个, 按问题数排序)")
    print("=" * 90)
    for r in sorted(problematic, key=lambda x: -len(x["problems"]))[:40]:
        print(f"  {r['file']:<28} {r['duration']:>6.2f}s SNR={r['snr_est']:>5.1f}dB "
              f"RMS={r['rms']:.4f} | {'; '.join(r['problems'])}")

    # 失败文件
    if failed:
        print("\n读取失败文件:")
        for r in failed[:20]:
            print(f"  {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
