"""
抽查验证脚本: 对代表性文件打印分段 RMS 时间分布,
确认"静音占比过高"是真实的长静音, 而非低音量语音误判。
"""
import os
import sys
import numpy as np
import soundfile as sf

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join("dataset_raw", "huawu")

# 待抽查文件: (文件名, 说明)
CHECK = [
    ("1155271209_383.wav", "静音占比最高(75%)"),
    ("1155591443_615.wav", "静音占比高(70%)"),
    ("1205891033_493.wav", "静音占比高(68%)"),
    ("1505956834_318.wav", "SNR最低(10.5dB)"),
    ("1205891033_770.wav", "SNR低(14.3dB)"),
    ("1205891033_574.wav", "高频噪声(15.4%)"),
]


def analyze_segments(path, seg_sec=0.5):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    seg_len = int(seg_sec * sr)
    segs = []
    for start in range(0, n, seg_len):
        chunk = data[start:start + seg_len]
        if len(chunk) < seg_len // 2:
            continue
        rms = np.sqrt(np.mean(chunk ** 2))
        segs.append(rms)
    return sr, data, segs, seg_sec


for name, desc in CHECK:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"[{name}] 文件不存在")
        continue
    sr, data, segs, seg_sec = analyze_segments(path)
    dur = len(data) / sr
    # 计算噪声底(最低5%帧)和SNR
    frame_len = 2048
    hop = 1024
    fr = []
    for start in range(0, max(1, len(data) - frame_len), hop):
        fr.append(np.sqrt(np.mean(data[start:start + frame_len] ** 2)))
    fr = np.array(fr)
    noise = np.percentile(fr, 5)
    snr = 20 * np.log10(max(np.sqrt(np.mean(data ** 2)), 1e-9) / max(noise, 1e-9))
    # 静音帧(低于噪声*3)比例
    silent = np.mean(fr < max(noise * 3, 1e-4))

    print(f"\n{'='*70}")
    print(f"[{name}] {desc}")
    print(f"时长={dur:.2f}s, 全局RMS={np.sqrt(np.mean(data**2)):.4f}, "
          f"SNR={snr:.1f}dB, 静音帧比例={silent*100:.0f}%")
    # 打印分段 RMS 条形图 (每 0.5s)
    rms_max = max(segs) if segs else 1
    bars = []
    for i, r in enumerate(segs):
        level = int(r / rms_max * 40)
        bar = "#" * level
        marker = "S" if r < max(noise * 3, 1e-4) else " "  # 静音段标记
        bars.append(f"  {i*seg_sec:>5.1f}s RMS={r:>7.4f} |{bar:<40}|{marker}")
    # 只打印前 16 段 + 后 4 段, 避免过长
    print("  (前16个0.5s分段的能量分布, #为相对能量, S=判定为静音)")
    for b in bars[:16]:
        print(b)
    if len(bars) > 20:
        print("  ...")
        for b in bars[-4:]:
            print(b)
