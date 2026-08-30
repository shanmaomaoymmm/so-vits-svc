"""
排除低质量音频脚本
将 A 类(11个噪声) + B 类(8个静音≥65%) 共 19 个文件从训练管线中移除:
1. 从 filelists/train.txt 删除对应行
2. 将 dataset/44k/huawu 中的对应预处理文件移到备份目录
3. 将 dataset_raw/huawu 中的对应原始 wav 移到备份目录
全部为可逆操作(移动而非删除)。
"""
import os
import sys
import glob
import shutil

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# A 类: 噪声问题(SNR偏低/高频噪声)
GROUP_A = [
    "1505956834_318.wav", "1755625659_951.wav", "1155591443_628.wav",
    "1155591443_649.wav", "1205891033_196.wav", "1205891033_770.wav",
    "1505956834_377.wav", "1505956834_72.wav", "1505956834_781.wav",
    "1205891033_574.wav", "1505956834_678.wav",
]
# B 类: 静音占比>=65%(已去除与A重复的 1755625659_951)
GROUP_B = [
    "1155271209_383.wav", "1155591443_615.wav", "1205891033_2114.wav",
    "1205891033_493.wav", "1155591443_829.wav", "1505956834_124.wav",
    "1505956834_616.wav", "1205891033_1849.wav",
]

EXCLUDE = GROUP_A + GROUP_B
BASE_NAMES = [f[:-4] for f in EXCLUDE]

P44 = os.path.join("dataset", "44k", "huawu")
PRAW = os.path.join("dataset_raw", "huawu")
B44 = os.path.join("dataset", "44k", "huawu_excluded_lowqual")
BRAW = os.path.join("dataset_raw", "huawu_excluded_lowqual")
TRAIN_TXT = os.path.join("filelists", "train.txt")
TRAIN_BAK = os.path.join("filelists", "train.txt.bak2_before_lowqual")

os.makedirs(B44, exist_ok=True)
os.makedirs(BRAW, exist_ok=True)


def main():
    print(f"待排除文件: {len(EXCLUDE)} 个")
    print(f"  A类(噪声): {len(GROUP_A)}, B类(静音高): {len(GROUP_B)}")
    print()

    # 1. 移动 dataset/44k 预处理文件
    moved44, missing44 = [], []
    for base in BASE_NAMES:
        for f in glob.glob(os.path.join(P44, base + "*")):
            dest = os.path.join(B44, os.path.basename(f))
            shutil.move(f, dest)
            moved44.append(os.path.basename(f))
        if not glob.glob(os.path.join(P44, base + "*.wav")):
            missing44.append(base + ".wav")
    print(f"[44k] 移出 {len(moved44)} 个预处理文件 -> dataset/44k/huawu_excluded_lowqual")
    if missing44:
        print(f"  警告: 44k 中找不到 {len(missing44)} 个 wav: {missing44}")

    # 2. 移动 dataset_raw 原始 wav
    movedraw, missraw = [], []
    for base in BASE_NAMES:
        src = os.path.join(PRAW, base + ".wav")
        if os.path.exists(src):
            shutil.move(src, os.path.join(BRAW, base + ".wav"))
            movedraw.append(base + ".wav")
        else:
            missraw.append(base + ".wav")
    print(f"[raw] 移出 {len(movedraw)} 个原始 wav -> dataset_raw/huawu_excluded_lowqual")
    if missraw:
        print(f"  警告: raw 中找不到 {len(missraw)} 个: {missraw}")

    # 3. 更新 filelists/train.txt
    lines = open(TRAIN_TXT, "r", encoding="utf-8").read().splitlines()
    keep, removed = [], []
    for line in lines:
        if any(f"/{base}." in line or line.endswith(base + ".wav") for base in BASE_NAMES):
            removed.append(line)
        else:
            keep.append(line)
    shutil.copy(TRAIN_TXT, TRAIN_BAK)
    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
    print(f"\n[train.txt] 原行数={len(lines)} -> 新行数={len(keep)}, 删除={len(removed)}")
    print(f"备份: {TRAIN_BAK}")

    # 验证
    new_lines = open(TRAIN_TXT, "r", encoding="utf-8").read().splitlines()
    resid = [l for l in new_lines if any(base in l for base in BASE_NAMES)]
    print(f"\n验证: train.txt 残留被排除文件的行数 = {len(resid)}")
    print(f"验证: 44k 残留 = {len(glob.glob(os.path.join(P44, '*')) and [f for b in BASE_NAMES for f in glob.glob(os.path.join(P44, b+'*'))])}")
    print(f"验证: raw 残留 = {sum(1 for b in BASE_NAMES if os.path.exists(os.path.join(PRAW, b + '.wav')))}")


if __name__ == "__main__":
    main()
