"""
修复脚本: 将 excluded_lowqual 中被误移的正常文件移回 dataset/44k/huawu/
误移原因: 之前用 glob "base*" 前缀匹配, 导致 "1505956834_72" 误匹配 "1505956834_720" 等。
修复逻辑: 只保留目标 19 个 base 对应的文件, 其余全部移回。
"""
import os
import re
import shutil

B44 = os.path.join("dataset", "44k", "huawu")
D44 = os.path.join("dataset", "44k", "huawu_excluded_lowqual")

# 目标 19 个 base (与 exclude_lowqual.py 保持一致)
GROUP_A = [
    "1505956834_318", "1755625659_951", "1155591443_628",
    "1155591443_649", "1205891033_196", "1205891033_770",
    "1505956834_377", "1505956834_72", "1505956834_781",
    "1205891033_574", "1505956834_678",
]
GROUP_B = [
    "1155271209_383", "1155591443_615", "1205891033_2114",
    "1205891033_493", "1155591443_829", "1505956834_124",
    "1505956834_616", "1205891033_1849",
]
TARGET = set(GROUP_A + GROUP_B)

BASE_RE = re.compile(r"^(\d+_\d+)")


def main():
    files = sorted(os.listdir(D44))
    moved_back = []
    kept = []
    unknown = []

    for f in files:
        m = BASE_RE.match(f)
        if not m:
            # 无法识别 base, 保守移回
            shutil.move(os.path.join(D44, f), os.path.join(B44, f))
            unknown.append(f)
            continue
        base = m.group(1)
        if base in TARGET:
            kept.append(f)
        else:
            shutil.move(os.path.join(D44, f), os.path.join(B44, f))
            moved_back.append(f)

    print(f"excluded_lowqual 原文件数: {len(files)}")
    print(f"移回误移文件数: {len(moved_back)}")
    print(f"保留(目标19个base)文件数: {len(kept)}")
    if unknown:
        print(f"无法识别而移回: {len(unknown)}")
        for u in unknown:
            print(f"  {u}")

    # 验证目标 base 在 44k 中应无残留
    residual = []
    for base in TARGET:
        for f in os.listdir(B44):
            if f.startswith(base + ".") or f == base:
                residual.append(f)
    print(f"\n验证: 44k 中目标 base 残留文件 = {len(residual)}")
    for r in residual[:20]:
        print(f"  {r}")

    # 验证 excluded_lowqual 中剩余文件都属于目标 base
    remain = os.listdir(D44)
    bad = [f for f in remain if BASE_RE.match(f) and BASE_RE.match(f).group(1) not in TARGET]
    print(f"验证: excluded_lowqual 剩余文件中非目标 base = {len(bad)}")


if __name__ == "__main__":
    main()
