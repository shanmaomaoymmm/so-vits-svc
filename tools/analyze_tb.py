"""
TensorBoard 训练曲线分析工具
解析 logs/44k 下的 tfevents 事件文件, 输出各标量的最新值、前后对比与趋势,
用于判断训练是否正常(收敛/波动/NaN/爆炸)。
用法: python tools/analyze_tb.py [logdir]
"""
import os
import sys
import glob
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tensorboard.backend.event_processing import event_accumulator


def analyze(logdir):
    files = sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))
    if not files:
        print(f"未找到 tfevents 文件: {logdir}")
        return
    print(f"logdir: {logdir}")
    print(f"tfevents 文件数: {len(files)} (最早的: {os.path.basename(files[0])}, "
          f"最新的: {os.path.basename(files[-1])})")

    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},  # 不截断, 加载全部
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print(f"标量 tag 数: {len(tags)}\n")

    # 优先关注的关键指标
    priority = [t for t in tags if any(k in t.lower() for k in
                ["g/loss", "g/mel", "g/kl", "g/fm", "d/loss", "loss", "perplexity"])]

    # 汇总所有标量的最新值
    rows = []
    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        vals = np.array([e.value for e in events])
        rows.append({
            "tag": tag, "n": len(vals), "start_step": steps[0],
            "end_step": steps[-1], "first": vals[0], "last": vals[-1],
            "min": vals.min(), "max": vals.max(), "mean": vals.mean(),
            "std": vals.std(), "nan": int(np.isnan(vals).sum()),
            "inf": int(np.isinf(vals).sum()),
        })
    rows.sort(key=lambda r: -r["n"])

    print("=" * 110)
    print(f"{'Tag':<38}{'点数':>6}{'范围步':>12}{'首值':>10}{'末值':>10}{'最小':>10}{'最大':>10}{'均值':>10}{'NaN':>5}{'Inf':>5}")
    print("=" * 110)
    for r in rows:
        step_range = f"{r['start_step']}-{r['end_step']}"
        print(f"{r['tag']:<38}{r['n']:>6}{step_range:>12}"
              f"{r['first']:>10.4f}{r['last']:>10.4f}{r['min']:>10.4f}"
              f"{r['max']:>10.4f}{r['mean']:>10.4f}{r['nan']:>5}{r['inf']:>5}")

    # 关键指标趋势 (最后 25 个点)
    print("\n" + "=" * 110)
    print("关键指标最近趋势 (每点=最新 25 个记录)")
    print("=" * 110)
    for tag in priority:
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue
        tail = events[-25:]
        print(f"\n  [{tag}]  共{len(events)}点")
        line = "    "
        for e in tail:
            line += f"{e.value:.3f} "
        print(line)
        # 前 5 个 vs 后 5 个对比
        first5 = [e.value for e in events[:5]]
        last5 = [e.value for e in events[-5:]]
        print(f"    前5均值={np.mean(first5):.4f}  末5均值={np.mean(last5):.4f}  "
              f"变化={np.mean(last5) - np.mean(first5):+.4f}")


if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else "logs/44k"
    analyze(logdir)
