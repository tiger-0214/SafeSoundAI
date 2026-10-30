"""
SafeSoundAI — S1 幅度轴预实验（s1_amplitude_pretest.py）

**这是 v3 编码里唯一还没验证的环节**：幅度 3 档（40/70/100%，跨 8 dB）
在耳内 LRA 上到底分不分得开。

设计（27 号 §四）：
  - 固定类别码 = car_horn（双脉冲，占空比 34%，最接近"可听化"条件）
  - 只变幅度 3 档，随机呈现，3AFC（1 弱 / 2 中 / 3 强）
  - 每人 30 试次（n=10/档），被试数 ≥12
  - 判据：总体 ≥80%，且相邻档不系统性混淆（弱↔中、中↔强的错答不应互相抵消）

两种结局：
  ✅ 通过 → 幅度轴成立，保留 9 码（3 类 × 3 档）
  ❌ 未通过 → 砍掉幅度轴，退化为纯时间编码 3 码，紧迫度改用重复间隔表达

用法：
  python3 s1_amplitude_pretest.py                # 交互式跑 30 试次
  python3 s1_amplitude_pretest.py --dry-run      # 只校验刺激与统计框架
  python3 s1_amplitude_pretest.py --merge outputs/reports/s1_sub01.json ... # 合并多被试
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import config

REPORT_DIR = config.OUTPUT_DIR / "reports"


def run_one(subject: str, n: int, dry: bool) -> Path | None:
    cmd = [sys.executable, "haptic_test.py", "--task", "amplitude", "--n", str(n)]
    if dry:
        cmd.append("--no-audio")
        subprocess.run(cmd, check=True)
        return None
    subprocess.run(cmd, check=True)
    src = REPORT_DIR / "haptic_test_amplitude.json"
    dst = REPORT_DIR / f"s1_{subject}.json"
    if src.exists():
        src.replace(dst)
        print(f"✅ 被试 {subject} 结果：{dst}")
    return dst


def merge(paths: list[Path]) -> None:
    """合并多被试：报总体准确率 + 逐档混淆，并给 S1 判定。"""
    all_trials, accs = [], []
    for p in paths:
        rep = json.loads(p.read_text(encoding="utf-8"))
        all_trials += rep["trials"]
        accs.append(rep["accuracy"])

    labels = ["alert", "high", "critical"]
    idx = {k: i for i, k in enumerate(labels)}
    cm = np.zeros((3, 3), dtype=int)
    for t in all_trials:
        truth = t["urgency"]          # 幅度任务的真值 = 紧迫度档，不是类别
        if truth in idx and t["answer"] in idx:
            cm[idx[truth], idx[t["answer"]]] += 1

    acc = float(np.trace(cm)) / cm.sum() if cm.sum() else 0.0
    per = {k: (cm[idx[k], idx[k]] / cm[idx[k]].sum() if cm[idx[k]].sum() else 0.0)
           for k in labels}
    # 单调性检查：有序量误判应主要落在相邻档，而不是跨两档
    cross2 = int(cm[0, 2] + cm[2, 0])
    passed = acc >= 0.80 and cross2 <= 0.1 * cm.sum()

    lines = [
        "# S1 幅度轴预实验 — 汇总",
        "",
        f"- 被试数：{len(paths)}　总试次：{int(cm.sum())}",
        f"- 逐人被试准确率：{[f'{a:.1%}' for a in accs]}",
        f"- 合并准确率：**{acc:.1%}**　判据 ≥80%：{'✅ 通过' if acc >= 0.80 else '❌ 未通过'}",
        f"- 跨两档混淆（弱↔强）：{cross2} 次（应 ≤10% 总试次）",
        "",
        "## 逐档召回",
        "| 档位 | 幅度 | 召回 |",
        "|:--|--:|--:|",
        "| alert (弱) | 40% | {:.1%} |".format(per["alert"]),
        "| high (中) | 70% | {:.1%} |".format(per["high"]),
        "| critical (强) | 100% | {:.1%} |".format(per["critical"]),
        "",
        "## 混淆矩阵（行=真值，列=作答）",
        "| 真值\\作答 | 弱 | 中 | 强 |",
        "|:--|--:|--:|--:|",
    ]
    for k, row in zip(labels, cm):
        lines.append(f"| {k} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## 判定", ""]
    if passed:
        lines.append("✅ **幅度轴成立**：保留 9 码编码（3 类 × 3 档），紧迫度用幅度承载。")
    else:
        lines.append("❌ **幅度轴不成立**：退化为纯时间编码 3 码，紧迫度改用重复间隔表达"
                     "（即回到 D4 方案）。论文中把这次阴性结果作为「多维不可靠」的自主证据。")
    out = REPORT_DIR / "s1_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n✅ 汇总写入 {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub01")
    ap.add_argument("--n", type=int, default=10, help="每档试次数（10 → 30 试次/人）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", nargs="*", default=None, help="合并多个被试 json")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        paths = [Path(p) for p in args.merge]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"找不到：{missing}")
        merge(paths)
        return

    run_one(args.subject, args.n, args.dry_run)


if __name__ == "__main__":
    main()
