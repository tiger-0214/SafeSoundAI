"""
SafeSoundAI — 触觉码强制选择测试框架（haptic_test.py, v3）

两个任务，分开报混淆矩阵（这是 27 号 §六 的要求：两个轴必须分别定位）：

  --task amplitude   幅度轴：固定一个类别码，只变 3 档幅度 → 3AFC（弱/中/强）
                     即 S1 预实验（s1_amplitude_pretest.py 是它的薄封装）
  --task category    类别轴：3 个类别，幅度随机 → 3AFC（枪声/喇叭/警笛）
                     幅度随机是故意的：若类别仍能高分，说明两轴正交

用法：
  python3 haptic_test.py --task amplitude --n 10            # 30 试次
  python3 haptic_test.py --task amplitude --n 10 --no-audio  # 只校验刺激设计
  python3 haptic_test.py --task category  --n 10
  python3 haptic_test.py --report outputs/reports/haptic_test_amplitude.json

输出：outputs/reports/haptic_test_<task>.json + .md（含混淆矩阵与判据）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import config
from haptic_encoder import PRIOR_URGENCY, SOUND_PATTERNS, URGENCY_LEVELS, V3_CLASSES
from haptic_render import SR, render_code

TMP = Path("/tmp/safesound_stim")
REPORT_DIR = config.OUTPUT_DIR / "reports"

# 被试看到的标签（不暴露内部命名，避免语义提示）
LABELS = {
    "amplitude": {"alert": "1 (弱)", "high": "2 (中)", "critical": "3 (强)"},
    "category": {
        "gun_shot": "1 (单次短促)",
        "car_horn": "2 (双次)",
        "siren":    "3 (连续起伏)",
    },
}
KEYS = {"amplitude": URGENCY_LEVELS, "category": V3_CLASSES}

# 被试输入的编号 → 内部键（报告里统一用内部键）
ANSWER_MAP = {
    "amplitude": {"1": "alert", "2": "high", "3": "critical"},
    "category": {"1": "gun_shot", "2": "car_horn", "3": "siren"},
}


@dataclass
class Trial:
    trial_id: int
    cls: str
    urgency: str
    answer: str
    rt_s: float


# ---------------------------------------------------------------------------
# 播放
# ---------------------------------------------------------------------------
def play(y: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "stim.wav"
    sf.write(path, y, sr)
    for cmd in (
        ["powershell.exe", "-c",
         f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
        ["aplay", "-q", str(path)],
        ["afplay", str(path)],
    ):
        try:
            subprocess.run(cmd, check=False, timeout=15, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            continue
    print(f"  ⚠️ 无法播放，请手动播放：{path}")


# ---------------------------------------------------------------------------
# 试次编排
# ---------------------------------------------------------------------------
def build_trials(task: str, n_per_code: int, seed: int) -> list[tuple[str, str]]:
    keys = KEYS[task]
    pool: list[tuple[str, str]] = []
    if task == "amplitude":
        cls = "car_horn"                      # 固定类别，只变幅度（S1 设计）
        for u in keys:
            pool += [(cls, u)] * n_per_code
    else:
        for cls in keys:
            for u in URGENCY_LEVELS:          # 幅度随机 → 测正交性
                pool += [(cls, u)] * n_per_code
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    return pool


def ask(task: str) -> str | None:
    opts = "  ".join(LABELS[task].values())
    print(f"    选项：{opts}")
    ans = input("    你的判断（输入编号，q 退出）：").strip().lower()
    if ans in ("q", "quit", ""):
        return None
    mapped = ANSWER_MAP[task].get(ans, ans)
    if mapped not in KEYS[task]:
        print("    ⚠️ 无效输入，按未作答处理")
        return "invalid"
    return mapped


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def truth_of(r: Trial, task: str) -> str:
    """幅度任务的真值是紧迫度档，类别任务的真值是类别。"""
    return r.urgency if task == "amplitude" else r.cls


def report(results: list[Trial], task: str, labels: list[str]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.answer == truth_of(r, task))
    acc = correct / n if n else 0.0

    idx = {k: i for i, k in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for r in results:
        t = truth_of(r, task)
        if t in idx and r.answer in idx:
            cm[idx[t], idx[r.answer]] += 1

    per_class = {}
    for k in labels:
        rows = [r for r in results if truth_of(r, task) == k]
        per_class[k] = sum(1 for r in rows if r.answer == k) / len(rows) if rows else 0.0

    # 判据：整体 ≥80%，且相邻档（有序量）不得系统性混淆
    passed = acc >= 0.80
    adjacent = {}
    if task == "amplitude":
        for a, b in zip(labels, labels[1:]):
            adjacent[f"{a}→{b}"] = int(cm[idx[a], idx[b]] + cm[idx[b], idx[a]])

    return {
        "task": task, "n_trials": n, "accuracy": acc,
        "per_class_recall": per_class, "confusion": cm.tolist(),
        "labels": labels, "adjacent_confusions": adjacent,
        "pass_80pct": passed,
        "median_rt_s": float(np.median([r.rt_s for r in results])) if results else None,
        "trials": [asdict(r) for r in results],
    }


def write_md(rep: dict, path: Path) -> None:
    lines = [
        f"# 触觉码测试报告 — {rep['task']}",
        "",
        f"- 试次：{rep['n_trials']}　总体准确率：**{rep['accuracy']:.1%}**"
        f"　判据（≥80%）：{'✅ 通过' if rep['pass_80pct'] else '❌ 未通过'}",
        f"- 中位反应时：{rep['median_rt_s']:.2f} s" if rep["median_rt_s"] else "",
        "",
        "## 各类召回率",
        "| 码 | 召回 |",
        "|:--|--:|",
    ]
    for k, v in rep["per_class_recall"].items():
        lines.append(f"| {k} | {v:.1%} |")
    lines += ["", "## 混淆矩阵（行=真值，列=作答）", "", "| 真值\\作答 | "
              + " | ".join(rep["labels"]) + " |", "|:--|" + "--:|" * len(rep["labels"])]
    for lab, row in zip(rep["labels"], rep["confusion"]):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    if rep["adjacent_confusions"]:
        lines += ["", "## 相邻档混淆（有序量应单向递增，不应回跳）", ""]
        for k, v in rep["adjacent_confusions"].items():
            lines.append(f"- {k}：{v} 次")
    path.write_text("\n".join(x for x in lines if x is not None) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["amplitude", "category"], default="amplitude")
    ap.add_argument("--n", type=int, default=10, help="每个码的试次数")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--no-audio", action="store_true", help="只校验刺激设计，不做试听")
    ap.add_argument("--report", type=str, default=None, help="从既有 json 生成报告")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.report:
        rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
        write_md(rep, Path(args.report).with_suffix(".md"))
        print(f"✅ 报告已生成：{Path(args.report).with_suffix('.md')}")
        return

    plan = build_trials(args.task, args.n, args.seed)
    print(f"=== {args.task} 测试：{len(plan)} 试次 ===")
    print("标签：" + "　".join(LABELS[args.task].values()))

    if args.no_audio:
        from collections import Counter
        cnt = Counter(f"{c}:{u}" for c, u in plan)
        for k, v in cnt.items():
            y = render_code(*k.split(":"))
            print(f"  {k:<20} {v} 次  RMS {20*np.log10(max(1e-9, float(np.sqrt(np.mean(y**2))))):.1f} dB"
                  f"  峰值 {np.abs(y).max():.2f}")
        print("\n✅ 刺激设计校验完成（未试听）")
        return

    results: list[Trial] = []
    for i, (cls, urg) in enumerate(plan, 1):
        y = render_code(cls, urg)
        print(f"\n试次 {i}/{len(plan)}")
        if i > 1:
            time.sleep(0.8)                     # 试次间静默，避免掩蔽
        play(y)
        t0 = time.time()
        ans = ask(args.task)
        if ans is None:
            break
        results.append(Trial(i, cls, urg, ans, time.time() - t0))

    labels = KEYS[args.task]
    rep = report(results, args.task, labels)
    out = REPORT_DIR / f"haptic_test_{args.task}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(rep, out.with_suffix(".md"))
    print(f"\n准确率 = {rep['accuracy']:.1%}"
          f"　判据 ≥80%：{'✅ 通过' if rep['pass_80pct'] else '❌ 未通过'}")
    print(f"结果：{out}\n报告：{out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
