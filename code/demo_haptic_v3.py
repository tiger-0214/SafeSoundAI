"""
SafeSoundAI — Demo 3：v3 触觉编码可听化（demo_haptic_v3.py）

把 3 类 × 3 档 = 9 个触觉码渲染成"能听出来的震动"（175 Hz 载波），
一次听全，便于自检两轴是否正交：

  轴 1  时间 → 类别   gun_shot 单次短促 / car_horn 双次 / siren 连续起伏
  轴 2  幅度 → 紧迫度  alert 40% / high 70% / critical 100%（跨 8 dB）

输出：outputs/audio/v3/<cls>_<urgency>.wav + reports/demo_haptic_v3.md

用法：
  python3 demo_haptic_v3.py            # 生成 9 个 wav + 报告
  python3 demo_haptic_v3.py --play     # 生成后依次播放（需音频输出）
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf

import config
from haptic_encoder import (
    PRIOR_URGENCY, SOUND_PATTERNS, URGENCY_INTERVALS, URGENCY_LEVELS,
    URGENCY_PWM, V3_CLASSES,
)
from haptic_render import SR, render_code, rms_db

OUT = config.OUTPUT_DIR / "audio" / "v3"
REPORTS = config.OUTPUT_DIR / "reports"
SHAPES = {"single": "███", "dual": "███ ·· ███", "ramp": "▁▃▅█▅▃▁"}


def play(path) -> None:
    for cmd in (
        ["powershell.exe", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
        ["aplay", "-q", str(path)],
    ):
        try:
            subprocess.run(cmd, check=False, timeout=15, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--dur", type=float, default=2.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in V3_CLASSES:
        ptype = SOUND_PATTERNS[cls][0]
        for u in URGENCY_LEVELS:
            y = render_code(cls, u, dur_s=args.dur)
            path = OUT / f"{cls}_{u}.wav"
            sf.write(path, y, SR)
            rows.append({
                "码": f"{cls}:{u}", "脉冲形状": SHAPES[ptype],
                "重复间隔ms": URGENCY_INTERVALS[cls],
                "幅度": f"{URGENCY_PWM[u]:.2f}", "峰值": round(float(np.abs(y).max()), 2),
                "RMS(dB)": round(rms_db(y), 1),
                "先验默认": "★" if PRIOR_URGENCY[cls] == u else "",
                "文件": path.name,
            })
            if args.play:
                play(path)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # 自检 1：幅度必须被保留（旧 render() 的峰值归一化会把三档压平）
    peaks = {r["码"]: r["峰值"] for r in rows}
    span = 20 * np.log10(max(peaks.values()) / min(peaks.values()))
    ok_amp = span > 6.0
    # 自检 2：同一紧迫度下三类峰值应一致（幅度轴与类别轴正交）
    same = all(
        abs(peaks[f"{c}:{u}"] - peaks[f"{V3_CLASSES[0]}:{u}"]) < 0.05
        for c in V3_CLASSES for u in URGENCY_LEVELS
    )

    lines = [
        "# Demo 3 — v3 触觉编码可听化（9 码）",
        "",
        "> 175 Hz 载波（LRA 共振）+ 震动包络 + 幅度。听到的节奏 = 手指感到的震动。",
        "",
        "| 码 | 脉冲形状 | 重复间隔 | 幅度 | 峰值 | RMS | 先验默认 |",
        "|:--|:--|--:|--:|--:|--:|:--:|",
    ]
    for r in rows:
        lines.append(f"| `{r['码']}` | `{r['脉冲形状']}` | {r['重复间隔ms']} ms | "
                     f"{r['幅度']} | {r['峰值']:.2f} | {r['RMS(dB)']} dB | {r['先验默认']} |")
    lines += [
        "",
        f"- 幅度跨档：**{span:.1f} dB**　{'✅' if ok_amp else '❌ 被归一化压平'}",
        f"- 同紧迫度下三类峰值一致：{'✅' if same else '❌ 存在类别能量泄漏'}",
        "",
        "*由 `demo_haptic_v3.py` 生成。真实 LRA 硬件到位后，同一套码直接下发到马达占空比。*",
    ]
    (REPORTS / "demo_haptic_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n幅度跨档 {span:.1f} dB {'✅' if ok_amp else '❌'}"
          f"　三类峰值一致 {'✅' if same else '❌'}")
    print(f"wav → {OUT}\n报告 → {REPORTS / 'demo_haptic_v3.md'}")


if __name__ == "__main__":
    main()
