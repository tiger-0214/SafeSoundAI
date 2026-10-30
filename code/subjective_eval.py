"""⚠️ LEGACY（2026-10-22 归档）—— 10 类时代的主观评估脚本，已被 v3 测试框架取代。

问题（27 号 §六）：只表达节奏，丢掉 interval_ms 与 pwm_base；
且 10 类口径已废弃，只保留 3 正类。

现行：
  python3 haptic_test.py --task category --n 10   # 类别轴
  python3 s1_amplitude_pretest.py                 # 幅度轴（S1 预实验）

SafeSoundAI — 主观评估脚本（subjective_eval.py）

【论文核心实验之一】验证「触觉模式可区分度」：被试能否通过震动节奏分辨
警报类型（喇叭 vs 枪声 vs 警笛 …）。

MVP 阶段没有真 LRA 硬件，用「可听化（sonification）」模拟震动节奏——
把触觉脉冲转成音频脉冲（ON=蜂鸣、OFF=静音），让被试先用听觉近似感受节奏，
后续换真 LRA 硬件即可复用同一套盲测框架。

用法：
  python subjective_eval.py --generate              # 生成 10 类可听化 wav
  python subjective_eval.py --blind-test            # 交互式盲测（需音频输出）
  python subjective_eval.py --report                # 从结果生成混淆矩阵报告
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import soundfile as sf

import config
from haptic_encoder import generate_vibration, SOUND_PATTERNS


# ---------------------------------------------------------------------------
# 触觉模式 → 可听化音频（震动节奏转成音频脉冲）
# ---------------------------------------------------------------------------
def _fill_tones(audio: np.ndarray, sr: int, on_segments: list[tuple[float, float]],
                freq: float = 220.0) -> np.ndarray:
    """在 on_segments（秒区间列表）填充正弦蜂鸣。"""
    t = np.arange(len(audio)) / sr
    for start, end in on_segments:
        mask = (t >= start) & (t < end)
        audio[mask] += 0.5 * np.sin(2 * np.pi * freq * t[mask])
    return np.clip(audio, -1, 1)


def pattern_to_audio(sound_type: str, duration_s: float = 2.5, sr: int = 16000) -> np.ndarray:
    """把某类的触觉模式转成音频脉冲序列。"""
    p = generate_vibration(sound_type, distance_m=8.0, scheme="ear")
    ptype, params = p.pattern_type, p.params
    audio = np.zeros(int(sr * duration_s))
    on_segments: list[tuple[float, float]] = []

    ms = lambda v: v / 1000.0  # noqa: E731

    if ptype == "single":
        on_segments.append((0.0, ms(params.get("on_ms", 50))))
    elif ptype == "continuous":
        on_segments.append((0.0, duration_s))
    elif ptype == "ramp":
        # 渐强渐弱：分两段
        d = ms(params.get("duration_ms", 1500))
        on_segments.append((0.0, d))
        on_segments.append((d, min(2 * d, duration_s)))
    elif ptype == "random_burst":
        lo, hi = params.get("count_range", (2, 5))
        n = np.random.default_rng().integers(lo, hi + 1)
        on_ms, off_ms = params.get("on_ms", 80), params.get("off_ms", 60)
        t = 0.0
        for _ in range(n):
            on_segments.append((t, t + ms(on_ms)))
            t += ms(on_ms + off_ms)
    elif ptype == "combo":
        burst_n, on_ms, off_ms = params.get("burst", (3, 200, 100))
        t = 0.0
        for _ in range(burst_n):
            on_segments.append((t, t + ms(on_ms)))
            t += ms(on_ms + off_ms)
        sustain = ms(params.get("sustain", 300))
        on_segments.append((t, t + sustain))
    else:  # pulse / dual_pulse
        on_ms = params.get("on_ms", 150)
        off_ms = params.get("off_ms", 150)
        count = params.get("count", 3)
        gap_ms = params.get("gap_ms", 0)
        t = 0.0
        for _ in range(count):
            on_segments.append((t, t + ms(on_ms)))
            t += ms(on_ms + off_ms)
        t += ms(gap_ms)

    return _fill_tones(audio, sr, on_segments)


# ---------------------------------------------------------------------------
# 生成 / 盲测 / 报告
# ---------------------------------------------------------------------------
def cmd_generate() -> None:
    out_dir = config.OUTPUT_DIR / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(config.TARGET_CLASSES):
        audio = pattern_to_audio(name)
        path = out_dir / f"{i:02d}_{name}.wav"
        sf.write(path, audio, config.SAMPLE_RATE)
        print(f"✅ {path}")


def _play(path: str) -> None:
    """尝试用系统播放器播放 wav（WSL 转发到 Windows）。"""
    for cmd in [
        ["powershell.exe", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
        ["aplay", path],
        ["afplay", path],
    ]:
        try:
            subprocess.run(cmd, check=False, timeout=10)
            return
        except Exception:
            continue
    print(f"  （无法自动播放，请手动播放：{path}）")


def cmd_blind_test() -> None:
    out_dir = config.OUTPUT_DIR / "audio"
    order = np.random.default_rng(config.RANDOM_SEED).permutation(config.NUM_CLASSES)
    results = []
    print("盲测开始：听到震动节奏后，输入你认为的类别编号（0-9）。")
    print("类别：", {i: c for i, c in enumerate(config.TARGET_CLASSES)})
    print("输入 q 提前结束。\n")

    for idx in order:
        name = config.TARGET_CLASSES[idx]
        path = out_dir / f"{idx:02d}_{name}.wav"
        _play(str(path))
        ans = input(f"试次 #{len(results)+1}：这是哪一类？(0-9/q) ").strip()
        if ans.lower() == "q":
            break
        try:
            pred = int(ans)
        except ValueError:
            print("  无效输入，跳过")
            continue
        results.append({"true": int(idx), "pred": pred})
        print(f"  正确答案：{idx} ({name})\n")

    out = config.OUTPUT_DIR / "subjective_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存：{out}")
    _report(results)


def _report(results: list[dict]) -> None:
    if not results:
        print("无结果。")
        return
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] for r in results]
    acc = accuracy_score(y_true, y_pred)
    print(f"\n区分度准确率 = {acc:.2%}（目标 >80%）")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    import pandas as pd
    print(pd.DataFrame(cm, index=config.TARGET_CLASSES,
                       columns=config.TARGET_CLASSES).to_string())


def cmd_report() -> None:
    path = config.OUTPUT_DIR / "subjective_results.json"
    if not path.exists():
        print(f"未找到结果文件：{path}")
        sys.exit(1)
    with open(path) as f:
        results = json.load(f)
    _report(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true")
    g.add_argument("--blind-test", action="store_true")
    g.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.generate:
        cmd_generate()
    elif args.blind_test:
        cmd_blind_test()
    else:
        cmd_report()
