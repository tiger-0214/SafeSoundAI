"""构造一段真实音频场景：从 trigger_v2 test 分片拼成 48 kHz wav。

用途：给流式运行时提供**真实音频**输入（不是正弦占位），并强制走 48 kHz→16 kHz 重采样。

拼接口径（v2，修过一个设计缺陷）：
  * 每个事件由 `--segs-per-event` 个同标签 1 s 分片连续拼接而成（默认 2 s）
  * 事件之间留 `--gap` 秒静音（默认 1.0 s）
  * 事件顺序按 siren/background/car_horn/background/gun_shot/background 轮转

v1 的缺陷：事件只有 1 s、间隔 0.5 s，导致任何 1 s 窗口里至少一半是静音，
模型一律判 background，事件级命中率被场景设计压到 20% —— 那是场景的问题，不是模型的问题。
真实警报声持续数秒，所以事件必须长于窗口。

产物：<out>.wav + <out>_events.txt（每行 `起点秒 \t 标签 \t 时长秒`）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

META = Path(config.PREPROCESSED) / "trigger_v2" / "metadata_v2.csv"
SR_OUT = 48000

_ap = argparse.ArgumentParser()
_ap.add_argument("--duration", type=float, default=60.0)
_ap.add_argument("--seed", type=int, default=7)
_ap.add_argument("--out", default="/tmp/scene48k.wav")
_ap.add_argument("--gap", type=float, default=1.0, help="事件间静音秒数")
_ap.add_argument("--segs-per-event", type=int, default=2, help="每个事件拼几个 1 s 分片")
_a = _ap.parse_args()

OUT = Path(_a.out)
TARGET_S = _a.duration

df = pd.read_csv(META)
te = df[(df.split == "test") & (df.label != "fireworks")]
rng = np.random.default_rng(_a.seed)

order = ["siren", "background", "car_horn", "background", "gun_shot", "background"]
gap = np.zeros(int(_a.gap * config.SAMPLE_RATE), dtype=np.float32)
chunks: list[np.ndarray] = []
events: list[tuple[float, str, float]] = []
t = 0.0
i = 0
while t < TARGET_S:
    label = order[i % len(order)]
    cand = te[te.label == label]
    idx = rng.integers(len(cand), size=max(1, _a.segs_per_event))
    segs = [np.load(cand.iloc[int(j)].npy_path).astype(np.float32) for j in idx]
    y = np.concatenate(segs)
    dur = len(y) / config.SAMPLE_RATE
    chunks.append(y)
    events.append((round(t, 3), label, round(dur, 3)))
    t += dur
    chunks.append(gap)
    t += len(gap) / config.SAMPLE_RATE
    i += 1

y16 = np.concatenate(chunks)[: int(TARGET_S * config.SAMPLE_RATE)]
y48 = resample_poly(y16, SR_OUT, config.SAMPLE_RATE).astype(np.float32)
y48 = (y48 / max(float(np.abs(y48).max()), 1e-9) * 0.9).astype(np.float32)
OUT.parent.mkdir(parents=True, exist_ok=True)
sf.write(OUT, y48, SR_OUT, subtype="PCM_16")

ev_path = OUT.with_name(OUT.stem + "_events.txt")
ev_path.write_text("\n".join(f"{tt}\t{lb}\t{du}" for tt, lb, du in events), encoding="utf-8")
print(f"写出 {OUT}: {len(y48) / SR_OUT:.2f}s @ {SR_OUT} Hz, peak {float(np.abs(y48).max()):.3f}")
print(f"事件 {len(events)} 个（gap {_a.gap}s，每事件 {_a.segs_per_event} s），前 6 个：")
for tt, lb, du in events[:6]:
    print(f"  {tt:7.2f}s  {lb:<10} 时长 {du}s")
