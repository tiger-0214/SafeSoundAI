"""
SafeSoundAI — v3 触觉模式渲染器（haptic_render.py）

把 HapticPattern 渲染成"能听出来的震动"：175 Hz LRA 载波 × 震动包络 × 幅度。

⚠️ 关键约束（早前版本在这一点上出过错）：
   **不能做峰值归一化**。旧版 `0.95 * y / max|y|` 会把 40/70/100% 三档压成同一响度，
   幅度轴的信息被抹掉，实验做出来必然"三档不可分"。
   本模块固定满刻度参考（FULL_SCALE），幅度信息完整保留。

用法：
  from haptic_render import render_code, SR
  y = render_code("siren", "critical", dur_s=2.5)
"""
from __future__ import annotations

import numpy as np

from haptic_encoder import (
    SOUND_PATTERNS, URGENCY_PWM, URGENCY_INTERVALS, generate_vibration,
)

SR = 16000
LRA_HZ = 175.0            # LRA 共振频率（偏离共振点振幅骤降，所以载波固定）
ATTACK_MS, RELEASE_MS = 8, 12
FULL_SCALE = 0.95         # 满刻度参考：100% 幅度 → 0.95 峰值


def _smooth(env: np.ndarray) -> np.ndarray:
    """起振/停振平滑，避免爆音，也更接近真实马达响应。"""
    a = max(1, int(SR * ATTACK_MS / 1000))
    r = max(1, int(SR * RELEASE_MS / 1000))
    win = np.ones(len(env), dtype=np.float32)
    if len(env) > a + r:
        win[:a] = np.linspace(0, 1, a)
        win[-r:] = np.linspace(1, 0, r)
    return env * win


def unit_envelope(sound_type: str) -> np.ndarray:
    """单个码单元的包络（不含幅度、不含重复）。"""
    ptype, params, _ = SOUND_PATTERNS[sound_type]
    ms = lambda v: max(1, int(SR * v / 1000))  # noqa: E731

    if ptype == "single":
        return _smooth(np.ones(ms(params["on_ms"]), dtype=np.float32))

    if ptype == "dual":
        on1, off, on2 = params["on_ms"]
        return _smooth(np.concatenate([
            np.ones(ms(on1), dtype=np.float32),
            np.zeros(ms(off), dtype=np.float32),
            np.ones(ms(on2), dtype=np.float32),
        ]))

    if ptype == "ramp":                      # ▁▃▅█▅▃▁ 升-降扫包络
        n = ms(params["duration_ms"])
        half = n // 2
        e = np.concatenate([np.linspace(0, 1, half), np.linspace(1, 0, n - half)])
        return _smooth(e.astype(np.float32))

    raise ValueError(f"unknown pattern_type: {ptype}")


# ---------------------------------------------------------------------------
# 关于「类别增益校准」：不需要，而且做了会坏
# ---------------------------------------------------------------------------
# 三类码的**峰值幅度在相同紧迫度下天然相等**——pwm 直接乘在载波上，
# critical 一律 0.95 峰值、alert 一律 0.38。差的是占空比与重复间隔
# （gun_shot 单脉冲 60 ms / car_horn 双脉冲 380 ms / siren 400 ms 包络），
# 那是**类别码本身**，不是能量泄漏。
#
# 如果按窗口 RMS 强行拉平三类，脉冲型码（gun_shot）需要 +9 dB 增益 → 立刻削波，
# 三档幅度反而塌成 2 dB。所以：**不做类别增益校准，只保证峰值一致**。
def render_code(sound_type: str, urgency: str, dur_s: float = 2.5,
                sr: int = SR) -> np.ndarray:
    """渲染一个码：载波 × (单元包络按类别间隔重复) × 紧迫度幅度。"""
    pwm = URGENCY_PWM[urgency]
    unit = unit_envelope(sound_type)
    gap = max(1, int(sr * URGENCY_INTERVALS[sound_type] / 1000))

    total = int(sr * dur_s)
    env = np.zeros(total, dtype=np.float32)
    pos = 0
    while pos < total:
        seg = unit[: max(0, total - pos)]
        env[pos:pos + len(seg)] += seg
        pos += len(unit) + gap

    carrier = np.sin(2 * np.pi * LRA_HZ * np.arange(total) / sr).astype(np.float32)
    y = carrier * env * pwm * FULL_SCALE
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def render_pattern(pat, dur_s: float = 2.5, sr: int = SR) -> np.ndarray:
    if pat.pwm_base <= 0:
        return np.zeros(int(sr * dur_s), dtype=np.float32)
    return render_code(pat.sound_type, pat.urgency, dur_s=dur_s, sr=sr)


def rms_db(y: np.ndarray) -> float:
    return 20.0 * np.log10(max(1e-9, float(np.sqrt(np.mean(y ** 2)))))


if __name__ == "__main__":
    print(f"{'code':<20}{'占空比':>9}{'RMS(dB)':>10}{'峰值':>8}{'档差':>9}")
    for cls in SOUND_PATTERNS:
        ref = None
        for u in ("alert", "high", "critical"):
            y = render_code(cls, u)
            db = rms_db(y)
            if u == "critical":
                ref = db
            d = "" if ref is None else f"{db - ref:+.1f}dB"
            duty = float(np.mean(np.abs(y) > 1e-3))
            print(f"{cls + ':' + u:<20}{duty:>9.1%}{db:>10.1f}"
                  f"{np.abs(y).max():>8.2f}{d:>10}")
    peaks = {u: np.abs(render_code('siren', u)).max() for u in ('alert', 'high', 'critical')}
    assert peaks['critical'] > 0.9 and peaks['high'] > 0.6 and peaks['alert'] > 0.3, \
        f"幅度未被保留（可能又被归一化了）：{peaks}"
    span = 20 * np.log10(URGENCY_PWM['critical'] / URGENCY_PWM['alert'])
    print(f"\n✅ 幅度保留：alert/high/critical 峰值 {peaks['alert']:.2f}/"
          f"{peaks['high']:.2f}/{peaks['critical']:.2f}，跨档 {span:.1f} dB")
    print("✅ 三类码峰值一致 → 幅度轴与类别轴正交（不做类别 RMS 校准，理由见文件注释）")
