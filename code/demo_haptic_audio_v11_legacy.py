"""⚠️ LEGACY（2026-10-22 归档）—— **本文件依赖 v2 的 11 类 SOUND_PATTERNS，v3 已移除，不可直接运行。**

v3 定稿后触觉编码只剩 3 类 × 3 档 = 9 码，渲染统一走 haptic_render.py。
保留本文件仅为历史参考（git 历史里有可运行版本）。
现行 demo：python3 demo_haptic_v3.py

SafeSoundAI — Demo 3：把触觉编码「听得见」（可听化渲染）
=====================================================
MVP 没有真 LRA 硬件，所以用「175 Hz 载波 + 震动包络」把 HapticPattern 渲染成音频，
听觉等价于手指触摸震动。这是主观评估（可区分度）的前置条件。

同时修掉旧版 subjective_eval.py 的缺陷：旧版只渲染「节奏」，丢掉了
`interval_ms`（紧急度维度）与 `pwm_base`（强度维度），导致 air_conditioner
与 engine_idling 输出音频完全相同（相似度 1.0），无法做盲测。本脚本把四个维度
全部编进音频：

  D1 类型 → 节奏形状（pulse / ramp / single / continuous / random_burst / combo）
  D2 方向 → （MVP 无 DOA，未启用；接口已留）
  D3 响度 → 振幅（pwm_base 0.15~1.0）
  D4 紧急度 → 重复间隔（300 / 500 / 800 / 1200 / 1500 ms）

产出：
  outputs/audio/haptic/<class>.wav         10 类震动可听化（各 3 s）
  outputs/audio/haptic/_contrast_old_vs_new.wav  旧 bug vs 修复后的对比
  outputs/audio/haptic/_alert_sequence.wav 车喇叭→警笛→枪声 连续报警
  outputs/figures/fig10_haptic_envelopes.png     10 类包络对比图
  outputs/reports/demo_haptic.md

用法：python3 demo_haptic_audio.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf

import config
import viz_style as vs
from data_audit import md_table
from haptic_encoder import SOUND_PATTERNS, URGENCY_INTERVALS, generate_vibration

SR = config.SAMPLE_RATE
LRA_HZ = 175.0          # 常见 LRA 共振频率
DUR_S = 3.0
ATTACK_MS, RELEASE_MS = 8, 12       # LRA 起振/停振时间常数
OUT = vs.AUDIO_DIR / "haptic"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 单个 pattern 周期 → 包络（单位 ms 网格）
# ---------------------------------------------------------------------------
def cycle_envelope(pat, rng: np.random.Generator | None = None) -> np.ndarray:
    """返回一个完整「震动—停止」周期的包络（0/1 幅度，长度 = 周期 ms）。"""
    p, kind = pat.params, pat.pattern_type
    segs: list[np.ndarray] = []

    def tone(ms: int) -> np.ndarray:
        return np.ones(int(ms))

    def rest(ms: int) -> np.ndarray:
        return np.zeros(int(ms))

    if kind == "pulse":
        for i in range(int(p.get("count", 1))):
            segs += [tone(p.get("on_ms", 150)), rest(p.get("off_ms", 100))]
        segs.append(rest(p.get("gap_ms", 0)))
    elif kind == "single":
        segs += [tone(p.get("on_ms", 50)), rest(150)]
    elif kind == "ramp":
        d = int(p.get("duration_ms", 1000))
        env = np.sin(np.linspace(0, np.pi, d)) ** 2       # up-down 渐变
        segs += [env, rest(200)]
    elif kind == "continuous":
        segs += [tone(600), rest(200)]
    elif kind == "random_burst":
        rng = rng or np.random.default_rng(0)
        cnt = int(rng.integers(*p.get("count_range", (2, 5))))
        for _ in range(cnt):
            segs += [tone(p.get("on_ms", 80)), rest(p.get("off_ms", 60))]
        segs.append(rest(p.get("gap_ms", 400)))
    elif kind == "combo":
        b_cnt, b_on, b_off = p.get("burst", (3, 200, 100))
        for _ in range(int(b_cnt)):
            segs += [tone(b_on), rest(b_off)]
        segs += [rest(80), tone(p.get("sustain", 300)), rest(p.get("gap_ms", 600))]
    else:
        segs += [tone(150), rest(150)]

    env = np.concatenate([np.asarray(s, dtype=np.float32) for s in segs])
    return env if len(env) else np.ones(100, dtype=np.float32)


def render(pat, dur_s: float = DUR_S, seed: int = 0) -> np.ndarray:
    """把一个 HapticPattern 渲染成 dur_s 秒音频（含 D3 强度 + D4 间隔）。"""
    rng = np.random.default_rng(seed)
    cyc = cycle_envelope(pat, rng)
    gap = max(1, int(pat.interval_ms))                  # 周期之间额外静默 = D4
    unit = np.concatenate([cyc, np.zeros(gap, dtype=np.float32)])
    n_ms = int(dur_s * 1000)
    reps = int(np.ceil(n_ms / len(unit)))
    env_ms = np.tile(unit, reps)[:n_ms]

    # ms 网格 → 采样点
    env = np.repeat(env_ms, SR // 1000)
    env = env[: int(dur_s * SR)]
    env = np.pad(env, (0, int(dur_s * SR) - len(env)))

    # LRA 起振/停振平滑（避免爆音，也更接近真实马达响应）
    k = max(1, ATTACK_MS * SR // 1000)
    kernel = np.hanning(k * 2 + 1)
    kernel /= kernel.sum()
    env = np.convolve(env, kernel, mode="same").astype(np.float32)

    carrier = np.sin(2 * np.pi * LRA_HZ * np.arange(len(env)) / SR).astype(np.float32)
    y = carrier * env * float(pat.pwm_base)            # D3 强度 = 振幅
    return (0.95 * y / max(1e-6, np.abs(y).max())).astype(np.float32) if np.abs(y).max() > 0 else y


# ---------------------------------------------------------------------------
# 旧版行为的复现（用于对比）
# ---------------------------------------------------------------------------
def render_old_bug(cls: str, dur_s: float = DUR_S) -> np.ndarray:
    """复现旧 subjective_eval.py：只看节奏，忽略 interval_ms 与 pwm_base。"""
    kind, params, _ = SOUND_PATTERNS[cls]
    pat = generate_vibration(cls)
    # 旧版：固定 30% 占空比脉冲串，间隔固定 500ms，强度固定
    on = int(params.get("on_ms", 150))
    env = np.concatenate([np.ones(on), np.zeros(max(1, int(on * 2)))])
    unit = np.concatenate([env, np.zeros(500)])
    n = int(dur_s * SR / 1000 * 1000)
    reps = int(np.ceil(n / len(unit)))
    env_ms = np.tile(unit, reps)[:n]
    e = np.repeat(env_ms, SR // 1000)[: int(dur_s * SR)]
    carrier = np.sin(2 * np.pi * LRA_HZ * np.arange(len(e)) / SR)
    y = carrier * e * 0.75
    return (0.95 * y / max(1e-6, np.abs(y).max())).astype(np.float32)


# ---------------------------------------------------------------------------
def fig_envelopes(envs: dict[str, np.ndarray]) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(13, 9), sharex=True)
    for ax, cls in zip(axes.ravel(), config.TARGET_CLASSES):
        pat = generate_vibration(cls, distance_m=8.0)
        y = envs[cls]
        t = np.arange(len(y)) / SR
        ax.fill_between(t, np.abs(y), color=vs.ROLE_COLOR[config.SAFETY_ROLE[cls]], alpha=0.85, lw=0)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_title(f"{cls.replace('_', ' ')} · {pat.pattern_type} · "
                     f"间隔 {pat.interval_ms} ms · 强度 {pat.pwm_base:.2f}",
                     fontsize=9, color=vs.ROLE_COLOR[config.SAFETY_ROLE[cls]])
    for ax in axes[-1]:
        ax.set_xlabel("时间 (s)")
    fig.suptitle("四维触觉编码的可听化包络（振幅=响度档 D3，周期=紧急度 D4，形状=类型 D1）",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return vs.save(fig, "fig10_haptic_envelopes.png")


def signature(y: np.ndarray) -> tuple:
    """粗粒度「指纹」：占空比 + 包络自相关峰 + 峰值，用来验证两两可区分。"""
    e = (np.abs(y) > 0.05).astype(np.float32)
    duty = round(float(e.mean()), 3)
    ac = np.correlate(e[: len(e) // 2], e[: len(e) // 2], "full")[len(e) // 2:]
    peak = int(np.argmax(ac[SR // 20:]) + SR // 20) if len(ac) > SR // 20 else 0
    return (duty, peak // (SR // 100))


def main() -> None:
    print("=" * 70)
    print("Demo 3 · 触觉编码可听化")
    print("=" * 70)

    envs, rows = {}, []
    for cls in config.TARGET_CLASSES:
        pat = generate_vibration(cls, distance_m=8.0)         # 8m → 强度 0.75
        y = render(pat, seed=config.CLASS_TO_IDX[cls])
        envs[cls] = y
        path = OUT / f"{config.CLASS_TO_IDX[cls]:02d}_{cls}.wav"
        sf.write(path, y, SR)
        rows.append({
            "类别": cls, "安全角色": config.SAFETY_ROLE[cls],
            "节奏模式": pat.pattern_type, "紧急度": pat.urgency,
            "重复间隔ms": pat.interval_ms, "强度": pat.pwm_base,
            "占空比": signature(y)[0], "文件": path.name,
        })

    df = pd.DataFrame(rows)
    print(df[["类别", "节奏模式", "紧急度", "重复间隔ms", "强度", "占空比"]].to_string(index=False))

    # 可区分性校验
    sigs = {c: signature(envs[c]) for c in envs}
    uniq = len(set(sigs.values()))
    print(f"\n[可区分性] 10 类指纹唯一数：{uniq}/10 "
          f"{'✅ 全部唯一' if uniq == 10 else '❌ 存在碰撞: ' + str(sigs)}")

    # 旧 bug 对比
    old_a, old_b = render_old_bug("air_conditioner"), render_old_bug("engine_idling")
    new_a, new_b = envs["air_conditioner"], envs["engine_idling"]
    sim_old = float(np.corrcoef(old_a, old_b)[0, 1])
    sim_new = float(np.corrcoef(new_a, new_b)[0, 1])
    print(f"[旧版 bug] air_conditioner vs engine_idling 波形相关："
          f"旧 {sim_old:.3f} → 新 {sim_new:.3f}")
    contrast = np.concatenate([np.zeros(int(0.3 * SR), dtype=np.float32),
                               old_a, np.zeros(int(0.6 * SR), dtype=np.float32),
                               old_b, np.zeros(int(1.0 * SR), dtype=np.float32),
                               new_a, np.zeros(int(0.6 * SR), dtype=np.float32),
                               new_b])
    sf.write(OUT / "_contrast_old_vs_new.wav", contrast, SR)

    # 报警序列：车喇叭 → 警笛 → 枪声
    seq = []
    for cls in ["car_horn", "siren", "gun_shot"]:
        seq.append(envs[cls][: int(1.6 * SR)])
        seq.append(np.zeros(int(0.4 * SR), dtype=np.float32))
    sf.write(OUT / "_alert_sequence.wav", np.concatenate(seq), SR)

    f = fig_envelopes(envs)
    print(f"包络图 → {f}")

    md = ["---\ntags: [project, SafeSoundAI, demo, haptic, sonification]\ncreated: 2026-10-17\n---\n",
          "# Demo 3 · 触觉编码可听化\n",
          "> 175 Hz 载波（LRA 共振）+ 震动包络 → 音频。听到的节奏 = 手指感到的震动。\n",
          "## 四维映射（本 demo 全部编进音频）\n",
          "| 维度 | 声学属性 | 触觉属性 | 实现 |\n|---|---|---|---|\n"
          "| D1 类型 | 事件类别 | 节奏形状 | pulse / ramp / single / continuous / random_burst / combo |\n"
          "| D3 响度/距离 | 能量 | 震动强度 | 振幅 = `pwm_base`（0.15 / 0.30 / 0.50 / 0.75 / 1.00）|\n"
          "| D4 紧急度 | 危险等级 | 重复间隔 | 300 / 500 / 800 / 1200 / 1500 ms |\n"
          "| D2 方向 | 方位角 | LRA 位置 | MVP 未启用（等 A3 DOA）|\n",
          "## 十类渲染结果（强度统一取 8 m 档 = 0.75）\n", md_table(df),
          f"\n**可区分性**：10 类包络指纹唯一数 **{uniq}/10**。\n",
          "## 修复的旧缺陷\n",
          f"旧版 `subjective_eval.py` 只表达节奏，丢掉 `interval_ms` 与 `pwm_base`，"
          f"导致 `air_conditioner` 与 `engine_idling` 的音频完全相同（相关 {sim_old:.3f}），盲测无意义。"
          f"本版把两维编入后，两者相关降至 **{sim_new:.3f}**（差异来自间隔 1500 vs 500 ms）。\n",
          "## 试听文件\n",
          "| 文件 | 内容 |\n|---|---|\n"
          "| `haptic/00_air_conditioner.wav` … `09_street_music.wav` | 10 类各 3 s |\n"
          "| `haptic/_contrast_old_vs_new.wav` | 前 6 s = 旧版（两段，听不出差别）；后 6 s = 修复后（间隔明显不同）|\n"
          "| `haptic/_alert_sequence.wav` | 车喇叭 → 警笛 → 枪声 连续报警 |\n",
          f"\n![包络]({f})\n",
          "\n---\n*由 `demo_haptic_audio.py` 生成。真实 LRA 硬件到位后，"
          "同一份 `HapticPattern` 直接驱动马达，无需改编码逻辑。*"]
    out = config.OUTPUT_DIR / "reports" / "demo_haptic.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"报告 → {out}")


if __name__ == "__main__":
    main()
