"""
SafeSoundAI — Demo 2：合成「城市街道」连续场景的实时检测回放
============================================================
把真实测试片段拼成一段 24 秒的城市街道音频（不同距离 = 不同增益 + 背景噪声），
然后以 1 s 窗 / 0.25 s 步进做滑窗推理，输出：

  1. 时间线图（真值 / 预测 / 置信度 / 触觉强度）   fig09_stream_timeline.png
  2. 报警回放动画（mp4 + gif）                      demo_stream.mp4 / .gif
  3. 场景音频（可直接听）                            demo_stream_scene.wav
  4. 事件日志                                       reports/demo_stream.md

用法：
  export SAFESOUND_DATA_DIR=/root/safesound_data
  python3 demo_stream.py            # 全流程（含动画，约 1-2 分钟）
  python3 demo_stream.py --no-anim  # 只出图和日志
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import viz_style as vs
from augmentation import mix_noise
from dataset import load_splits
from data_audit import md_table
from haptic_encoder import (
    DecisionEngine, Detection, LevelNormalizer, generate_vibration,
)
from mel import waveform_to_logmel
from models import DS_CNN

SR = config.SAMPLE_RATE
WIN = config.SEGMENT_SAMPLES
HOP = WIN // 4                      # 0.25 s 步进 = 更接近实时


# ---------------------------------------------------------------------------
# 场景编排
# ---------------------------------------------------------------------------
@dataclass
class SceneEvent:
    start_s: float
    dur_s: float
    cls: str
    gain: float      # 模拟距离：1.0 近 / 0.35 中 / 0.12 远


# 24 秒脚本：背景 → 喇叭（近）→ 警笛（中）→ 电钻（中）→ 枪声（近）→ 破碎锤（远）
SCENE: list[SceneEvent] = [
    SceneEvent(0.0, 3.0, "street_music", 0.55),      # 远处街头演出
    SceneEvent(3.0, 3.0, "car_horn", 0.95),          # 近距喇叭
    SceneEvent(6.0, 3.0, "air_conditioner", 0.45),   # 中距空调外机
    SceneEvent(9.0, 3.5, "siren", 0.75),             # 中距警笛
    SceneEvent(12.5, 3.0, "drilling", 0.60),         # 中距电钻
    SceneEvent(15.5, 2.5, "street_music", 0.40),     # 更远的街头演出
    SceneEvent(18.0, 2.5, "gun_shot", 1.00),         # 极近枪声
    SceneEvent(20.5, 3.5, "jackhammer", 0.50),       # 中距破碎锤
]
TOTAL_S = 24.0


def build_scene() -> tuple[np.ndarray, np.ndarray, list[SceneEvent]]:
    """拼场景 + 混噪声。返回 (波形, 每采样点真值 class_id, 事件表)。"""
    _, _, test = load_splits()
    n = int(TOTAL_S * SR)
    canvas = np.zeros(n, dtype=np.float32)
    truth = np.full(n, -1, dtype=np.int16)

    for ev in SCENE:
        sub = test[test["target_label"] == ev.cls]
        need = int(ev.dur_s * SR)
        k = int(np.ceil(need / SR))
        # 每次取该类不同文件的真实片段（而非把同一条复制几遍），更接近真实街景
        picks = sub.sample(n=min(k, len(sub)), replace=False, random_state=hash(ev.cls) % 2**31)
        seg = np.concatenate([np.load(p_).astype(np.float32) for p_ in picks["npy_path"]])[:need]
        a = int(ev.start_s * SR)
        b = min(a + need, n)
        canvas[a:b] += seg[: b - a] * ev.gain
        truth[a:b] = config.CLASS_TO_IDX[ev.cls]

    # 背景噪声（用 ESC-50 噪声池，混合到 18 dB SNR —— 模拟真实街道底噪）
    noise_df = pd.read_csv(config.METADATA_CLEAN)
    pool = noise_df[noise_df["role"] == "noise"]["npy_path"].tolist()
    rng = np.random.default_rng(7)
    noise = np.zeros(n, dtype=np.float32)
    for k in range(0, n, len(np.load(pool[0]))):
        p = pool[int(rng.integers(0, len(pool)))]
        z = np.load(p).astype(np.float32)
        m = min(len(z), n - k)
        noise[k:k + m] += z[:m]
    canvas = mix_noise(canvas.astype(np.float64), noise.astype(np.float64), snr_db=18.0).astype(np.float32)

    # 稳健电平归一：目标整体 RMS 0.10（≈「近 3-10 m」档），保留事件间相对距离
    canvas = canvas / max(1e-6, rms_of(canvas)) * 0.10
    canvas = np.clip(canvas, -1.0, 1.0)
    return canvas, truth, SCENE


# ---------------------------------------------------------------------------
# 滑窗推理 + 触觉
# ---------------------------------------------------------------------------
def rms_of(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def level_from_rms(rms: float) -> tuple[float, str]:
    if rms > 0.25:
        return 2.0, "极近"
    if rms > 0.10:
        return 8.0, "近"
    if rms > 0.04:
        return 20.0, "中"
    if rms > 0.015:
        return 40.0, "远"
    return 60.0, "极远"


def normalize_window(x: np.ndarray) -> np.ndarray:
    """逐窗 z-score —— 与训练期预处理完全一致（否则响度差异会系统性跑偏）。"""
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


@torch.no_grad()
def infer(model, y: np.ndarray, device, threshold: float):
    starts = np.arange(0, len(y) - WIN + 1, HOP)
    mels = torch.stack([waveform_to_logmel(normalize_window(y[s:s + WIN])).unsqueeze(0)
                        for s in starts]).to(device)
    probs = torch.softmax(model(mels), 1).cpu().numpy()
    eng = DecisionEngine(confidence_threshold=threshold)
    nz = LevelNormalizer()          # 相对电平分位（约掉链路增益，见双支路口径）
    active = {"cls": None, "start": 0.0, "recent": []}

    rows = []
    for s, p in zip(starts, probs):
        top = int(p.argmax())
        name = config.TARGET_CLASSES[top]
        rms = rms_of(y[s:s + WIN])
        level_db = 20.0 * np.log10(max(1e-9, rms))     # 电平取自**原始信号**（非降噪后）
        dist, lvl = level_from_rms(rms)                # 仅用于图注档位文字
        dets = [Detection(int(i), float(p[i])) for i in np.argsort(-p)[:3] if p[i] >= threshold * 0.6]
        win = eng.arbitrate(dets)
        fired = win is not None
        pat = None
        rel_q = None
        if fired:
            nz.push(name, level_db)
            rel_q = nz.quantile(name, level_db)
            t_now = s / SR
            if active["cls"] != name:
                active["cls"], active["start"] = name, t_now
            dur = t_now - active["start"]
            active["recent"] = [(tt, c) for tt, c in active["recent"] if t_now - tt <= 10.0]
            active["recent"].append((t_now, name))
            reps = sum(1 for _, c in active["recent"] if c == name) - 1
            pat = generate_vibration(name, duration_s=dur, repeats_10s=max(0, reps),
                                    rel_quantile=rel_q, scheme="ear")
        rows.append({
            "t": s / SR, "cls": name, "conf": float(p[top]), "rms": rms,
            "level": lvl, "fired": fired,
            "level_db": level_db, "rel_q": rel_q,
            "urgency": pat.urgency if pat else "-",
            "pattern": pat.pattern_type if pat else "-",
            "interval": pat.interval_ms if pat else 0,
            "strength": pat.pwm_base if pat else 0.0,
            "latent": p,
        })
    return starts, probs, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig_timeline(y, truth, df, mels_ok=True) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = df["t"].values
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.25, 1.1], hspace=0.42)

    # (a) 真值 vs 预测色带
    ax = fig.add_subplot(gs[0])
    tt = np.arange(len(y)) / SR
    ax.plot(tt, y, lw=0.35, color="#cfd8dc", zorder=1)
    # 真值色带
    for ev in SCENE:
        ax.axvspan(ev.start_s, ev.start_s + ev.dur_s, color=vs.ROLE_COLOR[config.SAFETY_ROLE[ev.cls]],
                   alpha=0.13, lw=0)
        ax.text(ev.start_s + ev.dur_s / 2, ax.get_ylim()[1] * 0.82, ev.cls.replace("_", "\n"),
                ha="center", va="top", fontsize=7.5, color=vs.ROLE_COLOR[config.SAFETY_ROLE[ev.cls]])
    # 预测点
    for x, name, conf in zip(t + 0.5, df["cls"], df["conf"]):
        ax.plot(x, -1.05, marker="o", ms=3.2,
                color=vs.ROLE_COLOR[config.SAFETY_ROLE[name]], alpha=min(1, 0.25 + conf))
    ax.set_xlim(0, TOTAL_S)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("时间 (s)")
    ax.set_title("(a) 场景波形 + 真值区间（色带）+ 预测点（底部圆点，颜色=安全角色）")

    # (b) 置信度 + 关键类概率
    ax = fig.add_subplot(gs[1])
    ax.plot(t + 0.5, df["conf"], "-", color=vs.INK, lw=1.4, label="Top-1 置信度")
    for cls, color in [("siren", vs.ACCENT), ("car_horn", vs.ACCENT2),
                       ("gun_shot", "#8e44ad"), ("jackhammer", vs.WARN)]:
        ax.plot(t + 0.5, [p[config.CLASS_TO_IDX[cls]] for p in df["latent"]], lw=1.1,
                color=color, label=cls)
    ax.axhline(0.8, ls="--", color=vs.ACCENT, lw=1.1)
    ax.text(0.2, 0.82, "触发阈值 0.8", fontsize=8, color=vs.ACCENT)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, TOTAL_S)
    ax.set_ylabel("概率")
    ax.set_title("(b) 关键危险类概率随时间变化")
    ax.legend(ncol=5, fontsize=8.5, loc="upper left")

    # (c) 触觉强度
    ax = fig.add_subplot(gs[2])
    fired = df["fired"].values
    colors = [vs.ROLE_COLOR[config.SAFETY_ROLE[c]] for c in df["cls"]]
    ax.bar(t + 0.5, np.where(fired, df["strength"], 0), width=HOP / SR * 0.85,
           color=colors, alpha=0.9)
    for i, r in df.iterrows():
        if r["fired"]:
            ax.text(r["t"] + 0.5, r["strength"] + 0.03, f"{r['pattern'][:6]}\n{r['interval']}ms",
                    ha="center", fontsize=6.5, color=vs.INK)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, TOTAL_S)
    ax.set_ylabel("震动强度 (PWM)")
    ax.set_xlabel("时间 (s)")
    ax.set_title("(c) 触觉输出：报警窗口的震动强度 / 模式 / 重复间隔")
    handles = [Patch(color=vs.ROLE_COLOR[r], label=vs.ROLE_LABEL[r]) for r in vs.ROLE_COLOR]
    ax.legend(handles=handles, fontsize=8, ncol=4, loc="upper right")

    fig.suptitle(f"SafeSoundAI 实时检测回放 · 合成城市街道 {TOTAL_S:.0f}s · "
                 f"1s 窗 / 0.25s 步进 · {len(df)} 帧", fontsize=13.5, fontweight="bold")
    return vs.save(fig, "fig09_stream_timeline.png")


def render_animation(y, df, outdir: Path) -> Path | None:
    """逐帧动画：波形游标 + 当前窗 Mel + 10 类概率条 + 触觉决策。"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    fig = plt.figure(figsize=(12.8, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.22)

    ax_w = fig.add_subplot(gs[0, 0])
    tt = np.arange(len(y)) / SR
    ax_w.plot(tt, y, lw=0.3, color="#b0bec5")
    ax_w.set_xlim(0, TOTAL_S); ax_w.set_ylim(-1.0, 1.0)
    ax_w.set_yticks([]); ax_w.set_xlabel("时间 (s)")
    ax_w.set_title("输入音频（游标 = 当前分析窗）")
    cursor = ax_w.axvspan(0, 1, color=vs.ACCENT, alpha=0.18)
    dot, = ax_w.plot([], [], "o", color=vs.ACCENT, ms=7, zorder=5)

    ax_m = fig.add_subplot(gs[1, 0])
    ax_m.set_title("当前 1 s 窗 log-Mel（128×100）")
    ax_m.set_xticks([]); ax_m.set_yticks([]); ax_m.grid(False)
    im = ax_m.imshow(waveform_to_logmel(y[:WIN]).numpy(), origin="lower", aspect="auto",
                     cmap="magma", vmin=-12, vmax=2)

    ax_p = fig.add_subplot(gs[:, 1])
    names = [c.replace("_", " ") for c in config.TARGET_CLASSES]
    bars = ax_p.barh(names, np.zeros(10), color=vs.class_colors(), height=0.68)
    ax_p.set_xlim(0, 1.02)
    ax_p.invert_yaxis()
    ax_p.set_xlabel("概率")
    ax_p.set_title("分类概率")
    ax_p.axvline(0.8, ls="--", lw=1, color=vs.ACCENT)

    txt = fig.text(0.5, 0.012, "", ha="center", fontsize=12, family="sans-serif")

    writer = FFMpegWriter(fps=6, bitrate=2200)
    out = outdir / "demo_stream.mp4"
    with writer.saving(fig, str(out), dpi=130):
        for _, r in df.iterrows():
            s = int(r["t"] * SR)
            cursor.set_x(r["t"]); cursor.set_width(1.0)
            dot.set_data([r["t"] + 0.5], [0])
            im.set_data(waveform_to_logmel(y[s:s + WIN]).numpy())
            for b, p in zip(bars, r["latent"]):
                b.set_width(float(p))
            if r["fired"]:
                msg = (f"🔔 警报 · {r['cls']} · {r['level']} · 震动 {r['pattern']}"
                       f" / {r['interval']}ms / 幅度 {r['strength']:.2f}")
                txt.set_color(vs.ROLE_COLOR[config.SAFETY_ROLE[r["cls"]]])
            else:
                msg = f"… {r['cls']}（置信度 {r['conf']:.2f} < 阈值，不打扰）"
                txt.set_color(vs.MUTED)
            txt.set_text(f"t = {r['t']:5.2f}s ｜ {msg}")
            writer.grab_frame()
    plt.close(fig)

    # mp4 → gif（可直接在聊天/网页里看）
    gif = outdir / "demo_stream.gif"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", str(out), "-vf",
                        "fps=8,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                        str(gif)], check=False, capture_output=True)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}")

    print("[1/4] 合成 24 s 城市街道场景 ...")
    y, truth, scene = build_scene()
    import soundfile as sf
    sf.write(vs.AUDIO_DIR / "demo_stream_scene.wav", y, SR)
    print(f"      场景音频 → {vs.AUDIO_DIR / 'demo_stream_scene.wav'}（{len(y) / SR:.1f}s, RMS {rms_of(y):.3f}）")

    print("[2/4] 加载 DS-CNN 滑窗推理 ...")
    model = DS_CNN().to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "ds_cnn_best.pth",
                                     map_location=device, weights_only=True))
    model.eval()
    starts, probs, df = infer(model, y, device, args.threshold)

    # 逐窗命中统计（窗中心落在真值区间内才算）
    pred_idx = np.array([config.CLASS_TO_IDX[c] for c in df["cls"]])
    centers = np.array([s + WIN // 2 for s in starts])
    gt = truth[centers]
    mask = gt >= 0
    hit = (pred_idx == gt)[mask]
    # 「稳态窗」= 窗口首尾都属于同一个事件（排除跨边界过渡窗，这是公平的评估口径）
    steady = []
    for s in starts:
        seg = truth[s:s + WIN]
        steady.append(seg[0] >= 0 and (seg == seg[0]).all())
    steady = np.array(steady)
    hit_steady = (pred_idx == truth[centers])[steady]
    print(f"      帧数 {len(df)} ｜ 全量命中率 {hit.mean():.1%}（{mask.sum()} 帧）"
          f" ｜ 稳态窗命中率 {hit_steady.mean():.1%}（{steady.sum()} 帧，排除跨事件边界过渡窗）")
    print(f"      报警帧 {int(df['fired'].sum())} / {len(df)}")

    print("[3/4] 时间线图 ...")
    f = fig_timeline(y, truth, df)
    print(f"      → {f}")

    if not args.no_anim:
        print("[4/4] 渲染动画（mp4 + gif）...")
        out = render_animation(y, df, vs.OUTPUT_DIR)
        print(f"      → {out}")
        g = vs.OUTPUT_DIR / "demo_stream.gif"
        if g.exists():
            print(f"      → {g}（{g.stat().st_size / 1e6:.1f} MB）")

    # 事件日志
    ev = df[df["fired"]][["t", "cls", "conf", "rms", "level", "urgency", "pattern", "interval", "strength"]].copy()
    ev["t"] = ev["t"].round(2)
    ev["conf"] = ev["conf"].round(3)
    ev["rms"] = ev["rms"].round(3)
    md = ["---\ntags: [project, SafeSoundAI, demo, streaming]\ncreated: 2026-10-17\n---\n",
          "# Demo 2 · 城市街道实时检测回放\n",
          f"> 场景：合成 {TOTAL_S:.0f} s（8 个事件 + ESC-50 背景噪声 @18 dB SNR）｜ "
          f"1 s 窗 / 0.25 s 步进 ｜ 阈值 {args.threshold}\n",
          "## 场景脚本\n",
          md_table(pd.DataFrame([{"起点s": e.start_s, "时长s": e.dur_s, "声音": e.cls,
                                  "模拟增益": e.gain,
                                  "安全角色": config.SAFETY_ROLE[e.cls]} for e in scene])),
          f"\n## 报警事件（{len(ev)} 帧）\n", md_table(ev),
          "\n## 指标\n",
          f"- 全量命中率：{mask.sum()} 帧中 **{hit.mean():.1%}**",
          f"- 稳态窗命中率：{steady.sum()} 帧中 **{hit_steady.mean():.1%}**"
          "（窗口完整落在一个事件内，排除边界过渡窗）",
          f"- 触发报警 {int(df['fired'].sum())}/{len(df)} 帧，"
          f"最小触发强度 {df[df['fired']]['strength'].min():.2f}"
          f" · 最大 {df[df['fired']]['strength'].max():.2f}（幅度=紧迫度 3 档，非距离）",
          "\n## 产物\n",
          "| 文件 | 说明 |\n|---|---|\n"
          f"| `{Path(f).name}` | 时间线三联图（真值/概率/触觉强度）|\n"
          "| `demo_stream.mp4` / `.gif` | 逐帧回放动画 |\n"
          "| `demo_stream_scene.wav` | 场景音频（可同步听）|",
          "\n---\n*由 `demo_stream.py` 自动生成。*"]
    out = config.OUTPUT_DIR / "reports" / "demo_stream.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n日志 → {out}")


if __name__ == "__main__":
    main()
