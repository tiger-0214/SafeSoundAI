"""
SafeSoundAI — 预处理可视化（viz_preprocess_note.py）
==================================================
给「3-预处理全流程」笔记出图。只读 metadata_v2.csv + 1 s npy，不改数据。

出的图（outputs/figures/preprocess/）：
  fig_funnel.png         数据漏斗：波形池 → 类别映射 → 背景下采样 → 去重 → 切分
  fig_class_counts.png   各类段数 vs 独立录音数（双轴，说明「段数≠独立样本」）
  fig_split_stack.png    切分 × 类别堆叠
  fig_wave_mel.png       4 类各一条：1 s 波形 + log-Mel(128×100)
  fig_sr_dist.png        源采样率分布（说明为什么必须统一重采样到 16 kHz）
  fig_win_per_rec.png    每条录音产出的窗数分布
  fig_bg_pool.png        背景池的原始类构成

用法：
    export SAFESOUND_DATA_DIR=/root/safesound_data
    python3 viz_preprocess_note.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import viz_style as vs
from mel import waveform_to_logmel

OUT = vs.FIG_DIR / "preprocess"
OUT.mkdir(parents=True, exist_ok=True)
DATA_V2 = config.PREPROCESSED / "trigger_v2"

CLS = ["siren", "car_horn", "gun_shot", "fireworks", "background"]
COLOR = {"siren": "#d62728", "car_horn": "#ff7f0e", "gun_shot": "#8c1d18",
         "fireworks": "#9467bd", "background": "#7f7f7f"}
SPLIT_COLOR = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}

# 漏斗各阶段（来自 preprocess_v2.py --skip-fsd50k 的实际打印）
FUNNEL = [
    ("波形池\n(urban+esc50)", 72201),
    ("背景下采样\n(预算 12k/2.5k)", 24495),
    ("近似重复剔除", 24451),
]


def fig_funnel(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.4))
    vals = [v for _, v in FUNNEL]
    names = [n for n, _ in FUNNEL]
    y = np.arange(len(vals))[::-1]
    bars = ax.barh(y, vals, color=["#c9d6e5", "#8fa8c8", "#4c72b0"], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("1 s 段数")
    # 右侧标出切分
    tr, va, te = (df.split == "train").sum(), (df.split == "val").sum(), (df.split == "test").sum()
    ax.annotate(f"最终切分：train {tr:,} / val {va:,} / test {te:,}",
                xy=(0.99, 0.06), xycoords="axes fraction", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f3f3f3", ec="0.7"))
    ax.set_title("数据漏斗：72,201 段全量池 → 24,451 段训练集")
    fig.tight_layout()
    fig.savefig(OUT / "fig_funnel.png")
    plt.close(fig)


def fig_class_counts(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    seg = df.label.value_counts().reindex(CLS).fillna(0)
    rec = df.groupby("label").rec_group.nunique().reindex(CLS).fillna(0)
    x = np.arange(len(CLS))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    b1 = ax.bar(x - 0.2, seg.values, 0.4, label="1 s 段数", color=[COLOR[c] for c in CLS], alpha=0.85)
    ax.set_ylabel("1 s 段数")
    ax.set_xticks(x)
    ax.set_xticklabels(CLS)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, rec.values, 0.4, label="独立录音数（rec_group）",
                 color=[COLOR[c] for c in CLS], alpha=0.45, hatch="//", edgecolor="white")
    ax2.set_ylabel("独立录音数")
    ax2.grid(False)
    for b, v in zip(b1, seg.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{int(v):,}", ha="center", fontsize=8)
    for b, v in zip(b2, rec.values):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{int(v):,}", ha="center", fontsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title("段数 vs 独立录音数：段数会把样本量夸大约 7 倍")
    fig.tight_layout()
    fig.savefig(OUT / "fig_class_counts.png")
    plt.close(fig)


def fig_split_stack(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    ct = pd.crosstab(df.label, df.split).reindex(CLS)[["train", "val", "test"]].fillna(0)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    bottom = np.zeros(len(ct))
    for s in ["train", "val", "test"]:
        ax.bar(ct.index, ct[s].values, bottom=bottom, label=s, color=SPLIT_COLOR[s])
        bottom += ct[s].values
    for i, tot in enumerate(bottom):
        ax.text(i, tot * 1.01, f"{int(tot):,}", ha="center", fontsize=8)
    ax.set_ylabel("1 s 段数")
    ax.set_title("切分 × 类别（fold0-7 训练 / fold8 验证 / fold9 测试，按录音分组）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_split_stack.png")
    plt.close(fig)


def fig_wave_mel(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(CLS), 2, figsize=(10, 8.2),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for i, c in enumerate(CLS):
        sub = df[(df.label == c) & (df.split == "test")]
        if sub.empty:
            sub = df[df.label == c]
        r = sub.iloc[len(sub) // 2]
        y = np.load(r.npy_path).astype(np.float32)
        t = np.arange(len(y)) / config.SAMPLE_RATE
        m = waveform_to_logmel(y)
        axes[i, 0].plot(t, y, lw=0.6, color=COLOR[c])
        axes[i, 0].set_ylabel(c)
        axes[i, 0].set_xlim(0, t[-1])
        if i == len(CLS) - 1:
            axes[i, 0].set_xlabel("时间 (s)")
        im = axes[i, 1].imshow(m, origin="lower", aspect="auto", cmap="magma")
        if i == len(CLS) - 1:
            axes[i, 1].set_xlabel("帧（100 帧 ≈ 1 s）")
        axes[i, 1].set_ylabel("Mel 通道 (128)")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.035, pad=0.02)
    axes[0, 0].set_title("1 s 波形（16 kHz, RMS 归一化）")
    axes[0, 1].set_title("log-Mel 特征 (128 × 100)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_wave_mel.png")
    plt.close(fig)


def fig_sr_dist(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    sr = df.orig_sr.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar([f"{int(s/1000)}k" if s >= 1000 else str(int(s)) for s in sr.index],
           sr.values, color="#4c72b0")
    for i, v in enumerate(sr.values):
        ax.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=8)
    ax.set_ylabel("1 s 段数")
    ax.set_xlabel("源采样率")
    ax.set_title(f"源采样率分布：{len(sr)} 种 → 统一重采样到 16 kHz")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sr_dist.png")
    plt.close(fig)


def fig_win_per_rec(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    n = df.groupby(["label", "rec_group"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    data = [n[n.label == c].n.values for c in CLS]
    bp = ax.boxplot(data, tick_labels=CLS, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], CLS):
        patch.set_facecolor(COLOR[c]); patch.set_alpha(0.6)
    for i, c in enumerate(CLS):
        v = n[n.label == c].n
        ax.text(i + 1, v.max() * 1.02 + 0.1, f"中位 {int(v.median())}｜最大 {int(v.max())}",
                ha="center", fontsize=7.5)
    ax.set_ylabel("每条录音产出的 1 s 窗数")
    ax.set_title("窗数≠独立样本：同一录音的多个窗高度相关")
    fig.tight_layout()
    fig.savefig(OUT / "fig_win_per_rec.png")
    plt.close(fig)


def fig_bg_pool(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    bg = df[df.label == "background"]
    top = (bg.groupby("label_raw").rec_group.nunique().sort_values(ascending=False).head(12))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.barh(top.index[::-1], top.values[::-1], color="#7f7f7f")
    for i, v in enumerate(top.values[::-1]):
        ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=8)
    ax.set_xlabel("独立录音数")
    ax.set_title(f"背景池 {bg.rec_group.nunique()} 条录音的原始类构成（top 12 / 共 {bg.label_raw.nunique()} 类）")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bg_pool.png")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_V2 / "metadata_v2.csv")
    print(f"metadata_v2: {len(df)} 段 | {df.rec_group.nunique()} 录音组 | 类 {df.label.value_counts().to_dict()}")
    fig_funnel(df)
    fig_class_counts(df)
    fig_split_stack(df)
    fig_wave_mel(df)
    fig_sr_dist(df)
    fig_win_per_rec(df)
    fig_bg_pool(df)
    for p in sorted(OUT.glob("*.png")):
        print(f"  → {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
