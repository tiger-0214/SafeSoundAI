"""
SafeSoundAI — Trigger-AED v2 数据审计

对 preprocess.py 产出的 metadata_v2.csv 做逐项体检，输出 Markdown 报告 + 图表。

检查项：
    A. 完整性   段文件是否存在 / 能否加载 / shape 是否 (16000,) / dtype / NaN / Inf
    B. 信号质量 峰值、直流偏置、有效值（归一化后应 ≈1）、削波比例、静音段
    C. 分布     每类段数 / 独立录音数 / 每库贡献 / 每折每类分布
    D. 泄漏     同一录音是否跨折；跨库近似重复是否已剔除
    E. 不平衡   最大类 / 最小类 比值；建议训练期处理方式

用法:
    export SAFESOUND_DATA_DIR=/root/safesound_data
    python data_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

DATA_V2 = config.PREPROCESSED / "trigger_v2"
META = DATA_V2 / "metadata_v2.csv"
REPORT = config.OUTPUT_DIR / "reports" / "data_audit.md"
FIG_DIR = config.OUTPUT_DIR / "figs" / "v2"
CLASSES = ["siren", "car_horn", "gun_shot", "fireworks", "background"]


def check_files(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        p = Path(r.npy_path)
        rec = {"ok": False, "shape": "", "dtype": "", "n_nan": 0, "n_inf": 0,
               "peak": np.nan, "rms": np.nan, "dc": np.nan, "plateau": np.nan,
               "dur_s": np.nan, "silent": False, "err": ""}
        try:
            y = np.load(p)
            rec["shape"] = str(tuple(y.shape))
            rec["dtype"] = str(y.dtype)
            if y.ndim != 1 or y.shape[0] != config.SEGMENT_SAMPLES:
                rec["err"] = "shape_mismatch"
            else:
                rec["ok"] = True
                y64 = y.astype(np.float64)
                rec["n_nan"] = int(np.isnan(y64).sum())
                rec["n_inf"] = int(np.isinf(y64).sum())
                peak = float(np.abs(y64).max())
                rec["peak"] = peak
                rec["rms"] = float(np.sqrt(np.mean(y64 ** 2)))
                rec["dc"] = float(np.mean(y64))
                absy = np.abs(y64)
                # 触顶平台比例：与段内峰值相差 <1e-6 的样本占比。
                # 正弦 ≈0.09%，真正源端削波 >1%（大量样本压在同一个最大值上）
                rec["plateau"] = float((absy >= peak * (1 - 1e-6)).mean())
                rec["dur_s"] = float(y.shape[0] / config.SAMPLE_RATE)
                rec["silent"] = peak < 1e-3
        except Exception as e:
            rec["err"] = type(e).__name__
        rows.append(rec)
    return pd.DataFrame(rows)


def leakage_check(df: pd.DataFrame) -> dict:
    out = {}
    for ds, sub in df.groupby("dataset"):
        g = sub.groupby("rec_group").cv_fold.nunique()
        out[ds] = {"groups": int(g.size), "cross_fold_groups": int((g > 1).sum())}
    return out


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.pivot_table(index="label", columns="split", values="class_id",
                           aggfunc="size", fill_value=0)
              .reindex(CLASSES).fillna(0).astype(int))


def make_figs(df: pd.DataFrame, checks: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fam in ("Noto Serif CJK SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        if any(f.name == fam for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    import torch

    from mel import waveform_to_logmel

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1) 类别 × 数据集堆叠柱状图
    ct = df.pivot_table(index="label", columns="dataset", values="class_id", aggfunc="size", fill_value=0)
    ct = ct.reindex(CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=140)
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][:ct.shape[1]])
    ax.set_ylabel("1 s 段数")
    ax.set_title("Trigger-AED v2 类别 × 数据集")
    ax.legend(title="dataset", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = FIG_DIR / "fig_v2_class_dist.png"
    plt.savefig(p); plt.close()
    paths.append(str(p))

    # 2) 每类 log-Mel 示例
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(7.2, 1.5 * len(CLASSES)), dpi=130)
    for i, c in enumerate(CLASSES):
        sub = df[df.label == c]
        for j in range(3):
            ax = axes[i, j]
            if j < len(sub.sample(min(3, len(sub)), random_state=42)):
                r = sub.sample(min(3, len(sub)), random_state=42).iloc[j]
                mel = waveform_to_logmel(np.load(r.npy_path)).numpy()
                ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(c, fontsize=8)
            else:
                ax.axis("off")
    fig.suptitle("每类 log-Mel 示例 (128×100)", fontsize=9)
    plt.tight_layout()
    p = FIG_DIR / "fig_v2_mel_examples.png"
    plt.savefig(p); plt.close()
    paths.append(str(p))

    # 3) 每类录音数 vs 段数（说明"段数≠样本量"）
    g = df.groupby("label").agg(segs=("class_id", "size"), files=("original_file", "nunique"))
    g = g.reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=140)
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.segs, width=0.4, label="1 s 段数", color="#4C78A8")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, g.files, width=0.4, label="独立录音数", color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20, ha="right")
    ax.set_ylabel("段数"); ax2.set_ylabel("录音数")
    ax.set_title("段数 vs 独立录音数（真正的有效样本量）")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    plt.tight_layout()
    p = FIG_DIR / "fig_v2_seg_vs_file.png"
    plt.savefig(p); plt.close()
    paths.append(str(p))
    return paths


def main() -> None:
    df = pd.read_csv(META)
    print(f"读取 {META}: {len(df)} 段")

    print("A/B 逐段体检…")
    checks = check_files(df)
    df = pd.concat([df.reset_index(drop=True), checks.reset_index(drop=True)], axis=1)

    per_class = (df.groupby("label")
                   .agg(segments=("class_id", "size"),
                        files=("original_file", "nunique"),
                        rec_groups=("rec_group", "nunique"),
                        med_dur=("dur_s", "median"))
                   .reindex(CLASSES))
    per_ds = df.groupby("dataset").size()
    splits = fold_table(df)
    leak = leakage_check(df)
    ratio = float(per_class.segments.max() / per_class.segments.min())

    bad = df[~df.ok]
    silent = df[df.silent.fillna(False)]
    cliphi = df[df.plateau.fillna(0) > 0.01]

    figs = make_figs(df, checks)

    lines = []
    A = lines.append
    A("# Trigger-AED v2 数据审计报告\n")
    A(f"数据源：`{META}`\n")
    A("## 0. 总览\n")
    A(f"- 总段数：**{len(df):,}**")
    A(f"- 独立来源录音：**{df.original_file.nunique():,}**（录音组 {df.rec_group.nunique():,}）")
    A(f"- 类别：{', '.join(CLASSES)}")
    A(f"- 总时长（1 s 段累加）：**{len(df) / 3600:.2f} 小时**")
    A(f"- 类不平衡比（最大/最小）：**{ratio:.1f}×**\n")

    A("## A. 完整性\n")
    A(f"- 可加载且 shape 正确：{int(df.ok.sum()):,} / {len(df):,} "
      f"（{df.ok.mean()*100:.2f}%）")
    A(f"- NaN 段：{int((df.n_nan > 0).sum())}，Inf 段：{int((df.n_inf > 0).sum())}")
    if len(bad):
        A(f"- 异常段（前 10）：")
        for r in bad.head(10).itertuples():
            A(f"  - `{r.npy_path}` → {r.err}")
    A("")

    A("## B. 信号质量\n")
    A("RMS 归一化后每段应有 `rms ≈ 1.0`；`peak` 反映峰均比（爆响类应明显 > 音调类）。\n")
    stat = df.groupby("label").agg(rms=("rms", "median"), peak=("peak", "median"),
                                  dc=("dc", lambda s: float(np.nanmax(np.abs(s)))),
                                  clip_frac=("plateau", "mean")).reindex(CLASSES)
    A("| 类 | rms 中位 | peak 中位 | max abs(dc) | 触顶平台比例均值 |")
    A("|---|---:|---:|---:|---:|")
    for k, r in stat.iterrows():
        A(f"| {k} | {r.rms:.3f} | {r.peak:.3f} | {r.dc:.4f} | {r.clip_frac*100:.2f}% |")
    A("")
    A(f"- 静音段（peak<1e-3）：**{len(silent)}**"
      + (f"，分布 {silent.label.value_counts().to_dict()}" if len(silent) else ""))
    A(f"- 疑似源端削波段（>1% 样本压在段内峰值上）：**{len(cliphi)}**"
      + (f"，分布 {cliphi.label.value_counts().to_dict()}" if len(cliphi) else ""))
    A("")

    A("## C. 分布\n")
    A("| 类 | 1 s 段数 | 源文件 | 录音组 | 段/录音 | 段长中位 (s) |")
    A("|---|---:|---:|---:|---:|---:|")
    for k, r in per_class.iterrows():
        A(f"| {k} | {int(r.segments):,} | {int(r.files):,} | {int(r.rec_groups):,} | "
          f"{r.segments/max(r.files,1):.1f} | {r.med_dur if pd.notna(r.med_dur) else 0.0:.2f} |")
    A("")
    A("按数据集：\n")
    A("| dataset | 段数 |")
    A("|---|---:|")
    for k, v in per_ds.items():
        A(f"| {k} | {int(v):,} |")
    A("")
    A("按切分 × 类别：\n")
    A("| 类 | " + " | ".join(splits.columns) + " |")
    A("|---|" + "---:|" * len(splits.columns))
    for k, r in splits.iterrows():
        A(f"| {k} | " + " | ".join(f"{int(v):,}" for v in r.values) + " |")
    A("")

    A("## D. 泄漏检查\n")
    A("录音级分组切分的判据：同一个 `rec_group` 只能出现在一个 fold。\n")
    A("| dataset | 录音组 | 跨折组数 | 判定 |")
    A("|---|---:|---:|---|")
    for k, v in leak.items():
        A(f"| {k} | {v['groups']:,} | {v['cross_fold_groups']} | "
          f"{'✅ 无泄漏' if v['cross_fold_groups']==0 else '❌ 存在泄漏'} |")
    A("")
    A(f"- 已剔除的近似重复录音：{int(df.dup_of.notna().sum()) if 'dup_of' in df.columns else 0} 段"
      "（跨库均值谱指纹命中，保留 urban > esc50 > fsd50k 优先级）")
    A("")

    A("## E. 不平衡与训练建议\n")
    A(f"- 最大/最小 = **{ratio:.1f}×**（{per_class.segments.idxmax()} vs {per_class.segments.idxmin()}）")
    A(f"- 若按独立录音看：最大/最小 = {per_class.files.max()/per_class.files.min():.1f}×")
    A("- 建议：训练用 `WeightedRandomSampler`（按类倒数频率）或 `CrossEntropyLoss(weight=...)`；"
      "评测一律报 **macro-F1** 与 **事件级召回/误报率**，不看总体准确率。")
    A("")
    A("## 图\n")
    for p in figs:
        A(f"![{Path(p).stem}]({p})")
    A("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 报告: {REPORT}")

    summary = {
        "n_segments": int(len(df)),
        "n_files": int(df.original_file.nunique()),
        "per_class": per_class.segments.astype(int).to_dict(),
        "per_class_files": per_class.files.astype(int).to_dict(),
        "splits": splits.to_dict(),
        "imbalance": round(ratio, 2),
        "bad_segments": int((~df.ok).sum()),
        "silent": int(len(silent)),
        "suspect_clipped": int(len(cliphi)),
        "leakage": leak,
    }
    (DATA_V2 / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """无 tabulate 依赖的 GitHub Markdown 表格。"""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:g}" if abs(v) < 1e5 else f"{v:.3g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
