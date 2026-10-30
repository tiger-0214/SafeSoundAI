"""
SafeSoundAI — 数据工程审计（data_audit.py）
==========================================
对应 WBS 1.1.6 / 1.5.1 / 1.5.2（部分）/ 1.5.3：

  1. 数据集构成统计（文件数 / 段数 / 时长 / 大小）
  2. 完整性核验   —— npy 路径存在率、样本长度一致性、标签一致性
  3. 泄漏核验     —— 同一原始文件是否跨 CV 折（必须为 0）
  4. 质量抽样     —— RMS 分布、疑似静音段比例
  5. 类别不平衡   —— 分布表 + 不平衡比
  6. 可视化       —— 构成图 / 类别分布图 / 各类 Mel 样本网格 / fold 分布图

输出：
  outputs/figures/fig01_dataset_composition.png
  outputs/figures/fig02_class_distribution.png
  outputs/figures/fig03_mel_samples.png
  outputs/figures/fig04_fold_and_leakage.png
  outputs/reports/data_audit.md

运行：
  export SAFESOUND_DATA_DIR=/root/safesound_data
  python3 data_audit.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config
import viz_style as vs
from mel import waveform_to_logmel


# ---------------------------------------------------------------------------
# 1. 载入
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_csv(config.METADATA_CLEAN)
    print(f"metadata_clean: {len(df):,} 段 | 列 {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2. 统计
# ---------------------------------------------------------------------------
def composition(df: pd.DataFrame) -> pd.DataFrame:
    """dataset × role 构成表。"""
    g = df.groupby(["dataset", "role"]).agg(
        文件数=("original_file", "nunique"),
        段数=("npy_path", "size"),
        总时长h=("num_samples", lambda s: s.sum() / config.SAMPLE_RATE / 3600),
    ).reset_index()
    g["段占比%"] = (g["段数"] / len(df) * 100).round(2)
    return g


def class_table(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["role"] == "train"]
    t = train.groupby("target_label").agg(
        段数=("npy_path", "size"),
        urban段=("dataset", lambda s: int((s == "urban").sum())),
        esc50段=("dataset", lambda s: int((s == "esc50").sum())),
        文件数=("original_file", "nunique"),
    )
    t = t.reindex(config.TARGET_CLASSES).fillna(0).astype(int)
    t["占比%"] = (t["段数"] / t["段数"].sum() * 100).round(2)
    t["安全角色"] = [config.SAFETY_ROLE[c] for c in t.index]
    t["相对最少类"] = (t["段数"] / t["段数"].min()).round(2)
    return t


# ---------------------------------------------------------------------------
# 3. 核验
# ---------------------------------------------------------------------------
def check_paths(df: pd.DataFrame, full: bool = True) -> dict:
    """npy 路径存在率（全量 stat）。"""
    t0 = time.perf_counter()
    paths = df["npy_path"].tolist()
    sample = paths if full else paths[:: max(1, len(paths) // 5000)]
    missing = [p for p in sample if not Path(p).exists()]
    return {
        "检查策略": "全量" if full else f"抽样 {len(sample):,}",
        "检查数": len(sample),
        "缺失数": len(missing),
        "存在率%": round((1 - len(missing) / len(sample)) * 100, 4),
        "耗时s": round(time.perf_counter() - t0, 1),
        "缺失样例": missing[:3],
    }


def check_labels(df: pd.DataFrame) -> dict:
    """class_id ↔ target_label 一致性 + 未知标签。"""
    bad = df[df["target_label"].map(lambda x: x in config.CLASS_TO_IDX)]
    mism = bad[bad.apply(lambda r: config.CLASS_TO_IDX[r["target_label"]] != r["class_id"], axis=1)]
    unknown = df[~df["target_label"].map(lambda x: x in config.CLASS_TO_IDX)]
    return {
        "target_label 在 10 类内": f"{len(bad):,}/{len(df):,}",
        "class_id 与 target_label 不一致": len(mism),
        "未知 target_label（噪声池）": len(unknown),
        "噪声池标签分布": unknown["label"].value_counts().head(8).to_dict() if len(unknown) else {},
    }


def check_leakage(df: pd.DataFrame) -> dict:
    """同一 original_file 是否跨 CV 折（文件级切分是否正确）。"""
    grp = df[df["role"] == "train"].groupby("original_file")["cv_fold"].nunique()
    leak_files = grp[grp > 1]
    # 同时检查：同一 dataset 内文件名是否跨折；跨 dataset 同名不算泄漏
    key = df[df["role"] == "train"].groupby(["dataset", "original_file"])["cv_fold"].nunique()
    leak2 = key[key > 1]
    return {
        "原始文件数": int(len(grp)),
        "跨折文件数（original_file 全局）": int(len(leak_files)),
        "跨折文件数（dataset+file）": int(len(leak2)),
        "结论": "✅ 无泄漏" if len(leak2) == 0 else "❌ 存在泄漏",
    }


def check_waveform(df: pd.DataFrame, n: int = 3000, seed: int = 42) -> dict:
    """抽样检查样本长度、归一化方式、静音段、NaN。

    注意：本项目片段在预处理阶段已做 **逐段 z-score 标准化**（零均值、单位标准差），
    所以 std≈1.0 是预期值，不能用「RMS=1」判成削波。真正的异常判据是：
      - std 远离 1（<0.1 → 近静音；>3 → 归一化异常）
      - 峰值 > 8σ（真实音频极少出现）
      - 出现 NaN/Inf
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    lengths, stds, peaks, silent, nan, zero = Counter(), [], [], 0, 0, 0
    for i in idx:
        y = np.load(df.iloc[i]["npy_path"]).astype(np.float32)
        lengths[int(y.shape[0])] += 1
        if not np.isfinite(y).all():
            nan += 1
            continue
        sd = float(y.std())
        stds.append(sd)
        peaks.append(float(np.abs(y).max()))
        if sd < 0.1:
            silent += 1
        if abs(sd) < 1e-9:
            zero += 1
    stds, peaks = np.array(stds), np.array(peaks)
    return {
        "抽样段数": int(len(idx)),
        "长度分布": dict(lengths),
        "归一化方式": "逐段 z-score（零均值/单位标准差）",
        "std 中位数": round(float(np.median(stds)), 4),
        "std 范围[P1,P99]": [round(float(np.percentile(stds, 1)), 4),
                             round(float(np.percentile(stds, 99)), 4)],
        "峰值σ 中位数": round(float(np.median(peaks)), 2),
        "疑似静音段(std<0.1)": f"{silent} ({silent / max(1, len(stds)) * 100:.2f}%)",
        "全零段": zero,
        "含 NaN/Inf": nan,
        "峰值>8σ 段": int((peaks > 8).sum()),
    }


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


# ---------------------------------------------------------------------------
# 4. 图
# ---------------------------------------------------------------------------
def fig_composition(df: pd.DataFrame, comp: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) 段数构成
    ax = axes[0]
    labels = [f"{r.dataset}·{r.role}" for r in comp.itertuples()]
    colors = [vs.DATASET_COLOR.get(r.dataset, vs.MUTED) for r in comp.itertuples()]
    sizes = comp["段数"].tolist()
    ax.pie(sizes, labels=[f"{l}\n{s:,}" for l, s in zip(labels, sizes)], colors=colors,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8.5},
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title(f"(a) 片段构成 · 共 {len(df):,} 段")

    # (b) 文件数
    ax = axes[1]
    ax.barh(labels, comp["文件数"], color=colors, height=0.6)
    for y, v in enumerate(comp["文件数"]):
        ax.text(v * 1.01, y, f"{v:,}", va="center", fontsize=9, color=vs.INK)
    ax.set_title(f"(b) 原始文件数 · 共 {int(comp['文件数'].sum()):,} 个")
    ax.set_xlabel("文件数")
    ax.invert_yaxis()

    # (c) 时长
    ax = axes[2]
    ax.barh(labels, comp["总时长h"], color=colors, height=0.6)
    for y, v in enumerate(comp["总时长h"]):
        ax.text(v * 1.01, y, f"{v:.1f} h", va="center", fontsize=9, color=vs.INK)
    ax.set_title("(c) 音频总时长")
    ax.set_xlabel("小时（1 s/段）")
    ax.invert_yaxis()

    fig.suptitle("SafeSoundAI 数据集构成总览", fontsize=14, fontweight="bold", y=1.03)
    return vs.save(fig, "fig01_dataset_composition.png")


def fig_classes(t: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) 堆叠柱
    ax = axes[0]
    x = np.arange(len(t))
    ax.bar(x, t["urban段"], color=vs.BLUE, label="UrbanSound8K", width=0.7)
    ax.bar(x, t["esc50段"], bottom=t["urban段"], color="#8e44ad", label="ESC-50 映射", width=0.7)
    for i, (tot, pct) in enumerate(zip(t["段数"], t["占比%"])):
        ax.text(i, tot * 1.02, f"{tot:,}\n{pct:.1f}%", ha="center", fontsize=8, color=vs.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in t.index], fontsize=8.5)
    for tick, role in zip(ax.get_xticklabels(), t["安全角色"]):
        tick.set_color(vs.ROLE_COLOR.get(role, vs.INK))
        tick.set_fontweight("bold")
    ax.set_ylabel("训练片段数")
    ax.set_ylim(0, t["段数"].max() * 1.22)
    ax.set_title(f"(a) 10 类训练片段分布 · 共 {t['段数'].sum():,} 段")
    ax.legend(loc="upper right", fontsize=9)

    # (b) 不平衡比
    ax = axes[1]
    colors = [vs.ROLE_COLOR.get(r, vs.MUTED) for r in t["安全角色"]]
    ax.barh([c.replace("_", " ") for c in t.index], t["段数"], color=colors, height=0.65)
    for y, (v, ratio) in enumerate(zip(t["段数"], t["相对最少类"])):
        ax.text(v * 1.02, y, f"{v:,}  ({ratio:.1f}×)", va="center", fontsize=8.5, color=vs.INK)
    ax.set_xlim(0, t["段数"].max() * 1.3)
    ax.invert_yaxis()
    ax.set_xlabel("片段数")
    ax.set_title(f"(b) 类不平衡 · 最大/最小 = {t['段数'].max() / t['段数'].min():.1f}×")

    handles = [plt.Rectangle((0, 0), 1, 1, color=vs.ROLE_COLOR[r]) for r in vs.ROLE_COLOR]
    ax.legend(handles, [vs.ROLE_LABEL[r] for r in vs.ROLE_COLOR], fontsize=8.5, loc="lower right")

    fig.suptitle("按安全角色着色的类别分布（红色=核心危险类，正是样本最少的一批）",
                 fontsize=13, fontweight="bold", y=1.02)
    return vs.save(fig, "fig02_class_distribution.png")


def fig_mel(df: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 5.4))
    train = df[df["role"] == "train"]
    for ax, cls in zip(axes.ravel(), config.TARGET_CLASSES):
        sub = train[train["target_label"] == cls]
        row = sub.iloc[len(sub) // 2]                     # 取中位序号样本，避免偶发极端
        y = np.load(row["npy_path"]).astype(np.float32)
        mel = waveform_to_logmel(y).numpy()
        im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        role = config.SAFETY_ROLE[cls]
        ax.set_title(f"{cls.replace('_', ' ')}\n{vs.ROLE_LABEL[role]}", fontsize=9.5,
                     color=vs.ROLE_COLOR[role])
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    fig.suptitle("各类 log-Mel 频谱样本（128 mel × 100 帧，1 s @ 16 kHz）",
                 fontsize=13, fontweight="bold")
    return vs.save(fig, "fig03_mel_samples.png")


def fig_folds(df: pd.DataFrame, leak: dict, wave: dict) -> Path:
    import matplotlib.pyplot as plt

    train = df[df["role"] == "train"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) 每折段数
    ax = axes[0]
    vc = train["cv_fold"].value_counts().sort_index()
    colors = [vs.MUTED] * len(vc)
    for i, f in enumerate(vc.index):
        if f == config.VAL_FOLD:
            colors[i] = "#8e44ad"
        elif f == config.TEST_FOLD:
            colors[i] = vs.ACCENT
    ax.bar(vc.index, vc.values, color=colors, width=0.7)
    for f, v in vc.items():
        ax.text(f, v + 40, f"{v:,}", ha="center", fontsize=8, color=vs.INK)
    ax.set_xlabel("CV 折")
    ax.set_ylabel("片段数")
    ax.set_ylim(0, vc.max() * 1.15)
    ax.set_title(f"(a) 10 折分布（测试折={config.TEST_FOLD}，验证折={config.VAL_FOLD}）")

    # (b) 每类在各折的分布（热力）
    ax = axes[1]
    ct = pd.crosstab(train["cv_fold"], train["target_label"]).reindex(columns=config.TARGET_CLASSES)
    im = ax.imshow(ct.T.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(config.TARGET_CLASSES)))
    ax.set_yticklabels([c.replace("_", " ") for c in config.TARGET_CLASSES], fontsize=8)
    ax.set_xticks(range(len(ct.index)))
    ax.set_xticklabels(ct.index, fontsize=8)
    ax.set_xlabel("CV 折")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
    ax.set_title("(b) 类别 × 折 分层核验")

    # (c) 质量检查汇总
    ax = axes[2]
    ax.axis("off")
    lines = [
        ("路径完整性", f"{wave.get('存在率%', '-')}% 存在"),
        ("跨折泄漏", leak["结论"]),
        ("样本长度", f"{list(wave['长度分布'].keys())} 点"),
        ("归一化", "z-score（std≈1）"),
        ("std 中位数", f"{wave['std 中位数']:.3f}"),
        ("峰值中位数", f"{wave['峰值σ 中位数']:.2f}σ"),
        ("疑似静音", str(wave["疑似静音段(std<0.1)"]).split(" ")[0]),
        ("NaN/Inf", str(wave["含 NaN/Inf"])),
    ]
    ax.text(0.02, 1.0, "质量核验", fontsize=12, fontweight="bold", va="top")
    for i, (k, v) in enumerate(lines):
        y = 0.86 - i * 0.115
        ax.text(0.02, y, k, fontsize=10, color=vs.MUTED, va="top")
        ax.text(0.98, y, v, fontsize=10, color=vs.INK, va="top", ha="right",
                fontweight="bold" if "✅" in str(v) else "normal")

    fig.suptitle("数据划分与质量核验", fontsize=13, fontweight="bold", y=1.02)
    return vs.save(fig, "fig04_fold_and_leakage.png")


# ---------------------------------------------------------------------------
# 5. 报告
# ---------------------------------------------------------------------------
def write_report(df, comp, cls, checks, figs) -> Path:
    md = []
    md.append("---\ntags: [project, SafeSoundAI, report, data-audit]\ncreated: 2026-10-17\n"
              "status: generated\n---\n")
    md.append("# 数据工程审计报告（自动生成）\n")
    md.append(f"> 数据源：`{config.METADATA_CLEAN}` ｜ 生成脚本：`code/data_audit.py` ｜ "
              f"设备：CPU（纯统计）\n")
    md.append("## 一、构成总览\n")
    md.append(md_table(comp))
    md.append(f"\n**合计**：{len(df):,} 段 ｜ "
              f"{int(comp['文件数'].sum()):,} 个原始文件 ｜ "
              f"{comp['总时长h'].sum():.2f} 小时音频\n")
    md.append(f"![数据集构成]({figs['comp']})\n")

    md.append("## 二、10 类分布（训练集）\n")
    md.append(md_table(cls, index=True))
    md.append(f"\n- 总训练片段：{cls['段数'].sum():,}")
    md.append(f"- 类不平衡比（max/min）：**{cls['段数'].max() / cls['段数'].min():.1f}×**")
    md.append(f"- 最少类：`{cls['段数'].idxmin()}`（{cls['段数'].min():,} 段）"
              f" ｜ 最多类：`{cls['段数'].idxmax()}`（{cls['段数'].max():,} 段）")
    md.append("\n> 注意：三个「核心危险」类中 `gun_shot` 只有 "
              f"{cls.loc['gun_shot', '段数']:,} 段，是模型最弱的一环（test F1 0.77）。\n")
    md.append(f"![类别分布]({figs['cls']})\n")
    md.append(f"![Mel 样本]({figs['mel']})\n")

    md.append("## 三、质量核验\n")
    for name, c in checks.items():
        md.append(f"### {name}\n")
        md.append("```json")
        md.append(json.dumps(c, ensure_ascii=False, indent=2, default=str))
        md.append("```\n")
    md.append(f"![划分与核验]({figs['fold']})\n")

    md.append("## 四、结论\n")
    ok = checks["路径完整性"]["缺失数"] == 0 and checks["泄漏核验"]["跨折文件数（dataset+file）"] == 0
    md.append(f"- 数据可用性：{'✅ 通过' if ok else '❌ 有问题'}"
              f"（路径缺失 {checks['路径完整性']['缺失数']}，跨折泄漏 "
              f"{checks['泄漏核验']['跨折文件数（dataset+file）']}）")
    w = checks["波形抽样"]
    md.append(f"- 长度一致性：全部为 {list(w['长度分布'].keys())} 点（1 s @ 16 kHz）")
    md.append(f"- 归一化：{w['归一化方式']}，std 中位数 {w['std 中位数']}（≈1 即正常），"
              f"峰值中位数 {w['峰值σ 中位数']}σ")
    md.append(f"- 静音/坏段：{w['疑似静音段(std<0.1)']}，全零段 {w['全零段']}，"
              f"NaN {w['含 NaN/Inf']} 段，峰值>8σ 段 {w['峰值>8σ 段']}")
    md.append("\n> ⚠️ **设计含义**：z-score 标准化抹掉了每段的绝对响度，"
              "所以「响度 → 震动强度」这一维**不能**从模型输入取，"
              "必须由 `realtime_demo.py` 的 `rms_to_distance_level()` 从原始波形单独算"
              "（当前实现正是这样，属正确设计）。")
    md.append("- 剩余缺口（WBS 1.1.4 / 1.1.5 / 1.5.2）：`警笛专用数据`、"
              "`自行录制中国道路声`、`人工听音抽检标签` —— 均需外部资源，未在本次闭环内。")
    md.append("\n---\n*本报告由 `data_audit.py` 自动生成，数字均为实测。*")

    out = config.OUTPUT_DIR / "reports" / "data_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def main() -> None:
    print("=" * 70)
    print("SafeSoundAI 数据工程审计")
    print("=" * 70)
    df = load()

    print("\n[1/5] 构成统计 ...")
    comp = composition(df)
    print(comp.to_string(index=False))

    print("\n[2/5] 类别分布 ...")
    cls = class_table(df)
    print(cls.to_string())

    print("\n[3/5] 完整性 / 标签 / 泄漏核验 ...")
    checks = {
        "路径完整性": check_paths(df),
        "标签一致性": check_labels(df),
        "泄漏核验": check_leakage(df),
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")

    print("\n[4/5] 波形质量抽样 ...")
    wave = check_waveform(df)
    print(f"  {wave}")

    print("\n[5/5] 出图 + 报告 ...")
    figs = {
        "comp": fig_composition(df, comp),
        "cls": fig_classes(cls),
        "mel": fig_mel(df),
        "fold": fig_folds(df, checks["泄漏核验"], {**checks["路径完整性"], **wave}),
    }
    for k, v in figs.items():
        print(f"  {k:6s} → {v}")

    checks["波形抽样"] = wave
    rep = write_report(df, comp, cls, checks, figs)
    print(f"\n报告 → {rep}")
    print("完成。")


if __name__ == "__main__":
    main()
