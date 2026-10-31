"""
SafeSoundAI — 检测侧数字门禁（audit_detection_numbers.py, P1-1）

【为什么需要】
降噪侧已有 `audit_filter_numbers.py`（57 项断言，任何一处对不上就退出码非 0）。
检测侧此前**没有任何门禁**：91.19% / 93.11% / .8269 / .9259 / .9113 这些数字
只存在于笔记和 PPT 里，仓库里找不到能重跑的脚本。这是「数字对不上原始记录」
最容易发生的地方，也是审稿人第一刀。

本脚本给出两级证据：

  ① 逐样本记录（per-sample records）——唯一可信的原始层
     `outputs/reports/detection_per_sample.csv`
     每行 = 一个 1 s 段：真实标签、预测、置信度、四类概率、录音组、npy 路径

  ② 汇总数字（summary）——全部由 ① 派生，不允许手工填写
     `outputs/reports/detection_eval.json`

`--check` 模式**不加载模型、不读 metadata**，只从 ① 重算 ② 的每一个字段。
只要有人手改了汇总数字、或换了模型没重跑逐样本记录，check 就失败。

【口径（写死在代码里，改口径必须同步改这里与笔记）】
  · 段级（segment-level）：3560 条 test 段去掉 fireworks → 3533 段，逐段 argmax
  · 录音级（recording-level / 运营口径）：按 `rec_group` 把该类概率取 max 再 argmax
    —— 这是「一段录音里出现过该事件」的定义，与产品语义一致
    （人耳判断「这首有警笛」看的是整段，不是某一秒）
  · macro-F1：四类等权（不是按样本数加权）；background 也计入
  · 两者都报，禁止只报对自己有利的那个

用法：
    # 生成 / 刷新逐样本记录与汇总
    python3 audit_detection_numbers.py --eval --prep raw

    # CI 门禁：只校验汇总与逐样本记录一致（不加载模型，秒级）
    python3 audit_detection_numbers.py --check

退出码：0 = 全部通过；1 = 有断言失败；2 = 产物缺失。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

REPORT_DIR = Path(config.OUTPUT_DIR) / "reports"
CSV = REPORT_DIR / "detection_per_sample.csv"
JSON = REPORT_DIR / "detection_eval.json"
META_V2 = Path(config.PREPROCESSED) / "trigger_v2" / "metadata_v2.csv"
DEFAULT_CKPT = Path(config.MODEL_DIR) / "dscnn_v2_4cls_fft1024.pt"

#: 与 `train_dscnn_v2.py --exclude-class fireworks` 对齐：fireworks 因数据不足剔除
EXCLUDE_LABELS = ("fireworks",)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def macro_f1(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> float:
    """四类等权 macro-F1（手写，避免 sklearn 版本口径差异）。"""
    f1s = []
    for c in labels:
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def per_class_report(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict:
    out = {}
    for c in labels:
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        tn = int(np.sum((pred != c) & (y != c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[str(c)] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "precision": round(p, 4), "recall": round(r, 4),
                       "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                       "support": int(np.sum(y == c))}
    return out


def confusion(y: np.ndarray, pred: np.ndarray, n: int) -> list[list[int]]:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred, strict=False):
        m[int(t), int(p)] += 1
    return m.tolist()


# ---------------------------------------------------------------------------
# ① 生成逐样本记录
# ---------------------------------------------------------------------------
def run_eval(ckpt: Path, prep: str, batch_size: int, limit: int | None, device: str) -> dict:
    import torch
    import vocab
    from mel import waveform_to_logmel

    if not META_V2.exists():
        raise SystemExit(f"缺 {META_V2}（先跑 trigger_v2 预处理）")
    head = vocab.load_aed(ckpt, device=device)
    classes = list(head.classes)
    print(f"AED 头: {vocab.describe(head)} | prep={prep} | device={device}")

    df = pd.read_csv(META_V2)
    te = df[(df.split == "test") & (~df.label.isin(EXCLUDE_LABELS))].copy()
    missing = [c for c in classes if c not in set(te.label)]
    if missing:
        raise SystemExit(f"测试集缺类别 {missing}，无法算四类 macro-F1")
    cls2idx = {c: i for i, c in enumerate(classes)}
    te["y"] = te.label.map(cls2idx).astype(int)
    if limit:
        te = te.groupby("y", group_keys=False).head(max(20, limit // len(classes)))
    te = te.reset_index(drop=True)
    print(f"test 段 {len(te)} | 分布 {te.label.value_counts().to_dict()} "
          f"| 录音组 {te.rec_group.nunique()}")

    def prep_one(y: np.ndarray) -> np.ndarray:
        if prep == "zscore":
            y = (y - y.mean()) / (y.std() + 1e-8)
        elif prep == "rms":
            y = y / (np.sqrt(np.mean(y ** 2)) + 1e-8) * 0.05
        return y

    probs_all = np.zeros((len(te), len(classes)), dtype=np.float32)
    model = head.model
    buf, idxs = [], []
    with torch.no_grad():
        for i in range(len(te)):
            row = te.iloc[i]
            y = np.load(row.npy_path).astype(np.float32)
            mel = waveform_to_logmel(prep_one(y), n_fft=head.mel["n_fft"],
                                     hop_length=head.mel["hop_length"],
                                     win_length=head.mel["win_length"],
                                     frames=head.mel["frames"])
            buf.append(mel)
            idxs.append(i)
            if len(buf) == batch_size or i == len(te) - 1:
                x = torch.stack(buf).unsqueeze(1).to(device)      # (B,1,128,100)
                p = torch.softmax(model(x), 1).cpu().numpy()
                probs_all[idxs] = p
                buf, idxs = [], []
    pred = probs_all.argmax(1)
    y = te["y"].values
    out = te[["rec_group", "npy_path", "label", "y"]].copy()
    out["pred"] = pred
    out["p_max"] = probs_all.max(1).round(6)
    for i, c in enumerate(classes):
        out[f"p_{c}"] = probs_all[:, i].round(6)
    out["correct"] = (out.pred == out.y).astype(int)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV, index=False)

    summary = summarize(out, classes)
    summary.update({
        "model_role": "deployment_4cls",
        "scope_note": ("本门禁审计的是**4 类部署头**（siren/car_horn/gun_shot/background），"
                       "即耳机软件实际调用的那个模型。v3 多标签+拒识架构（段级 91.19%、"
                       "录音级 93.11%、mAP .9113）是论文主模型，其逐样本记录在 "
                       "experiments/aed/ 下单独复现，不在本文件覆盖范围内 —— "
                       "两者的数字不可互相引用。"),
        "prep": prep,
        "ckpt": str(ckpt),
        "ckpt_md5": _md5(ckpt),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "mel": head.mel,
        "excluded_labels": list(EXCLUDE_LABELS),
        "csv": str(CSV.relative_to(Path(config.OUTPUT_DIR).parent))
        if str(CSV).startswith(str(Path(config.OUTPUT_DIR).parent)) else str(CSV),
        "csv_md5": _md5(CSV),
    })
    JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n段级 acc {summary['segment']['accuracy']:.4f} | "
          f"macro-F1 {summary['segment']['macro_f1']:.4f} | "
          f"录音级 acc {summary['recording']['accuracy']:.4f} | "
          f"macF1 {summary['recording']['macro_f1']:.4f}")
    print(f"产物: {CSV.name}, {JSON.name}")
    return summary


# ---------------------------------------------------------------------------
# ② 从逐样本记录派生汇总（唯一入口）
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, classes: list[str]) -> dict:
    """**唯一**允许产生检测侧汇总数字的函数。--check 走的也是它。"""
    n = len(classes)
    y = df["y"].to_numpy()
    pred = df["pred"].to_numpy()
    labels = list(range(n))

    seg = {
        "n": int(len(df)),
        "n_recordings": int(df.rec_group.nunique()),
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(macro_f1(y, pred, labels), 4),
        "per_class": per_class_report(y, pred, labels),
        "confusion": confusion(y, pred, n),
        "n_classes": n,
        "classes": classes,
        "label_counts": {str(c): int((y == c).sum()) for c in labels},
    }

    pcols = [f"p_{c}" for c in classes]
    g = df.groupby("rec_group", sort=True)
    agg = g[pcols].max()
    rec_y = g["y"].max()                       # 一个录音组只有一个真实标签
    rec_pred = agg.to_numpy().argmax(1)
    rec = {
        "n": int(len(agg)),
        "accuracy": round(float((rec_pred == rec_y.to_numpy()).mean()), 4),
        "macro_f1": round(macro_f1(rec_y.to_numpy(), rec_pred, labels), 4),
        "per_class": per_class_report(rec_y.to_numpy(), rec_pred, labels),
        "confusion": confusion(rec_y.to_numpy(), rec_pred, n),
        "label_counts": {str(c): int((rec_y.to_numpy() == c).sum()) for c in labels},
    }

    # 运营口径：危险类召回 / 背景误报（背景录音被判成危险类）
    hazard_idx = [i for i, c in enumerate(classes) if c != "background"]
    bg_idx = classes.index("background") if "background" in classes else None
    if bg_idx is not None:
        is_haz = np.isin(y, hazard_idx)
        pred_haz = np.isin(pred, hazard_idx)
        seg["hazard_recall"] = round(float((pred_haz & is_haz).sum() /
                                          max(1, is_haz.sum())), 4)
        seg["background_false_alarm"] = round(float((pred_haz & ~is_haz).sum() /
                                                    max(1, (~is_haz).sum())), 4)
        r_pred_haz = np.isin(rec_pred, hazard_idx)
        r_is_haz = np.isin(rec_y.to_numpy(), hazard_idx)
        rec["hazard_recall"] = round(float((r_pred_haz & r_is_haz).sum() /
                                          max(1, r_is_haz.sum())), 4)
        rec["background_false_alarm"] = round(float((r_pred_haz & ~r_is_haz).sum() /
                                                    max(1, (~r_is_haz).sum())), 4)
    return {"segment": seg, "recording": rec}


# ---------------------------------------------------------------------------
# ③ 门禁
# ---------------------------------------------------------------------------
def run_check(tol: float = 1e-4) -> int:
    if not CSV.exists() or not JSON.exists():
        print(f"❌ 缺产物：{CSV.name} / {JSON.name}。先跑 --eval", file=sys.stderr)
        return 2
    saved = json.loads(JSON.read_text(encoding="utf-8"))
    df = pd.read_csv(CSV)
    classes = saved["segment"]["classes"]
    fresh = summarize(df, classes)

    fails: list[str] = []
    n_ok = 0

    def check(name: str, a, b) -> None:
        nonlocal n_ok
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ok = abs(float(a) - float(b)) <= tol
        else:
            ok = (a == b)
        if ok:
            n_ok += 1
            print(f"✅ {name} = {a}")
        else:
            fails.append(f"{name}: 汇总 {a} != 复算 {b}")
            print(f"❌ {name}: 汇总 {a} != 复算 {b}")

    print("=" * 74)
    print("【L1】逐样本记录完整性")
    print("=" * 74)
    check("CSV 行数 == 汇总 n", saved["segment"]["n"], fresh["segment"]["n"])
    check("CSV 录音组数 == 汇总", saved["segment"]["n_recordings"],
          fresh["segment"]["n_recordings"])
    check("CSV md5 未变", saved["csv_md5"], _md5(CSV))
    check("每行 p_max == 四类概率最大值",
          round(float(np.abs(df.p_max - df[[f"p_{c}" for c in classes]].max(1)).max()), 6), 0.0)
    check("pred == 概率 argmax",
          int((df.pred != df[[f"p_{c}" for c in classes]].to_numpy().argmax(1)).sum()), 0)
    check("每行概率和 == 1（±1e-4）",
          round(float(np.abs(df[[f"p_{c}" for c in classes]].sum(1) - 1).max()), 4) <= 1e-4,
          True)
    check("correct 列与 pred/y 一致", int((df.correct != (df.pred == df.y).astype(int)).sum()), 0)
    check("标签分布未被改动", saved["segment"]["label_counts"],
          fresh["segment"]["label_counts"])

    print("=" * 74)
    print("【L2】段级汇总可由逐样本记录重算")
    print("=" * 74)
    for k in ("accuracy", "macro_f1"):
        check(f"段级 {k}", saved["segment"][k], fresh["segment"][k])
    for k in ("hazard_recall", "background_false_alarm"):
        if k in saved["segment"]:
            check(f"段级 {k}", saved["segment"][k], fresh["segment"][k])
    for c, row in fresh["segment"]["per_class"].items():
        check(f"段级 类{c} F1", saved["segment"]["per_class"][c]["f1"], row["f1"])
    check("段级混淆矩阵总和 == n",
          int(np.array(saved["segment"]["confusion"]).sum()), saved["segment"]["n"])
    check("段级混淆矩阵 == 复算", saved["segment"]["confusion"], fresh["segment"]["confusion"])

    print("=" * 74)
    print("【L3】录音级（运营口径）汇总可由逐样本记录重算")
    print("=" * 74)
    for k in ("accuracy", "macro_f1"):
        check(f"录音级 {k}", saved["recording"][k], fresh["recording"][k])
    for k in ("hazard_recall", "background_false_alarm"):
        if k in saved["recording"]:
            check(f"录音级 {k}", saved["recording"][k], fresh["recording"][k])
    for c, row in fresh["recording"]["per_class"].items():
        check(f"录音级 类{c} F1", saved["recording"]["per_class"][c]["f1"], row["f1"])

    print("=" * 74)
    print("【L4】权重与口径锚点")
    print("=" * 74)
    ckpt = Path(saved["ckpt"])
    if ckpt.exists():
        check("ckpt md5 未变（模型没被换掉）", saved["ckpt_md5"], _md5(ckpt))
    else:
        print(f"⚠ 找不到 {ckpt}，跳过 md5 校验（不影响数字溯源）")
    check("prep 口径已记录", saved.get("prep") in ("raw", "zscore", "rms"), True)
    check("录音级 ≥ 段级（录音级聚合应更高，否则口径写反了）",
          saved["recording"]["accuracy"] >= saved["segment"]["accuracy"] - 1e-9, True)

    print("=" * 74)
    if fails:
        print(f"❌ {len(fails)} 项失败 / {n_ok + len(fails)} 项")
        for f in fails:
            print("   -", f)
        return 1
    print(f"✅ {n_ok} / {n_ok} 项通过 —— 检测侧汇总数字全部可由逐样本记录重算")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检测侧数字门禁（P1-1）")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--eval", action="store_true", help="重跑推理，刷新逐样本记录与汇总")
    g.add_argument("--check", action="store_true", help="只校验汇总与逐样本记录一致")
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--prep", choices=["raw", "zscore", "rms"], default="raw",
                    help="推理前预处理，必须与训练一致（train_dscnn_v2.py = raw）")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="调试用：每类最多 N 段")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    if args.eval:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_eval(Path(args.ckpt), args.prep, args.batch_size, args.limit, device)
        return 0
    return run_check(args.tol)


if __name__ == "__main__":
    raise SystemExit(main())
