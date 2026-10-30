"""
SafeSoundAI — Trigger-AED v2 训练（5 类：4 触发 + background）

与 v1 的差别：
    1. 数据源 = trigger_v2/metadata_v2.csv（urban + esc50 + fsd50k，录音级分组切分）
    2. 输出头 5 类 one-hot softmax（siren / car_horn / gun_shot / fireworks / background）
    3. 目标域按 dataset 分层报告 → 同库 in-domain 与跨库 cross-corpus 一目了然
    4. 温度标定（val 上拟合）后报 ECE

运行：
    export SAFESOUND_DATA_DIR=/root/safesound_data
    python train_dscnn.py --epochs 30
    python train_dscnn.py --cross-corpus fsd50k      # 训练集剔除 fsd50k，fsd50k 全部当外部测试
    python train_dscnn.py --epochs 2 --limit 2000    # 冒烟
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset

import config
from mel import waveform_to_logmel
from models import DS_CNN, count_parameters, model_size_mb

CLASSES_V2 = ["siren", "car_horn", "gun_shot", "fireworks", "background"]
NUM_CLASSES_V2 = len(CLASSES_V2)
DATA_V2 = config.PREPROCESSED / "trigger_v2"
META_V2 = DATA_V2 / "metadata_v2.csv"
OUT = config.OUTPUT_DIR


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SegDataset(Dataset):
    """读 1 s .npy 波形 → log-Mel(128,100)。在线计算，不缓存（与 v1 一致）。"""

    def __init__(self, df: pd.DataFrame, train: bool = False, augment: bool = True,
                 n_fft: int | None = None):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.augment = augment and train
        self.n_fft = n_fft            # None = 用 config.N_FFT(512)；1024 可消除退化滤波器

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = np.load(r.npy_path).astype(np.float32)
        if self.augment:
            # 随机增益 ±6 dB
            y = y * float(10 ** (np.random.uniform(-6, 6) / 20))
        mel = (waveform_to_logmel(y) if self.n_fft is None
               else waveform_to_logmel(y, n_fft=self.n_fft))   # (128, 100)
        if self.augment:
            # SpecAugment：2 段频率掩蔽 + 2 段时间掩蔽
            for _ in range(2):
                f = np.random.randint(0, 12)
                f0 = np.random.randint(0, mel.shape[0] - f)
                mel[f0:f0 + f, :] = mel.mean()
            for _ in range(2):
                t = np.random.randint(0, 12)
                t0 = np.random.randint(0, mel.shape[1] - t)
                mel[:, t0:t0 + t] = mel.mean()
        return mel.unsqueeze(0), int(r.class_id)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def predict(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for mel, y in loader:
            logits_all.append(model(mel.to(device)).cpu())
            y_all.append(y)
    return torch.cat(logits_all), torch.cat(y_all).numpy()


def fit_temperature(logits: torch.Tensor, y: np.ndarray) -> float:
    """在 val 上拟合温度 T（单一标量，LBFGS）。"""
    logT = torch.zeros(1, requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    yt = torch.from_numpy(y).long().to(logits.device)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(logT), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def describe(tag: str, logits: torch.Tensor, y: np.ndarray, T: float = 1.0) -> dict:
    probs = F.softmax(logits / T, dim=1).numpy()
    pred = probs.argmax(1)
    rep = classification_report(y, pred, labels=list(range(NUM_CLASSES_V2)),
                                target_names=CLASSES_V2, output_dict=True, zero_division=0)
    return {
        "tag": tag,
        "n": int(len(y)),
        "acc": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
        "per_class": {c: {k: round(float(rep[c][k]), 4) for k in ("precision", "recall", "f1-score")}
                      for c in CLASSES_V2},
        "support": {c: int(rep[c]["support"]) for c in CLASSES_V2},
        "ece": round(ece(probs, y), 4),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    global NUM_CLASSES_V2
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cross-corpus", default="", help="把该数据源整体从训练集剔除，当外部测试域")
    ap.add_argument("--exclude-class", default="",
                    help="剔除若干类（逗号分隔，不能剔 background），class_id 会重映射。例: fireworks")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    ap.add_argument("--n-fft", type=int, default=None,
                    help="STFT 窗长（默认 config.N_FFT=512）。128 Mel 在 257 bin 下有退化滤波器，1024 可消除")
    args = ap.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| torch", torch.__version__)

    df = pd.read_csv(META_V2)
    print(f"metadata_v2: {len(df)} 段 | 类 {df.label.value_counts().to_dict()}")
    print(f"数据集 {df.dataset.value_counts().to_dict()} | split {df.split.value_counts().to_dict()}")

    # --- 剔除指定类（类目实验用）：fireworks 等数据不足的类可先停掉 ---
    excl = [c.strip() for c in args.exclude_class.split(",") if c.strip()]
    if excl:
        if "background" in excl:
            raise SystemExit("background 不能剔除（否则没有负类）")
        missing = [c for c in excl if c not in CLASSES_V2]
        if missing:
            raise SystemExit(f"未知类名 {missing}，可选: {CLASSES_V2}")
        n_before = len(df)
        df = df[~df.label.isin(excl)].copy()
        CLASSES_V2[:] = [c for c in CLASSES_V2 if c not in excl]   # 原地改，全局生效
        NUM_CLASSES_V2 = len(CLASSES_V2)
        label2new = {name: i for i, name in enumerate(CLASSES_V2)}
        df["class_id"] = df.label.map(label2new).astype(int)
        print(f"[exclude-class] 剔除 {excl}: {n_before} → {len(df)} 段 | 剩余类 {CLASSES_V2}")

    ext = args.cross_corpus.strip().lower()
    if ext:
        external = df[df.dataset == ext].copy()
        df = df[df.dataset != ext].copy()
        print(f"[cross-corpus] 外部域 = {ext} ({len(external)} 段) 已从训练/验证剔除")

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    if args.limit:
        tr = tr.groupby("class_id", group_keys=False).head(max(50, args.limit // NUM_CLASSES_V2))
        va = va.head(max(100, args.limit // 10))
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    # class weight：平方根阻尼，防 40:1 不平衡时权重爆炸
    cnt = tr.class_id.value_counts().sort_index().values.astype(float)
    w = (cnt.sum() / (len(cnt) * np.maximum(cnt, 1))) ** 0.5
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("class weight:", {c: round(float(x), 3) for c, x in zip(CLASSES_V2, w.cpu())})

    dl_tr = DataLoader(SegDataset(tr, train=True, n_fft=args.n_fft), batch_size=args.batch_size, shuffle=True,
                       num_workers=4, pin_memory=(device == "cuda"), drop_last=True)
    dl_va = DataLoader(SegDataset(va, n_fft=args.n_fft), batch_size=128, shuffle=False, num_workers=4)
    dl_te = DataLoader(SegDataset(te, n_fft=args.n_fft), batch_size=128, shuffle=False, num_workers=4)

    model = DS_CNN(n_classes=NUM_CLASSES_V2).to(device)
    print(f"DS-CNN v2: {count_parameters(model):,} 参数 / {model_size_mb(model):.2f} MB")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    best_f1, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for mel, y in dl_tr:
            mel, y = mel.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(mel), y, weight=w, label_smoothing=0.05)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        vl, vy = predict(model, dl_va, device)
        vf1 = f1_score(vy, vl.argmax(1).numpy(), average="macro", zero_division=0)
        vacc = accuracy_score(vy, vl.argmax(1).numpy())
        hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 4),
                     "val_acc": round(float(vacc), 4), "val_macro_f1": round(float(vf1), 4)})
        print(f"  ep{ep:02d} loss {tot/max(nb,1):.4f} | val acc {vacc:.4f} F1 {vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  早停于 ep{ep}（val macro-F1 {best_f1:.4f}）")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": CLASSES_V2, "arch": "DS_CNN",
                "n_classes": NUM_CLASSES_V2, "best_val_macro_f1": best_f1},
               OUT / "models" / f"dscnn_v2{('_' + args.tag) if args.tag else ''}.pt")

    # --- 评测 ---
    vl, vy = predict(model, dl_va, device)
    T = fit_temperature(vl, vy)
    te_logits, te_y = predict(model, dl_te, device)
    reports = [describe("test_all", te_logits, te_y, T)]

    # 按数据源分层（in-domain / cross-corpus）
    te_df = te.reset_index(drop=True)
    for ds in sorted(te_df.dataset.unique()):
        m = (te_df.dataset == ds).values
        if m.sum() >= 20:
            reports.append(describe(f"test_{ds}", te_logits[m], te_y[m], T))

    if ext and len(external):
        dl_ex = DataLoader(SegDataset(external, n_fft=args.n_fft), batch_size=128, shuffle=False, num_workers=4)
        ex_logits, ex_y = predict(model, dl_ex, device)
        reports.append(describe(f"external_{ext}", ex_logits, ex_y, T))

    untuned = describe("test_all_uncalibrated", te_logits, te_y, 1.0)
    cm = confusion_matrix(te_y, F.softmax(te_logits / T, 1).argmax(1).numpy(),
                          labels=list(range(NUM_CLASSES_V2))).tolist()

    result = {
        "model": "DS_CNN", "classes": CLASSES_V2,
        "params": count_parameters(model), "size_mb": round(model_size_mb(model), 2),
        "device": device, "epochs_run": len(hist), "best_val_macro_f1": round(best_f1, 4),
        "temperature": round(T, 4),
        "ece_before": untuned["ece"], "ece_after": reports[0]["ece"],
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "cross_corpus": ext or None,
        "reports": reports, "confusion_matrix": cm,
        "history": hist, "elapsed_s": round(time.time() - t0, 1),
    }
    tag = ("_" + args.tag) if args.tag else ""
    (OUT / "reports" / f"aed_v2_eval{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # --- Markdown ---
    L = ["# Trigger-AED v2 评测报告", "",
         f"- 模型：DS-CNN，{result['params']:,} 参数 / {result['size_mb']:.2f} MB",
         f"- 类别：{' / '.join(CLASSES_V2)}",
         f"- 训练 {result['n_train']} / 验证 {result['n_val']} / 测试 {result['n_test']} 段",
         f"- 训练轮数 {result['epochs_run']}（早停），最佳 val macro-F1 {best_f1:.4f}",
         f"- 温度标定 T = {T:.3f}，ECE {untuned['ece']:.4f} → {reports[0]['ece']:.4f}",
         f"- 跨库设置：{ext or '无（各库内部切分）'}", ""]
    for rep in reports:
        L += [f"## {rep['tag']}（n={rep['n']}）", "",
              f"Acc **{rep['acc']:.4f}** · macro-F1 **{rep['macro_f1']:.4f}** · weighted-F1 {rep['weighted_f1']:.4f} · ECE {rep['ece']:.4f}", "",
              "| 类 | P | R | F1 | 支持数 |", "|---|---:|---:|---:|---:|"]
        for c in CLASSES_V2:
            p = rep["per_class"][c]
            L.append(f"| {c} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1-score']:.3f} | {rep['support'][c]} |")
        L.append("")
    L += ["## 混淆矩阵（行=真值，列=预测）", "",
          "| 真\\预 | " + " | ".join(CLASSES_V2) + " |", "|---|" + "---:|" * NUM_CLASSES_V2]
    for i, c in enumerate(CLASSES_V2):
        L.append(f"| {c} | " + " | ".join(str(x) for x in cm[i]) + " |")
    (OUT / "reports" / f"aed_v2_eval{tag}.md").write_text("\n".join(L))

    print("\n" + "=" * 60)
    print(f"测试集 Acc {reports[0]['acc']:.4f} | macro-F1 {reports[0]['macro_f1']:.4f} | T={T:.3f}")
    for rep in reports:
        print(f"  {rep['tag']:22s} n={rep['n']:5d} acc={rep['acc']:.4f} F1={rep['macro_f1']:.4f}")
    print(f"耗时 {result['elapsed_s']:.0f}s → outputs/reports/aed_v2_eval{tag}.md")


if __name__ == "__main__":
    main()
