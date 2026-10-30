"""
SafeSoundAI — 紧急声音检测有效性评估（eval_emergency.py）
=========================================================
回答一个问题：**模型真的能在真实片段里检出我们规划的那几类紧急声音吗？**

普通分类报告只给「段级 F1」，但产品的验收标准是「一声喇叭/枪响/警笛出现时，
用户必须在 1 秒内被提醒，且不能整天乱震」。所以本脚本从 5 个角度测：

  E1 段级检出率    : p(真实类) ≥ τ 的段占比（即"会不会触发触觉"）
  E2 事件级召回    : 按原始文件聚合成事件，任一窗命中即算检出（真实场景语义）
  E3 首次检出时延  : 事件内第一个命中窗的相对时间（0.5 s 步进精度）
  E4 误报率        : 非危险事件被判成危险类 → 次/事件、次/小时
  E5 噪声鲁棒性    : 叠加 ESC-50 噪声池，SNR 20/10/5/0/-5 dB 下的检出率（真实街道）

产出：reports/emergency_eval.md / emergency_metrics.json + fig12~fig14

用法：
  python3 eval_emergency.py [--ckpt ../outputs/models/ds_cnn_best.pth]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
import viz_style as vs
from augmentation import mix_noise
from dataset import AudioSegmentDataset, build_noise_pool, load_splits
from mel import waveform_to_logmel
from models import DS_CNN

REP = vs.REPORT_DIR
CRITICAL = [c for c, r in config.SAFETY_ROLE.items() if r == "critical"]
DANGER = [c for c, r in config.SAFETY_ROLE.items() if r == "danger"]
SAFETY_RELEVANT = CRITICAL + DANGER          # 需要触觉报警的 6 类
HOP_S = 0.5                                   # 滑窗步进（与 50% overlap 一致）
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# 模型 & 推理
# ---------------------------------------------------------------------------
def load_model(ckpt: str, device: torch.device):
    model = DS_CNN()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


@torch.no_grad()
def infer_probs(model, ds: Dataset, device, batch_size: int = 128) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for x, _ in loader:
        logits = model(x.to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 带噪测试集（E5）
# ---------------------------------------------------------------------------
class NoisySegmentDataset(Dataset):
    """把测试段按指定 SNR 与噪声池混合（与训练时 augmentation 完全一致的路径）。"""

    def __init__(self, df: pd.DataFrame, noise_paths: list[str], snr_db: float, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.noise_paths = noise_paths
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = np.load(row["npy_path"]).astype(np.float32)
        if self.snr_db is not None:
            nz = np.load(self.rng.choice(self.noise_paths)).astype(np.float32)
            y = mix_noise(y, nz, self.snr_db)
        mel = waveform_to_logmel(y).unsqueeze(0)
        return mel, int(row["class_id"])


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def segment_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    out = {"accuracy": float((pred == y).mean())}
    per = {}
    for i, c in enumerate(config.TARGET_CLASSES):
        m = y == i
        tp = int(((pred == i) & m).sum())
        fp = int(((pred == i) & ~m).sum())
        fn = int((m & (pred != i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # 触发口径：该类概率 ≥ τ（触觉报警的真实判据，允许多类同时超阈）
        rec_at = {f"R@{t:.1f}": float((probs[m, i] >= t).mean()) if m.sum() else None for t in TAUS}
        per[c] = {"n": int(m.sum()), "precision": prec, "recall": rec, "f1": f1, **rec_at}
    out["per_class"] = per
    return out


def event_frame(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """把段级预测聚合成事件级（同一原始文件 = 一个事件）。"""
    d = df.reset_index(drop=True).copy()
    d["seg_idx"] = d.groupby(["dataset", "original_file"]).cumcount()
    for i, c in enumerate(config.TARGET_CLASSES):
        d[f"p_{c}"] = probs[:, i]
    return d


def event_metrics(ev: pd.DataFrame, taus=TAUS) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (ds, f), g in ev.groupby(["dataset", "original_file"]):
        lab = g["target_label"].iloc[0]
        n = len(g)
        dur = (n - 1) * HOP_S + config.SEGMENT_S
        p_lab = g[f"p_{lab}"].to_numpy()
        row = {"dataset": ds, "file": f, "label": lab, "n_seg": n, "dur_s": dur,
               "role": config.SAFETY_ROLE.get(lab, "background")}
        for t in taus:
            hit = np.where(p_lab >= t)[0]
            row[f"hit@{t:.1f}"] = len(hit) > 0
            row[f"ttd@{t:.1f}"] = float(hit[0] * HOP_S) if len(hit) else None
        # 误报：任何危险类概率 ≥ τ（对所有 τ 都算，供阈值权衡表用）
        for t in taus:
            row[f"fa@{t:.1f}"] = any((g[f"p_{c}"].to_numpy() >= t).any() for c in SAFETY_RELEVANT)
        rows.append(row)
    ev_df = pd.DataFrame(rows)

    res = {}
    for t in taus:
        key = f"@{t:.1f}"
        det = ev_df[ev_df.label.isin(SAFETY_RELEVANT)]
        bg = ev_df[~ev_df.label.isin(SAFETY_RELEVANT)]
        res[key] = {
            "event_recall_all_danger": float(det[f"hit{key}"].mean()) if len(det) else None,
            "event_recall_critical": float(det[det.label.isin(CRITICAL)][f"hit{key}"].mean()),
            "event_recall_danger": float(det[det.label.isin(DANGER)][f"hit{key}"].mean()),
            "per_class_event_recall": {
                c: float(det[det.label == c][f"hit{key}"].mean()) for c in SAFETY_RELEVANT
            },
            "ttd_median_s": float(np.nanmedian(det[f"ttd{key}"].astype(float))) if det[f"ttd{key}"].notna().any() else None,
        }
        # 误报率：非危险事件里，有多少被判成危险（次/事件），换算成次/小时
        if len(bg) and f"fa{key}" in bg:
            fa_ev = bg[f"fa{key}"]
            hours = bg.dur_s.sum() / 3600
            res[key]["false_alarm_events"] = int(fa_ev.sum())
            res[key]["false_alarm_per_event"] = float(fa_ev.mean())
            res[key]["false_alarm_per_hour"] = float(fa_ev.sum() / hours) if hours else None
        hours_all = ev_df.dur_s.sum() / 3600
        res[key]["test_audio_hours"] = float(hours_all)
    return ev_df, res


def snr_sweep(model, test: pd.DataFrame, noise_paths: list[str], device, snr_list, tau=0.8) -> pd.DataFrame:
    sub = test[test.target_label.isin(SAFETY_RELEVANT)]
    # 控制成本：每类最多 250 段
    sub = (sub.groupby("target_label", group_keys=False)
              .apply(lambda g: g.sample(min(250, len(g)), random_state=0)))
    rows = []
    for snr in snr_list:
        ds = NoisySegmentDataset(sub, noise_paths, snr, seed=1)
        probs = infer_probs(model, ds, device)
        y = sub.class_id.to_numpy()
        pred = probs.argmax(1)
        for i, c in enumerate(config.TARGET_CLASSES):
            m = y == i
            if not m.any():
                continue
            rows.append({"snr": "clean" if snr is None else snr, "class": c, "n": int(m.sum()),
                         "top1_acc": float((pred[m] == i).mean()),
                         "recall_at_tau": float((probs[m, i] >= tau).mean()),
                         "hit_anywhere": float((probs[m][:, [config.CLASS_TO_IDX[x] for x in SAFETY_RELEVANT]].max(1) >= tau).sum() / m.sum())})
        print(f"  SNR {snr}: 完成")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.MODEL_DIR / "ds_cnn_best.pth"))
    ap.add_argument("--tau", type=float, default=0.8, help="主阈值（与决策引擎一致）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    _, _, test = load_splits()
    test = test.reset_index(drop=True)
    print(f"测试集 fold {config.TEST_FOLD}: {len(test)} 段 / {test.original_file.nunique()} 事件")

    ds = AudioSegmentDataset(test, augment=False)
    probs = infer_probs(model, ds, device)
    y = test.class_id.to_numpy()
    seg = segment_metrics(probs, y)

    ev = event_frame(test, probs)
    ev_df, ev_res = event_metrics(ev)

    noise_paths = build_noise_pool()
    snr_df = snr_sweep(model, test, noise_paths, device,
                       snr_list=[None, 20, 10, 5, 0, -5], tau=args.tau)

    # ---------------- 报告 ----------------
    def fmt(x, nd=3):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    L = ["---", "tags: [project, SafeSoundAI, aed, emergency]", "created: 2026-10-17", "---", "",
         "# 紧急声音检测有效性评估（2026-10-17）", "",
         f"检查点：`{args.ckpt}`｜测试集 fold {config.TEST_FOLD}：{len(test):,} 段 / "
         f"{test.original_file.nunique():,} 个事件 / {ev_df.dur_s.sum()/3600:.2f} h 音频", "",
         "## 一、段级（模型单窗能力）", "",
         f"整体准确率 **{seg['accuracy']*100:.2f}%**", "",
         "| 类别 | 角色 | n | P | R | F1 | R@0.5 | R@0.8 |", "|---|:--:|---|---|---|---|---|---|"]
    for c in config.TARGET_CLASSES:
        m = seg["per_class"][c]
        role = {"critical": "🔴 核心", "danger": "🟠 高危", "warning": "🟡 警示", "background": "⚪ 背景"}[config.SAFETY_ROLE[c]]
        L.append(f"| {c} | {role} | {m['n']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                 f"{fmt(m['R@0.5'])} | {fmt(m['R@0.8'])} |")

    L += ["", "## 二、事件级（真实场景语义：一次声音事件至少命中一窗）", "",
          "| 阈值 τ | 核心危险 3 类 | 高危 3 类 | 6 类合计 | 首次检出时延(中位) | 误报 次/事件 | 误报 次/小时 |",
          "|---|---|---|---|---|---|---|"]
    for t in TAUS:
        r = ev_res[f"@{t:.1f}"]
        L.append(f"| {t:.1f} | {fmt(r['event_recall_critical'])} | {fmt(r['event_recall_danger'])} | "
                 f"{fmt(r['event_recall_all_danger'])} | {fmt(r['ttd_median_s'], 2)} s | "
                 f"{fmt(r.get('false_alarm_per_event'))} | {fmt(r.get('false_alarm_per_hour'), 1)} |")

    L += ["", f"### 主阈值 τ={args.tau} 下的逐类事件召回", "",
          "| 类别 | 事件数 | 事件召回 | 首次检出时延中位 |", "|---|---|---|---|"]
    key = f"@{args.tau:.1f}"
    for c in SAFETY_RELEVANT:
        sub = ev_df[ev_df.label == c]
        L.append(f"| {c} | {len(sub)} | {fmt(float(sub[f'hit{key}'].mean()))} | "
                 f"{fmt(float(np.nanmedian(sub[f'ttd{key}'].astype(float))), 2)} s |")

    L += ["", "## 三、噪声鲁棒性（叠加 ESC-50 噪声池，每类 ≤250 段）", "",
          "| 类别 | clean | 20 dB | 10 dB | 5 dB | 0 dB | -5 dB |", "|---|---|---|---|---|---|---|"]
    piv = snr_df.pivot_table(index="class", columns="snr", values="recall_at_tau")
    order = ["clean", 20, 10, 5, 0, -5]
    for c in SAFETY_RELEVANT:
        if c in piv.index:
            L.append("| " + c + " | " + " | ".join(fmt(piv.loc[c].get(k)) for k in order) + " |")
    mean_row = [snr_df[snr_df.snr == k].recall_at_tau.mean() for k in order]
    L.append("| **6 类均值** | " + " | ".join(f"{v:.3f}" for v in mean_row) + " |")

    # 结论判据
    r05 = ev_res["@0.5"]; r08 = ev_res["@0.8"]
    verdict = []
    verdict.append(("核心危险类事件召回 @τ=0.5", r05["event_recall_critical"], 0.95))
    verdict.append(("6 类事件召回 @τ=0.5", r05["event_recall_all_danger"], 0.90))
    verdict.append(("核心危险类事件召回 @τ=0.8（当前默认）", r08["event_recall_critical"], 0.90))
    L += ["", "## 四、结论（对照验收标准）", "",
          "| 验收项 | 实测 | 目标 | 判定 |", "|---|---|---|---|"]
    for name, val, target in verdict:
        ok = (val is not None) and val >= target
        L.append(f"| {name} | {fmt(val)} | ≥ {target:.2f} | {'✅ 达标' if ok else '❌ 未达标'} |")
    fa = r08.get("false_alarm_per_hour")
    L.append(f"| 误报（τ=0.8，背景/警示事件被判危险） | {fmt(fa, 1)} 次/小时 | ≤ 5 | "
             f"{'✅' if fa is not None and fa <= 5 else '⚠️'} |")

    (REP / "emergency_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (REP / "emergency_metrics.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "segment": seg, "event": ev_res,
         "snr": snr_df.to_dict(orient="records"),
         "test_segments": len(test), "test_events": int(test.original_file.nunique())},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 图 ----------------
    import matplotlib.pyplot as plt

    # fig12：段级 vs 事件级召回
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    cls = SAFETY_RELEVANT
    x = np.arange(len(cls))
    for i, (t, col) in enumerate([(0.5, "#f1c40f"), (0.8, "#c0392b")]):
        vals = [seg["per_class"][c][f"R@{t}"] or 0 for c in cls]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=f"段级 R@{t}", color=col)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in cls], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("检出率")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("段级单窗检出率（紧急 6 类）", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    vals5 = [ev_df[ev_df.label == c][f"hit@0.5"].mean() for c in cls]
    vals8 = [ev_df[ev_df.label == c][f"hit@0.8"].mean() for c in cls]
    ax.bar(x - 0.19, vals5, 0.38, label="事件召回 @τ=0.5", color="#e67e22")
    ax.bar(x + 0.19, vals8, 0.38, label="事件召回 @τ=0.8", color="#2c6fb1")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in cls], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("事件召回")
    ax.axhline(0.95, ls=":", c="#c0392b", lw=1)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("事件级召回（任一窗命中即算检出）", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle("SafeSoundAI · 紧急声音检测有效性", fontsize=13, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    vs.save(fig, "fig12_emergency_recall")

    # fig13：SNR 鲁棒性
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    xs = list(range(len(order)))
    labels = ["clean", "20", "10", "5", "0", "-5"]
    for c in SAFETY_RELEVANT:
        if c in piv.index:
            ys = [piv.loc[c].get(k, np.nan) for k in order]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=c,
                    color=vs.ROLE_COLOR.get(config.SAFETY_ROLE[c], "#888"))
    ax.plot(xs, mean_row, marker="s", ms=5, lw=2.6, ls="--", color="#0f1720", label="6 类均值")
    ax.set_xticks(xs); ax.set_xticklabels([f"{l} dB" if l != "clean" else "clean" for l in labels])
    ax.set_ylim(0, 1.05); ax.set_ylabel(f"检出率 R@τ={args.tau}")
    ax.set_xlabel("信噪比（噪声来自 ESC-50 噪声池，训练时仅见过 0~20 dB）")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.set_title("噪声鲁棒性：真实街道条件下的检出率", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    vs.save(fig, "fig13_snr_robustness")

    # fig14：时延 ECDF + 误报曲线
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    for c in CRITICAL:
        t = ev_df[(ev_df.label == c) & ev_df[f"ttd@{args.tau:.1f}"].notna()][f"ttd@{args.tau:.1f}"].astype(float)
        if len(t):
            ax.step(np.sort(t.values), np.arange(1, len(t) + 1) / len(t), where="post", label=f"{c} (n={len(t)})", lw=1.8)
    ax.set_xlabel(f"首次检出时延 (s)，τ={args.tau}"); ax.set_ylabel("累积比例")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.02)
    ax.axvline(1.0, ls=":", c="#c0392b", lw=1); ax.text(1.05, 0.05, "1 s 目标", color="#c0392b", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("核心危险类首次检出时延", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ts = TAUS
    fa_vals = [ev_res[f"@{t:.1f}"].get("false_alarm_per_hour", np.nan) for t in ts]
    rec_vals = [ev_res[f"@{t:.1f}"]["event_recall_all_danger"] or np.nan for t in ts]
    ax.plot(ts, rec_vals, marker="o", label="6 类事件召回", color="#27ae60")
    ax.set_xlabel("触发阈值 τ"); ax.set_ylabel("事件召回", color="#27ae60")
    ax.set_ylim(0.4, 1.02)
    ax2 = ax.twinx()
    ax2.plot(ts, fa_vals, marker="s", ls="--", label="误报 次/小时", color="#c0392b")
    ax2.set_ylabel("误报（次/小时，非危险事件被判危险）", color="#c0392b")
    ax2.set_ylim(bottom=0)
    ax.set_title("阈值权衡：检出率 vs 误报", fontsize=11, loc="left")
    for s in ("top",):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    fig.tight_layout()
    vs.save(fig, "fig14_threshold_ttd")

    print("\n".join(L[-14:]))
    print(f"\n报告 → {REP/'emergency_eval.md'}")


if __name__ == "__main__":
    main()
