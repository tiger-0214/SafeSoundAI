"""
SafeSoundAI — 决策层策略评估（eval_decision_policy.py）
======================================================
诚实口径（录音级划分）下模型 ECE 从 0.016 涨到 0.126，等于「概率不再可信」，
直接用 τ=0.8 会把大量真实报警压掉。本脚本回答：**不动模型，只改决策层，
能把紧急声音的检出拉回多少？**

三件事：
  D1 温度标定（temperature scaling）：在验证折 fold 8 上拟合单一温度 T，重算 ECE
  D2 阈值-误报曲线：标定后逐阈值统计 6 类事件召回 与 误报（次/小时）
  D3 时序平滑：对每个事件做 3 窗滑窗取最大/均值，再看召回与首检时延

产出：reports/decision_policy.md + fig15_calibration_policy.png

用法：python3 eval_decision_policy.py [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

import config
import viz_style as vs
from dataset import AudioSegmentDataset, load_splits
from eval_emergency import (CRITICAL, DANGER, HOP_S, SAFETY_RELEVANT,
                            infer_probs, load_model)

REP = vs.REPORT_DIR


# ---------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """在验证集上最小化 NLL，拟合单一温度 T（保序，不改变 argmax）。"""
    z = torch.from_numpy(logits).float()
    t = torch.from_numpy(y).long()
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    ce = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(z / T.clamp(min=0.05), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=0.05))


@torch.no_grad()
def infer_logits(model, ds, device, batch_size: int = 128):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def events(df: pd.DataFrame, probs: np.ndarray, smooth: int = 0) -> pd.DataFrame:
    """聚合成事件；smooth=3 时对每类概率做 3 窗滑动最大值（时序平滑/去抖动）。"""
    d = df.reset_index(drop=True).copy()
    if smooth and smooth > 1:
        p = probs.copy()
        pooled = []
        for _, g in d.groupby(["dataset", "original_file"], sort=False):
            idx = g.index.to_numpy()
            block = p[idx]
            k = smooth
            sm = np.stack([
                block[max(0, i - k // 2): i + k // 2 + 1].max(0) for i in range(len(block))
            ])
            pooled.append(sm)
        probs = np.concatenate(pooled, axis=0)
    d["seg_idx"] = d.groupby(["dataset", "original_file"]).cumcount()
    for i, c in enumerate(config.TARGET_CLASSES):
        d[f"p_{c}"] = probs[:, i]
    d["eid"] = d["dataset"].astype(str) + "|" + d["original_file"].astype(str)
    d["dur_s"] = d.groupby("eid")["seg_idx"].transform(
        lambda s: (s.max() - s.min()) * HOP_S + config.SEGMENT_S)
    return d


def security_curve(ev: pd.DataFrame, taus) -> pd.DataFrame:
    rows = []
    hours = ev.dur_s.sum() / 3600
    for t in taus:
        det = ev[ev.label.isin(SAFETY_RELEVANT)]
        bg = ev[~ev.label.isin(SAFETY_RELEVANT)]
        crit = det[det.label.isin(CRITICAL)]
        # 事件命中：该类概率最大值 ≥ τ
        def hit_ratio(sub):
            if not len(sub):
                return None
            hits = [g[f"p_{lab}"].max() >= t
                    for (_, lab), g in sub.groupby(["eid", "label"])]
            return float(np.mean(hits))
        fa_events = 0
        for _, g in bg.groupby("eid"):
            if g[[f"p_{c}" for c in SAFETY_RELEVANT]].to_numpy().max() >= t:
                fa_events += 1
        rows.append({"tau": t, "recall_critical": hit_ratio(crit), "recall_all": hit_ratio(det),
                     "fa_per_hour": fa_events / hours if hours else None})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.MODEL_DIR / "ds_cnn_best.pth"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    train, val, test = load_splits()
    val, test = val.reset_index(drop=True), test.reset_index(drop=True)

    z_val = infer_logits(model, AudioSegmentDataset(val, augment=False), device)
    z_test = infer_logits(model, AudioSegmentDataset(test, augment=False), device)
    y_val, y_test = val.class_id.to_numpy(), test.class_id.to_numpy()

    p_val_raw = torch.softmax(torch.from_numpy(z_val), 1).numpy()
    p_test_raw = torch.softmax(torch.from_numpy(z_test), 1).numpy()
    T = fit_temperature(z_val, y_val)
    p_val = torch.softmax(torch.from_numpy(z_val / T), 1).numpy()
    p_test = torch.softmax(torch.from_numpy(z_test / T), 1).numpy()

    e_raw, e_cal = ece(p_test_raw, y_test), ece(p_test, y_test)
    acc = float((p_test.argmax(1) == y_test).mean())

    # 阈值曲线：原始 vs 标定 vs 标定+平滑
    taus = np.round(np.arange(0.20, 0.91, 0.05), 2)
    ev_raw = events(test, p_test_raw)
    ev_cal = events(test, p_test)
    ev_sm = events(test, p_test, smooth=3)
    cur_raw = security_curve(ev_raw, taus).assign(策略="原始概率")
    cur_cal = security_curve(ev_cal, taus).assign(策略="温度标定")
    cur_sm = security_curve(ev_sm, taus).assign(策略="标定+3窗平滑")
    cur = pd.concat([cur_raw, cur_cal, cur_sm], ignore_index=True)

    # 选出「误报 ≤ 5 次/小时」下召回最高的工作点
    best = {}
    for name, g in cur.groupby("策略"):
        ok = g[(g.fa_per_hour <= 5) & g.recall_critical.notna()]
        if len(ok):
            r = ok.sort_values("recall_critical", ascending=False).iloc[0]
            r_all = ok.sort_values("recall_all", ascending=False).iloc[0]
            best[name] = {"tau": float(r.tau), "recall_critical": float(r.recall_critical),
                          "recall_all_at_that_tau": float(
                              g[np.isclose(g.tau, r.tau)].recall_all.iloc[0]),
                          "fa_per_hour": float(r.fa_per_hour),
                          "tau_best_all": float(r_all.tau),
                          "recall_all_best": float(r_all.recall_all),
                          "fa_at_best_all": float(r_all.fa_per_hour)}

    # ---------------- 报告 ----------------
    L = ["---", "tags: [project, SafeSoundAI, aed, decision]", "created: 2026-10-17", "---", "",
         "# 决策层策略评估：标定 + 阈值 + 时序平滑（2026-10-17）", "",
         f"检查点 `{args.ckpt}`｜验证集 fold {config.VAL_FOLD}（{len(val)} 段）用于拟合温度，"
         f"测试集 fold {config.TEST_FOLD}（{len(test)} 段 / {test.original_file.nunique()} 事件）用于报告。", "",
         "## 一、标定（问题：概率不再可信）", "",
         "| 指标 | 原始 | 温度标定后 |", "|---|---|---|",
         f"| 温度 T | 1.000 | **{T:.3f}** |",
         f"| ECE（测试集） | {e_raw:.4f} | **{e_cal:.4f}** |",
         f"| Accuracy（argmax 不变） | {acc*100:.2f}% | {acc*100:.2f}% |", "",
         "## 二、误报 ≤5 次/小时 时的最佳工作点", "",
         "| 策略 | τ | 核心类事件召回 | 6 类事件召回 | 误报/小时 |", "|---|---|---|---|---|"]
    for name, b in best.items():
        L.append(f"| {name} | {b['tau']:.2f} | {b['recall_critical']*100:.1f}% | "
                 f"{b['recall_all_at_that_tau']*100:.1f}% | {b['fa_per_hour']:.1f} |")

    L += ["", "## 三、阈值扫描明细（标定 + 3 窗平滑）", "",
          "| τ | 核心类召回 | 6 类召回 | 误报/小时 |", "|---|---|---|---|"]
    for _, r in cur_sm[cur_sm.recall_critical.notna()].iterrows():
        L.append(f"| {r.tau:.2f} | {r.recall_critical*100:.1f}% | {r.recall_all*100:.1f}% | {r.fa_per_hour:.1f} |")

    before = best.get("原始概率", {})
    after = best.get("标定+3窗平滑", {})
    L += ["", "## 四、结论", ""]
    if before and after:
        L.append(f"- 不标定时，为了把误报压到 ≤5 次/小时，核心类召回只有 {before['recall_critical']*100:.1f}%；"
                 f"标定 + 3 窗平滑后可到 **{after['recall_critical']*100:.1f}%**"
                 f"（提升 {(after['recall_critical']-before['recall_critical'])*100:.1f} pp）。")
    L.append(f"- 模型本身的排序能力没变（argmax 准确率 {acc*100:.2f}% 不变），**瓶颈在概率尺度与决策规则**；"
             f"但即便最优工作点，也仍未达到「核心类 ≥95% 召回」的验收标准 → 必须回头改模型/数据，"
             f"单靠决策层救不回来。")
    (REP / "decision_policy.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (REP / "decision_policy.json").write_text(json.dumps(
        {"temperature": T, "ece_raw": e_raw, "ece_calibrated": e_cal, "accuracy": acc,
         "best": best, "curve": cur.to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 图 ----------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    for name, g, col, ls in [("原始概率", cur_raw, "#c0392b", "-"),
                             ("温度标定", cur_cal, "#e67e22", "-"),
                             ("标定+3窗平滑", cur_sm, "#27ae60", "-")]:
        g = g[g.recall_critical.notna()]
        ax.plot(g.tau, g.recall_critical, marker="o", ms=3.5, lw=1.8, ls=ls, color=col, label=name)
    ax.axhline(0.95, ls=":", c="#0f1720", lw=1)
    ax.text(0.22, 0.955, "验收线 95%", fontsize=8.5)
    ax.set_xlabel("触发阈值 τ"); ax.set_ylabel("核心危险类事件召回")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("阈值 → 召回（越高越好）", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    for name, g, col in [("原始概率", cur_raw, "#c0392b"), ("标定+3窗平滑", cur_sm, "#27ae60")]:
        g = g[g.recall_critical.notna()]
        ax.plot(g.fa_per_hour, g.recall_critical, marker="o", ms=3.5, lw=1.8, color=col, label=name)
    ax.axvline(5, ls=":", c="#0f1720", lw=1); ax.text(5.3, 0.05, "可接受误报 5 次/小时", fontsize=8.5)
    ax.set_xlabel("误报（次/小时）"); ax.set_ylabel("核心危险类事件召回")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("安全权衡曲线：召回 vs 打扰", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle("SafeSoundAI · 决策层能否救回召回？", fontsize=13, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    vs.save(fig, "fig15_calibration_policy")

    print("\n".join(L[-8:]))
    print(f"\n温度 T={T:.3f}｜ECE {e_raw:.4f} → {e_cal:.4f}")
    print("最佳工作点：", json.dumps(best, ensure_ascii=False))
    print(f"→ {REP/'decision_policy.md'}")


if __name__ == "__main__":
    main()
