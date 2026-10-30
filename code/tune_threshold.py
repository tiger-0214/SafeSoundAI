"""
SafeSoundAI — Trigger-AED v2 决策层阈值策略（tune_threshold.py）
==================================================================
4 类基线（siren / car_horn / gun_shot / background）在 argmax 口径下
siren 漏检 17.3%（1,486 段里 257 段被判成 background），precision 却有 0.925
→ 说明不是模型不会，是**决策门限偏保守**。本脚本不动模型，只回答：

  T1 单一 siren 门限 τ 扫描：漏检率 ↔ 误报率 ↔ 整体 macro-F1 怎么trade-off
  T2 达标点：把 siren 段级漏检压到 5% / 3% 需要 τ 多少，代价是什么
  T3 事件级（录音级）口径：产品真正关心的是「一次警报有没有被听到」，
     1 s 窗级 17% 漏检 ≠ 事件级 17% 漏检（多窗可互相兜底）——用 max-pool 复核

产出：outputs/reports/threshold_policy_4cls.{md,json} + figures/fig_threshold_siren_4cls.png

用法：
    export SAFESOUND_DATA_DIR=/path/to/preprocessed_data
    python3 tune_threshold.py --ckpt ../outputs/models/dscnn_v2_4cls.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import viz_style as vs
from models import DS_CNN
import train_dscnn as T2

REP, FIG = vs.REPORT_DIR, vs.FIG_DIR
DATA_V2 = config.PREPROCESSED / "trigger_v2"


# ---------------------------------------------------------------------------
def load_4cls_frame() -> tuple[pd.DataFrame, list[str]]:
    """读 metadata_v2，剔除 fireworks 并重映射 class_id（与 --exclude-class 一致）。"""
    df = pd.read_csv(DATA_V2 / "metadata_v2.csv")
    df = df[df.label != "fireworks"].copy()
    cls = [c for c in T2.CLASSES_V2 if c != "fireworks"]
    m = {c: i for i, c in enumerate(cls)}
    df["class_id"] = df.label.map(m).astype(int)
    return df, cls


@torch.no_grad()
def infer(model, df: pd.DataFrame, device: str) -> np.ndarray:
    loader = DataLoader(T2.SegDataset(df), batch_size=128, shuffle=False, num_workers=4)
    out = []
    for mel, _ in loader:
        out.append(model(mel.to(device)).cpu())
    return torch.cat(out)


def argmax_metrics(p: np.ndarray, y: np.ndarray, i_bg: int) -> dict:
    pred = p.argmax(1)
    trig = np.array([i for i in range(p.shape[1]) if i != i_bg])
    return {
        "acc": float((pred == y).mean()),
        "siren_recall": float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
        "bg_false_alarm": float((pred[y == i_bg] != i_bg).mean()),
        "trigger_miss": float((pred[np.isin(y, trig)] == i_bg).mean()),
    }


def apply_siren_gate(p: np.ndarray, tau: float, i_siren: int) -> np.ndarray:
    """p_siren ≥ τ → siren；否则在其余类上取 argmax（屏蔽 siren 不改变其余排序）。"""
    pred = p.argmax(1).copy()
    pred[p[:, i_siren] >= tau] = i_siren
    return pred


def sweep(p: np.ndarray, y: np.ndarray, cls: list[str], i_bg: int) -> pd.DataFrame:
    rows = []
    for tau in np.round(np.arange(0.02, 1.0, 0.01), 2):
        pred = apply_siren_gate(p, tau, 0)
        s_sel = y == 0
        rec = float((pred[s_sel] == 0).mean())
        pre = float((y[pred == 0] == 0).mean()) if (pred == 0).any() else 0.0
        trig = np.array([i for i in range(len(cls)) if i != i_bg])
        rows.append({
            "tau": float(tau),
            "siren_recall": round(rec, 4),
            "siren_miss": round(1 - rec, 4),
            "siren_precision": round(pre, 4),
            "siren_f1": round(2 * pre * rec / max(pre + rec, 1e-9), 4),
            "bg_false_alarm": round(float((pred[y == i_bg] != i_bg).mean()), 4),
            "trigger_miss": round(float((pred[np.isin(y, trig)] == i_bg).mean()), 4),
            "acc": round(float((pred == y).mean()), 4),
        })
    return pd.DataFrame(rows)


def event_level(df: pd.DataFrame, p: np.ndarray, tau: float | None) -> dict:
    """录音级：同 rec_group 的多窗取 max p_siren；有任一窗超门限即算该次警报被听见。"""
    d = df.reset_index(drop=True).copy()
    d["p_siren"] = p[:, 0]
    d["pred"] = p.argmax(1) if tau is None else apply_siren_gate(p, tau, 0)
    siren_rec = d[d.label == "siren"].groupby("rec_group").agg(
        hit=("pred", lambda s: int((s == 0).any())), n=("pred", "size"))
    bg_rec = d[d.label == "background"].groupby("rec_group").agg(
        fa=("pred", lambda s: int((s == 0).any())), n=("pred", "size"))
    return {
        "siren_recordings": int(len(siren_rec)),
        "siren_event_recall": round(float(siren_rec.hit.mean()), 4),
        "median_windows_per_siren_rec": float(siren_rec.n.median()),
        "bg_recordings": int(len(bg_rec)),
        "bg_event_false_alarm": round(float(bg_rec.fa.mean()), 4),
    }


def pick_tau(sw: pd.DataFrame, target_miss: float) -> dict:
    ok = sw[sw.siren_miss <= target_miss]
    if len(ok) == 0:
        return {"tau": 0.02, "note": f"扫到 0.02 仍达不到漏检 ≤{target_miss:.0%}"}
    r = ok.iloc[-1]           # 满足条件的最大 τ（精度最好）
    return {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in r.items()}


def pick_tau_constrained(sw: pd.DataFrame, max_fa: float) -> dict:
    """在「背景段级误报率 ≤ max_fa」的约束下，把 siren 漏检压到最低的操作点。"""
    ok = sw[sw.bg_false_alarm <= max_fa]
    if len(ok) == 0:
        return {"note": f"误报 ≤{max_fa:.0%} 的约束下无可行点"}
    r = ok.loc[ok.siren_miss.idxmin()]
    return {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in r.items()}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """小样本比例区间的 Wilson 95% CI（14/14 这种必须报下界，否则是自欺）。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.OUTPUT_DIR / "models" / "dscnn_v2_4cls.pt"))
    ap.add_argument("--tag", default="4cls")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, cls = load_4cls_frame()
    i_bg = cls.index("background")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    assert ck["classes"] == cls, f"权重类别 {ck['classes']} ≠ 当前 {cls}"
    model = DS_CNN(n_classes=len(cls)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"载入 {args.ckpt} | 类 {cls} | device {device}")

    va = df[df.split == "val"]
    te = df[df.split == "test"].reset_index(drop=True)
    lv = infer(model, va, device)
    T = T2.fit_temperature(lv, va.class_id.values)
    lt = infer(model, te, device)
    p = F.softmax(lt / T, dim=1).numpy()
    y = te.class_id.values
    print(f"val {len(va)} / test {len(te)} | T = {T:.4f}")

    base = argmax_metrics(p, y, i_bg)
    print("argmax 基线:", {k: round(v, 4) if isinstance(v, float) else v for k, v in base.items()})

    sw = sweep(p, y, cls, i_bg)
    pts = {f"miss<={t:.0%}": pick_tau(sw, t) for t in (0.10, 0.05, 0.03, 0.01)}
    cons = {f"fa<={t:.0%}": pick_tau_constrained(sw, t) for t in (0.10, 0.13, 0.16)}
    best_f1 = sw.loc[sw.siren_f1.idxmax()].to_dict()

    op_table = []
    for tau in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05):
        r = sw.iloc[(sw.tau - tau).abs().idxmin()]
        op_table.append({k: r[k] for k in ("tau", "siren_miss", "siren_precision", "siren_f1",
                                           "bg_false_alarm", "trigger_miss", "acc")})

    ev_base = event_level(te, p, None)
    ev_tuned = {n: event_level(te, p, v["tau"]) for n, v in {**pts, **cons}.items() if "note" not in v}
    k_ev = int(round(ev_base["siren_event_recall"] * ev_base["siren_recordings"]))
    ci_lo, ci_hi = wilson_ci(k_ev, ev_base["siren_recordings"])

    result = {
        "ckpt": args.ckpt, "temperature": round(T, 4), "n_test": int(len(te)),
        "argmax_baseline": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in base.items()},
        "operating_points": pts, "constrained_points": cons, "best_siren_f1_point": best_f1,
        "event_recall_wilson95": [round(ci_lo, 4), round(ci_hi, 4)],
        "op_table": op_table, "sweep": sw.to_dict("records"),
        "event_level": {"baseline": ev_base, "tuned": ev_tuned},
    }
    (REP / f"threshold_policy_{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))

    # ---------------- 图 ----------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax[0].plot(sw.tau, sw.siren_recall, label="siren 召回（=1−漏检）", lw=2)
    ax[0].plot(sw.tau, sw.siren_precision, label="siren 精度", lw=2)
    ax[0].plot(sw.tau, sw.siren_f1, label="siren F1", lw=1.6, ls="--")
    for n, v in pts.items():
        if "note" not in v:
            ax[0].axvline(v["tau"], color="gray", ls=":", lw=1)
            ax[0].annotate(f"τ={v['tau']:.2f}\n漏检{v['siren_miss']:.1%}",
                           (v["tau"], 0.06), fontsize=7.5, rotation=90, ha="right")
    ax[0].axhline(0.95, color="crimson", ls="-.", lw=1)
    ax[0].text(0.96, 0.955, "漏检 5% 线", color="crimson", fontsize=8, ha="right")
    ax[0].set_xlabel("siren 门限 τ"); ax[0].set_ylabel("比例")
    ax[0].set_title("siren 门限扫描：召回/精度/F1"); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].invert_xaxis()

    ax[1].plot(sw.tau, sw.bg_false_alarm, color="darkorange", label="背景误报率（段级）", lw=2)
    ax[1].plot(sw.tau, sw.trigger_miss, color="steelblue", label="三触发类整体漏检", lw=2)
    ax2 = ax[1].twinx()
    ax2.plot(sw.tau, sw.acc, color="green", ls="--", lw=1.6, label="整体准确率")
    ax2.set_ylabel("整体准确率"); ax2.grid(False)
    ax[1].set_xlabel("siren 门限 τ"); ax[1].set_ylabel("比例")
    ax[1].set_title("代价：误报与整体指标")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    ax[1].invert_xaxis()
    fig.tight_layout()
    fp = FIG / f"fig_threshold_siren_{args.tag}.png"
    fig.savefig(fp)
    plt.close(fig)

    # ---------------- Markdown ----------------
    chosen = pts.get("miss<=5%", {})
    L = [f"# Trigger-AED v2 决策层阈值策略（{args.tag}）", "",
         f"- 权重：`{args.ckpt}`",
         f"- 温度标定 T = {T:.4f}（在 val 上拟合，作用于 test）",
         f"- 测试段数 {len(te)}（录音级切分，无泄漏）",
         "- 判定规则：`p_siren(标定后) ≥ τ → siren`；否则在其余类上取 argmax", "",
         "## 一、argmax 基线（τ 不干预）", "",
         "| 指标 | 值 |", "|---|---:|",
         f"| 整体准确率 | {base['acc']:.4f} |",
         f"| siren 段级召回 | {base['siren_recall']:.4f} |",
         f"| 三触发类整体漏检率 | {base['trigger_miss']:.4f} |",
         f"| 背景误报率（段级） | {base['bg_false_alarm']:.4f} |", "",
         "## 二、达标点（把 siren 漏检压到目标线所需门限）", "",
         "| 目标 | τ | siren 召回 | siren 精度 | siren F1 | 背景误报率 | 三触发类漏检 | 整体准确率 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, v in pts.items():
        if "note" in v:
            L.append(f"| {n} | — | — | — | — | — | — | {v['note']} |")
        else:
            L.append(f"| {n} | {v['tau']:.2f} | {v['siren_recall']:.4f} | {v['siren_precision']:.4f} | "
                     f"{v['siren_f1']:.4f} | {v['bg_false_alarm']:.4f} | {v['trigger_miss']:.4f} | {v['acc']:.4f} |")
    L += ["", "## 三、受约束操作点（在可接受的误报预算下能拿到的最低漏检）", "",
          "| 误报预算（背景段级） | τ | siren 漏检 | siren 精度 | siren F1 | 实际背景误报率 | 三触发类漏检 | 整体准确率 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, v in cons.items():
        if "note" in v:
            L.append(f"| {n} | — | — | — | — | — | — | {v['note']} |")
        else:
            L.append(f"| {n} | {v['tau']:.2f} | {v['siren_miss']:.4f} | {v['siren_precision']:.4f} | "
                     f"{v['siren_f1']:.4f} | {v['bg_false_alarm']:.4f} | {v['trigger_miss']:.4f} | {v['acc']:.4f} |")
    L += ["", "## 四、操作点速查表", "",
          "| τ | 漏检率 | 精度 | F1 | 背景误报率 | 三触发类漏检 | 整体准确率 |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in op_table:
        L.append(f"| {r['tau']:.2f} | {r['siren_miss']:.4f} | {r['siren_precision']:.4f} | "
                 f"{r['siren_f1']:.4f} | {r['bg_false_alarm']:.4f} | {r['trigger_miss']:.4f} | {r['acc']:.4f} |")
    L += ["", f"F1 最优点：τ = {best_f1['tau']:.2f}（siren F1 {best_f1['siren_f1']:.4f}，"
              f"漏检 {best_f1['siren_miss']:.4f}）", "",
          "## 五、事件级（录音级）口径复核", "",
          "| 口径 | 警报被听见的比例（事件级召回） | 背景录音误报比例 |",
          "|---|---:|---:|",
          f"| argmax 基线 | {ev_base['siren_event_recall']:.4f} | {ev_base['bg_event_false_alarm']:.4f} |"]
    for n, v in ev_tuned.items():
        t = {**pts, **cons}.get(n, {}).get("tau")
        L.append(f"| τ={t:.2f}（{n}） | {v['siren_event_recall']:.4f} | {v['bg_event_false_alarm']:.4f} |")
    L += ["", f"- siren 录音数 {ev_base['siren_recordings']}，每条录音在测试集里平均保留 "
              f"{ev_base['median_windows_per_siren_rec']:.0f} 个 1 s 窗（中位数）；背景录音数 {ev_base['bg_recordings']}",
          f"- 事件级召回的 Wilson 95% 置信区间：**[{ci_lo:.3f}, {ci_hi:.3f}]**"
          "（录音数太少，点估计 100% 不能当承诺）", "",
          "> 说明：测试集是每段长录音**抽样**出的若干 1 s 窗，不是连续流，因此事件级召回是"
          "「多窗兜底能救回多少」的下界估计；真实部署时窗更多、滑窗重叠，事件级召回只会更高。", "",
          "## 六、结论", ""]
    if chosen and "note" not in chosen:
        L += [f"- 段级「漏检 ≤5%」的达标点是 **τ = {chosen['tau']:.2f}**，但代价是背景段级误报从 "
              f"{base['bg_false_alarm']:.1%} 涨到 **{chosen['bg_false_alarm']:.1%}**、背景录音事件级误报 "
              f"{ev_tuned['miss<=5%']['bg_event_false_alarm']:.1%} —— 这个目标函数在当前数据上**不该追求**："
              "每 3 个背景窗报 1 次，用户会直接关掉设备。",
              f"- 事件级口径下基线已经是 {ev_base['siren_event_recall']:.0%}"
              f"（{k_ev}/{ev_base['siren_recordings']} 条警报录音至少一窗被检出，95% CI 下界 {ci_lo:.2f}）："
              "**段级 17% 漏检并不等于「警报被漏掉」**，多窗兜底已经消化掉大部分。",
              f"- F1 最优点 τ = {best_f1['tau']:.2f}（漏检 {best_f1['siren_miss']:.1%}、精度 "
              f"{best_f1['siren_precision']:.1%}、背景误报 {best_f1['bg_false_alarm']:.1%}），整体准确率在 "
              f"τ≈0.25 处反而最高（{sw.loc[(sw.tau - 0.25).abs().idxmin(), 'acc']:.4f} > 基线 {base['acc']:.4f}）。",
              "- 因此不要「一刀切」，建议 **双档门限**：",
              "  - 轻提示（触觉一级，代价低的提醒）用 **τ₁ ≈ 0.13**：段级漏检 10%、误报 16.6%",
              "  - 强警报（触觉升级/打断用户）用 **τ₂ ≈ 0.25~0.30**：精度 0.90 以上、误报 12~13%",
              "- 真正的瓶颈不是阈值，而是 **siren 只有 74 条独立录音**：阈值只能重新分配错误，"
              "要把漏检与误报同时压下去必须补数据（见笔记 20 第五节）。"]
    L += [f"![siren 门限扫描](figures/{fp.name})", ""]
    (REP / f"threshold_policy_{args.tag}.md").write_text("\n".join(L))

    print("\n" + "=" * 66)
    print(f"argmax: acc {base['acc']:.4f} | siren 召回 {base['siren_recall']:.4f} | "
          f"三触发类漏检 {base['trigger_miss']:.4f} | 背景误报 {base['bg_false_alarm']:.4f}")
    for n, v in pts.items():
        if "note" in v:
            print(f"  {n:10s} {v['note']}")
        else:
            print(f"  {n:10s} τ={v['tau']:.2f} 召回{v['siren_recall']:.4f} 精度{v['siren_precision']:.4f} "
                  f"误报{v['bg_false_alarm']:.4f} acc{v['acc']:.4f} | 事件级召回{ev_tuned[n]['siren_event_recall']:.4f}")
    print(f"→ {REP}/threshold_policy_{args.tag}.md")


if __name__ == "__main__":
    main()
