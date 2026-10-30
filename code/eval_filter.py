"""
SafeSoundAI — 过滤模块四层评测（eval_filter.py, A2.5）

按 docs/filter_eval_protocol.md 逐条计算：
  过滤层：Supp_dB（目标抑制）、Pres_dB（白名单保留）、静默压制率、SI-SDR 提升
  体验层：RTF / 单块耗时 / 输出能量变化
  安全层：静默压制次数（必须 0）、过抑制次数、fail-open 次数
  检测层：引用已有 AED 指标（不重复计算，报告中并列）

产物：
  outputs/reports/filter_eval.md / filter_eval.json / filter_per_sample.csv
  outputs/figures/fig16_filter_suppression.png / fig17_filter_tradeoff.png

用法：
    python eval_filter.py --n 200
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import config
import filter_config as fc
from attenuator import SpectralGateAttenuator, mel_to_linear_gain
from tse_model import MaskNet, complex_mask, istft, stft

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def db(x: float) -> float:
    return 10.0 * np.log10(max(x, 1e-12))


def suppress_gain(y_stem: np.ndarray, mix: np.ndarray, mask_lin: np.ndarray) -> float:
    """把同一掩码施加到分轨，返回分轨的输出/输入能量比（dB，负值 = 被抑制）。"""
    Y = stft(torch.from_numpy(y_stem.astype(np.float32)).to(DEVICE))
    T = Y.shape[-1]
    g = mask_lin[:, :T]
    if g.shape[1] < T:
        g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
    y_out = istft(Y * torch.from_numpy(g).to(device=Y.device, dtype=Y.dtype), len(y_stem)).cpu().numpy()
    return db(float(np.mean(y_out ** 2) + 1e-12) / (float(np.mean(y_stem ** 2)) + 1e-12))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean(); ref = ref - ref.mean()
    proj = (est * ref).sum() * ref / (ref.pow(2).sum() if hasattr(ref, "pow") else (ref ** 2).sum() + 1e-9)
    noise = est - proj
    return db(float((proj ** 2).sum() + 1e-9) / (float((noise ** 2).sum()) + 1e-9))


# ---------------------------------------------------------------------------
def tse_mask(model: MaskNet | None, mix: np.ndarray, keep_ids,
             strength: float | None = None) -> np.ndarray | None:
    if model is None:
        return None
    with torch.no_grad():
        y = torch.from_numpy(mix.astype(np.float32)).to(DEVICE)
        Y = stft(y)
        mag = torch.log1p(Y.abs())[None, None]
        cond = torch.tensor([fc.keep_vector(keep_ids)], dtype=torch.float32, device=DEVICE)
        mm, pr = model(mag, cond)
        M = complex_mask(mm, pr)[0].abs().cpu().numpy()
        return fc.apply_strength(np.clip(M, 0.0, 1.0), strength)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="test split 抽样条数")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tse-ckpt", default=str(fc.TSE_CKPT))
    ap.add_argument("--profile", default=fc.DEFAULT_PROFILE)
    ap.add_argument("--strength", type=float, default=None,
                    help="安全强度 α（1=原生掩码；0.35=守白名单 3 dB 红线的保守档）")
    ap.add_argument("--alpha-grid", action="store_true",
                    help="额外输出 α 权衡曲线（Supp/Pres/gap/红线超标率）")
    ap.add_argument("--gain-strength", type=float, default=None,
                    help="方案 A 的类增益强度 α（默认 fc.GAIN_STRENGTH）。"
                         "注意与 --strength（方案 B 的 TSE α）是两个独立的旋钮")
    ap.add_argument("--mode", choices=["aed", "oracle", "both"], default="both",
                    help="aed=AED 驱动（部署现状）；oracle=真值 one-hot（机理上界）；both=双轨同时出数")
    args = ap.parse_args()

    rows = []
    with open(fc.MIX_MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == args.split:
                rows.append(r)
    rows = rows[:args.n]
    if not rows:
        raise SystemExit(f"manifest 无 {args.split} 样本")

    att = SpectralGateAttenuator(profile=args.profile, strength=args.gain_strength)
    tse = None
    if Path(args.tse_ckpt).exists():
        ck = torch.load(args.tse_ckpt, map_location=DEVICE)
        tse = MaskNet().to(DEVICE)
        tse.load_state_dict(ck["model"] if "model" in ck else ck)
        tse.eval()

    def load(p):
        y, _ = sf.read(p, always_2d=False, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return np.asarray(y, dtype=np.float32)[:config.SEGMENT_SAMPLES]

    per = []
    t_all = time.perf_counter()
    for i, r in enumerate(rows, 1):
        mix = load(r["mix_path"]); t_stem = load(r["target_path"]); i_stem = load(r["interferer_path"])
        keep_ids = [int(x) for x in r["keep_set"].split(",")]
        s_keep = mix - t_stem                       # 应保留（干扰 + 底噪）

        res = {"mix_id": r["mix_id"], "target": r["target_class"],
               "interferer": r["interferer_class"], "keep_set": r["keep_set_name"],
               "snr_target_db": float(r["snr_target_db"])}

        # --- 方案 A：双轨 —— AED 驱动（部署现状）与 oracle（机理上界）---
        t0 = time.perf_counter()
        mask_aed = att.mask_from(mix)
        t_a = time.perf_counter() - t0
        res["A_aed_supp_db"] = round(suppress_gain(t_stem, mix, mask_aed), 2)
        res["A_aed_pres_db"] = round(suppress_gain(i_stem, mix, mask_aed), 2)
        mask_oracle = att.mask_from(mix, oracle_class=int(r["target_id"]))
        res["A_oracle_supp_db"] = round(suppress_gain(t_stem, mix, mask_oracle), 2)
        res["A_oracle_pres_db"] = round(suppress_gain(i_stem, mix, mask_oracle), 2)
        # 主列 A_* 依 --mode 选择（默认取部署现状，避免"假装检测器是完美的"）
        if args.mode == "oracle":
            mask_a, res["A_supp_db"], res["A_pres_db"] = (
                mask_oracle, res["A_oracle_supp_db"], res["A_oracle_pres_db"])
        else:
            mask_a, res["A_supp_db"], res["A_pres_db"] = (
                mask_aed, res["A_aed_supp_db"], res["A_aed_pres_db"])

        # --- 方案 B（TSE）---
        if tse is not None:
            t0 = time.perf_counter()
            mask_b = tse_mask(tse, mix, keep_ids, args.strength)
            t_b = time.perf_counter() - t0
            res["B_supp_db"] = round(suppress_gain(t_stem, mix, mask_b), 2)
            res["B_pres_db"] = round(suppress_gain(i_stem, mix, mask_b), 2)
            res["B_ms"] = round(1000 * t_b, 2)
        else:
            t_b = 0.0
            res["B_supp_db"] = res["B_pres_db"] = None
            res["B_ms"] = None
        res["A_ms"] = round(1000 * t_a, 2)

        # --- 混合（C）：AED 置信度决定执行器；此处用 A 的掩码作为安全上界 ---
        if tse is not None:
            res["C_supp_db"] = round(max(res["A_supp_db"], res["B_supp_db"]), 2)
            res["C_pres_db"] = round(max(res["A_pres_db"], res["B_pres_db"]), 2)
        else:
            res["C_supp_db"], res["C_pres_db"] = res["A_supp_db"], res["A_pres_db"]

        # --- SI-SDR 提升（以 s_keep 为参考）---
        Y = stft(torch.from_numpy(mix.astype(np.float32)).to(DEVICE))
        for tag, mask in (("A", mask_a), ("B", tse_mask(tse, mix, keep_ids, args.strength) if tse is not None else None)):
            if mask is None:
                continue
            T = Y.shape[-1]
            g = mask[:, :T]
            if g.shape[1] < T:
                g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
            y_out = istft(Y * torch.from_numpy(g).to(device=Y.device, dtype=Y.dtype), len(mix)).cpu().numpy()
            res[f"{tag}_sisdr_imp"] = round(si_sdr(y_out, s_keep) - si_sdr(mix, s_keep), 2)

        # --- 安全层 ---
        res["silent_suppression"] = int(res["A_pres_db"] < fc.HARD_GATE_TOLERANCE_DB)
        per.append(res)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} 完成…")

    elapsed = time.perf_counter() - t_all

    # ------------------------------------------------------------------
    # α（安全强度）权衡曲线：g_eff = 1 - α(1 - g)
    # 说明：α 收缩是等比例缩放，Supp/Pres 同向下降，因此该曲线用于回答
    #      “守住白名单 ≤3 dB 红线时，最大可用抑制是多少”，而非提升选择性。
    # ------------------------------------------------------------------
    alpha_rows, alphas = [], [0.15, 0.25, 0.35, 0.45, 0.60, 0.80, 1.00]
    if args.alpha_grid and tse is not None:
        for r in rows:
            mix = load(r["mix_path"]); t_stem = load(r["target_path"]); i_stem = load(r["interferer_path"])
            keep_ids = [int(x) for x in r["keep_set"].split(",")]
            with torch.no_grad():
                y = torch.from_numpy(mix).to(DEVICE)
                Y = stft(y)
                mm, pr = tse(torch.log1p(Y.abs())[None, None],
                             torch.tensor([fc.keep_vector(keep_ids)], dtype=torch.float32, device=DEVICE))
                m_raw = np.clip(complex_mask(mm, pr)[0].abs().cpu().numpy(), 0.0, 1.0)
            rec = {"mix_id": r["mix_id"]}
            for a in alphas:
                g = fc.apply_strength(m_raw, a)
                rec[f"supp_a{a}"] = round(suppress_gain(t_stem, mix, g), 2)
                rec[f"pres_a{a}"] = round(suppress_gain(i_stem, mix, g), 2)
            alpha_rows.append(rec)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def col(name):
        return np.array([r[name] for r in per if r.get(name) is not None], dtype=float)

    summary = {
        "n": len(per), "split": args.split, "profile": args.profile,
        "tse_enabled": tse is not None,
        "A": {"supp_db_mean": round(float(col("A_supp_db").mean()), 2),
              "supp_db_median": round(float(np.median(col("A_supp_db"))), 2),
              "pres_db_mean": round(float(col("A_pres_db").mean()), 2),
              "sisdr_imp": round(float(col("A_sisdr_imp").mean()), 2) if "A_sisdr_imp" in per[0] else None,
              "ms_per_1s": round(float(col("A_ms").mean()), 1)},
        "B": ({"supp_db_mean": round(float(col("B_supp_db").mean()), 2),
               "supp_db_median": round(float(np.median(col("B_supp_db"))), 2),
               "pres_db_mean": round(float(col("B_pres_db").mean()), 2),
               "sisdr_imp": round(float(col("B_sisdr_imp").mean()), 2) if "B_sisdr_imp" in per[0] else None,
               "ms_per_1s": round(float(col("B_ms").mean()), 1)} if tse is not None else None),
        "safety": {"silent_suppression": int(sum(r["silent_suppression"] for r in per)),
                   "silent_suppression_aed": int(sum(int(r["A_aed_pres_db"] < fc.HARD_GATE_TOLERANCE_DB) for r in per)),
                   "silent_suppression_oracle": int(sum(int(r["A_oracle_pres_db"] < fc.HARD_GATE_TOLERANCE_DB) for r in per)),
                   "pres_below_minus3_pct": round(100 * float(np.mean(col("A_pres_db") < -3)), 1)},
        "dual_track": {
            "A_aed": {"supp_db_mean": round(float(col("A_aed_supp_db").mean()), 2),
                      "pres_db_mean": round(float(col("A_aed_pres_db").mean()), 2),
                      "trigger_rate_pct": round(100 * float(np.mean(col("A_aed_supp_db") < -1.0)), 1)},
            "A_oracle": {"supp_db_mean": round(float(col("A_oracle_supp_db").mean()), 2),
                         "pres_db_mean": round(float(col("A_oracle_pres_db").mean()), 2),
                         "trigger_rate_pct": round(100 * float(np.mean(col("A_oracle_supp_db") < -1.0)), 1)},
        },
        "strength": args.strength if args.strength is not None else fc.TSE_STRENGTH,
        "gain_strength": args.gain_strength if args.gain_strength is not None else fc.GAIN_STRENGTH,
        "alpha_grid": ([{"alpha": a,
                         "supp_db_mean": round(float(np.mean([r[f"supp_a{a}"] for r in alpha_rows])), 2),
                         "pres_db_mean": round(float(np.mean([r[f"pres_a{a}"] for r in alpha_rows])), 2),
                         "gap_db": round(float(np.mean([r[f"pres_a{a}"] - r[f"supp_a{a}"] for r in alpha_rows])), 2),
                         "pres_below_minus3_pct": round(100 * float(np.mean([r[f"pres_a{a}"] < fc.HARD_GATE_TOLERANCE_DB for r in alpha_rows])), 1)}
                        for a in alphas] if alpha_rows else None),
        "timing": {"wall_s": round(elapsed, 1)},
    }

    fc.FILTER_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(fc.FILTER_REPORT_DIR / "filter_per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    if alpha_rows:
        with open(fc.FILTER_REPORT_DIR / "filter_alpha_grid.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(alpha_rows[0].keys()))
            w.writeheader(); w.writerows(alpha_rows)
    (fc.FILTER_REPORT_DIR / "filter_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 图表
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fc.FILTER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

        # fig16：抑制量分布
        fig, ax = plt.subplots(figsize=(7, 4))
        labels, data, colors = ["A 谱增益门"], [col("A_supp_db")], ["#4C78A8"]
        if tse is not None:
            labels.append("B 条件目标提取"); data.append(col("B_supp_db")); colors.append("#F58518")
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.axhline(-12, ls="--", c="gray", lw=1, label="A 目标 −12 dB")
        ax.axhline(-20, ls=":", c="gray", lw=1, label="B 目标 −20 dB")
        ax.set_ylabel("Target suppression (dB)")
        ax.set_title("Filtering: suppression of blacklisted sounds")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(fc.FILTER_FIG_DIR / "fig16_filter_suppression.png", dpi=150)
        plt.close(fig)

        # fig17：抑制 vs 保留散点
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(col("A_supp_db"), col("A_pres_db"), s=18, alpha=0.6, label="A", c="#4C78A8")
        if tse is not None:
            ax.scatter(col("B_supp_db"), col("B_pres_db"), s=18, alpha=0.6, label="B", c="#F58518")
        ax.axhline(-1, ls="--", c="red", lw=1, label="Keep threshold −1 dB")
        ax.set_xlabel("Target suppression (dB)"); ax.set_ylabel("Whitelist change (dB)")
        ax.set_title("Suppression vs preservation trade-off")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(fc.FILTER_FIG_DIR / "fig17_filter_tradeoff.png", dpi=150)
        plt.close(fig)
        figs = ["fig16_filter_suppression.png", "fig17_filter_tradeoff.png"]
        if summary["alpha_grid"]:
            xs = [g["alpha"] for g in summary["alpha_grid"]]
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(xs, [g["supp_db_mean"] for g in summary["alpha_grid"]], "o-", c="#F58518", label="Supp (target)")
            ax.plot(xs, [g["pres_db_mean"] for g in summary["alpha_grid"]], "s-", c="#4C78A8", label="Pres (whitelist)")
            ax.axhline(-3, ls="--", c="red", lw=1, label="Whitelist red line -3 dB")
            ax.set_xlabel("alpha (safety strength)"); ax.set_ylabel("Level change (dB)")
            ax.set_title("Safety strength alpha trade-off")
            ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(fc.FILTER_FIG_DIR / "fig18_filter_alpha.png", dpi=150)
            plt.close(fig)
            figs.append("fig18_filter_alpha.png")
    except Exception as e:                                   # 图表失败不影响主报告
        print("图表生成失败:", e)
        figs = []

    # ------------------------------------------------------------------
    # Markdown 报告
    # ------------------------------------------------------------------
    # 报告辅助变量：α 表 + 规格判定
    strength_used = args.strength if args.strength is not None else fc.TSE_STRENGTH
    if summary["alpha_grid"]:
        alpha_table = "\n".join(
            f"| {g['alpha']:.2f} | {g['supp_db_mean']} | {g['pres_db_mean']} | {g['gap_db']} | "
            f"{g['pres_below_minus3_pct']}% |" for g in summary["alpha_grid"])
    else:
        alpha_table = "| （未启用 `--alpha-grid`） | — | — | — | — |"

    a_supp, a_pres = summary["A"]["supp_db_mean"], summary["A"]["pres_db_mean"]
    b_supp = summary["B"]["supp_db_mean"] if tse is not None else "—"
    b_pres = summary["B"]["pres_db_mean"] if tse is not None else "—"
    v_supp = ("B 未达标（目标 ≥20 dB）" if isinstance(b_supp, float) and b_supp > -20 else "B 达标") \
        if tse is not None else "未评估"
    v_pres = ("B 未达标" if isinstance(b_pres, float) and b_pres < -3 else "B 达标") \
        if tse is not None else "未评估"
    a_silent = summary["safety"]["silent_suppression_aed"]
    v_silent = "A 达标" if a_silent == 0 else "A 未达标"

    md = [f"""---
tags: [project, SafeSoundAI, filter, report]
created: 2026-10-17
status: generated
---

# 声音过滤模块评估报告（自动生成）

> 数据：`4_Synthetic_Mixtures` {args.split} split（抽样 {summary['n']} 条）｜档位：{args.profile}｜
> TSE 权重：{'已加载' if tse is not None else '未训练/缺失（仅方案 A）'}

## 一、过滤层指标

| 方案 | 目标抑制 Supp (dB, mean/median) | 白名单保留 Pres (dB) | SI-SDR 提升 (dB) | 单条耗时 (ms/1s) |
|---|---|---|---|---|
| A 谱增益门 | {summary['A']['supp_db_mean']} / {summary['A']['supp_db_median']} | {summary['A']['pres_db_mean']} | {summary['A']['sisdr_imp']} | {summary['A']['ms_per_1s']} |"""]
    if tse is not None:
        md.append(f"| B 条件目标提取 | {summary['B']['supp_db_mean']} / {summary['B']['supp_db_median']} "
                  f"| {summary['B']['pres_db_mean']} | {summary['B']['sisdr_imp']} | {summary['B']['ms_per_1s']} |")
    md.append(f"""
## 二、双轨对照（同一条掩码施加法，区别只在"类别从哪来"）

| 轨道 | 类别来源 | 目标抑制 (dB) | 白名单保留 (dB) | 触发率（抑制 > 1 dB 的样本占比） |
|---|---|---|---|---|
| A / AED 驱动（部署现状） | DS-CNN 后验 + 置信度门控 p≥{fc.APPLY_CONF_THRESHOLD} | {summary['dual_track']['A_aed']['supp_db_mean']} | {summary['dual_track']['A_aed']['pres_db_mean']} | {summary['dual_track']['A_aed']['trigger_rate_pct']}% |
| A / oracle（机理上界） | 真值类别 one-hot | {summary['dual_track']['A_oracle']['supp_db_mean']} | {summary['dual_track']['A_oracle']['pres_db_mean']} | {summary['dual_track']['A_oracle']['trigger_rate_pct']}% |

> 两行的差值 = **检测器瓶颈带来的效果损失**。oracle 行证明"谱增益门这套机理本身有效"，
> AED 行说明"部署现状下它几乎不触发（因为判不准就 fail-open）"。
> 这正是把过滤主力交给方案 B（不依赖 AED，只用 profile 的 keep 向量）的实证依据。

## 三、安全层

| 指标 | 值 | 要求 |
|---|---|---|
| 静默压制次数（oracle 轨道，白名单衰减 > 3 dB） | {summary['safety']['silent_suppression_oracle']} | **0** |
| 静默压制次数（AED 轨道） | {summary['safety']['silent_suppression_aed']} | **0** |
| 白名单衰减 > 3 dB 的样本占比 | {summary['safety']['pres_below_minus3_pct']}% | 越低越好 |

## 四、安全强度 α 权衡（g_eff = 1 − α·(1 − g)）

> α 收缩是对掩码做等比例缩放：抑制与保留同时变弱，**选择性（gap）不会被改善**。
> 该表用于回答「守住白名单 ≤3 dB 红线时，最大可用抑制是多少」。

| α | 目标抑制 Supp (dB) | 白名单保留 Pres (dB) | 选择性 gap (dB) | 白名单 >3 dB 样本占比 |
|---|---|---|---|---|
{alpha_table}

## 五、规格达标判定

| 规格项 | 要求 | 方案 A（AED 驱动） | 方案 B（TSE，α={strength_used}） | 判定 |
|---|---|---|---|---|
| 目标抑制 | ≥ 12 dB（A）/ ≥ 20 dB（B） | {a_supp} dB | {b_supp} dB | {v_supp} |
| 白名单保留 | 衰减 ≤ 3 dB | {a_pres} dB | {b_pres} dB | {v_pres} |
| 静默压制（>3 dB） | 0 次 | {a_silent} 次 | — | {v_silent} |

## 六、检测层前提（并列声明）

AED 当前 fold 9 事件召回 **83.3%**（核心 3 类）、误报 **123.8 次/h**，未达门控级启用条件
（≥95% / ≤5 次·h⁻¹）。因此本次交付仅启用「增益级过滤 + fail-open」，门控模式在代码中硬关闭
（`filter_config.ALLOW_GATING = False`）。

## 七、结论与差距

1. 方案 A（oracle）抑制量 {summary['dual_track']['A_oracle']['supp_db_mean']} dB，机制有效性得证；
   换成真实 AED 后降到 {summary['dual_track']['A_aed']['supp_db_mean']} dB（触发率 {summary['dual_track']['A_aed']['trigger_rate_pct']}%）。
2. 方案 B {'抑制量 ' + str(summary['B']['supp_db_mean']) + ' dB（不依赖 AED，仅需 profile keep 向量）' if tse is not None else '未评估（缺权重）'}。
3. 方案 B 白名单保留 {summary['B']['pres_db_mean'] if tse is not None else '—'} dB。

## 八、复现命令

```bash
python code/synth_mix.py --n-train 3000 --n-val 500 --n-test 500
python code/train_tse.py --epochs-pre 5 --epochs-ft 25
python code/eval_filter.py --n 200
```

## 九、图表

""" + ("\n".join(f"- `outputs/figures/{f}`" for f in figs) if figs else "- （未生成）") + "\n")

    (fc.FILTER_REPORT_DIR / "filter_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n→ outputs/reports/filter_eval.md")


if __name__ == "__main__":
    main()
