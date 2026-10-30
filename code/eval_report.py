"""
SafeSoundAI — AED 模块评估报告（eval_report.py）
=================================================
对应 WBS 2.4.2 / 2.4.4 / 2.4.5 / 2.4.6 —— 把「训练日志里的数字」变成可放进论文的图与表。

产出：
  1. 混淆矩阵（计数 + 行归一化）           fig05_confusion_matrix.png
  2. 每类 P/R/F1 + 支持数                   fig06_per_class_prf.png
  3. 置信度分布 + 校准曲线（ECE）            fig07_confidence_calibration.png
  4. 安全阈值扫描：触发率 / 触发正确率 / 危险类漏报率  fig08_threshold_sweep.png
  5. 延迟分解（Mel / 推理 CPU / 推理 GPU / 触觉编码）与模型体积（fp32 / TorchScript / int8）
  6. 报告：outputs/reports/aed_eval.md

运行：
  export SAFESOUND_DATA_DIR=/path/to/preprocessed_data
  python3 eval_report.py                 # 全量 test fold 9
  python3 eval_report.py --limit 800     # 快速自检
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

import config
import viz_style as vs
from dataset import AudioSegmentDataset, load_splits
from data_audit import md_table
from haptic_encoder import DecisionEngine, Detection, generate_vibration
from mel import waveform_to_logmel
from models import DS_CNN, count_parameters, model_size_mb


CRITICAL = [c for c in config.TARGET_CLASSES if config.SAFETY_ROLE[c] == "critical"]
DANGER = [c for c in config.TARGET_CLASSES if config.SAFETY_ROLE[c] in ("critical", "danger")]


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (y_true, probs, latent 省) —— probs 为 softmax 后 (N, 10)。"""
    y_true, probs = [], []
    for mel, label in loader:
        logits = model(mel.to(device, non_blocking=True))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.append(label.numpy())
    return np.concatenate(y_true), np.concatenate(probs), None


# ---------------------------------------------------------------------------
# 阈值扫描（安全视角）
# ---------------------------------------------------------------------------
def threshold_sweep(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """
    以 max(prob) 为触发阈值，统计：
      触发率        —— 有多少帧会真的发警报（关系到打扰程度）
      触发时正确率  —— 已触发帧里 Top-1 判对的比例（关系到误报）
      危险类漏报率  —— 真值为 critical/danger 类的帧中，未触发（低置信）的比例（关系到漏报）
    """
    pred = probs.argmax(1)
    conf = probs.max(1)
    danger_idx = [config.CLASS_TO_IDX[c] for c in DANGER]
    crit_idx = [config.CLASS_TO_IDX[c] for c in CRITICAL]
    is_danger_true = np.isin(y_true, danger_idx)
    is_crit_true = np.isin(y_true, crit_idx)

    rows = []
    for t in np.arange(0.30, 1.001, 0.025):
        fired = conf >= t
        rows.append({
            "阈值": round(float(t), 3),
            "触发率": round(float(fired.mean()), 4),
            "触发时正确率": round(float((pred[fired] == y_true[fired]).mean()), 4) if fired.any() else np.nan,
            "危险类漏报率": round(float((~fired[is_danger_true]).mean()), 4) if is_danger_true.any() else np.nan,
            "核心类漏报率": round(float((~fired[is_crit_true]).mean()), 4) if is_crit_true.any() else np.nan,
            "触发帧占比(危险类内)": round(float(fired[is_danger_true].mean()), 4) if is_danger_true.any() else np.nan,
        })
    return pd.DataFrame(rows)


def ece(y_true, probs, n_bins: int = 12) -> tuple[float, pd.DataFrame]:
    """期望校准误差 + 可靠性数据（每 bin 的置信度 vs 实际准确率）。"""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(conf, bins) - 1
    rows, total = [], len(conf)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        e += m.sum() / total * abs(acc - cf)
        rows.append({"bin": f"{bins[b]:.2f}-{bins[b + 1]:.2f}", "n": int(m.sum()),
                     "mean_conf": round(float(cf), 4), "accuracy": round(float(acc), 4),
                     "gap": round(float(acc - cf), 4)})
    return float(e), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 延迟 / 体积
# ---------------------------------------------------------------------------
def measure_latency(model, device, n_runs: int = 300) -> dict:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(config.SEGMENT_SAMPLES).astype(np.float32)

    mel_t = []
    for _ in range(50):
        t0 = time.perf_counter()
        mel = waveform_to_logmel(y)
        mel_t.append((time.perf_counter() - t0) * 1000)

    x = mel.unsqueeze(0).unsqueeze(0).to(device)
    infer_t = []
    warm = 20
    for i in range(n_runs + warm):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warm:
            infer_t.append((time.perf_counter() - t0) * 1000)

    # CPU 推理（端侧部署的真实参考）
    cpu = torch.device("cpu")
    m_cpu = model.to(cpu)
    xc = x.to(cpu)
    cpu_t = []
    for i in range(n_runs + warm):
        t0 = time.perf_counter()
        with torch.no_grad():
            m_cpu(xc)
        if i >= warm:
            cpu_t.append((time.perf_counter() - t0) * 1000)
    model.to(device)

    enc_t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate_vibration("siren", distance_m=8.0, scheme="ear")
        enc_t.append((time.perf_counter() - t0) * 1000)

    r = {
        "Mel 转换(CPU)": float(np.median(mel_t)),
        "推理(GPU)": float(np.median(infer_t)) if device.type == "cuda" else float("nan"),
        "推理(CPU)": float(np.median(cpu_t)),
        "触觉编码": float(np.median(enc_t)),
    }
    r["端到端(GPU)"] = r["Mel 转换(CPU)"] + (r["推理(GPU)"] if not np.isnan(r["推理(GPU)"]) else 0) + r["触觉编码"]
    r["端到端(CPU)"] = r["Mel 转换(CPU)"] + r["推理(CPU)"] + r["触觉编码"]
    return r


def model_footprint(model) -> dict:
    """体积：参数量 / fp32 state_dict / TorchScript / int8 动态量化。"""
    param = count_parameters(model)
    fp32 = model_size_mb(model)
    tmp = Path("/tmp/_ss_dscnn.pt")
    torch.save(model.state_dict(), tmp)
    ckpt_kb = tmp.stat().st_size / 1024

    scripted = torch.jit.trace(model.to("cpu").eval(),
                               torch.randn(1, 1, config.N_MELS, config.FRAMES))
    scripted.save("/tmp/_ss_dscnn_scripted.pt")
    script_kb = Path("/tmp/_ss_dscnn_scripted.pt").stat().st_size / 1024

    q = torch.quantization.quantize_dynamic(model.to("cpu"), {nn.Linear}, dtype=torch.qint8)
    torch.save(q.state_dict(), "/tmp/_ss_dscnn_int8.pt")
    int8_kb = Path("/tmp/_ss_dscnn_int8.pt").stat().st_size / 1024

    return {
        "参数量": param,
        "fp32 理论体积MB": round(fp32, 3),
        "checkpoint KB": round(ckpt_kb, 1),
        "TorchScript KB": round(script_kb, 1),
        "int8(动态量化) KB": round(int8_kb, 1),
    }


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig_confusion(y_true, y_pred) -> Path:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(config.TARGET_CLASSES))))
    cmn = cm / cm.sum(1, keepdims=True)
    short = [c.replace("_", "\n") for c in config.TARGET_CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, mat, title, fmt, cmap in [
        (axes[0], cm, "(a) 计数", "d", "Blues"),
        (axes[1], cmn, "(b) 行归一化（召回）", ".2f", "RdPu"),
    ]:
        im = ax.imshow(mat, cmap=cmap)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(short, fontsize=8)
        ax.grid(False)
        thresh = mat.max() / 2
        for i in range(10):
            for j in range(10):
                v = mat[i, j]
                if v == 0:
                    continue
                s = format(int(v), fmt) if fmt == "d" else format(v, fmt)
                ax.text(j, i, s, ha="center", va="center", fontsize=7.5,
                        color="white" if v > thresh else "#222")
        ax.set_xlabel("预测"); ax.set_ylabel("真值")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(f"DS-CNN 混淆矩阵 · test fold 9（n={len(y_true):,}）",
                 fontsize=14, fontweight="bold", y=1.0)
    return vs.save(fig, "fig05_confusion_matrix.png")


def fig_per_class(y_true, y_pred) -> Path:
    import matplotlib.pyplot as plt

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(10)), zero_division=0)
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.26
    colors = vs.class_colors()
    ax.bar(x - w, p, w, label="Precision", color="#2c6fb1")
    ax.bar(x, r, w, label="Recall", color="#27ae60")
    ax.bar(x + w, f, w, label="F1", color="#c0392b")
    for i in range(10):
        ax.text(i + w, f[i] + 0.012, f"{f[i]:.2f}", ha="center", fontsize=7.5, color=vs.INK)
    ax.axhline(0.85, color=vs.MUTED, ls="--", lw=1)
    ax.text(9.5, 0.862, "目标 0.85", ha="right", fontsize=8, color=vs.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in config.TARGET_CLASSES], fontsize=8.5)
    for tick, role in zip(ax.get_xticklabels(), [config.SAFETY_ROLE[c] for c in config.TARGET_CLASSES]):
        tick.set_color(vs.ROLE_COLOR[role]); tick.set_fontweight("bold")
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("score")
    ax.set_title(f"每类 P/R/F1（macro F1 = {f.mean():.3f}，加权 F1 = "
                 f"{f1_score(y_true, y_pred, average='weighted'):.3f}）")
    ax.legend(ncol=3, loc="lower left", fontsize=9)
    ax.text(0.01, 0.02, f"支持数最少：gun_shot {s[6]} 帧、car_horn {s[1]} 帧",
            transform=ax.transAxes, fontsize=8, color=vs.MUTED)
    return vs.save(fig, "fig06_per_class_prf.png")


def fig_confidence(y_true, probs) -> tuple[Path, float]:
    import matplotlib.pyplot as plt

    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == y_true
    e, rel = ece(y_true, probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(conf[correct], bins=bins, color=vs.OK, alpha=0.75, label=f"判对 n={correct.sum():,}")
    ax.hist(conf[~correct], bins=bins, color=vs.ACCENT, alpha=0.75, label=f"判错 n={(~correct).sum():,}")
    ax.set_yscale("log")
    ax.set_xlabel("Top-1 置信度"); ax.set_ylabel("帧数（对数）")
    ax.set_title("(a) 判对 vs 判错的置信度分布")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color=vs.MUTED, lw=1, label="完美校准")
    ax.plot(rel["mean_conf"], rel["accuracy"], "o-", color=vs.BLUE, label="实测")
    ax.set_xlabel("平均置信度"); ax.set_ylabel("实际准确率")
    ax.set_title(f"(b) 校准曲线 · ECE = {e:.4f}")
    ax.legend(fontsize=9)
    fig.suptitle("置信度可信度评估（决定阈值能否安全使用）", fontsize=13, fontweight="bold", y=1.02)
    return vs.save(fig, "fig07_confidence_calibration.png"), e


def fig_threshold(sw: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    ax.plot(sw["阈值"], sw["触发率"], "-o", ms=3.5, color=vs.BLUE, label="触发率（打扰程度）")
    ax.plot(sw["阈值"], sw["触发时正确率"], "-o", ms=3.5, color=vs.OK, label="触发时正确率（1-误报）")
    ax.axvline(0.8, color=vs.ACCENT, ls="--", lw=1.2)
    ax.text(0.805, 0.05, "MVP 默认阈值 0.8", fontsize=8, color=vs.ACCENT, rotation=90)
    ax.set_xlabel("置信度阈值"); ax.set_ylabel("比例")
    ax.set_ylim(0, 1.03)
    ax.set_title("(a) 阈值 vs 触发行为")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(sw["阈值"], sw["危险类漏报率"], "-o", ms=3.5, color=vs.ACCENT2, label="高危类漏报率")
    ax.plot(sw["阈值"], sw["核心类漏报率"], "-o", ms=3.5, color=vs.ACCENT, label="核心危险类漏报率")
    ax.axvline(0.8, color=vs.ACCENT, ls="--", lw=1.2)
    ax.set_xlabel("置信度阈值"); ax.set_ylabel("漏报率（越低越安全）")
    ax.set_ylim(0, max(0.35, float(sw["核心类漏报率"].max()) * 1.1))
    ax.set_title("(b) 阈值 vs 漏报（PTSD 场景最怕漏报）")
    ax.legend(fontsize=9)
    fig.suptitle("安全阈值扫描：误报（打扰）与漏报（危险）的权衡",
                 fontsize=13, fontweight="bold", y=1.02)
    return vs.save(fig, "fig08_threshold_sweep.png")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.MODEL_DIR / "ds_cnn_best.pth"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 段（快速自检）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"AED 评估 | 设备 {device} | 模型 {args.ckpt}")
    print("=" * 70)

    # 模型
    model = DS_CNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    # 数据
    _, _, test = load_splits()
    if args.limit:
        test = test.iloc[: args.limit].reset_index(drop=True)
    ds = AudioSegmentDataset(test, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                         num_workers=config.NUM_WORKERS)
    print(f"test 片段：{len(ds):,}")

    t0 = time.perf_counter()
    y_true, probs, _ = run_inference(model, loader, device)
    infer_s = time.perf_counter() - t0
    y_pred = probs.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    print(f"Acc = {acc:.4f} | 推理耗时 {infer_s:.1f}s（含 Mel 计算与数据加载）")

    rep = classification_report(y_true, y_pred, target_names=config.TARGET_CLASSES, zero_division=0)
    print(rep)

    print("[1/4] 混淆矩阵 ...")
    f_cm = fig_confusion(y_true, y_pred)
    print("[2/4] 每类 P/R/F1 ...")
    f_prf = fig_per_class(y_true, y_pred)
    print("[3/4] 置信度校准 ...")
    f_conf, ece_val = fig_confidence(y_true, probs)
    print(f"      ECE = {ece_val:.4f}")
    print("[4/4] 阈值扫描 ...")
    sw = threshold_sweep(y_true, probs)
    f_thr = fig_threshold(sw)
    print(sw.iloc[::4].to_string(index=False))

    print("\n延迟 / 体积 ...")
    lat = measure_latency(model, device)
    foot = model_footprint(model)
    for k, v in {**lat, **foot}.items():
        print(f"  {k:22s}: {v}")

    # ---------------- 报告 ----------------
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(10)),
                                                 zero_division=0)
    cls_tbl = pd.DataFrame({
        "类别": config.TARGET_CLASSES, "安全角色": [config.SAFETY_ROLE[c] for c in config.TARGET_CLASSES],
        "支持数": s, "Precision": p.round(3), "Recall": r.round(3), "F1": f.round(3),
    }).sort_values("F1")

    md = ["---\ntags: [project, SafeSoundAI, report, aed, evaluation]\ncreated: 2026-10-17\n"
          "status: generated\n---\n", "# AED 模块评估报告（自动生成）\n",
          f"> 模型：`{Path(args.ckpt).name}` ｜ 测试集：UrbanSound8K+ESC-50 CV fold {config.TEST_FOLD}"
          f"（{len(y_true):,} 段，文件级切分）｜ 设备：{device}\n",
          "## 一、总体指标\n",
          md_table(pd.DataFrame([
              {"指标": "Accuracy", "值": f"{acc:.4f}"},
              {"指标": "Macro F1", "值": f"{f.mean():.4f}"},
              {"指标": "Weighted F1", "值": f"{f1_score(y_true, y_pred, average='weighted'):.4f}"},
              {"指标": "Macro Precision", "值": f"{p.mean():.4f}"},
              {"指标": "Macro Recall", "值": f"{r.mean():.4f}"},
              {"指标": "ECE（校准误差）", "值": f"{ece_val:.4f}"},
              {"指标": "参数量", "值": f"{count_parameters(model):,}"},
          ])),
          "\n## 二、分类报告（sklearn 原样）\n", "```", rep, "```\n",
          "## 三、各类明细（按 F1 升序 —— 最弱项在最上面）\n", md_table(cls_tbl),
          f"\n![混淆矩阵]({f_cm})\n", f"![每类指标]({f_prf})\n",
          "## 四、置信度与阈值\n",
          f"校准误差 ECE = **{ece_val:.4f}**（0 为完美校准）。"
          "ECE 越小，'置信度 0.8' 才越能当作安全阈值使用。\n",
          f"![置信度]({f_conf})\n",
          "### 阈值扫描表\n", md_table(sw),
          f"\n![阈值扫描]({f_thr})\n",
          "> 读法：PTSD 场景**漏报的代价远高于误报**，因此实际部署阈值应低于 0.8，"
          "或在核心危险类（car_horn / gun_shot / siren）上单独降阈值。\n",
          "## 五、延迟与体积\n",
          md_table(pd.DataFrame([{"项": k, "值": v} for k, v in {**lat, **foot}.items()])),
          "\n**预算对照**：采集<10ms + Mel<2ms + 分类<10ms + 决策<3ms → 端到端 < 30ms（MVP 验收线）。\n",
          "## 六、结论\n",
          f"- 精度：Acc {acc:.2%} ｜ Macro F1 {f.mean():.3f} —— "
          f"{'✅ 达标' if acc >= 0.85 else '❌ 未达 85% 目标'}",
          f"- 体积：checkpoint {foot['checkpoint KB']} KB ｜ int8 动态量化 {foot['int8(动态量化) KB']} KB"
          f" —— {'✅ <2MB 达标' if foot['checkpoint KB'] < 2048 else '❌'}",
          f"- 延迟：端到端(CPU) {lat['端到端(CPU)']:.2f} ms ｜ 端到端(GPU) {lat['端到端(GPU)']:.2f} ms "
          f"—— {'✅ <30ms 达标' if lat['端到端(CPU)'] < 30 else '❌'}",
          f"- 最弱类：`{cls_tbl.iloc[0]['类别']}`（F1 {cls_tbl.iloc[0]['F1']}，"
          f"支持 {int(cls_tbl.iloc[0]['支持数'])}）—— 样本量不足是主因，需补数据或类别重加权。",
          f"- 阈值建议：0.8 时核心危险类漏报率 "
          f"{float(sw[sw['阈值'].round(3) == 0.8]['核心类漏报率'].iloc[0]):.2%}（见阈值表）。",
          "\n---\n*本报告由 `eval_report.py` 自动生成，所有数字为实测。*"]

    out = config.OUTPUT_DIR / "reports" / "aed_eval.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告 → {out}")

    # 机器可读快照，供 demo/README 与以后对比
    snap = config.OUTPUT_DIR / "reports" / "aed_metrics.json"
    snap.write_text(json.dumps({
        "accuracy": round(acc, 4), "macro_f1": round(float(f.mean()), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "ece": round(ece_val, 6), "n_test": int(len(y_true)),
        "per_class_f1": {c: round(float(v), 4) for c, v in zip(config.TARGET_CLASSES, f)},
        "latency_ms": {k: round(float(v), 3) for k, v in lat.items()},
        "footprint": foot,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标快照 → {snap}")


if __name__ == "__main__":
    main()
