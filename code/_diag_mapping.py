"""映射合理性诊断：ESC-50 源类 → 10 类目标，谱特征上是否真的同分布？

方法：
 1. 每类取 log-Mel 均值谱（npy 已切段的波形 → 128×100 log-Mel → 沿时间平均 → 128 维），
    再对每维做全库 z-score（避免低频能量主导余弦相似度）。
 2. 计算 ESC-50 源类均值谱 vs 每个目标类均值谱的余弦相似度 → 排名 + 与目标类的差值 margin。
 3. 用 urban 各类内部的分散度（类内平均距离）作标尺：若源类到目标类的距离 > 类内距离，
    说明"注入的是一批外来的、目标类里原本没有的声音"。
不写报告，只打印。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from mel import waveform_to_logmel

DATA = os.environ.get("SAFESOUND_DATA_DIR", str(config.PREPROCESSED))
META = os.path.join(DATA, "metadata_clean.csv")
rng = np.random.default_rng(0)

MAPPING = config.ESC50_TO_TARGET


def class_profile(paths, n=60):
    """随机抽 n 段的 log-Mel 均值谱（128 维）。"""
    sel = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    profs = []
    for i in sel:
        y = np.load(paths[i])
        m = waveform_to_logmel(y)                     # (128, 100)
        profs.append(m.mean(dim=1).numpy())
    return np.mean(profs, axis=0), np.array(profs)     # (128,), (n,128)


def main():
    df = pd.read_csv(META)
    train = df[df.role == "train"]

    # ---- 1. 目标类（urban）均值谱 + 类内分散度 ----
    urban_stats = {}
    for t in config.TARGET_CLASSES:
        p = train[(train.dataset == "urban") & (train.target_label == t)].npy_path.tolist()
        mean, arr = class_profile(p)
        urban_stats[t] = (mean, arr)
        print(f"[urban] {t:18s} n_sample={len(arr)}")

    P = np.stack([urban_stats[t][0] for t in config.TARGET_CLASSES])   # (10,128)
    mu, sd = P.mean(0), P.std(0) + 1e-6
    Pz = (P - mu) / sd

    def z(v):
        return (v - mu) / sd

    # 类内分散度标尺：每个 urban 类内部各段到其类均值的平均余弦距离
    intra = {}
    for t in config.TARGET_CLASSES:
        mean, arr = urban_stats[t]
        az = (arr - mu) / sd
        cz = np.linalg.norm(az, axis=1) * np.linalg.norm(z(mean)) + 1e-9
        cos = az @ z(mean) / cz
        intra[t] = float(np.mean(1 - cos))

    # ---- 2. ESC-50 源类 ----
    rows = []
    for src, tgt in MAPPING.items():
        p = train[(train.dataset == "esc50") & (train.label == src)].npy_path.tolist()
        smean, sarr = class_profile(p)
        sz = z(smean)
        sims = (Pz @ sz) / (np.linalg.norm(Pz, axis=1) * np.linalg.norm(sz) + 1e-9)
        order = np.argsort(-sims)
        rank_of_target = [config.TARGET_CLASSES[i] for i in order].index(tgt) + 1
        best = config.TARGET_CLASSES[order[0]]
        rows.append({
            "esc50_src": src,
            "mapped_to": tgt,
            "top1": best,
            "sim_mapped": float(sims[config.TARGET_CLASSES.index(tgt)]),
            "sim_top1": float(sims[order[0]]),
            "margin_top1_vs_mapped": float(sims[order[0]] - sims[config.TARGET_CLASSES.index(tgt)]),
            "target_rank": rank_of_target,
            "intra_target": intra[tgt],
            "top3": ", ".join(f"{config.TARGET_CLASSES[i]}:{sims[i]:.3f}" for i in order[:3]),
        })

    out = pd.DataFrame(rows).sort_values("margin_top1_vs_mapped", ascending=False)
    pd.set_option("display.width", 250)
    print("\n=== 映射一致性（sim 越高越像；target_rank=1 表示映射成立）===")
    print(out[["esc50_src", "mapped_to", "top1", "sim_mapped", "sim_top1",
               "margin_top1_vs_mapped", "target_rank", "intra_target"]].to_string(index=False))
    print("\n=== 每个源类最像的 3 个目标类 ===")
    for r in out.itertuples():
        print(f"{r.esc50_src:16s} → 映射 {r.mapped_to:16s} | {r.top3}")

    print("\n=== urban 类内分散度（1-cos，作为'同类'的标尺）===")
    for t, v in sorted(intra.items(), key=lambda x: -x[1]):
        print(f"{t:18s} {v:.4f}")


if __name__ == "__main__":
    main()
