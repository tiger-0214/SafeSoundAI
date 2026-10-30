"""映射合理性诊断 v2：用「只在 urban 上训练的线性探针」去判别 ESC-50 映射段。

为什么不用均值谱余弦：log-Mel 均值谱被低频能量主导，urban 类内 1-cos 距离高达 0.55–0.89，
均值谱相似度基本是噪声（实测几乎所有源类都"最像" dog_bark）。改用判别式探针：

 1. 特征：每段 log-Mel(128×100) 的均值 + 标准差 + MFCC(20) 均值 + 标准差。
 2. 训练：仅用 urban（fold 0–7）的段，LogisticRegression(class_weight=balanced)。
 3. 探针自检：urban fold 9 上的准确率（应与 DS-CNN 同量级，证明探针有判别力）。
 4. 核心：把 ESC-50 的 3,960 段映射段喂进去，看每类被预测成什么 —— 
    「与映射目标一致率」高 = 映射在特征空间成立；低 = 这批声音落在别的类的区域里。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from mel import waveform_to_logmel

DATA = os.environ.get("SAFESOUND_DATA_DIR", str(config.PREPROCESSED))
META = os.path.join(DATA, "metadata_clean.csv")
CLS = config.TARGET_CLASSES
rng = np.random.default_rng(0)

MFCC = torchaudio.transforms.MFCC(
    sample_rate=config.SAMPLE_RATE, n_mfcc=20,
    melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 128,
               "f_min": 50.0, "f_max": 7500.0},
)


def feats(paths):
    out = []
    for p in paths:
        y = np.load(p)
        m = waveform_to_logmel(y)                      # (128,100)
        c = MFCC(torch.from_numpy(y).float())          # (20,T)
        f = np.concatenate([m.mean(1).numpy(), m.std(1).numpy(),
                            c.mean(1).numpy(), c.std(1).numpy()])
        out.append(f)
    return np.array(out)


def main():
    df = pd.read_csv(META)
    tr = df[(df.role == "train") & (df.dataset == "urban") & (df.cv_fold <= 7)]
    te = df[(df.role == "train") & (df.dataset == "urban") & (df.cv_fold == 9)]

    # 每类抽样，控制耗时
    sub = (tr.groupby("target_label", group_keys=False)
             .apply(lambda g: g.sample(min(250, len(g)), random_state=0)))
    Xtr = feats(sub.npy_path.tolist()); ytr = sub.class_id.values
    Xte = feats(te.npy_path.tolist());  yte = te.class_id.values

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xtr), ytr)

    acc = (clf.predict(sc.transform(Xte)) == yte).mean()
    print(f"[探针自检] urban fold9 段级准确率 = {acc:.4f}（DS-CNN 为 0.6411）")

    # ESC-50 映射段
    esc = df[(df.dataset == "esc50") & (df.role == "train")]
    Xe = feats(esc.npy_path.tolist())
    proba = clf.predict_proba(sc.transform(Xe))
    pred = clf.classes_[proba.argmax(1)]
    esc = esc.assign(pred=pred, p_max=proba.max(1))

    print("\n=== ESC-50 源类 → 探针预测分布（行归一化）===")
    rows = []
    for src, tgt in config.ESC50_TO_TARGET.items():
        s = esc[esc.label == src]
        d = s.pred.value_counts(normalize=True)
        tgt_id = CLS.index(tgt)
        cons = float((s.pred == tgt_id).mean())
        top = f"{CLS[d.index[0]]}:{d.iloc[0]:.2f}"
        rows.append({
            "esc50_src": src, "mapped_to": tgt,
            "agree_with_mapping": round(cons, 3),
            "pred_top1": top,
            "pred_2nd": f"{CLS[d.index[1]]}:{d.iloc[1]:.2f}" if len(d) > 1 else "-",
            "pred_3rd": f"{CLS[d.index[2]]}:{d.iloc[2]:.2f}" if len(d) > 2 else "-",
            "mean_conf": round(float(s.p_max.mean()), 3),
        })
    out = pd.DataFrame(rows).sort_values("agree_with_mapping", ascending=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))

    print("\n=== 汇总：一致率 <0.5 的映射（应重新考虑）===")
    bad = out[out.agree_with_mapping < 0.5]
    print(f"{len(bad)} / {len(out)} 条映射在特征空间不成立")
    print(bad[["esc50_src", "mapped_to", "agree_with_mapping", "pred_top1"]].to_string(index=False))

    # 落回噪声池的风险：ESC-50 非映射类里有多少被探针判成"高置信目标类"
    noise = df[(df.dataset == "esc50") & (df.role == "noise")]
    ns = noise.sample(min(1500, len(noise)), random_state=0)
    Xn = feats(ns.npy_path.tolist())
    pn = clf.predict_proba(sc.transform(Xn))
    hi = (pn.max(1) > 0.6).mean()
    print(f"\n[噪声池] {len(ns)} 段非映射 ESC-50 中，探针高置信(>0.6)判为目标类的比例 = {hi:.1%}"
          f"（这些段作为纯噪声混入，会带进未标注的目标类能量）")


if __name__ == "__main__":
    main()
