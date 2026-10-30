"""短窗（流式）精度与延迟基准：DS-CNN 4 类模型，1s → 200ms 窗口扫描。

目的：回答「低延迟检测到底差多少」——不是猜，而是实测精度随窗口缩短的衰减。
做法：测试集每类取至多 150 段，裁取前 N 个采样点（模拟事件刚发生、缓冲区已有背景），
      零填充到 1s，其余流程与训练一致（waveform_to_logmel, n_fft=1024）。
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from mel import waveform_to_logmel
from models import DS_CNN

CKPT = Path("/mnt/d/learning/ObsidianVault/Project-SafeSoundAI/0-SafeSound AI workspace/outputs/models/dscnn_v2_4cls_fft1024.pt")
CLASSES = ["siren", "car_horn", "gun_shot", "background"]
SR = config.SAMPLE_RATE
NFFT = 1024
WINDOWS = [1.0, 0.5, 0.4, 0.3, 0.25]
PER_CLASS = 150


def load_model():
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
    m = DS_CNN(n_classes=len(CLASSES))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def main():
    df = pd.read_csv("/root/safesound_data/metadata_clean.csv")
    df = df[(df.cv_fold == config.TEST_FOLD) & (df.label != "fireworks")].copy()
    # 4 类池：正中类保留，其余全部归 background（与 train_dscnn_v2 默认映射一致）
    df["class_id"] = df.label.map(lambda x: CLASSES.index(x) if x in CLASSES[:3] else 3)
    rows = []
    for c in CLASSES:
        sub = df[df.label == c]
        if len(sub):
            rows.append(sub.sample(min(PER_CLASS, len(sub)), random_state=0))
    df = pd.concat(rows, ignore_index=True)
    print(f"[data] test subset n={len(df)}  分布={df.label.value_counts().to_dict()}")

    # 预读波形
    waves = []
    for r in df.itertuples():
        y = np.load(r.npy_path).astype(np.float32)
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))
        waves.append(y[:SR])
    y_true = df.class_id.values

    model = load_model()
    out = {}
    for norm in (False, True):
     for win in WINDOWS:
        n = int(SR * win)
        mels = []
        for y in waves:
            yy = np.zeros(SR, dtype=np.float32)
            yy[:n] = y[:n]
            if norm:
                yy = (yy - yy.mean()) / (yy.std() + 1e-8)
            mels.append(waveform_to_logmel(torch.from_numpy(yy), n_fft=NFFT))
        X = torch.stack([m.unsqueeze(0) for m in mels])  # (B,1,128,F)
        with torch.no_grad():
            # 延迟：单条批，取 5 次均值
            t0 = time.perf_counter()
            for _ in range(5):
                model(X[:1])
            lat = (time.perf_counter() - t0) / 5 * 1000
            pred = model(X).argmax(1).numpy()
        acc = float((pred == y_true).mean())
        rec = {c: float((pred[y_true == i] == i).mean()) if (y_true == i).any() else None
               for i, c in enumerate(CLASSES)}
        key = f"zscore={norm}|win={win}"
        out[key] = dict(acc=round(acc, 4), latency_ms=round(lat, 2),
                        recall={k: (round(v, 3) if v is not None else None) for k, v in rec.items()})
        print(f"zscore={norm!s:<5} win={win:>4}s  acc={acc:.4f}  单条推理={lat:6.2f} ms  recall={out[key]['recall']}", flush=True)

    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
