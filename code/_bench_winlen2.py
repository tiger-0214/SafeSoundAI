"""短窗基准 v2：把零填充改成「真实背景噪声填充」——模拟事件刚开始、缓冲区里是街道噪声。

对比 v1（零填充）为什么要换：零填充让一半窗口是静音，模型只见过「整段 1s 全是该声」，
必然把半静音的窗判成 background，会**高估**短窗损失。真实流式里缓冲区是背景噪声。
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
WINDOWS = [1.0, 0.75, 0.5, 0.4, 0.3]
PER_CLASS = 120
N_BG = 200


def load_model():
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
    m = DS_CNN(n_classes=len(CLASSES))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def mel_of(y: np.ndarray) -> torch.Tensor:
    return waveform_to_logmel(torch.from_numpy(y.astype(np.float32)), n_fft=NFFT)


def main():
    df = pd.read_csv("/root/safesound_data/metadata_clean.csv")
    df = df[(df.cv_fold == config.TEST_FOLD) & (df.label != "fireworks")].copy()
    pos = df[df.label.isin(CLASSES[:3])]
    bg = df[~df.label.isin(CLASSES[:3])]

    rows = [pos[pos.label == c].sample(min(PER_CLASS, len(pos[pos.label == c])), random_state=0)
            for c in CLASSES[:3]]
    rows.append(bg.sample(min(N_BG, len(bg)), random_state=0))
    df = pd.concat(rows, ignore_index=True)
    df["y"] = df.label.map(lambda x: CLASSES.index(x) if x in CLASSES[:3] else 3)

    waves, y_true = [], []
    for r in df.itertuples():
        y = np.load(r.npy_path).astype(np.float32)
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))
        waves.append(y[:SR]); y_true.append(CLASSES.index(r.label) if r.label in CLASSES[:3] else 3)
    y_true = np.array(y_true)
    filler_pool = [w for w, t in zip(waves, y_true) if t == 3]
    print(f"[data] n={len(waves)} {pd.Series(y_true).value_counts().to_dict()}", flush=True)

    model = load_model()
    out = {}
    rng = np.random.default_rng(0)
    for win in WINDOWS:
        n = int(SR * win)
        mels = []
        for w, t in zip(waves, y_true):
            yy = np.zeros(SR, dtype=np.float32)
            yy[:n] = w[:n]
            if n < SR and t != 3:                      # 正类：剩余部分填真实背景
                yy[n:] = rng.choice(filler_pool)[:SR - n]
            elif n < SR:                               # 背景自身也按同比例截断+背景续
                yy[n:] = rng.choice(filler_pool)[:SR - n]
            mels.append(mel_of(yy))
        X = torch.stack([m.unsqueeze(0) for m in mels])
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(5):
                model(X[:1])
            lat = (time.perf_counter() - t0) / 5 * 1000
            pred = model(X).argmax(1).numpy()
        acc = float((pred == y_true).mean())
        rec = {c: (round(float((pred[y_true == i] == i).mean()), 3) if (y_true == i).any() else None)
               for i, c in enumerate(CLASSES)}
        bg_fp = (round(float((pred[y_true == 3] != 3).mean()), 3))
        out[win] = dict(acc=round(acc, 4), latency_ms=round(lat, 2), recall=rec, bg_false_alarm=bg_fp)
        print(f"win={win:>4}s acc={acc:.4f} 推理={lat:5.2f}ms recall={rec} bg虚警={bg_fp}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
