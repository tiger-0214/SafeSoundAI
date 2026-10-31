"""P0-2 可行性实验：短窗（0.5 s）到底值不值。

问题背景
--------
1 s 窗给出 1.0–2.0 s 的整段告警延迟（见 `demo_e2e.py`），因为窗口必须**整个**
落进事件才能判出；250 ms 只是 hop，不是延迟。想压低延迟只能缩窗，但同一个
1 s 权重重跑 0.5 s 窗精度从 .8865 掉到 .7192 —— 那份数字来自**补零**：老代码把
0.5 s 的 50 帧 pad 回 100 帧，模型实际看到「半窗信号 + 半窗静音」。

本脚本要回答的唯一问题：把 50 帧真当 50 帧训练之后，0.5 s 窗能不能把精度
拉回到可用的程度（口径：与 1 s 窗同测试集、同样本、同 offset 网格）。

跑法::

    export SAFESOUND_DATA_DIR=/root/safesound_data
    python3 eval_short_window.py --ckpt1s outputs/models/dscnn_v2_4cls_fft1024.pt \
                                 --ckpt05 outputs/models/dscnn_v2_short05.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config
import vocab
from mel import get_mel_transform

EXCLUDE = ("fireworks",)
OUT = Path(config.OUTPUT_DIR) / "reports"


def load_test_df() -> pd.DataFrame:
    meta = Path(config.PREPROCESSED) / "trigger_v2" / "metadata_v2.csv"
    df = pd.read_csv(meta)
    df = df[df.split == "test"]
    df = df[~df.label.isin(EXCLUDE)]
    return df.reset_index(drop=True)


@torch.no_grad()
def predict(head, waves: list[np.ndarray], win_s: float, frames: int,
            device: str, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """对每个波形取 offset 网格上的窗口，返回 (probs, 每窗耗时 ms)。"""
    tf = head.mel_transform().to(device)
    model = head.model
    n = int(round(config.SAMPLE_RATE * win_s))
    feats = []
    for y in waves:
        seg = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        m = torch.log(tf(torch.from_numpy(seg).float().to(device)) + 1e-9)
        m = m[:, :frames] if m.shape[1] >= frames else torch.nn.functional.pad(
            m, (0, frames - m.shape[1]))
        feats.append(m.cpu())
    x = torch.stack(feats).unsqueeze(1)               # (N,1,128,frames)
    probs, times = [], []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0 / len(xb)
        times.append((torch.softmax(logits, 1).cpu().numpy(), dt))
    return np.concatenate([p for p, _ in times]), np.array([t for _, t in times])


def score(y: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    return {
        "n": int(len(y)),
        "acc": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "per_class_recall": {c: round(float((pred[y == i] == i).mean()), 4)
                             for i, c in enumerate(classes) if (y == i).any()},
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt1s", default=str(Path(config.MODEL_DIR) / "dscnn_v2_4cls_fft1024.pt"))
    ap.add_argument("--ckpt05", default=str(Path(config.MODEL_DIR) / "dscnn_v2_short05.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT / "short_window_eval.json"))
    args = ap.parse_args()

    df = load_test_df()
    waves = [np.load(p).astype(np.float32) for p in df.npy_path]
    # class_id 重映射到 4 类头空间
    names = [c for c in vocab.AED4_CLASSES]
    y = df.label.map({c: i for i, c in enumerate(names)}).values.astype(int)
    print(f"测试段 {len(waves)} | 类别 {names}")

    runs = []
    heads = {}
    for tag, path in (("1s", args.ckpt1s), ("05", args.ckpt05)):
        if not Path(path).exists():
            print(f"[skip] {tag}: {path} 不存在")
            continue
        heads[tag] = vocab.load_aed(path, device=args.device)
        print(f"{tag}: {Path(path).name} mel={heads[tag].mel} "
              f"params={sum(p.numel() for p in heads[tag].model.parameters()):,}")

    # 1 s 模型：1.0 s 窗（基线）/ 0.5 s 窗（补零路径 vs 真 50 帧）
    for tag, win, frames, note in (
        ("1s", 1.0, 100, "基线：1 s 窗 + 1 s 权重"),
        ("1s", 0.5, 100, "老口径：0.5 s 窗但补零到 100 帧（复现 .7192 的来源）"),
        ("1s", 0.5, 50,  "同一 1 s 权重，0.5 s 窗只取 50 帧（无补零，未重训）"),
        ("05", 0.5, 50,  "0.5 s 权重 + 0.5 s 窗（本实验要验证的方案）"),
    ):
        if tag not in heads:
            continue
        t0 = time.perf_counter()
        probs, lat = predict(heads[tag], waves, win, frames, args.device)
        s = score(y, probs, names)
        s.update({"model": tag, "win_s": win, "frames": frames, "note": note,
                  "wall_s": round(time.perf_counter() - t0, 2),
                  "latency_ms_p50": round(float(np.percentile(lat, 50)), 4),
                  "latency_ms_p95": round(float(np.percentile(lat, 95)), 4)})
        runs.append(s)
        print(f"  {note:52s} acc={s['acc']:.4f} F1={s['macro_f1']:.4f} "
              f"p50={s['latency_ms_p50']:.3f}ms")

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "scope_note": ("段级单窗离线评测，与 detection_numbers.json 同测试集/同口径；"
                       "窗口一律取段首（真实流式里窗口随机落点，见 demo_e2e 的事件级指标）。"
                       "**不可**与 v3 多标签数字互相引用。"),
        "prep": "raw npy → logmel，无 z-score（与训练一致）",
        "runs": runs,
    }, ensure_ascii=False, indent=2))
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
