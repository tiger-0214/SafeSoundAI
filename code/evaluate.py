"""
SafeSoundAI — 评估脚本（evaluate.py）

1. 加载训练好的 DS-CNN，评估 test 集（Acc / F1 / 混淆矩阵）；
2. 测量端到端延迟：波形 → Mel → 推理 → 触觉编码（对照 <30ms 预算）。

运行：
  python evaluate.py                        # 用 ds_cnn_best.pth
  python evaluate.py --ckpt path/to.pth
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from dataset import AudioSegmentDataset, load_splits
from haptic_encoder import generate_vibration, DecisionEngine, Detection
from mel import waveform_to_logmel
from models import DS_CNN


def load_model(ckpt: str, device: torch.device) -> DS_CNN:
    model = DS_CNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def evaluate_accuracy(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    preds, truths = [], []
    with torch.no_grad():
        for mel, label in loader:
            logits = model(mel.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            truths.append(label.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return float(accuracy_score(truths, preds)), preds, truths


def measure_latency(model, device, n_runs: int = 200) -> dict[str, float]:
    """测量各阶段延迟（中位数，单位 ms）。"""
    y = np.random.default_rng(0).standard_normal(config.SEGMENT_SAMPLES).astype(np.float32)

    # 1. Mel 转换（CPU）
    mel_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        mel = waveform_to_logmel(y)
        mel_times.append((time.perf_counter() - t0) * 1000)
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, frames)

    # 2. 推理（GPU/CPU）
    infer_times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(mel)
            torch.cuda.synchronize()
            infer_times.append((time.perf_counter() - t0) * 1000)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                model(mel)
            infer_times.append((time.perf_counter() - t0) * 1000)

    # 3. 触觉编码（CPU）
    enc_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate_vibration("siren", distance_m=8.0, scheme="ear")
        enc_times.append((time.perf_counter() - t0) * 1000)

    mel_ms = float(np.median(mel_times))
    infer_ms = float(np.median(infer_times))
    enc_ms = float(np.median(enc_times))
    return {
        "Mel 转换": mel_ms,
        "分类推理": infer_ms,
        "触觉编码": enc_ms,
        "端到端(合计)": mel_ms + infer_ms + enc_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(config.MODEL_DIR / "ds_cnn_best.pth"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模型: {args.ckpt}")
    model = load_model(args.ckpt, device)

    # --- 准确率评估 ---
    _, _, test = load_splits()
    test_ds = AudioSegmentDataset(test, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )
    acc, y_pred, y_true = evaluate_accuracy(model, test_loader, device)
    print(f"\n[test] Acc = {acc:.4f}")
    print("\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred, target_names=config.TARGET_CLASSES,
                                zero_division=0))
    print("=== 混淆矩阵 ===")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred),
                       index=config.TARGET_CLASSES,
                       columns=config.TARGET_CLASSES).to_string())

    # --- 延迟评估 ---
    print("\n=== 端到端延迟（中位数，预算 <30ms）===")
    lat = measure_latency(model, device)
    for k, v in lat.items():
        flag = "✅" if v < 30 else "❌"
        print(f"  {k:16s}: {v:6.2f} ms  {flag if k.startswith('端到端') else ''}")
    print("\n对照预算：采集<10ms + Mel<2ms + 分类<10ms + DOA<5ms + 决策<3ms")


if __name__ == "__main__":
    main()
