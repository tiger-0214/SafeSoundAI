"""诊断：逐级比对，找出流式路径与离线评测不一致的环节。

对一个真实的 siren 段，依次做：
  A. 原始 npy（离线评测口径）
  B. 重采样到 48k 再训回 16k（流式路径的重采样器）
  C. 峰值归一化到 0.9 + B
  D. 从拼好的场景 wav 里按窗口切出来（整条链路）
每步都跑 DualPathPipeline.classify，看概率怎么变。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from streaming_runtime import StreamResampler
from dual_path_pipeline import DualPathPipeline

CKPT = Path(config.MODEL_DIR) / "dscnn_v2_4cls_fft1024.pt"
META = Path(config.PREPROCESSED) / "trigger_v2" / "metadata_v2.csv"

pipe = DualPathPipeline(ckpt=CKPT, threshold=0.8, use_filter=False)
print("classes:", pipe.classes)

df = pd.read_csv(META)
te = df[(df.split == "test") & (df.label == "siren")]
rows = te.head(5)

rs = StreamResampler(48000, 16000)


def show(tag: str, y: np.ndarray) -> None:
    p = pipe.classify(y.astype(np.float32))
    cls = pipe.classes[int(np.argmax(p))]
    print(f"  {tag:<42} rms {np.sqrt(np.mean(y**2)):.4f} peak {np.abs(y).max():.3f} "
          f"→ {cls:<10} p={p.max():.3f}  " +
          " ".join(f"{c}={v:.2f}" for c, v in zip(pipe.classes, p)))


for _, r in rows.iterrows():
    y16 = np.load(r.npy_path).astype(np.float32)
    print(f"\n=== {Path(r.npy_path).name}  ({r.label})  len={len(y16)}")
    show("A 原始 16k", y16)

    up = np.repeat(y16, 3)                        # 16k → 48k（整数倍，理想插零近似）
    down = rs.process(up.astype(np.float32))
    down = np.concatenate([down, np.zeros(0, np.float32)])
    if len(down) >= len(y16):
        show("B 16k→48k(重复)→16k(流式重采样器)", down[:len(y16)])
    else:
        print(f"  B 输出长度不足 {len(down)} < {len(y16)}，跳过")

    yp = y16 / max(np.abs(y16).max(), 1e-9) * 0.9
    show("C 峰值归一化到 0.9（16k）", yp)

    if len(down) >= len(y16):
        dp = down[:len(y16)]
        dp = dp / max(np.abs(dp).max(), 1e-9) * 0.9
        show("D 归一化 + 重采样往返", dp)
