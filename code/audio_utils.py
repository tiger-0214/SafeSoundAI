"""
SafeSoundAI — 1.1 音频预处理（audio_utils.py）

复现已产出的 72,201 段 .npy 的预处理逻辑，同时供实时推理复用。
四步：重采样 16kHz → 切 1s 段（50% overlap）→ RMS 归一化 → 返回段列表。

注意：本模块是「参考实现 + 推理复用」，已产出的 .npy 无需重跑；
      真实训练直接从 2_Preprocessed_Data 读取 .npy，见 dataset.py。
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample

from config import SAMPLE_RATE, SEGMENT_S, OVERLAP


def load_and_resample(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    读取音频并重采样到目标采样率。

    - 优先 soundfile（鲁棒）；多声道 → 取均值转单声道。
    - 采样率不一致时用 scipy.signal.resample 做纯数值插值。
    """
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:                      # 立体声 → 单声道
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        y = resample(y, int(len(y) * target_sr / sr)).astype(np.float32)
    return y


def rms_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RMS 归一化：抹掉绝对响度，强制模型学频谱形状。"""
    rms = float(np.sqrt(np.mean(y ** 2)))
    return y / max(rms, eps)


def segment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_s: float = SEGMENT_S,
    overlap: float = OVERLAP,
) -> list[np.ndarray]:
    """切 1s 固定窗口，50% overlap。返回段列表（每段 shape=(16000,)）。"""
    seg_len = int(segment_s * sr)
    hop = int(seg_len * (1.0 - overlap))
    segs = []
    start = 0
    while start + seg_len <= len(y):
        segs.append(y[start:start + seg_len])
        start += hop
    return segs


def preprocess_file(
    path: str,
    segment_s: float = SEGMENT_S,
    overlap: float = OVERLAP,
) -> list[np.ndarray]:
    """完整预处理单文件：重采样 → 单声道 → RMS 归一化 → 切段。"""
    y = load_and_resample(path)
    y = rms_normalize(y)
    return segment_audio(y, segment_s=segment_s, overlap=overlap)


def n_segments(duration_s: float, segment_s: float = SEGMENT_S, overlap: float = OVERLAP) -> int:
    """片段数公式：N = ⌊(duration − 1.0) / 0.5⌋ + 1（1s 窗口、50% overlap）。"""
    return max(0, int((duration_s - segment_s) / (segment_s * (1 - overlap))) + 1)


if __name__ == "__main__":
    # 自检：对一段 3s 音频应切出 5 段
    print("3s →", n_segments(3.0), "段（应为 5）")
    print("5s →", n_segments(5.0), "段（应为 9）")
