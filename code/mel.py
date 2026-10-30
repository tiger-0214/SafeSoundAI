"""
SafeSoundAI — 1.4 Mel 频谱生产（mel.py）

波形 → log-Mel 频谱。默认参数对应 DS-CNN（128×100、n_fft=512、hop=160，
与笔记一致）；PANNs 迁移时改用其官方参数（n_mels=64 等），见 train_panns.py。

设计决策：不预缓存 Mel 到磁盘（72k 段 × 128×100 float32 ≈ 14GB 不划算），
训练时在线计算（torchaudio 可 GPU 加速），由 dataset.py 调用。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, FRAMES, F_MIN, F_MAX


def get_mel_transform(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    sample_rate: int = SAMPLE_RATE,
) -> torchaudio.transforms.MelSpectrogram:
    """构造 MelSpectrogram 变换（幂谱，未取 log）。"""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
        normalized=False,
    )


def waveform_to_logmel(
    y: np.ndarray | torch.Tensor,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    frames: int = FRAMES,
) -> torch.Tensor:
    """
    波形 → log-Mel 频谱。

    y      : (T,) 单声道波形，16kHz
    返回   : (n_mels, frames) 的 log-Mel 张量
    """
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    else:
        y = y.float()

    transform = get_mel_transform(
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
    )
    mel = transform(y)                       # (n_mels, T')
    mel = torch.log(mel + 1e-9)              # log 域

    # 截断 / 填充到固定帧数
    if mel.shape[1] > frames:
        mel = mel[:, :frames]
    elif mel.shape[1] < frames:
        mel = F.pad(mel, (0, frames - mel.shape[1]))
    return mel


if __name__ == "__main__":
    # 自检：1s 随机波形 → (128, 100)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    mel = waveform_to_logmel(y)
    print("log-Mel shape:", tuple(mel.shape), "（应为 (128, 100)）")
