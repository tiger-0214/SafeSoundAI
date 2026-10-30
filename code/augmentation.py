"""
SafeSoundAI — 1.2 数据增强（augmentation.py）

两类增强，训练时在线调用（不落盘）：
  1. 噪声混合：从 ESC-50 噪声池随机取样本，按 SNR 0~20dB 混合；
  2. SpecAugment：对 Mel 频谱做时间/频率掩码。

用法（训练时在 dataset 的 __getitem__ 中调用）：
    from augmentation import mix_noise, SpecAugment
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# 1. 噪声混合（波形级）
# ---------------------------------------------------------------------------
def mix_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    把噪声按指定 SNR 混合进信号。

    signal / noise : 等长 1D 数组（16kHz）
    snr_db         : 目标信噪比（0~20dB）
    返回：混合后的信号（幅度可能放大，可后续再 RMS 归一化）
    """
    signal = signal.astype(np.float32)
    noise = noise.astype(np.float32)
    if len(noise) < len(signal):
        # 噪声不足时循环填充
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))[:len(signal)]
    else:
        noise = noise[:len(signal)]

    sig_power = float(np.mean(signal ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(sig_power / (noise_power * snr_linear))
    return signal + scale * noise


# ---------------------------------------------------------------------------
# 2. SpecAugment（Mel 级，torchaudio 现成）
# ---------------------------------------------------------------------------
class SpecAugment:
    """对 (C, T, F) 的 Mel 张量做时间/频率掩码。"""

    def __init__(
        self,
        freq_mask_param: int = 12,
        time_mask_param: int = 20,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (C, T, F) 或 (T, F)
        for _ in range(self.n_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.n_time_masks):
            mel = self.time_mask(mel)
        return mel


if __name__ == "__main__":
    # 自检
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000).astype(np.float32)
    noise = rng.standard_normal(16000).astype(np.float32)
    out = mix_noise(sig, noise, snr_db=10.0)
    print("mix_noise 输出 shape:", out.shape)

    mel = torch.randn(1, 100, 128)
    aug = SpecAugment()
    print("SpecAugment 输出 shape:", aug(mel).shape)
