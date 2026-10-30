"""
SafeSoundAI — PyTorch 数据集（dataset.py）

从 metadata_clean.csv 加载 .npy 片段，按 cv_fold 划分 train/val/test，
在线计算 log-Mel，训练时可选噪声混合 + SpecAugment。

⚠️ 必须先运行 build_metadata.py 生成 metadata_clean.csv。
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from augmentation import mix_noise, SpecAugment
from mel import waveform_to_logmel


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取干净 metadata，按 cv_fold 切 train/val/test。"""
    df = pd.read_csv(config.METADATA_CLEAN)
    train = df[df["cv_fold"].isin(config.TRAIN_FOLDS)]
    val = df[df["cv_fold"] == config.VAL_FOLD]
    test = df[df["cv_fold"] == config.TEST_FOLD]
    return train, val, test


class AudioSegmentDataset(Dataset):
    """
    音频片段数据集（波形 → log-Mel 在线计算）。

    metadata : 包含 npy_path / class_id 的 DataFrame（只含 role='train' 样本）
    noise_pool : 噪声池（role='noise' 样本的 npy 路径列表），None 则不做噪声混合
    augment   : 是否启用增强（噪声混合 + SpecAugment），仅训练集应为 True
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        noise_pool: list[str] | None = None,
        augment: bool = False,
        mel_kwargs: dict | None = None,
    ):
        self.df = metadata.reset_index(drop=True)
        self.noise_pool = noise_pool or []
        self.augment = augment
        self.mel_kwargs = mel_kwargs or {}
        self.spec_aug = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        y = np.load(row["npy_path"]).astype(np.float32)   # (16000,)
        y = torch.from_numpy(y)

        if self.augment:
            # 噪声混合（50% 概率，SNR 0~20dB）
            if self.noise_pool and random.random() < 0.5:
                noise_path = random.choice(self.noise_pool)
                noise = np.load(noise_path).astype(np.float32)
                snr = random.uniform(0.0, 20.0)
                y = torch.from_numpy(mix_noise(y.numpy(), noise, snr))

        mel = waveform_to_logmel(y, **self.mel_kwargs)     # (n_mels, frames)
        if self.spec_aug is not None:
            mel = self.spec_aug(mel.unsqueeze(0)).squeeze(0)

        return mel.unsqueeze(0), int(row["class_id"])      # (1, n_mels, frames), label


def build_noise_pool() -> list[str]:
    """从 metadata_clean 提取噪声池（role='noise'）的 npy 路径。"""
    df = pd.read_csv(config.METADATA_CLEAN)
    noise = df[df["role"] == "noise"]
    return noise["npy_path"].tolist()


if __name__ == "__main__":
    train, val, test = load_splits()
    print(f"train: {len(train)} 段 | val: {len(val)} | test: {len(test)}")
    noise_pool = build_noise_pool()
    print(f"噪声池: {len(noise_pool)} 段")

    ds = AudioSegmentDataset(train, noise_pool=noise_pool, augment=True)
    mel, label = ds[0]
    print(f"样本 Mel shape: {tuple(mel.shape)}, label: {label}")
