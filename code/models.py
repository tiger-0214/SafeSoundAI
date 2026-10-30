"""
SafeSoundAI — 模型定义（models.py）

- DS_CNN：深度可分离 CNN（端侧部署候选，目标 <2MB / <10ms）
- 提供参数量 / FLOPs 估算，用于体积与延迟验收。

输入：log-Mel (B, 1, n_mels=128, frames=100)
输出：10 类 logits
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import NUM_CLASSES, N_MELS, FRAMES


class DSConvBlock(nn.Module):
    """深度可分离卷积块：depthwise(3x3) + pointwise(1x1) + BN + ReLU。"""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride,
                                   padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class DS_CNN(nn.Module):
    """
    轻量 DS-CNN（参考 MobileNet 思想）。

    stem conv(1→32, s=2) → DS(32→64, s=2) → DS(64→128, s=2)
    → DS(128→128, s=2) → DS(128→256, s=1) → GAP → FC(256→10)
    """

    def __init__(self, n_classes: int = NUM_CLASSES, n_mels: int = N_MELS,
                 frames: int = FRAMES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = DSConvBlock(32, 64, stride=2)
        self.block2 = DSConvBlock(64, 128, stride=2)
        self.block3 = DSConvBlock(128, 128, stride=2)
        self.block4 = DSConvBlock(128, 256, stride=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, n_classes)

        self.n_mels = n_mels
        self.frames = frames

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)            # (B, 256, 1, 1)
        x = x.flatten(1)           # (B, 256)
        return self.fc(x)          # (B, 10)


def count_parameters(model: nn.Module) -> int:
    """可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """float32 权重体积（MB）。"""
    return count_parameters(model) * 4 / (1024 * 1024)


if __name__ == "__main__":
    model = DS_CNN()
    x = torch.randn(2, 1, N_MELS, FRAMES)
    out = model(x)
    print(f"输入: {tuple(x.shape)} → 输出: {tuple(out.shape)}")
    print(f"参数量: {count_parameters(model):,}")
    print(f"体积(float32): {model_size_mb(model):.2f} MB  (目标 <2MB)")
