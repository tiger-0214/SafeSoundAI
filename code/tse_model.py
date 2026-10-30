"""
SafeSoundAI — 方案 B：类别条件目标提取网络（tse_model.py, A2.3）

核心思想：把 SoundBubble (Nat. Electron. 2024) 的「距离条件嵌入」换成
「**类别条件嵌入**」——同一套权重，运行时切换 keep 集合即可改变"保留什么"，
无需重训。详见 docs/filter_arch.md §三。

网络：因果深度可分离卷积 U-Net + FiLM 条件注入 + 复数掩码输出
  * 输入：log1p(|STFT|)  (B, 1, 257, T)
  * 条件：10 维 keep 向量 → MLP → 32 维 → 每层 FiLM(γ, β)
  * 输出：(B, 2, 257, T) → (幅值掩码, 相位残差)
  * 因果：时间维只做左填充，不做未来帧卷积
  * 目标：≤ 1 M 参数（实测约 0.2 M，见报告）

用法：
    from tse_model import MaskNet, extract_waveform
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import filter_config as fc


# ---------------------------------------------------------------------------
# 基础模块
# ---------------------------------------------------------------------------
class CausalConv2d(nn.Module):
    """时间维因果（左填充），频率维对称填充。"""

    def __init__(self, in_c: int, out_c: int, kernel=(3, 3), stride=(1, 1),
                 dilation=(1, 1), groups: int = 1):
        super().__init__()
        # 张量布局 (B,C,F,T)：kernel[0]=频率核, kernel[1]=时间核
        self.pad_t = dilation[1] * (kernel[1] - 1)
        self.pad_f = dilation[0] * (kernel[0] - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, dilation=dilation,
                              groups=groups, bias=False)

    def forward(self, x):
        x = F.pad(x, (self.pad_t, 0, self.pad_f, self.pad_f))
        return self.conv(x)


class FiLM(nn.Module):
    """条件仿射：h ← h·(1+γ) + β"""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h, cond):
        gb = self.to_gamma_beta(cond)[:, :, None, None]          # (B,2C,1,1)
        gamma, beta = gb.chunk(2, dim=1)
        return h * (1 + gamma) + beta


class DSBlock(nn.Module):
    """深度可分离卷积 + BN + PReLU，可下采样（stride）。"""

    def __init__(self, in_c: int, out_c: int, stride=(1, 1)):
        super().__init__()
        self.dw = CausalConv2d(in_c, in_c, (3, 3), stride=stride, groups=in_c)
        self.pw = CausalConv2d(in_c, out_c, (1, 1))
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CausalDilated(nn.Module):
    """瓶颈处的因果空洞卷积（时间维扩感受野）。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dw = CausalConv2d(channels, channels, (1, 3), dilation=(1, dilation),
                               groups=channels)
        self.pw = CausalConv2d(channels, channels, (1, 1))
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.PReLU(channels)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# 主网络
# ---------------------------------------------------------------------------
class MaskNet(nn.Module):
    """类别条件复数掩码网络（因果）。"""

    def __init__(self, n_freqs: int = config.N_FFT // 2 + 1, cond_dim: int = 32,
                 ch: tuple[int, ...] = (32, 48, 64, 80), dilations: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.n_freqs = n_freqs
        self.cond_mlp = nn.Sequential(
            nn.Linear(config.NUM_CLASSES, 64), nn.PReLU(64),
            nn.Linear(64, cond_dim), nn.PReLU(cond_dim),
        )
        self.stem = nn.Sequential(
            CausalConv2d(1, ch[0], (3, 3)),
            nn.BatchNorm2d(ch[0]), nn.PReLU(ch[0]),
        )
        self.films = nn.ModuleList()
        self.down = nn.ModuleList()
        c_prev = ch[0]
        for c in ch[1:]:
            self.down.append(DSBlock(c_prev, c, stride=(2, 1)))   # 频率维下采样 ×2
            self.films.append(FiLM(cond_dim, c))
            c_prev = c
        self.bottleneck = nn.Sequential(*[CausalDilated(c_prev, d) for d in dilations])
        self.film_bottleneck = FiLM(cond_dim, c_prev)
        self.up = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.Sequential(
                nn.ConvTranspose2d(c_prev, c, (2, 1), stride=(2, 1)),
                nn.BatchNorm2d(c), nn.PReLU(c),
            ))
            c_prev = c
        self.head = nn.Conv2d(c_prev, 2, 1)
        # 关键初始化：幅值掩码初值 ≈0.85（sigmoid(1.735)），相位残差初值 0
        # 否则未训练的随机 head 会把相位乱转，SI-SNR 直接崩到 -20 dB 量级
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias[0] = 1.735
            self.head.bias[1] = 0.0

    def forward(self, mag, cond):
        """mag: (B,1,F,T) log1p 幅值；cond: (B,10) keep 向量"""
        c = self.cond_mlp(cond)
        h = self.stem(mag)
        skips = [h]
        for block, film in zip(self.down, self.films):
            h = film(block(h), c)
            skips.append(h)
        h = self.film_bottleneck(self.bottleneck(h), c)
        for up, skip in zip(self.up, reversed(skips[:-1])):
            h = up(h)
            # 频率对齐（奇数长度下采样后上采样会差 1 格）
            df = skip.shape[-2] - h.shape[-2]
            if df > 0:
                h = F.pad(h, (0, 0, 0, df))
            elif df < 0:
                h = h[..., :skip.shape[-2], :]
            h = h + skip
        out = self.head(h)
        mag_mask = torch.sigmoid(out[:, :1])                 # 0..1
        phase_res = torch.tanh(out[:, 1:2]) * 0.5            # ±0.5π 残差
        return mag_mask, phase_res


# ---------------------------------------------------------------------------
# STFT / 掩码施加（与 attenuator 一致的参数）
# ---------------------------------------------------------------------------
def stft(y: torch.Tensor) -> torch.Tensor:
    win = torch.hann_window(fc.FILTER_WIN, device=y.device)
    return torch.stft(y, n_fft=fc.FILTER_N_FFT, hop_length=fc.FILTER_HOP,
                      win_length=fc.FILTER_WIN, window=win, center=True,
                      return_complex=True)


def istft(Y: torch.Tensor, length: int) -> torch.Tensor:
    win = torch.hann_window(fc.FILTER_WIN, device=Y.device)
    return torch.istft(Y, n_fft=fc.FILTER_N_FFT, hop_length=fc.FILTER_HOP,
                       win_length=fc.FILTER_WIN, window=win, center=True, length=length)


def complex_mask(mag_mask: torch.Tensor, phase_res: torch.Tensor) -> torch.Tensor:
    """(B,1,F,T)+(B,1,F,T) → (B,F,T) 复数掩码"""
    gain = mag_mask[:, 0]
    ph = phase_res[:, 0] * torch.pi
    return torch.polar(gain, ph)


@torch.no_grad()
def extract_waveform(model: MaskNet, y: torch.Tensor, keep_ids, device=None) -> torch.Tensor:
    """对整段波形做一次类别条件分离（离线/评测用）。"""
    device = device or next(model.parameters()).device
    model.eval()
    y = y.to(device).float()
    Y = stft(y)
    mag = torch.log1p(Y.abs())[None, None]
    cond = torch.tensor([fc.keep_vector(keep_ids)], dtype=torch.float32, device=device)
    mag_mask, phase_res = model(mag, cond)
    Y_out = Y * complex_mask(mag_mask, phase_res)[0]
    return istft(Y_out, len(y))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MaskNet()
    x = torch.randn(2, 1, config.N_FFT // 2 + 1, 100)
    cond = torch.tensor([fc.keep_vector(fc.KEEP_SETS["all_safety"])] * 2, dtype=torch.float32)
    mm, pr = m(x, cond)
    print("参数量:", f"{count_params(m):,}")
    print("输出:", tuple(mm.shape), tuple(pr.shape))
    # 因果性自检（eval 模式：BN 用 running stats，否则 batch 统计会跨时间泄漏）
    m.eval()
    mm, _ = m(x, cond)
    x2 = x.clone(); x2[:, :, :, 80:] += 1.0
    mm2, _ = m(x2, cond)
    diff = (mm[..., :70] - mm2[..., :70]).abs().max().item()
    print(f"因果性自检（未来扰动对过去帧影响，应≈0）: {diff:.2e}")
