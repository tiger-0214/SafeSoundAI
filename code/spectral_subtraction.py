#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VAD 调制非对称谱减（A 路线）—— 不依赖 AED 分类的降噪主干。

## 为什么需要它

V2.1 的语义门控已经把「非语音帧」压到 −18 dB，但它是**二值的**：
Silero VAD 判为语音的那 10 ms 帧，整帧增益被抬回 0 dB，帧内的稳态噪声
（空调嗡鸣、引擎低频、街道底噪）原样进入耳朵。实测 E2 的 SI-SDR 提升只有
+0.12 dB —— 语音是保住了，但语音**所在的那段噪声**没动。

谱减补的正是这个洞：它不是「整帧放行/整帧压掉」，而是逐时频点估计
「这个 bin 里有多少是噪声」，然后把噪声减掉、语音留着。所以在语音帧内
也能降噪。

## 非对称在哪里

经典谱减对全帧一视同仁，压得深就伤语音、压得浅就吵。这里过减因子 α 被
VAD 概率调制：

    α(t) = α_noise + (α_speech − α_noise) · p_speech(t)

- 无语音帧（p≈0）：α = α_noise ≈ 3.0 → 激进过减，反正没有内容怕伤
- 语音帧（p≈1）：α = α_speech ≈ 1.0 → 保守，只减掉噪声的期望功率

这就是标题里「非对称」的含义，也是与 Bhat 等（2020）「保护警报声」路线的
区别 —— 他们做的是**让位**，这里做的是**按语音存在概率分配攻击性**。

## 与语义增益的关系

两者正交，可以叠：
- 语义增益（attenuator）回答「这是什么声音」→ 类别级、整窗
- 谱减回答「这个时频点里噪声占多少」→ 频点级、10 ms

叠加方式取逐点较小值（更保守的那个生效）。见 `combine_gain_db()`。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter1d, uniform_filter, uniform_filter1d

import filter_config as fc
import config

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
@dataclass
class SpectralSubParams:
    """谱减参数。全部有物理含义，可逐项消融（这是选 DSP 路线的主要理由）。"""

    n_fft: int = fc.FILTER_N_FFT
    hop: int = fc.FILTER_HOP
    win: int = fc.FILTER_WIN
    sr: int = config.SAMPLE_RATE

    alpha_noise: float = 8.0        # 无语音帧的过减因子（激进）
    alpha_speech: float = 4.0       # 语音帧的过减因子（保守）
    # 谱底同样非对称：语音帧留高谱底防音乐噪声，非语音帧踩到底（那里没有
    # 内容可伤，唯一代价是潜在的 musical noise，而它在纯噪声段几乎听不出来）。
    beta_speech: float = 0.08       # 语音帧幅度下限（−22 dB）
    beta_noise: float = 0.003       # 非语音帧幅度下限（−50 dB）
    noise_bias: float = 1.5         # 最小值统计的偏置补偿（Martin 2001）
    min_stat_sec: float = 1.2       # 最小值跟踪窗长
    smooth_hz: int = 3              # 频域平滑（bin）
    smooth_frames: int = 2          # 时域平滑（帧）
    # 功率谱时间平滑帧数。谱减对瞬时功率的指数分布方差极其敏感：单个 bin 的
    # 瞬时值可能达到均值的 10 倍，此时 1−α/γ 直接趋近 1，这些点会吃掉几乎
    # 全部残留能量（实测 dB 均值 −16 dB、能量加权只有 −0.75 dB）。先在功率
    # 域沿时间做 ~50 ms 平滑再算 SNR，是压制这个方差最直接的手段。
    smooth_ps_frames: int = 5
    # 噪声估计是否只用**过去**的帧。流式部署必须是 True，否则流式版本会偷看
    # 未来帧、离线评测与实时表现不一致。默认 True 是对齐部署口径。
    causal_noise: bool = True
    floor_speech_db: float = -12.0  # 语音帧允许的最深衰减
    floor_noise_db: float = -40.0   # 非语音帧允许的最深衰减
    max_atten_db: float | None = None   # 外部硬下限（None = 用 profile 的）

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 噪声功率谱估计
# --------------------------------------------------------------------------- #
def estimate_noise_psd(power: np.ndarray, vad: np.ndarray | None,
                       p: SpectralSubParams) -> np.ndarray:
    """最小值统计（minimum statistics）估计噪声功率谱。

    power : (F, T) 功率谱
    vad   : (T,) 语音概率 ∈[0,1]，None = 不做语音加权

    为什么用最小值统计而不是「非语音帧平均」：后者要求录音里必须有一段
    纯噪声；用户戴着耳机走路的场景往往整段都在说话或整段都在噪声里，
    拿不到干净的噪声样本。最小值统计逐帧自适应，不需要「静音段」。

    做法：对每个频点取过去 min_stat_sec 内的功率最小值，乘以偏置因子
    补偿「最小值系统性低估均值」这个偏差（高斯噪声下约 1.5×）。
    """
    T = power.shape[1]
    D = max(int(round(p.min_stat_sec * p.sr / p.hop)), 1)
    D = min(D, T)
    if p.causal_noise:
        # 因果最小值：第 t 帧只看 [t-D+1, t]。开头 D-1 帧没有足够历史，
        # 用第一个有效窗口的值回填（等价于"假设开头就是稳态"）。
        pmin = minimum_filter1d(power, size=D, axis=1, mode="nearest")
        if T >= D:
            win = sliding_window_view(power, D, axis=1)      # (F, T-D+1, D)
            cmin = win.min(axis=-1)
            pmin = np.empty_like(power)
            pmin[:, :D - 1] = cmin[:, :1]
            pmin[:, D - 1:] = cmin
        else:
            pmin = np.repeat(power.min(axis=1, keepdims=True), T, axis=1)
    else:
        pmin = minimum_filter1d(power, size=D, axis=1, mode="nearest")
    noise = pmin * p.noise_bias
    if vad is not None and np.any(vad < 0.5):
        # 非语音帧是噪声的"直接观测"，用它逐频点校正最小值估计：
        # 最小值可能落在语音间隙里的低谷，取二者较大者避免过减。
        quiet = vad < 0.5
        if quiet.sum() >= max(3, int(0.1 * T)):
            direct = np.median(power[:, quiet], axis=1, keepdims=True)
            noise = np.maximum(noise, direct)
    return np.maximum(noise, EPS)


# --------------------------------------------------------------------------- #
# 增益
# --------------------------------------------------------------------------- #
def vad_modulated_gain(power: np.ndarray, vad: np.ndarray | None,
                       p: SpectralSubParams | None = None) -> tuple[np.ndarray, dict]:
    """返回 (F, T) 的 dB 增益与诊断信息。"""
    p = p or SpectralSubParams()
    F, T = power.shape
    if vad is None:
        vad = np.zeros(T, dtype=np.float32)
    vad = np.clip(np.asarray(vad, dtype=np.float32), 0.0, 1.0)

    # 先平滑功率谱（只用于 SNR 估计，增益仍施加到原始 STFT）
    ps = power
    if p.smooth_ps_frames > 1:
        ps = uniform_filter1d(power, size=int(p.smooth_ps_frames), axis=1, mode="nearest")

    noise = estimate_noise_psd(ps, vad, p)

    # 后验 SNR 与过减因子（逐帧由 VAD 概率插值）
    snr_post = ps / noise
    alpha = p.alpha_noise + (p.alpha_speech - p.alpha_noise) * vad[None, :]

    # 功率域谱减 → 幅度增益；开方后谱底也要开方
    g2 = 1.0 - alpha / np.maximum(snr_post, 1e-6)
    g = np.sqrt(np.maximum(g2, 0.0))
    beta_eff = p.beta_noise + (p.beta_speech - p.beta_noise) * vad[None, :]
    g = np.maximum(g, np.sqrt(beta_eff))

    # 时频平滑：单帧谱减的经典毛病是 musical noise（随机残留的窄带哨音），
    # 平滑把孤立的过减谷填上，听感上从"水声"变"闷"。
    if p.smooth_hz > 1 or p.smooth_frames > 1:
        g = uniform_filter(g, size=(max(p.smooth_hz, 1), max(p.smooth_frames, 1)),
                           mode="nearest")

    g_db = 20.0 * np.log10(np.maximum(g, EPS))

    # 衰减下限：语音帧不许压穿 floor_speech_db（保可懂度），
    # 非语音帧宽松得多（那里没有内容可伤）
    sp = (vad > 0.5)[None, :]
    sp_full = np.broadcast_to(sp, g_db.shape)
    g_db = np.where(sp, np.maximum(g_db, p.floor_speech_db),
                    np.maximum(g_db, p.floor_noise_db))
    if p.max_atten_db is not None:
        g_db = np.maximum(g_db, p.max_atten_db)

    info = {
        "vad_mean": round(float(vad.mean()), 4),
        "vad_active_pct": round(float((vad > 0.5).mean() * 100.0), 1),
        "alpha_mean": round(float(alpha.mean()), 3),
        "gain_db_mean": round(float(g_db.mean()), 2),
        "gain_db_p10": round(float(np.percentile(g_db, 10)), 2),
        "noise_psd_mean_db": round(float(10 * np.log10(noise.mean() + EPS)), 2),
        "floor_hits_speech_pct": round(
            float((g_db[sp_full] <= p.floor_speech_db + 1e-6).mean() * 100.0) if sp.any() else 0.0, 1),
    }
    return g_db, info


# --------------------------------------------------------------------------- #
# 与语义增益合并
# --------------------------------------------------------------------------- #
def combine_gain_db(semantic_db: np.ndarray, spectral_db: np.ndarray) -> np.ndarray:
    """逐时频点取较小值（更保守的那个生效）。

    语义增益是「整窗一个类别的判决」，谱减是「这个频点里的噪声估计」。
    两者都不该被对方否决：语义说"siren −12"时，谱减认为"这个频点 SNR 很高
    不用压"也不该把 siren 放过去；反之谱减要压的地方，语义的 0 dB（人声类）
    也不该保它。取 min 是唯一同时满足这两条的合成方式。
    """
    T = min(semantic_db.shape[1], spectral_db.shape[1])
    a, b = semantic_db[:, :T], spectral_db[:, :T]
    F = min(a.shape[0], b.shape[0])
    return np.minimum(a[:F], b[:F])


# --------------------------------------------------------------------------- #
# 供评测使用的独立封装
# --------------------------------------------------------------------------- #
def _stft(y: np.ndarray, p: SpectralSubParams):
    import torch
    win = torch.hann_window(p.win)
    Y = torch.stft(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
                   n_fft=p.n_fft, hop_length=p.hop, win_length=p.win,
                   window=win, center=True, return_complex=True)
    return Y


def _istft(Y, length: int, p: SpectralSubParams) -> np.ndarray:
    import torch
    win = torch.hann_window(p.win)
    y = torch.istft(Y, n_fft=p.n_fft, hop_length=p.hop, win_length=p.win,
                    window=win, center=True, length=length)
    return y.numpy().astype(np.float32)


def apply_spectral_subtraction(y: np.ndarray, vad: np.ndarray | None = None,
                               p: SpectralSubParams | None = None):
    """端到端：纯谱减（不与语义增益叠加）。返回 (y_out, gain_db, info)。"""
    p = p or SpectralSubParams()
    Y = _stft(y, p)
    power = (Y.abs() ** 2).numpy()
    T = power.shape[1]
    if vad is not None:
        vad = _align_vad(np.asarray(vad, dtype=np.float32), T)
    gain_db, info = vad_modulated_gain(power, vad, p)
    g = np.power(10.0, gain_db / 20.0).astype(np.float32)
    import torch
    Y2 = Y * torch.from_numpy(g).to(Y.dtype)
    info["gain_db_mean"] = round(float(gain_db.mean()), 2)
    e_in = float(np.mean(y ** 2) + EPS)
    y_out = _istft(Y2, len(y), p)
    info["energy_delta_db"] = round(float(10 * np.log10(float(np.mean(y_out ** 2) + EPS) / e_in)), 2)
    return y_out, gain_db, info


def _align_vad(vad: np.ndarray, T: int) -> np.ndarray:
    """把任意帧率的 VAD 权重线性重采样到谱减的帧数 T。"""
    n = len(vad)
    if n == T:
        return vad
    if n == 0:
        return np.zeros(T, dtype=np.float32)
    idx = np.linspace(0, n - 1, T)
    return np.interp(idx, np.arange(n), vad).astype(np.float32)


# --------------------------------------------------------------------------- #
# CLI：独立验证（不经 pipeline）
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="VAD 调制非对称谱减（独立验证）")
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha-noise", type=float, default=3.0)
    ap.add_argument("--alpha-speech", type=float, default=1.0)
    ap.add_argument("--beta-speech", type=float, default=0.08)
    ap.add_argument("--beta-noise", type=float, default=0.003)
    ap.add_argument("--no-vad", action="store_true", help="消融：关掉 VAD 调制（全场激进）")
    ap.add_argument("--save-gain", action="store_true", help="保存 dB 增益矩阵 npy")
    args = ap.parse_args()

    import soundfile as sf
    import attenuator as A

    y, sr = sf.read(args.mix, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    p = SpectralSubParams(alpha_noise=args.alpha_noise,
                          alpha_speech=args.alpha_speech,
                          beta_speech=args.beta_speech, beta_noise=args.beta_noise)
    vad = None
    if not args.no_vad:
        att = A.SpectralGateAttenuator(profile=fc.DEFAULT_PROFILE)
        T_full = 1 + len(y) // p.hop
        vad = att.voice_gate_frames(y, T_full)

    y_out, gain_db, info = apply_spectral_subtraction(y, vad, p)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, y_out, sr)
    if args.save_gain:
        np.save(str(Path(args.out).with_suffix(".gain.npy")), gain_db)
    print(json.dumps({"out": args.out, **info, "params": p.to_dict()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
