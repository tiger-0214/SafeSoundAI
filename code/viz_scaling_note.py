"""
SafeSoundAI — 「scaling 变换」可视化（viz_scaling_note.py）
=========================================================
给「3-预处理全流程」笔记的 scaling 一节出图。只读 metadata_v2.csv + 1 s npy，不改数据。

出的图（outputs/figures/preprocess/）：
  fig_melbank.png        128 个 Mel 滤波器组的形状 + 带宽随中心频率变化
                         （说明：低频分得细、高频分得粗 = 在关键频段"放大分辨率"）
  fig_scaling_chain.png  同一条 siren 的四步：波形 → 线性 STFT(dB) → Mel(未取对数) → log-Mel

用法：
    export SAFESOUND_DATA_DIR=/root/safesound_data
    python3 viz_scaling_note.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torchaudio

import config
import viz_style as vs

OUT = vs.FIG_DIR / "preprocess"
OUT.mkdir(parents=True, exist_ok=True)
DATA_V2 = config.PREPROCESSED / "trigger_v2"

N_FFT, HOP, WIN = 512, 160, 512
SR, N_MELS, F_MIN, F_MAX, FRAMES = 16000, 128, 50.0, 7500.0, 100


def mel_filterbank() -> np.ndarray:
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1, f_min=F_MIN, f_max=F_MAX,
        n_mels=N_MELS, sample_rate=SR, norm=None,
    )
    return fb.numpy()                                    # (257, 128)


def fig_melbank() -> dict:
    """128 个三角形滤波器的形状 + 带宽/中心频率关系。"""
    import matplotlib.pyplot as plt

    F = mel_filterbank()
    freqs = np.arange(F.shape[0]) * SR / N_FFT              # 每个 bin 的中心频率

    bw_hz, cen_hz, wsum = [], [], []
    for i in range(N_MELS):
        nz = np.nonzero(F[:, i] > 1e-12)[0]
        if len(nz) == 0:                                     # 退化滤波器（全零）
            bw_hz.append(np.nan); cen_hz.append(np.nan); wsum.append(0.0); continue
        lo, hi = nz[0], nz[-1]
        bw_hz.append((hi - lo + 1) * SR / N_FFT)
        cen_hz.append((lo + hi) / 2 * SR / N_FFT)
        wsum.append(float(F[:, i].sum()))
    bw_hz, cen_hz = np.array(bw_hz), np.array(cen_hz)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    im = ax.imshow(F.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, SR / 2 / 1000, 1, N_MELS])
    ax.set_xlabel("频率 (kHz) — 线性刻度")
    ax.set_ylabel("Mel 滤波器序号（低 → 高）")
    ax.set_title("128 个 Mel 滤波器：低频铺得密、高频铺得疏")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="权重")
    for f in [500, 1000, 2000, 4000, 7000]:
        ax.axvline(f / 1000, color="white", lw=0.5, alpha=0.35)

    ax = axes[1]
    ok = ~np.isnan(bw_hz)
    ax.plot(cen_hz[ok], bw_hz[ok], "o-", ms=3.5, lw=1.4, color="#c44e52")
    ax.set_xscale("log")
    ax.set_xlabel("滤波器中心频率 (Hz) — 对数刻度")
    ax.set_ylabel("滤波器带宽 (Hz)")
    ax.set_title("带宽随中心频率增长：低频细、高频粗")
    for i in (0, 63, 127):
        if not np.isnan(bw_hz[i]):
            ax.annotate(f"#{i+1}\n{cen_hz[i]:.0f} Hz / {bw_hz[i]:.0f} Hz",
                        (cen_hz[i], bw_hz[i]), textcoords="offset points",
                        xytext=(8, -14), fontsize=8.5)
    ax.axhline(np.nanmean(bw_hz[:16]), color="#4c72b0", ls="--", lw=1.1,
               label=f"前 16 个平均 {np.nanmean(bw_hz[:16]):.0f} Hz")
    ax.axhline(np.nanmean(bw_hz[-16:]), color="#55a868", ls="--", lw=1.1,
               label=f"末 16 个平均 {np.nanmean(bw_hz[-16:]):.0f} Hz")
    ax.legend(fontsize=8.5, loc="upper left")

    p = OUT / "fig_melbank.png"
    fig.savefig(p)
    plt.close(fig)

    return {
        "path": str(p),
        "n_filters": N_MELS,
        "n_effective": int(ok.sum()),
        "bw_min_hz": float(np.nanmin(bw_hz)),
        "bw_max_hz": float(np.nanmax(bw_hz)),
        "bw_first16_mean": float(np.nanmean(bw_hz[:16])),
        "bw_last16_mean": float(np.nanmean(bw_hz[-16:])),
        "example": {f"#{i+1}": {"center_hz": float(cen_hz[i]), "bw_hz": float(bw_hz[i])}
                    for i in (0, 15, 31, 63, 95, 127) if not np.isnan(bw_hz[i])},
    }


def fig_scaling_chain(df: pd.DataFrame) -> str:
    """同一条 siren：波形 → 线性 STFT(dB) → Mel 谱（线性幅度）→ log-Mel（模型输入）。"""
    import matplotlib.pyplot as plt

    row = df[(df.label == "siren") & (df.split == "train")].sort_values("n_windows_total").iloc[-1]
    y = np.load(row.npy_path).astype(np.float32)

    wav = torch.from_numpy(y).float()
    spec = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                      window=torch.hann_window(WIN), center=True, return_complex=True).abs()
    spec_db = 20 * torch.log10(spec + 1e-6)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0,
    )(wav)
    mel = mel[:, :FRAMES]
    logmel = torch.log(mel + 1e-9)[:, :FRAMES]

    t = np.arange(len(y)) / SR
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})

    ax = axes[0]
    ax.plot(t, y, lw=0.7, color="#1f77b4")
    ax.set_xlim(0, t[-1]); ax.set_xlabel("时间 (s)"); ax.set_ylabel("幅值")
    ax.set_title(f"① 波形：1 s / 16,000 点（RMS=1.0）")

    ax = axes[1]
    im = ax.imshow(spec_db.numpy(), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, spec_db.shape[1] * HOP / SR, 0, SR / 2 / 1000])
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("频率 (kHz) — 线性")
    ax.set_title("② STFT：257 个线性频点 (dB)")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="dB")

    ax = axes[2]
    im = ax.imshow(mel.numpy(), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, mel.shape[1] * HOP / SR, 1, N_MELS])
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("Mel 通道（低 → 高）")
    ax.set_title("③ Mel 刻度：128 通道（幂谱）")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    ax = axes[3]
    im = ax.imshow(logmel.numpy(), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, logmel.shape[1] * HOP / SR, 1, N_MELS])
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("Mel 通道（低 → 高）")
    ax.set_title(f"④ log 压缩 → 模型输入 {tuple(logmel.shape)}")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.tight_layout()
    p = OUT / "fig_scaling_chain.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def main() -> None:
    df = pd.read_csv(DATA_V2 / "metadata_v2.csv")
    info = fig_melbank()
    p2 = fig_scaling_chain(df)

    print("== Mel 滤波器组 ==")
    for k, v in info.items():
        if k != "path":
            print(f"  {k}: {v}")
    print(f"  → {info['path']}")
    print(f"  → {p2}")


if __name__ == "__main__":
    main()
