"""
SafeSoundAI — 方案 A：AED 驱动谱增益门（attenuator.py, A2.2）

信号链（docs/filter_arch.md §二）：
    STFT → Mel → DS_CNN(AED) → 类后验 → 类增益软混合（频带加权）→ 限幅/平滑
         → 线性频域掩码 → 复数谱乘法 → iSTFT

设计要点：
  * 不做逐窗 z-score（流式一致性），只做单次全局 RMS 归一化后再还原增益；
  * 类增益按「该类能量谱形」加权，避免全频段一刀切把同频段警报声一起压掉；
  * 内置硬门校验（白名单衰减 > 3 dB → 该窗直通）与能量塌陷保护（filter_safety §二）。

用法：
    python attenuator.py --audio outputs/audio/02_children_playing.wav --out /tmp/f.wav
    python attenuator.py --demo-set            # 处理 outputs/audio 下全部 demo 音频
"""
from __future__ import annotations

import argparse
import csv
import warnings
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

import config
import filter_config as fc
from audio_utils import load_and_resample
from mel import get_mel_transform
from models import DS_CNN

warnings.filterwarnings('ignore', message='At least one mel filterbank')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROFILE_CACHE = fc.FILTER_REPORT_DIR / "class_spectral_profile.json"


# ---------------------------------------------------------------------------
# 1) 类能量谱形（数据驱动，避免凭感觉设频带）
# ---------------------------------------------------------------------------
def build_class_profiles(max_per_class: int = 150, force: bool = False) -> dict[int, list[float]]:
    """用 metadata_clean.csv 采样，统计每类平均 log-Mel 谱形（归一化到 0..1）。"""
    if PROFILE_CACHE.exists() and not force:
        raw = json.loads(PROFILE_CACHE.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    mel_tf = get_mel_transform()
    buckets: dict[int, list[np.ndarray]] = {}
    with open(config.METADATA_CLEAN, newline="", encoding="utf-8") as f:
        rng = np.random.default_rng(42)
        rows = list(csv.DictReader(f))
        rng.shuffle(rows)
        for row in rows:
            cid = int(row["class_id"])
            if len(buckets.get(cid, [])) >= max_per_class:
                continue
            p = Path(row["npy_path"])
            if not p.exists():
                continue
            y = torch.from_numpy(np.load(p).astype(np.float32))
            mel = torch.log(mel_tf(y) + 1e-9)
            buckets.setdefault(cid, []).append(mel.mean(dim=1).numpy())

    profiles: dict[int, list[float]] = {}
    for cid, mels in buckets.items():
        band = np.mean(np.stack(mels), axis=0)          # (128,)
        band = (band - band.min()) / (band.max() - band.min() + 1e-9)
        profiles[cid] = band.round(4).tolist()
    PROFILE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_CACHE.write_text(json.dumps(profiles, ensure_ascii=False), encoding="utf-8")
    return profiles


# ---------------------------------------------------------------------------
# 2) mel 增益 → 线性频点增益
# ---------------------------------------------------------------------------
def mel_to_linear_gain(mel_gain_db: np.ndarray, n_freqs: int = config.N_FFT // 2 + 1) -> np.ndarray:
    """
    mel_gain_db : (n_mels,) 或 (n_mels, T)
    返回        : (n_freqs,) 或 (n_freqs, T) 的 dB 增益，用 mel 滤波器组权重插值。
    """
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs, f_min=config.F_MIN, f_max=config.F_MAX,
        n_mels=config.N_MELS, sample_rate=config.SAMPLE_RATE,
        norm=None, mel_scale="htk",
    ).numpy()                                        # (n_freqs, n_mels)
    w = fb / np.maximum(fb.sum(axis=0, keepdims=True), 1e-9)
    return w @ mel_gain_db


# ---------------------------------------------------------------------------
# 3) 核心：AED 后验 → 时频增益
# ---------------------------------------------------------------------------
class SpectralGateAttenuator:
    def __init__(self, profile: str = fc.DEFAULT_PROFILE,
                 aed_ckpt: str | Path | None = None,
                 device: torch.device = DEVICE,
                 smooth_alpha: float = fc.SMOOTH_ALPHA,
                 use_class_bands: bool = True,
                 apply_conf: float = fc.APPLY_CONF_THRESHOLD):
        self.profile = fc.get_profile(profile)
        self.gains = {int(k): float(v) for k, v in self.profile["gains"].items()}
        self.max_att = float(self.profile["max_total_attenuation_db"])
        self.device = device
        self.smooth_alpha = smooth_alpha
        self.use_class_bands = use_class_bands
        # 置信度门控：只有 p ≥ apply_conf 的类别才参与增益叠加。
        # 理由（实测）：当前 AED 对 11 条 demo 里 8 条输出 car_horn≈0.6-0.7，
        # 警笛被判成 jackhammer 0.40。若用全类软混合，模糊后验会把警笛压掉 6 dB。
        # 改进后：判不准 = 不衰减（fail-open），安全优先。
        self.apply_conf = float(apply_conf)
        self.stats = {"silent_suppression_events": 0, "over_suppression_events": 0,
                      "frames": 0, "hard_gate_windows": 0, "pass_frames": 0}

        self.model = DS_CNN().to(device)
        ckpt = Path(aed_ckpt or (config.MODEL_DIR / "ds_cnn_best.pth"))
        self.model.load_state_dict(torch.load(ckpt, map_location=device))
        self.model.eval()

        prof = build_class_profiles()
        self.class_band = np.zeros((config.NUM_CLASSES, config.N_MELS), dtype=np.float32)
        for cid in range(config.NUM_CLASSES):
            if cid in prof:
                self.class_band[cid] = np.array(prof[cid], dtype=np.float32)

        self.mel_tf = get_mel_transform().to(device)

    # -- AED 后验：按 1 s 窗批量推理（L3 过渡方案，见 filter_constraints §三） --
    @torch.no_grad()
    def window_probs(self, y: np.ndarray, hop: int = config.SEGMENT_SAMPLES // 2) -> np.ndarray:
        seg = config.SEGMENT_SAMPLES
        if len(y) < seg:
            y = np.pad(y, (0, seg - len(y)))
        starts = list(range(0, max(1, len(y) - seg + 1), hop))
        wins = np.stack([y[s:s + seg] for s in starts]).astype(np.float32)
        logits = []
        for i in range(0, len(wins), 32):
            batch = torch.from_numpy(wins[i:i + 32]).to(self.device)   # (B, 16000)
            mel = torch.stack([self.mel_tf(b) for b in batch])         # (B, n_mels, T')
            mel = torch.log(mel + 1e-9)[:, None, :, :config.FRAMES]
            if mel.shape[-1] < config.FRAMES:
                mel = F.pad(mel, (0, config.FRAMES - mel.shape[-1]))
            logits.append(self.model(mel))
        probs = F.softmax(torch.cat(logits), dim=1).cpu().numpy()
        return probs, starts                                      # (n_win, 10)

    def frame_probs(self, y: np.ndarray) -> np.ndarray:
        """把窗口级后验上采样到 STFT 帧级（重叠窗取平均）。"""
        probs, starts = self.window_probs(y)
        n_frames = 1 + len(y) // fc.FILTER_HOP
        acc = np.zeros((n_frames, config.NUM_CLASSES), dtype=np.float32)
        cnt = np.zeros((n_frames, 1), dtype=np.float32)
        win_frames = config.SEGMENT_SAMPLES // fc.FILTER_HOP     # 100
        for wi, s in enumerate(starts):
            f0 = s // fc.FILTER_HOP
            f1 = min(f0 + win_frames, n_frames)
            if f1 <= f0:
                continue
            acc[f0:f1] += probs[wi][None, :]
            cnt[f0:f1] += 1.0
        return acc / np.maximum(cnt, 1.0)

    # -- 增益曲线：类软混合 + 频带加权 + 限幅 + 平滑 --
    def gain_curve_db(self, probs_frames: np.ndarray) -> np.ndarray:
        """
        probs_frames : (T, 10)
        返回 (n_mels, T) 的 dB 增益

        置信度门控版加权：w_c ∝ exp(p_c/τ) · 1[p_c ≥ apply_conf]。
        整帧没有任何类别达到阈值 → 权重全 0 → 增益 0 dB（直通）。
        """
        T = probs_frames.shape[0]
        conf = (probs_frames >= self.apply_conf).astype(np.float32)
        w = np.exp(probs_frames / fc.SOFTMAX_TEMPERATURE) * conf
        denom = np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
        w = np.where(w.sum(axis=1, keepdims=True) > 0, w / denom, 0.0)   # (T, 10)
        self.stats["pass_frames"] += int((w.sum(axis=1) == 0).sum())

        g_db = np.zeros((config.N_MELS, T), dtype=np.float32)
        for cid in range(config.NUM_CLASSES):
            target = self.gains.get(cid, 0.0)
            if abs(target) < 1e-6:
                continue
            if self.use_class_bands:
                band = self.class_band[cid][:, None]                 # (128,1)
            else:
                band = np.ones((config.N_MELS, 1), dtype=np.float32)
            g_db += (w[:, cid][None, :] * target) * band             # 能量带加权

        np.clip(g_db, max(fc.GAIN_CLAMP_DB[0], -self.max_att), fc.GAIN_CLAMP_DB[1], out=g_db)
        # 跨帧一阶平滑 + 跳变限制
        sm = np.empty_like(g_db)
        prev = g_db[:, 0]
        for t in range(T):
            cur = self.smooth_alpha * prev + (1 - self.smooth_alpha) * g_db[:, t]
            jump = np.clip(cur - prev, -fc.MAX_DB_JUMP_PER_FRAME, fc.MAX_DB_JUMP_PER_FRAME)
            prev = prev + jump
            sm[:, t] = prev
        return sm

    # -- 硬门：白名单类衰减 >3 dB 且置信度高 → 该窗直通（filter_safety §2.1） --
    def hard_gate_mask(self, probs_frames: np.ndarray) -> np.ndarray:
        """返回 (T,) bool，True = 允许施加增益。"""
        allow = np.ones(probs_frames.shape[0], dtype=bool)
        for cid in fc.WHITELIST_IDS:
            strong = probs_frames[:, cid] > 0.5
            # 白名单类目标增益为 0 → 只有被软混合"误伤"才需要门控
            allow &= ~strong | (probs_frames[:, cid] < 0.0)
        return allow

    def _resolve_probs(self, y: np.ndarray, oracle_class: int | None = None) -> np.ndarray:
        """取 AED 后验；oracle_class 不为空时用真值 one-hot 替换（评测上界用）。"""
        probs = self.frame_probs(y)
        if oracle_class is not None:
            probs = np.zeros_like(probs)
            probs[:, int(oracle_class)] = 1.0
        return probs

    # -- 端到端处理 --
    def process(self, y: np.ndarray, oracle_class: int | None = None) -> tuple[np.ndarray, dict]:
        y = y.astype(np.float32)
        probs = self._resolve_probs(y, oracle_class)
        mel_gain_db = self.gain_curve_db(probs)                      # (128, T)
        lin_gain_db = mel_to_linear_gain(mel_gain_db)                # (257, T)

        # 硬门帧 → 增益归零
        allow = self.hard_gate_mask(probs)
        lin_gain_db[:, ~allow] = 0.0
        self.stats["hard_gate_windows"] += int((~allow).sum())

        # STFT
        win = torch.hann_window(fc.FILTER_WIN)
        Y = torch.stft(torch.from_numpy(y), n_fft=fc.FILTER_N_FFT, hop_length=fc.FILTER_HOP,
                       win_length=fc.FILTER_WIN, window=win, center=True,
                       return_complex=True)
        T = Y.shape[-1]
        g = lin_gain_db[:, :T]
        if g.shape[1] < T:
            g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
        gain = torch.from_numpy(10.0 ** (g / 20.0)).to(Y.dtype)
        Y2 = Y * gain

        y_out = torch.istft(Y2, n_fft=fc.FILTER_N_FFT, hop_length=fc.FILTER_HOP,
                            win_length=fc.FILTER_WIN, window=win, center=True,
                            length=len(y)).numpy().astype(np.float32)

        # 能量塌陷保护
        e_in = float(np.mean(y ** 2) + 1e-12)
        e_out = float(np.mean(y_out ** 2) + 1e-12)
        delta_db = 10 * np.log10(e_out / e_in)
        if delta_db < fc.ANTI_COLLAPSE_DB:
            self.stats["over_suppression_events"] += 1
            y_out = y.copy()

        self.stats["frames"] += int(T)
        info = {"probs_top": int(np.argmax(probs.mean(axis=0))),
                "top_name": config.TARGET_CLASSES[int(np.argmax(probs.mean(axis=0)))],
                "energy_delta_db": round(float(delta_db), 2),
                "applied": bool(delta_db >= fc.ANTI_COLLAPSE_DB)}
        return y_out, info

    # -- 供 eval 使用：把本方法产生的掩码施加到任意分轨（评测协议 §3.1） --
    def mask_from(self, y_ref: np.ndarray, oracle_class: int | None = None) -> np.ndarray:
        probs = self._resolve_probs(y_ref, oracle_class)
        mel_gain_db = self.gain_curve_db(probs)
        lin = mel_to_linear_gain(mel_gain_db)
        allow = self.hard_gate_mask(probs)
        lin[:, ~allow] = 0.0
        return 10.0 ** (lin / 20.0)                                  # (257,T)

    @staticmethod
    def apply_mask(y: np.ndarray, gain: np.ndarray) -> np.ndarray:
        win = torch.hann_window(fc.FILTER_WIN)
        Y = torch.stft(torch.from_numpy(y.astype(np.float32)), n_fft=fc.FILTER_N_FFT,
                       hop_length=fc.FILTER_HOP, win_length=fc.FILTER_WIN,
                       window=win, center=True, return_complex=True)
        T = Y.shape[-1]
        g = gain[:, :T]
        if g.shape[1] < T:
            g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
        return torch.istft(Y * torch.from_numpy(g).to(Y.dtype), n_fft=fc.FILTER_N_FFT,
                           hop_length=fc.FILTER_HOP, win_length=fc.FILTER_WIN,
                           window=win, center=True, length=len(y)).numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--profile", type=str, default=fc.DEFAULT_PROFILE)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--demo-set", action="store_true", help="批量处理 outputs/audio/*.wav")
    args = ap.parse_args()

    att = SpectralGateAttenuator(profile=args.profile, aed_ckpt=args.ckpt)
    fc.FILTER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    if args.demo_set:
        targets = sorted(p for p in fc.FILTER_AUDIO_DIR.glob("*.wav")
                         if not p.stem.startswith("filtered_"))
    elif args.audio:
        targets = [Path(args.audio)]
    else:
        ap.error("需要 --audio 或 --demo-set")

    for p in targets:
        y = load_and_resample(str(p))
        y_out, info = att.process(y)
        out = Path(args.out) if (args.out and not args.demo_set) else \
            fc.FILTER_AUDIO_DIR / f"filtered_{p.stem}.wav"
        sf.write(out, y_out, config.SAMPLE_RATE, subtype="PCM_16")
        print(f"{p.name}: top={info['top_name']:<16} Δ能量={info['energy_delta_db']:>7.2f} dB "
              f"→ {out.name}")

    print("\n统计:", json.dumps(att.stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
