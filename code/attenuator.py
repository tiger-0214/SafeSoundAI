"""
SafeSoundAI — 方案 A：AED 驱动谱增益门（attenuator.py, A2.2）

信号链（docs/filter_arch.md §二）：
    STFT → Mel → DS_CNN(AED) → 类后验 → 类增益软混合（频带加权）→ 限幅/平滑
         → 线性频域掩码 → 复数谱乘法 → iSTFT

设计要点（2026-10-31 修订，依据真实混合集实测，见 reports/filter_ab_report.md）：
  * **逐窗 z-score**：训练段是 z-score 过的，推理不做同样归一化会让 AED 后验整体错位
    （200 条真实混合上 top-1 从 0.17 回升到 0.53）。开关 `fc.ZSCORE_WINDOWS`。
  * 类增益按「该类能量谱形」加权，避免全频段一刀切把同频段警报声一起压掉；
  * 白名单保护从「整窗否决」改为「频谱让位」：只在白名单类的能量带内把增益抬回
    ≈0 dB，其余频段照常衰减。开关 `fc.WHITELIST_CARVEOUT`。
  * 语义一律经 `vocab` 翻译到**当前检查点自己的类别空间**，不再硬编码 10 类 id。

用法：
    python attenuator.py --audio in.wav --out /tmp/f.wav
    python attenuator.py --demo-set --ckpt outputs/models/ds_cnn_best.pth
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
import vocab
from audio_utils import load_and_resample
from mel import get_mel_transform

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
                 apply_conf: float = fc.APPLY_CONF_THRESHOLD,
                 zscore_windows: bool | None = None,
                 carveout: bool | None = None,
                 strength: float | None = None,
                 noise_conditional: bool | None = None):
        self.profile = fc.get_profile(profile)
        self.profile_name = profile
        self.max_att = float(self.profile["max_total_attenuation_db"])
        self.device = device
        self.smooth_alpha = smooth_alpha
        self.use_class_bands = use_class_bands
        self.apply_conf = float(apply_conf)
        self.zscore = fc.ZSCORE_WINDOWS if zscore_windows is None else bool(zscore_windows)
        self.carveout = fc.WHITELIST_CARVEOUT if carveout is None else bool(carveout)
        self.strength = float(fc.GAIN_STRENGTH if strength is None else strength)
        self.noise_cond = (fc.NOISE_CONDITIONAL_GAIN if noise_conditional is None
                           else bool(noise_conditional))
        self.stats = {"silent_suppression_events": 0, "over_suppression_events": 0,
                      "frames": 0, "hard_gate_windows": 0, "pass_frames": 0,
                      "carveout_frames": 0}

        # ---- AED 头：类名 / 类数 / mel 参数全部由检查点决定（vocab 是唯一入口）----
        self.head = vocab.load_aed(aed_ckpt, device=device)
        self.classes: list[str] = list(self.head.classes)
        self.n_classes = self.head.n_classes
        self.mel_tf = self.head.mel_transform().to(device)
        self.frames = int(self.head.mel["frames"])
        # 语义表按名字翻译到本头空间（白名单类永久 0 dB，未知类 fail-open 0 dB）
        self.gains: dict[int, float] = fc.gains_for(self.classes, profile)
        self.whitelist_ids = fc.whitelist_ids_in(self.classes)
        self.noise_ids = [i for i in range(self.n_classes) if i not in self.whitelist_ids]
        self.model = self.head.model

        prof = build_class_profiles()
        self.class_band = np.zeros((self.n_classes, config.N_MELS), dtype=np.float32)
        noise_ref = [band for cid, band in prof.items()
                     if cid < len(vocab.SOURCE_CLASSES)
                     and vocab.role_of(vocab.SOURCE_CLASSES[cid]) == "noise"]
        for cid, name in enumerate(self.classes):
            src = config.CLASS_TO_IDX.get(name)
            if src is not None and src in prof:
                self.class_band[cid] = np.array(prof[src], dtype=np.float32)
            elif noise_ref:                        # 如 4 类头的 background：取噪声类均值
                self.class_band[cid] = np.mean(np.stack(noise_ref), axis=0)
            else:
                self.class_band[cid] = 1.0

    # -- AED 后验：按 1 s 窗批量推理（L3 过渡方案，见 filter_constraints §三） --
    @torch.no_grad()
    def window_probs(self, y: np.ndarray, hop: int = config.SEGMENT_SAMPLES // 2) -> np.ndarray:
        seg = config.SEGMENT_SAMPLES
        if len(y) < seg:
            y = np.pad(y, (0, seg - len(y)))
        starts = list(range(0, max(1, len(y) - seg + 1), hop))
        wins = np.stack([y[s:s + seg] for s in starts]).astype(np.float32)
        if self.zscore:                            # ← 与训练口径一致（核心修复）
            wins = (wins - wins.mean(axis=1, keepdims=True)) / (wins.std(axis=1, keepdims=True) + 1e-8)
        logits = []
        for i in range(0, len(wins), 32):
            batch = torch.from_numpy(wins[i:i + 32]).to(self.device)   # (B, 16000)
            mel = torch.stack([self.mel_tf(b) for b in batch])         # (B, n_mels, T')
            mel = torch.log(mel + 1e-9)[:, None, :, :self.frames]
            if mel.shape[-1] < self.frames:
                mel = F.pad(mel, (0, self.frames - mel.shape[-1]))
            logits.append(self.model(mel))
        probs = F.softmax(torch.cat(logits), dim=1).cpu().numpy()
        return probs, starts                                      # (n_win, n_classes)

    def frame_probs(self, y: np.ndarray) -> np.ndarray:
        """把窗口级后验上采样到 STFT 帧级（重叠窗取平均）。"""
        probs, starts = self.window_probs(y)
        n_frames = 1 + len(y) // fc.FILTER_HOP
        acc = np.zeros((n_frames, self.n_classes), dtype=np.float32)
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

    # -- 增益曲线：类软混合 + 频带加权 + 白名单让位 + 限幅 + 平滑 --
    def gain_curve_db(self, probs_frames: np.ndarray) -> np.ndarray:
        """
        probs_frames : (T, n_classes)
        返回 (n_mels, T) 的 dB 增益

        置信度门控加权：w_c ∝ exp(p_c/τ) · 1[p_c ≥ apply_conf]。
        整帧没有任何类别达到阈值 → 权重全 0 → 增益 0 dB（直通）。
        白名单让位（carveout）：按 p_白名单 逐频带把增益抬回 ≈0 dB。
        """
        T = probs_frames.shape[0]
        raw = probs_frames                                   # 绝对后验（白名单让位用）
        if self.noise_cond and self.noise_ids:
            probs_frames = probs_frames.copy()
            mask_ids = np.array(self.noise_ids, dtype=int)
            sub = probs_frames[:, mask_ids]
            ssum = sub.sum(axis=1, keepdims=True)
            keep = np.zeros_like(probs_frames)
            keep[:, mask_ids] = np.where(ssum > 1e-6, sub / np.maximum(ssum, 1e-9), 0.0)
            probs_frames = keep
        conf = (probs_frames >= self.apply_conf).astype(np.float32)
        w = np.exp(probs_frames / fc.SOFTMAX_TEMPERATURE) * conf
        denom = np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
        w = np.where(w.sum(axis=1, keepdims=True) > 0, w / denom, 0.0)   # (T, n_classes)
        self.stats["pass_frames"] += int((w.sum(axis=1) == 0).sum())

        g_db = np.zeros((config.N_MELS, T), dtype=np.float32)
        for cid in range(self.n_classes):
            target = self.gains.get(cid, 0.0)
            if abs(target) < 1e-6:
                continue
            if self.use_class_bands:
                band = self.class_band[cid][:, None]                 # (n_mels,1)
            else:
                band = np.ones((config.N_MELS, 1), dtype=np.float32)
            g_db += (w[:, cid][None, :] * target) * band             # 能量带加权

        if self.carveout:
            # 让位强度 = 白名单类的逐帧置信度 × 该类能量带（只让出该类的频谱，不否决整帧）
            protect = np.zeros((config.N_MELS, T), dtype=np.float32)
            for cid in self.whitelist_ids:
                protect = np.maximum(protect,
                                     self.class_band[cid][:, None] * raw[:, cid][None, :])
            self.stats["carveout_frames"] += int((protect.max(axis=0) > 0.5).sum())
            g_db *= (1.0 - np.clip(protect, 0.0, 1.0))
            # 硬底线：白名单能量带内衰减不得深于 CARVEOUT_GUARD_DB
            g_db = np.where(protect >= 0.5, np.maximum(g_db, fc.CARVEOUT_GUARD_DB), g_db)

        np.clip(g_db, max(fc.GAIN_CLAMP_DB[0], -self.max_att), fc.GAIN_CLAMP_DB[1], out=g_db)
        if self.strength != 1.0:                 # 安全强度：向 0 dB 收缩
            g_db *= self.strength
        # 跨帧一阶平滑 + 跳变限制
        sm = np.empty_like(g_db)
        prev = g_db[:, 0]
        for t in range(T):
            cur = self.smooth_alpha * prev + (1 - self.smooth_alpha) * g_db[:, t]
            jump = np.clip(cur - prev, -fc.MAX_DB_JUMP_PER_FRAME, fc.MAX_DB_JUMP_PER_FRAME)
            prev = prev + jump
            sm[:, t] = prev
        return sm

    # -- 硬门（保守模式）：白名单类极高置信 → 该窗直通（filter_safety §2.1） --
    def hard_gate_mask(self, probs_frames: np.ndarray) -> np.ndarray:
        """返回 (T,) bool，True = 允许施加增益。

        carveout 模式下只剩「极高置信白名单」才整窗否决；旧的全量否决逻辑
        （p>0.5 即直通）在真实混合上会把 80% 的帧放空，实测见报告 §三。
        """
        allow = np.ones(probs_frames.shape[0], dtype=bool)
        if self.carveout:
            thr = fc.WHITELIST_VETO_CONF
            for cid in self.whitelist_ids:
                allow &= probs_frames[:, cid] < thr
        else:
            for cid in self.whitelist_ids:
                allow &= probs_frames[:, cid] < 0.5
        return allow

    def _resolve_probs(self, y: np.ndarray, oracle_class: int | None = None) -> np.ndarray:
        """取 AED 后验；oracle_class 不为空时用真值 one-hot 替换（评测上界用）。

        oracle_class 是**源类空间（10 类）** id，这里经 vocab 桥接进本头空间；
        桥接不到（4 类头没有该类）时退回「全零 → fail-open 不衰减」。
        """
        probs = self.frame_probs(y)
        if oracle_class is not None:
            probs = np.zeros_like(probs)
            hid = vocab.source_to_bridge(int(oracle_class), self.classes)
            if hid is None:
                hid = self.classes.index("background") if "background" in self.classes else 0
            probs[:, hid] = 1.0
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
        top_i = int(np.argmax(probs.mean(axis=0)))
        info = {"probs_top": top_i,
                "top_name": self.classes[top_i],           # ← 本头自己的类名，不再反查 10 类表
                "top_prob": round(float(probs.mean(axis=0)[top_i]), 3),
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
    ap.add_argument("--no-zscore", action="store_true", help="关闭逐窗 z-score（消融用）")
    ap.add_argument("--no-carveout", action="store_true", help="回退整窗否决式白名单保护（消融用）")
    args = ap.parse_args()

    att = SpectralGateAttenuator(profile=args.profile, aed_ckpt=args.ckpt,
                                 zscore_windows=not args.no_zscore,
                                 carveout=not args.no_carveout)
    print(f"AED 头: {vocab.describe(att.head)}")
    print(f"模式: zscore={att.zscore} carveout={att.carveout} "
          f"gains={ {att.classes[i]: g for i, g in att.gains.items()} }")
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
        print(f"{p.name}: top={info['top_name']}({info['top_prob']:.2f}) "
              f"Δ能量={info['energy_delta_db']:>7.2f} dB applied={info['applied']} → {out.name}")

    print("\n统计:", json.dumps(att.stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
