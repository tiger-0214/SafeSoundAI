"""SafeSoundAI — 生成「干净语音 + 噪声」配对集（SoundBubble 路线的训练数据）

**与 `4_Synthetic_Mixtures` 的区别（容易混，先看这段）**：
  * `4_Synthetic_Mixtures`：target 是**噪声类**（空调/电钻…），interferer 是危险声。
    它服务的是「降噪支路压得对不对」的**评测**，里面**一条语音都没有**。
  * 本脚本：target 是**干净语音**，噪声是「除人声外的一切」（含危险声）。
    它服务的是单通道语音增强（TF-GridNet 掩码）的**训练** —— 即「除人声外全压」。

两者不能互相替代：用评测集训模型会学成「压噪声、放大危险声」，正好反了。

用法：
    python make_speech_mixtures.py --n 50 --split smoke        # 烟测
    python make_speech_mixtures.py --n 10000 --split train     # 正式训练集
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

import config

SPEECH_ROOT = Path("/root/safesound_data/speech/LibriSpeech/dev-clean")
MIX_ROOT = Path("/root/safesound_data/4_Synthetic_Mixtures")
OUT_ROOT = Path("/root/safesound_data/speech_mixtures")
SR = config.SAMPLE_RATE                      # 16 kHz，与 filter_pipeline 一致


def _load_mono(path: Path) -> np.ndarray | None:
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:                                    # noqa: BLE001
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:                                         # LibriSpeech 就是 16 k，噪声源同为 16 k
        import torch, torchaudio
        y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR).numpy()
    return y.astype(np.float32)


def _fit(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """裁/循环到恰好 n 个样本；语音用「居中裁剪」，噪声用「循环平铺」。"""
    if len(y) >= n:
        start = int(rng.integers(0, len(y) - n + 1))
        return y[start:start + n]
    reps = int(np.ceil(n / max(len(y), 1)))
    return np.tile(y, reps)[:n]


def collect_noise_pool(limit: int = 6000) -> list[Path]:
    """噪声池 = 现有混合集的全部分轨（target/interferer/noise 都算「非语音」）。"""
    pool: list[Path] = []
    with open(MIX_ROOT / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in ("target_path", "interferer_path", "noise_path"):
                p = Path(row[k])
                if p.exists():
                    pool.append(p)
    if len(pool) > limit:
        random.Random(0).shuffle(pool)
        pool = pool[:limit]
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--split", default="smoke")
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--snr-min", type=float, default=-5.0)
    ap.add_argument("--snr-max", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    n_samples = int(args.dur * SR)

    speech_files = sorted(SPEECH_ROOT.rglob("*.flac"))
    if not speech_files:
        raise SystemExit(f"找不到语音源：{SPEECH_ROOT}")
    noise_pool = collect_noise_pool()
    if not noise_pool:
        raise SystemExit(f"找不到噪声源：{MIX_ROOT}/manifest.csv")

    out = OUT_ROOT / args.split
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "noise").mkdir(parents=True, exist_ok=True)
    (out / "mix").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    i = 0
    tries = 0
    while len(rows) < args.n and tries < args.n * 20:
        tries += 1
        sp_path = speech_files[int(rng.integers(0, len(speech_files)))]
        sp = _load_mono(sp_path)
        if sp is None or len(sp) < SR:              # 短于 1 s 的不要（能量/RMS 估计不稳）
            continue
        nz_path = noise_pool[int(rng.integers(0, len(noise_pool)))]
        nz = _load_mono(nz_path)
        if nz is None:
            continue

        clean = _fit(sp, n_samples, rng)
        noise = _fit(nz, n_samples, rng)

        # 按 SNR 定噪声音量：speech 与 noise 都用 RMS（不用峰值，避免瞬态主导）
        e_s = float(np.mean(clean ** 2) + 1e-12)
        e_n = float(np.mean(noise ** 2) + 1e-12)
        snr_db = float(rng.uniform(args.snr_min, args.snr_max))
        target_e_n = e_s / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_e_n / e_n)

        mix = clean + noise
        peak = float(np.max(np.abs(mix)) + 1e-9)
        scale = 0.9 / peak if peak > 0.9 else 1.0
        clean, noise, mix = clean * scale, noise * scale, mix * scale

        i += 1
        tag = f"sp{i:06d}"
        sf.write(out / "clean" / f"{tag}.wav", clean, SR, subtype="PCM_16")
        sf.write(out / "noise" / f"{tag}.wav", noise, SR, subtype="PCM_16")
        sf.write(out / "mix" / f"{tag}.wav", mix, SR, subtype="PCM_16")
        rows.append({
            "mix_id": tag,
            "clean_path": str(out / "clean" / f"{tag}.wav"),
            "noise_path": str(out / "noise" / f"{tag}.wav"),
            "mix_path": str(out / "mix" / f"{tag}.wav"),
            "speech_src": str(sp_path),
            "noise_src": str(nz_path),
            "snr_db": round(snr_db, 2),
            "dur_s": args.dur,
            "split": args.split,
            "seed": args.seed,
        })

    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "stats.json").write_text(json.dumps({
        "n": len(rows), "split": args.split, "dur_s": args.dur,
        "snr_range_db": [args.snr_min, args.snr_max],
        "speech_pool": len(speech_files), "noise_pool": len(noise_pool),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {len(rows)} 条 → {out}（speech 池 {len(speech_files)}｜noise 池 {len(noise_pool)}）")


if __name__ == "__main__":
    main()
