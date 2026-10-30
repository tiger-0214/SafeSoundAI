"""
SafeSoundAI — 合成混音数据生成器（synth_mix.py, A2.1）

按 docs/synth_mixer_spec.md 生成带分轨（stems）的混音样本，供：
  - 方案 A（attenuator.py）离线验证
  - 方案 B（tse_model.py / train_tse.py）训练与评测

五元组：target(黑名单/中性) + interferer(白名单) + noise_bed + snr_target_db + keep_set

用法：
    python synth_mix.py --n-train 3000 --n-val 500 --n-test 500
    python synth_mix.py --n-train 20 --n-val 5 --n-test 5 --workers 2   # 冒烟测试
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

import config
import filter_config as fc

SR = config.SAMPLE_RATE
SEG = config.SEGMENT_SAMPLES           # 16000
# 数据源可指向 ext4 本地副本（drvfs 上逐文件读取是 10x 慢的瓶颈）
#   export SAFESOUND_SRC_DIR=/path/to/raw_corpora
SRC = Path(os.environ.get("SAFESOUND_SRC_DIR", str(config.RAW_DATA)))
USD8K_ROOT = config.RAW_DATA / "UrbanSound8K"
ESC50_ROOT = config.RAW_DATA / "ESC-50-master"
AUDIO_OVERRIDE = {
    "usd8k": Path(os.environ["SAFESOUND_SRC_USD8K"]) if os.environ.get("SAFESOUND_SRC_USD8K") else None,
    "esc50": Path(os.environ["SAFESOUND_SRC_ESC50"]) if os.environ.get("SAFESOUND_SRC_ESC50") else None,
}

TARGET_IDS = sorted(fc.BLACKLIST_IDS | fc.NEUTRAL_IDS)     # 可被抑制的事件声
INTERF_IDS = sorted(fc.WHITELIST_IDS)                      # 必须保留的事件声
KEEP_SET_NAMES = ["critical", "emergency", "all_safety", "passthrough"]


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------
def index_usd8k() -> dict[int, list[Path]]:
    """UrbanSound8K → {class_id: [wav...]}"""
    csv_path = USD8K_ROOT / "metadata" / "UrbanSound8K.csv"
    pool: dict[int, list[Path]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["classID"])
            base = AUDIO_OVERRIDE["usd8k"] or (USD8K_ROOT / "audio")
            wav = base / f"fold{row['fold']}" / row["slice_file_name"]
            if wav.exists():
                pool.setdefault(cid, []).append(wav)
    return pool


def index_esc50() -> tuple[dict[int, list[Path]], list[Path]]:
    """ESC-50 → (映射到目标类的补充池, 噪声池)"""
    csv_path = ESC50_ROOT / "meta" / "esc50.csv"
    mapped: dict[int, list[Path]] = {}
    noise: list[Path] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base = AUDIO_OVERRIDE["esc50"] or (ESC50_ROOT / "audio")
            wav = base / row["filename"]
            if not wav.exists():
                continue
            tgt = config.ESC50_TO_TARGET.get(row["category"])
            if tgt is None:
                noise.append(wav)
            else:
                mapped.setdefault(config.CLASS_TO_IDX[tgt], []).append(wav)
    return mapped, noise


# ---------------------------------------------------------------------------
# 音频读取（带 LRU 缓存，避免重复解码）
# ---------------------------------------------------------------------------
_CACHE: dict[str, np.ndarray] = {}
_CACHE_ORDER: list[str] = []
_CACHE_MAX = 2000      # 1 s 段缓存，命中率高且内存 <500 MB


def load_16k(path: Path) -> np.ndarray:
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    y, sr = sf.read(key, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = resample(y, int(round(len(y) * SR / sr))).astype(np.float32)
    _CACHE[key] = y
    _CACHE_ORDER.append(key)
    if len(_CACHE_ORDER) > _CACHE_MAX:
        _CACHE.pop(_CACHE_ORDER.pop(0), None)
    return y


def crop_1s(y: np.ndarray, rng: random.Random) -> np.ndarray:
    """随机裁 1 s；不足则循环平铺后裁。"""
    if len(y) < SEG:
        reps = int(np.ceil(SEG / max(len(y), 1)))
        y = np.tile(y, reps)
    start = rng.randint(0, len(y) - SEG)
    seg = y[start:start + SEG].copy()
    peak = float(np.max(np.abs(seg))) + 1e-9
    if peak > 0.9:                       # 峰值安全归一
        seg *= 0.9 / peak
    return seg


def rms_scale(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg ** 2) + 1e-12))


# ---------------------------------------------------------------------------
# 单条样本合成
# ---------------------------------------------------------------------------
def make_sample(idx: int, split: str, pools, rng: random.Random) -> dict:
    usd, esc_mapped, noise_pool = pools

    def pick(cid: int) -> Path:
        src = []
        if cid in usd:
            src.extend(usd[cid])
        if cid in esc_mapped:
            src.extend(esc_mapped[cid])
        if not src:
            raise RuntimeError(f"类别 {config.TARGET_CLASSES[cid]} 无可用音频")
        return rng.choice(src)

    t_id = rng.choice(TARGET_IDS)
    i_id = rng.choice([c for c in INTERF_IDS if c != t_id])
    # 目标/干扰尽量不同源文件
    for _ in range(20):
        t_path, i_path = pick(t_id), pick(i_id)
        if t_path != i_path:
            break

    s_t = crop_1s(load_16k(t_path), rng)
    s_i = crop_1s(load_16k(i_path), rng)
    s_n = crop_1s(load_16k(rng.choice(noise_pool)), rng)

    snr_t = rng.uniform(-5.0, 10.0)        # target 相对 interferer
    snr_n = rng.uniform(10.0, 20.0)        # 底噪相对混合物

    # 幅度标定
    a_i = 1.0
    a_t = float(np.sqrt(10 ** (snr_t / 10) * (rms_scale(s_i) ** 2) / (rms_scale(s_t) ** 2)))
    mix2 = a_t ** 2 * rms_scale(s_t) ** 2 + a_i ** 2 * rms_scale(s_i) ** 2
    a_n = float(np.sqrt(10 ** (-snr_n / 10) * mix2 / (rms_scale(s_n) ** 2)))

    t_s = (a_t * s_t).astype(np.float32)
    i_s = (a_i * s_i).astype(np.float32)
    n_s = (a_n * s_n).astype(np.float32)
    mix = t_s + i_s + n_s

    peak = float(np.max(np.abs(mix))) + 1e-9
    scale = (0.95 / peak) if peak > 0.95 else 1.0
    mix = (mix * scale).astype(np.float32)
    t_s = (t_s * scale).astype(np.float32)
    i_s = (i_s * scale).astype(np.float32)
    n_s = (n_s * scale).astype(np.float32)

    keep_name = rng.choice(KEEP_SET_NAMES)
    keep_ids = list(fc.KEEP_SETS[keep_name])
    # keep 集合必须包含本样本的 interferer（语义自洽）
    if i_id not in keep_ids:
        keep_ids.append(i_id)

    mix_id = f"mix_{idx:06d}"
    paths = {
        "mix_path": fc.MIX_AUDIO / f"{mix_id}.wav",
        "target_path": fc.MIX_STEMS / f"{mix_id}_target.wav",
        "interferer_path": fc.MIX_STEMS / f"{mix_id}_interf.wav",
        "noise_path": fc.MIX_STEMS / f"{mix_id}_noise.wav",
    }
    sf.write(paths["mix_path"], mix, SR, subtype="PCM_16")
    sf.write(paths["target_path"], t_s, SR, subtype="PCM_16")
    sf.write(paths["interferer_path"], i_s, SR, subtype="PCM_16")
    sf.write(paths["noise_path"], n_s, SR, subtype="PCM_16")

    return {
        "mix_id": mix_id,
        **{k: str(v) for k, v in paths.items()},
        "target_class": config.TARGET_CLASSES[t_id],
        "interferer_class": config.TARGET_CLASSES[i_id],
        "target_id": t_id,
        "interferer_id": i_id,
        "snr_target_db": round(snr_t, 2),
        "snr_noise_db": round(snr_n, 2),
        "keep_set": ",".join(map(str, keep_ids)),
        "keep_set_name": keep_name,
        "split": split,
        "seed": rng.randint(0, 10 ** 9),
        "mix_peak_before_scale": round(peak, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1, help="预留（当前单进程）")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    if args.out_dir:
        fc.MIX_DIR = Path(args.out_dir)
        fc.MIX_AUDIO = fc.MIX_DIR / "audio"
        fc.MIX_STEMS = fc.MIX_DIR / "stems"
        fc.MIX_MANIFEST = fc.MIX_DIR / "manifest.csv"
    fc.ensure_mix_dirs()

    t0 = time.time()
    usd = index_usd8k()
    esc_mapped, noise_pool = index_esc50()
    print(f"[index] UrbanSound8K {sum(len(v) for v in usd.values())} 文件 / "
          f"{len(usd)} 类；ESC-50 补充 {sum(len(v) for v in esc_mapped.values())} 文件，"
          f"噪声池 {len(noise_pool)} 文件（{time.time()-t0:.1f}s）")

    pools = (usd, esc_mapped, noise_pool)
    rows: list[dict] = []
    idx = 0
    plan = [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]
    for split, n in plan:
        t_split = time.time()
        for k in range(n):
            idx += 1
            rng = random.Random(args.seed * 100000 + idx)
            rows.append(make_sample(idx, split, pools, rng))
        print(f"[{split}] {n} 条完成（{time.time()-t_split:.1f}s）")

    fc.MIX_DIR.mkdir(parents=True, exist_ok=True)
    with open(fc.MIX_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 统计
    from collections import Counter
    stats = {
        "n_total": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "target_class": dict(Counter(r["target_class"] for r in rows)),
        "interferer_class": dict(Counter(r["interferer_class"] for r in rows)),
        "keep_set": dict(Counter(r["keep_set_name"] for r in rows)),
        "snr_target_db": {
            "min": min(r["snr_target_db"] for r in rows),
            "max": max(r["snr_target_db"] for r in rows),
            "mean": round(float(np.mean([r["snr_target_db"] for r in rows])), 2),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "manifest": str(fc.MIX_MANIFEST),
        "sr": SR,
        "seg_samples": SEG,
    }
    (fc.MIX_DIR / "synth_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] {len(rows)} 条 → {fc.MIX_DIR}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
