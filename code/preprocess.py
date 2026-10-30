"""
SafeSoundAI — Trigger-AED v2 预处理（4 触发类 + 背景）

目标（与笔记 19 一致）：
    4 个 PTSD 触发类  : siren / car_horn / gun_shot / fireworks
    1 个背景类        : background（不含任何触发类的城市声、机械声、人声、音乐、动物、自然、设备提示音）

数据来源：
    urban  (UrbanSound8K, 10 类)   → 4 类正样本 + 7 类进背景
    esc50  (ESC-50, 50 类)         → siren/car_horn/fireworks 正样本 + 其余进背景
    fsd50k (FSD50K, 200 类)        → Siren / Horn / Gunshot / Fireworks 正样本 + 其余进背景

处理流程（与 1.1 原方案逐参数一致）：
    多源解码 → 单声道(多声道取均值) → 重采样 16 kHz → RMS 归一 → 1 s 窗 / 50% overlap
    → float32 .npy 落盘 → 近似去重 → 录音级分组切分（8/1/1）

用法：
    export SAFESOUND_DATA_DIR=/path/to/preprocessed_data
    python preprocess.py                     # 全流程
    python preprocess.py --skip-fsd50k       # 只用已有 urban/esc50 段
    python preprocess.py --with-explosion    # 把 FSD50K Explosion/Boom 并入 gun_shot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample

import config

# ---------------------------------------------------------------------------
# v2 类别定义
# ---------------------------------------------------------------------------
TARGET_CLASSES_V2 = ["siren", "car_horn", "gun_shot", "fireworks"]
BACKGROUND = "background"
CLASSES_V2 = TARGET_CLASSES_V2 + [BACKGROUND]
CLASS_TO_IDX_V2 = {c: i for i, c in enumerate(CLASSES_V2)}
NUM_CLASSES_V2 = len(CLASSES_V2)

# UrbanSound8K 原始类名 → v2 类
URBAN_TO_V2 = {
    "siren":       "siren",
    "car_horn":    "car_horn",
    "gun_shot":    "gun_shot",
    # 以下 7 类全部进背景
    "air_conditioner":  BACKGROUND,
    "children_playing": BACKGROUND,
    "dog_bark":         BACKGROUND,
    "drilling":         BACKGROUND,
    "engine_idling":    BACKGROUND,
    "jackhammer":       BACKGROUND,
    "street_music":     BACKGROUND,
}

# ESC-50 原始类名 → v2 类（只映射真正同族的，其余一律背景）
ESC50_TO_V2 = {
    "siren":     "siren",
    "car_horn":  "car_horn",
    "fireworks": "fireworks",   # v1 曾错误并入 gun_shot，v2 恢复为独立类
    "clock_alarm": BACKGROUND,  # 闹钟 ≠ 火警/安防报警 → 背景 device_beep
}

# FSD50K 官方类名 → v2 类
FSD50K_TO_V2 = {
    "Siren": "siren",
    "Vehicle_horn_and_car_horn_and_honking": "car_horn",
    "Gunshot_and_gunfire": "gun_shot",
    "Fireworks": "fireworks",
}
# 可选：单次爆炸并入 gun_shot（默认关闭，保持类内纯净）
FSD50K_EXPLOSION = ["Explosion", "Boom"]

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
DATA_V2 = config.PREPROCESSED / "trigger_v2"
SEG_DIR = DATA_V2 / "segments"
FSD50K_RAW = config.PREPROCESSED / "_newdata" / "fsd50k_raw" / "FSD50K"
FSD50K_META = config.PREPROCESSED / "_newdata" / "fsd50k"

# 每文件最多保留多少段（防长录音/长 clip 用切片数量刷榜）
MAX_WIN_PER_FILE = 10
# 背景段预算（控类不平衡，且保留录音级多样性）
BG_BUDGET = {"urban": 12000, "esc50": 2500, "fsd50k": 8000}

SEED = config.RANDOM_SEED


# ---------------------------------------------------------------------------
# 基础音频工具（与 audio_utils.py 完全一致的参数）
# ---------------------------------------------------------------------------
def load_mono_16k(path: Path, target_sr: int = config.SAMPLE_RATE):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        y = resample(y, int(round(len(y) * target_sr / sr))).astype(np.float32)
    return y, sr


def rms_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return (y / max(rms, eps)).astype(np.float32)


def segment(y: np.ndarray):
    seg_len = config.SEGMENT_SAMPLES
    hop = int(seg_len * (1.0 - config.OVERLAP))
    out, start = [], 0
    while start + seg_len <= len(y):
        out.append(y[start:start + seg_len])
        start += hop
    return out


def pick_windows(n: int, cap: int = MAX_WIN_PER_FILE):
    """超过 cap 时按等间距抽样，保证覆盖整段录音而不是只看开头。"""
    if n <= cap:
        return list(range(n))
    return sorted(np.unique(np.linspace(0, n - 1, cap).round().astype(int)).tolist())


# ---------------------------------------------------------------------------
# 1) urban / esc50：直接复用 v1 已产出的 1 s 段（同参数，不重复解码）
# ---------------------------------------------------------------------------
def collect_existing(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in raw.itertuples(index=False):
        ds = str(r.dataset).strip().lower()
        fold = int(r.fold)
        stem = str(r.original_file).rsplit(".", 1)[0]
        npy = config.PREPROCESSED / ds / f"fold{fold}" / f"{stem}_seg{int(r.segment_idx):03d}.npy"
        if not npy.exists():
            continue
        if ds == "urban":
            label = URBAN_TO_V2.get(str(r.label))
        elif ds == "esc50":
            label = ESC50_TO_V2.get(str(r.label), BACKGROUND)
        else:
            continue
        if label is None:
            continue
        rows.append({
            "dataset": ds,
            "original_file": str(r.original_file),
            "rec_group": stem if ds == "esc50" else stem.split("-")[0],  # urban: fsID
            "label_raw": str(r.label),
            "label": label,
            "class_id": CLASS_TO_IDX_V2[label],
            "segment_idx": int(r.segment_idx),
            "npy_path": str(npy),
            "src_fold": fold,
        })
    df = pd.DataFrame(rows)

    # 补全来源信息：原始采样率 / 时长 / license（只读文件头，不重新解码）
    cache: dict[tuple, tuple] = {}
    lic = {"urban": "CC-BY-NC-3.0", "esc50": "CC-BY-NC-3.0"}   # US8K / ESC-50 均为 CC BY-NC 3.0
    for i, r in enumerate(df.itertuples(index=False), 1):
        key = (r.dataset, r.original_file, r.src_fold)
        if key not in cache:
            if r.dataset == "urban":
                p = config.RAW_DATA / "UrbanSound8K" / "audio" / f"fold{r.src_fold}" / r.original_file
            else:
                p = config.RAW_DATA / "ESC-50-master" / "audio" / r.original_file
            try:
                inf = sf.info(str(p))
                cache[key] = (inf.frames / inf.samplerate, inf.samplerate)
            except Exception:
                cache[key] = (np.nan, np.nan)
        dur, sr0 = cache[key]
        df.at[i - 1, "duration_s"] = dur
        df.at[i - 1, "orig_sr"] = sr0
        df.at[i - 1, "license"] = lic[r.dataset]
        if i % 2000 == 0:
            print(f"  来源信息补全 {i}/{len(df)}", flush=True)

    # 每个来源录音的总段数（含被下采样丢掉的），用于审计与"段数≠独立样本"的说明
    tot = df.groupby(["dataset", "original_file"]).size().rename("n_windows_total")
    df = df.merge(tot, left_on=["dataset", "original_file"], right_index=True, how="left")
    return df


# ---------------------------------------------------------------------------
# 2) FSD50K：解码 → 切片 → 落盘
# ---------------------------------------------------------------------------
def fsd50k_selection(with_explosion: bool) -> pd.DataFrame:
    gt = pd.concat([
        pd.read_csv(FSD50K_META / "FSD50K.ground_truth" / "dev.csv"),
        pd.read_csv(FSD50K_META / "FSD50K.ground_truth" / "eval.csv"),
    ], ignore_index=True)
    gt["labels"] = gt["labels"].fillna("")

    info = {}
    for name in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        p = FSD50K_META / "FSD50K.metadata" / name
        info.update(json.load(open(p)))

    def lic(u: str) -> str:
        u = str(u)
        if "zero" in u:
            return "CC0"
        if "by-nc" in u or "by_nc" in u:
            return "CC-BY-NC"
        if "sampling" in u:
            return "Sampling+"
        if "by" in u:
            return "CC-BY"
        return "other"

    gt["license"] = [lic(info.get(str(f), {}).get("license", "?")) for f in gt.fname]
    gt["duration"] = [float(info.get(str(f), {}).get("duration", np.nan)) for f in gt.fname]

    # 正类
    pos_rows = []
    for fsd_cls, v2_cls in FSD50K_TO_V2.items():
        sub = gt[gt.labels.str.contains(fsd_cls, regex=False)]
        sub = sub.assign(label=v2_cls)
        pos_rows.append(sub)

    if with_explosion:
        sub = gt[gt.labels.str.contains("|".join(FSD50K_EXPLOSION), regex=True)]
        # 已带 Fireworks 的不重复计入
        sub = sub[~sub.labels.str.contains("Fireworks", regex=False)]
        pos_rows.append(sub.assign(label="gun_shot", label_raw_extra="explosion"))

    pos = pd.concat(pos_rows, ignore_index=True)
    # 一个 clip 命中多个目标类时只保留一个（优先级 siren > gun_shot > fireworks > car_horn）
    prio = {"siren": 0, "gun_shot": 1, "fireworks": 2, "car_horn": 3}
    pos["_p"] = pos.label.map(prio)
    pos = pos.sort_values("_p").drop_duplicates(subset="fname", keep="first").drop(columns="_p")

    # 背景：不含任何目标标签的 clip，按类别分层抽样
    banned = list(FSD50K_TO_V2.keys()) + (FSD50K_EXPLOSION if with_explosion else [])
    mask = pd.Series(True, index=gt.index)
    for b in banned:
        mask &= ~gt.labels.str.contains(b, regex=False)
    bg = gt[mask].copy()
    bg["label"] = BACKGROUND
    rng = np.random.default_rng(SEED)
    n_bg_clips = max(1, BG_BUDGET["fsd50k"] // 4)      # 平均每 clip ~4 段
    bg = bg.sample(n=min(len(bg), n_bg_clips), random_state=SEED).copy()

    sel = pd.concat([pos, bg], ignore_index=True)
    sel["fname"] = sel.fname.astype(str)
    return sel


def build_fsd50k(sel: pd.DataFrame, only_split: str | None = None) -> list[dict]:
    out_dir = SEG_DIR / "fsd50k"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    t0 = time.time()
    for i, r in enumerate(sel.itertuples(index=False), 1):
        if only_split and not str(r.fname).startswith(only_split):
            pass
        wav = FSD50K_RAW / ("FSD50K.dev_audio" if r.split == "dev" else "FSD50K.eval_audio") / f"{r.fname}.wav"
        if not wav.exists():
            skipped += 1
            continue
        try:
            y, sr0 = load_mono_16k(wav)
        except Exception as e:                                   # 损坏文件跳过
            skipped += 1
            continue
        dur = len(y) / config.SAMPLE_RATE
        y = rms_normalize(y)
        segs = segment(y)
        if not segs:
            skipped += 1
            continue
        keep = pick_windows(len(segs))
        stem = str(r.fname)
        for k in keep:
            p = out_dir / f"{stem}_seg{k:03d}.npy"
            if not p.exists():
                np.save(p, segs[k])
            rows.append({
                "dataset": "fsd50k",
                "original_file": f"{stem}.wav",
                "rec_group": stem,
                "label_raw": str(r.labels),
                "label": str(r.label),
                "class_id": CLASS_TO_IDX_V2[str(r.label)],
                "segment_idx": int(k),
                "npy_path": str(p),
                "src_fold": 0 if r.split == "dev" else 1,
                "duration_s": dur,
                "orig_sr": int(sr0),
                "license": str(r.license),
                "n_windows_total": len(keep),
            })
        if i % 200 == 0:
            print(f"  fsd50k {i}/{len(sel)}  clips, {len(rows)} segs, {time.time()-t0:.0f}s", flush=True)
    print(f"  fsd50k 完成: {len(rows)} 段, 跳过 {skipped} 个文件, {time.time()-t0:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# 3) 背景下采样（按文件抽，保留录音级多样性）
# ---------------------------------------------------------------------------
def subsample_background(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for ds, budget in BG_BUDGET.items():
        sub = df[(df.dataset == ds) & (df.label == BACKGROUND)]
        if sub.empty:
            continue
        files = sub.original_file.unique()
        rng = np.random.default_rng(SEED)
        rng.shuffle(files)
        cnt, chosen = 0, []
        per_file = sub.groupby("original_file").size().to_dict()
        for f in files:
            if cnt >= budget:
                break
            chosen.append(f)
            cnt += per_file[f]
        keep.append(sub[sub.original_file.isin(chosen)])
        print(f"  背景下采样 {ds}: {len(files)} → {len(chosen)} 文件, {len(sub)} → {cnt} 段")
    tgt = df[df.label != BACKGROUND]
    return pd.concat([tgt] + keep, ignore_index=True)


# ---------------------------------------------------------------------------
# 4) 近似重复检测（均值 log-Mel 指纹）
# ---------------------------------------------------------------------------
def fingerprint(npy_paths: list[str], n_mels: int = 32) -> str:
    acc = np.zeros(n_mels, dtype=np.float64)
    for p in npy_paths[:4]:
        y = np.load(p).astype(np.float64)
        spec = np.abs(np.fft.rfft(y))
        edges = np.linspace(0, len(spec), n_mels + 1).astype(int)
        band = np.array([spec[edges[i]:edges[i + 1]].mean() for i in range(n_mels)])
        acc += np.log(band + 1e-9)
    v = np.round(acc / min(len(npy_paths), 4), 1)
    return hashlib.md5(v.tobytes()).hexdigest()


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("  计算录音级指纹…")
    order = {"urban": 0, "esc50": 1, "fsd50k": 2}
    df = df.assign(_prio=df.dataset.map(order).fillna(9))
    groups = df.groupby(["dataset", "original_file"], sort=False)
    fps = {}
    for (ds, f), g in groups:
        fps[(ds, f)] = fingerprint(g.npy_path.tolist())

    # 跨库重复：优先保留 urban > esc50 > fsd50k
    seen: dict[str, tuple] = {}
    dup_flag = {}
    for key, fp in sorted(fps.items(), key=lambda kv: order.get(kv[0][0], 9)):
        if fp in seen:
            dup_flag[key] = seen[fp]
        else:
            seen[fp] = key
    print(f"  检出近似重复录音 {len(dup_flag)} 个")
    df["dup_of"] = [dup_flag.get((d, f)) for d, f in zip(df.dataset, df.original_file)]
    df["is_duplicate"] = df.dup_of.notna()
    return df.drop(columns="_prio")


# ---------------------------------------------------------------------------
# 5) 录音级分组切分（每库内部切，再拼折）
# ---------------------------------------------------------------------------
def group_split(df: pd.DataFrame, n_splits: int = config.N_FOLDS) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df["cv_fold"] = -1
    for ds, sub in df.groupby("dataset"):
        grp = sub.groupby("rec_group").agg(y=("class_id", lambda s: s.mode().iloc[0])).reset_index()
        # 样本极少时退化为普通 group 切分
        min_cnt = grp.y.value_counts().min()
        n_splits_ds = min(n_splits, len(grp)) if min_cnt >= n_splits else min(n_splits, max(2, len(grp)))
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits_ds, shuffle=True, random_state=SEED)
            assign = {}
            for fold, (_, va) in enumerate(sgkf.split(grp, grp.y, grp.rec_group)):
                for g in grp.rec_group.iloc[va]:
                    assign[g] = fold
        except ValueError:
            files = grp.rec_group.sample(frac=1.0, random_state=SEED).tolist()
            assign = {g: i % n_splits_ds for i, g in enumerate(files)}
        df.loc[df.dataset == ds, "cv_fold"] = df.loc[df.dataset == ds, "rec_group"].map(assign)
    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fsd50k", action="store_true")
    ap.add_argument("--with-explosion", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    DATA_V2.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()

    print("=" * 70)
    print("Trigger-AED v2 预处理")
    print("  类别:", CLASSES_V2)
    print("=" * 70)

    # --- 1) 已有 urban / esc50 ---
    print("\n[1/5] 复用已产出的 1 s 段（urban + esc50）")
    raw = pd.read_csv(config.METADATA_RAW)
    old = collect_existing(raw)
    print("  ", old.groupby(["dataset", "label"]).size().to_string().replace("\n", "\n   "))

    # --- 2) FSD50K ---
    frames = [old]
    if not args.skip_fsd50k:
        print("\n[2/5] FSD50K 解码 + 切片")
        sel = fsd50k_selection(args.with_explosion)
        print("  选中 clips:", sel.groupby("label").size().to_dict())
        rows = build_fsd50k(sel)
        frames.append(pd.DataFrame(rows))
    else:
        print("\n[2/5] 跳过 FSD50K")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n合并后原始段数: {len(df)}")

    # --- 3) 背景下采样 ---
    print("\n[3/5] 背景下采样")
    df = subsample_background(df)
    print(f"  下采样后: {len(df)} 段")

    # --- 4) 去重 ---
    print("\n[4/5] 近似重复检测")
    df = mark_duplicates(df)
    n_before = len(df)
    df = df[~df.is_duplicate].copy()
    print(f"  去重: {n_before} → {len(df)} 段")

    # --- 5) 切分 ---
    print("\n[5/5] 录音级分组切分")
    df = group_split(df)
    df["split"] = np.select(
        [df.cv_fold.isin(config.TRAIN_FOLDS), df.cv_fold == config.VAL_FOLD, df.cv_fold == config.TEST_FOLD],
        ["train", "val", "test"], default="unused",
    )

    # --- 落盘 ---
    for c in ("duration_s", "orig_sr", "license", "src_fold", "n_windows_total"):
        if c not in df.columns:
            df[c] = np.nan
    cols = ["dataset", "original_file", "rec_group", "label_raw", "label", "class_id",
            "cv_fold", "split", "segment_idx", "npy_path", "duration_s", "orig_sr", "license",
            "src_fold", "n_windows_total"]
    out = df[cols].sort_values(["split", "dataset", "label", "original_file", "segment_idx"])
    path = DATA_V2 / "metadata_v2.csv"
    out.to_csv(path, index=False)
    print(f"\n✅ metadata_v2: {path}  ({len(out)} 段, {out.original_file.nunique()} 个来源录音)")

    summary = {
        "n_segments": int(len(out)),
        "n_files": int(out.original_file.nunique()),
        "n_rec_groups": int(out.rec_group.nunique()),
        "classes": CLASSES_V2,
        "per_class": out.label.value_counts().to_dict(),
        "per_dataset": out.dataset.value_counts().to_dict(),
        "per_split": out.split.value_counts().to_dict(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (DATA_V2 / "preprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
