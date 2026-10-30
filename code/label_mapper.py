"""
SafeSoundAI — 1.3 标签映射 + 文件级 Stratified 10-fold（防泄漏）

核心职责：
  1. map_label()：把 (dataset, label) 映射到 10 目标类（或 None=噪声池）
  2. assign_role()：标记样本角色 train / noise
  3. file_level_stratified_split()：按「源文件」级别做 Stratified K-Fold，
     确保同一录音的所有片段落在同一 fold（防止数据泄漏导致准确率虚高 5-15%）

⚠️ 防泄漏铁律：
   绝不按「片段(segment)」划分，必须按「源文件(original_file)」划分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import (
    TARGET_CLASSES,
    CLASS_TO_IDX,
    ESC50_TO_TARGET,
    RANDOM_SEED,
    N_FOLDS,
)


def map_label(dataset: str, label: str) -> str | None:
    """
    把 (dataset, label) 映射到 10 目标类之一；返回 None 表示进入噪声池。

    - urban：label 本身即目标类（10 类之一），直接返回；
    - esc50：查 ESC50_TO_TARGET 映射表，命中则返回目标类，否则 None。
    """
    dataset = str(dataset).strip().lower()
    label = str(label).strip()

    if dataset == "urban":
        # urban 的 label 就是 10 目标类；若出现异常值，返回 None 兜底
        return label if label in CLASS_TO_IDX else None
    elif dataset == "esc50":
        return ESC50_TO_TARGET.get(label, None)
    else:
        raise ValueError(f"未知 dataset: {dataset!r}")


def assign_role(target_label: str | None) -> str:
    """target_label 非空 → 'train'；空 → 'noise'（噪声池，供 1.2 增强使用）。"""
    return "train" if target_label is not None else "noise"


def map_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    在 metadata 上追加三列：
      - target_label : 10 目标类之一，或 NaN（噪声）
      - role         : 'train' / 'noise'
      - class_id     : 0-9 的类别编号（噪声为 -1）
    原地返回新 DataFrame（不修改入参）。
    """
    df = metadata.copy()
    mapped = df.apply(
        lambda r: map_label(r["dataset"], r["label"]), axis=1
    )
    df["target_label"] = mapped
    df["role"] = df["target_label"].apply(assign_role)
    df["class_id"] = df["target_label"].apply(
        lambda t: CLASS_TO_IDX[t] if t is not None else -1
    )
    return df


def file_level_stratified_split(
    metadata: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    文件级 Stratified K-Fold 划分（防泄漏核心）。

    只对 role='train' 的样本划分 fold；噪声池（role='noise'）赋 fold=-1。
    返回追加了 'cv_fold'（0..n_folds-1 或 -1）列的新 DataFrame。
    """
    df = metadata.copy()
    df["cv_fold"] = -1

    train = df[df["role"] == "train"]
    if train.empty:
        return df

    # 文件级：每个源文件取一个代表标签
    file_level = (
        train.groupby("original_file", as_index=False)["target_label"].first()
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    file_to_fold = {}
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(file_level, file_level["target_label"])
    ):
        for f in file_level.iloc[val_idx]["original_file"]:
            file_to_fold[f] = fold_idx

    df.loc[train.index, "cv_fold"] = train["original_file"].map(file_to_fold)
    return df


def recording_group(dataset: str, original_file: str) -> str:
    """
    返回「同一条录音」的分组键（比 original_file 更粗一级）。

    为什么需要：UrbanSound8K 对同一条录音做了 0.5 s 平移切片，同一 fsID 下会有
    `72567-1-0-0.wav` / `72567-1-0-1.wav` 这类文件，二者 1 s 窗有 50% 重叠，
    甚至逐比特相同的段（实测 87 组）。只按 original_file 划分 → 训练/测试共享同一段音频，
    准确率虚高且恰好高估我们最关心的 car_horn / siren / drilling。

    - urban：取文件名第一段 fsID（8,732 个文件 → 1,297 条录音；其中 5 条录音被官方
      标了 2 个类别，属 UrbanSound8K 已知重复标注，分组时整条录音一起走）
    - esc50：每个文件本身就是独立录音，直接用文件名（去扩展名）
    """
    dataset = str(dataset).strip().lower()
    name = str(original_file)
    if dataset == "urban":
        return f"urban:{name.split('-')[0]}"
    return f"{dataset}:{name.rsplit('.', 1)[0]}"


def group_level_stratified_split(
    metadata: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    录音级（group-level）Stratified 10-fold —— 比 file_level_stratified_split 更严格的防泄漏。

    分组单位由 recording_group() 决定（urban 按 fsID，esc50 按文件）。
    组内类别可能不唯一（UrbanSound8K 有 5 条录音被标两类别），分层标签取组内众数。

    分配算法：组规模差异极大（urban 单条录音最多 100 个切片），StratifiedKFold /
    StratifiedGroupKFold 都会把某些折灌成 7,000+ 段、另一些只剩 4,300 段
    （实测 min 类-折低至 35 段，紧急类样本被抽干）。因此改用**贪心装箱**：
    组按段数降序，每组放进「该类当前段数最少、其次总段数最少」的折。
    seed 仅用于打散同规模组的顺序，保证可复现。
    """
    from collections import Counter

    df = metadata.copy()
    df["cv_fold"] = -1
    df["rec_group"] = [
        recording_group(d, f) for d, f in zip(df["dataset"], df["original_file"])
    ]

    train = df[df["role"] == "train"]
    if train.empty:
        return df

    grp = (
        train.groupby("rec_group")
        .agg(
            label=("target_label", lambda s: Counter(s).most_common(1)[0][0]),
            n=("class_id", "size"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    grp = grp.assign(_jit=rng.random(len(grp))).sort_values(
        ["n", "_jit"], ascending=[False, True]
    )

    cls_cnt = {c: np.zeros(n_folds) for c in grp["label"].unique()}
    tot = np.zeros(n_folds)
    group_to_fold: dict[str, int] = {}
    for label, sub in grp.groupby("label"):
        for rec, n in zip(sub["rec_group"], sub["n"]):
            f = int(np.lexsort((tot, cls_cnt[label]))[0])
            group_to_fold[rec] = f
            cls_cnt[label][f] += n
            tot[f] += n

    df.loc[train.index, "cv_fold"] = train["rec_group"].map(group_to_fold)
    return df


def split_summary(metadata: pd.DataFrame) -> str:
    """生成划分概览（每个 fold 的样本数 + 类别分布），用于人工核查。"""
    df = metadata[metadata["role"] == "train"]
    lines = []
    for fold in sorted(df["cv_fold"].unique()):
        sub = df[df["cv_fold"] == fold]
        n_files = sub["original_file"].nunique()
        lines.append(f"fold {fold}: {len(sub):>6} 段 / {n_files:>4} 文件")
    lines.append(f"噪声池: {len(metadata[metadata['role']=='noise']):>6} 段")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自检：加载原始 metadata，跑一遍映射 + 划分，打印概览
    import config

    raw = pd.read_csv(config.METADATA_RAW)
    print(f"原始 metadata: {len(raw)} 段")

    mapped = map_metadata(raw)
    print("\n=== 目标类分布 ===")
    print(
        mapped[mapped["role"] == "train"]
        .groupby("target_label")["original_file"]
        .nunique()
        .to_string()
    )

    split = file_level_stratified_split(mapped)
    print("\n=== Fold 划分概览 ===")
    print(split_summary(split))
