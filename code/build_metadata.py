"""
SafeSoundAI — 重建干净 metadata

问题背景：
  1.1 已产出 72,201 段 .npy + metadata.csv，但 csv 里的 npy_path 是
  「旧绝对路径」（项目曾位于 /mnt/d/Program Files/...），现已失效。

本脚本职责：
  1. 用当前工作区路径重建 npy_path（基于 dataset + fold + 文件名规律）；
  2. 调用 label_mapper 完成 1.3 标签映射 + 文件级 10-fold 划分；
  3. 产出 metadata_clean.csv —— 后续所有训练/评估的唯一数据源。

运行：
  python build_metadata.py
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

import config
from label_mapper import (
    map_metadata,
    file_level_stratified_split,
    group_level_stratified_split,
    split_summary,
)


def rebuild_npy_path(row: pd.Series) -> str:
    """按数据集目录规律重建 .npy 绝对路径。"""
    dataset = str(row["dataset"]).strip().lower()
    fold = int(row["fold"])
    segment_idx = int(row["segment_idx"])
    # original_file 形如 "1-100032-A-0.wav" → 去掉 .wav 得到 stem
    stem = str(row["original_file"]).rsplit(".", 1)[0]
    npy_name = f"{stem}_seg{segment_idx:03d}.npy"
    return str(config.PREPROCESSED / dataset / f"fold{fold}" / npy_name)


def main() -> None:
    print(f"读取原始 metadata: {config.METADATA_RAW}")
    raw = pd.read_csv(config.METADATA_RAW)
    print(f"  共 {len(raw)} 段")

    # 1. 重建路径
    raw["npy_path"] = raw.apply(rebuild_npy_path, axis=1)

    # 2. 标签映射 + 角色标记
    mapped = map_metadata(raw)

    # 3. 划分：默认「录音级」（urban 按 fsID 分组），彻底消除 0.5 s 平移切片的跨折泄漏
    if args.file_level:
        split = file_level_stratified_split(mapped)          # 旧行为（有 87 组内容泄漏）
        print("  划分方式: 文件级（不推荐，仅用于复现旧结果）")
    else:
        split = group_level_stratified_split(mapped)         # 推荐
        print("  划分方式: 录音级（urban 按 fsID 分组，防 0.5 s 平移泄漏）")

    # 4. 保存干净 metadata（覆盖前备份旧版）
    config.ensure_dirs()
    if config.METADATA_CLEAN.exists():
        bak = config.METADATA_CLEAN.with_name(config.METADATA_CLEAN.stem + "_prev.csv")
        shutil.copy2(config.METADATA_CLEAN, bak)
        print(f"  旧版已备份: {bak.name}")
    split.to_csv(config.METADATA_CLEAN, index=False)
    print(f"\n✅ 已写出: {config.METADATA_CLEAN}")
    print(f"   train 样本: {len(split[split['role']=='train'])} 段")
    print(f"   noise 样本: {len(split[split['role']=='noise'])} 段")

    # 5. 概览
    print("\n=== 目标类分布（按源文件数）===")
    train = split[split["role"] == "train"]
    print(train.groupby("target_label")["original_file"].nunique().to_string())
    print("\n=== Fold 划分概览 ===")
    print(split_summary(split))

    # 6. 完整性自检：抽查若干 npy 路径是否真实存在
    import os
    sample = split.sample(min(200, len(split)), random_state=42)
    missing = [p for p in sample["npy_path"] if not os.path.exists(p)]
    print(f"\n=== 路径自检（抽查 {len(sample)} 段）===")
    if missing:
        print(f"❌ {len(missing)} 段路径失效，示例:")
        for p in missing[:5]:
            print("   ", p)
    else:
        print(f"✅ 抽查 {len(sample)} 段全部存在")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-level", action="store_true",
                    help="用旧的文件级划分（仅复现历史结果；存在 87 组跨折内容泄漏）")
    args = ap.parse_args()
    main()
