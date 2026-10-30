"""
SafeSoundAI — 2.1 Baseline：MFCC + SVM（train_baseline.py）

最快的性能下界验证：MFCC 统计特征 → LinearSVC。
不依赖 GPU，用于确认数据管线可用、给出精度基线。

运行：
  python train_baseline.py                 # 全量训练
  python train_baseline.py --limit 2000    # 快速冒烟测试（每 fold 抽 2000 段）
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from functools import partial

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

import config
from dataset import load_splits


def extract_mfcc(path: str, n_mfcc: int = 13) -> np.ndarray:
    """单段 MFCC 统计特征：13 系数 × (mean, std) = 26 维。"""
    y = np.load(path).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=config.SAMPLE_RATE, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def extract_features(paths: list[str], n_jobs: int = 8) -> np.ndarray:
    """并行提取 MFCC 特征。"""
    with mp.Pool(n_jobs) as pool:
        feats = pool.map(extract_mfcc, paths, chunksize=64)
    return np.vstack(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="每 fold 最多采样多少段（快速冒烟测试）")
    args = ap.parse_args()

    train, val, test = load_splits()
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=config.RANDOM_SEED)
        val = val.sample(min(args.limit // 4, len(val)), random_state=config.RANDOM_SEED)
        test = test.sample(min(args.limit // 4, len(test)), random_state=config.RANDOM_SEED)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    print("提取 train 特征...")
    X_train = extract_features(train["npy_path"].tolist())
    y_train = train["class_id"].values
    print("提取 val 特征...")
    X_val = extract_features(val["npy_path"].tolist())
    y_val = val["class_id"].values
    print("提取 test 特征...")
    X_test = extract_features(test["npy_path"].tolist())
    y_test = test["class_id"].values

    clf = LinearSVC(max_iter=2000, random_state=config.RANDOM_SEED, dual="auto")
    print("训练 SVM...")
    clf.fit(X_train, y_train)

    for name, X, y in [("train", X_train, y_train),
                       ("val", X_val, y_val),
                       ("test", X_test, y_test)]:
        acc = accuracy_score(y, clf.predict(X))
        print(f"[{name}] Acc = {acc:.4f}")

    print("\n=== Test 分类报告 ===")
    y_pred = clf.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=config.TARGET_CLASSES, zero_division=0,
    ))
    print("\n=== Test 混淆矩阵 ===")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=config.TARGET_CLASSES,
                       columns=config.TARGET_CLASSES).to_string())


if __name__ == "__main__":
    main()
