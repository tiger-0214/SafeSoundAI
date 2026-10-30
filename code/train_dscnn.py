"""
SafeSoundAI — 2.2/2.3 轻量 CNN 训练（train_dscnn.py，MVP 主模型）

DS-CNN 从零训练，输入 log-Mel(128×100)，输出 10 类。
类不平衡用 class-weight 加权交叉熵处理。

运行：
  python train_dscnn.py --epochs 20 --batch_size 64
  python train_dscnn.py --epochs 1 --batch_size 32 --limit 2000   # 快速冒烟测试
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import config
from dataset import AudioSegmentDataset, load_splits, build_noise_pool
from models import DS_CNN, count_parameters, model_size_mb


def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(device)
            logits = model(mel)
            preds.append(logits.argmax(1).cpu().numpy())
            truths.append(label.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return float(accuracy_score(truths, preds)), preds, truths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="每 fold 采样上限（冒烟测试）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    train, val, test = load_splits()
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=config.RANDOM_SEED)
        val = val.sample(min(args.limit // 4, len(val)), random_state=config.RANDOM_SEED)
        test = test.sample(min(args.limit // 4, len(test)), random_state=config.RANDOM_SEED)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    noise_pool = build_noise_pool()
    train_ds = AudioSegmentDataset(train, noise_pool=noise_pool, augment=True)
    val_ds = AudioSegmentDataset(val, augment=False)
    test_ds = AudioSegmentDataset(test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    model = DS_CNN().to(device)
    print(f"参数量: {count_parameters(model):,} | 体积: {model_size_mb(model):.2f} MB")

    # class-weight 处理类不平衡
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train["class_id"]), y=train["class_id"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(label)
            n += len(label)
        scheduler.step()

        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={total_loss / n:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            config.ensure_dirs()
            torch.save(model.state_dict(), config.MODEL_DIR / "ds_cnn_best.pth")

    print(f"\n最佳 val_acc = {best_val:.4f}")

    # 加载最佳权重测 test
    model.load_state_dict(torch.load(config.MODEL_DIR / "ds_cnn_best.pth"))
    test_acc, y_pred, y_true = evaluate(model, test_loader, device)
    print(f"[test] Acc = {test_acc:.4f}")
    print("\n=== Test 分类报告 ===")
    print(classification_report(y_true, y_pred, target_names=config.TARGET_CLASSES,
                                zero_division=0))


if __name__ == "__main__":
    main()
