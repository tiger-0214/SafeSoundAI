"""
SafeSoundAI — 划分协议对照（compare_protocols.py）
=================================================
量化「文件级划分」到底把指标抬高了多少。

两个检查点 × 两套划分 = 4 个数字：
  ① v1 文件级划分 + v1 旧模型 → 复现旧口径 89.82% 一类的结果
  ② v2 录音级划分 + v2 新模型 → 诚实口径（测试录音从未出现在训练集）
  ③④ 交叉组合，用于说明「模型其实记住了录音，而不是学会了声音」

用法：python3 compare_protocols.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import config
import viz_style as vs
from dataset import AudioSegmentDataset
from eval_emergency import infer_probs, load_model

REP = vs.REPORT_DIR
V1 = config.METADATA_CLEAN.with_name("metadata_clean_v1_filelevel.csv")


def split_test(path, fold: int = config.TEST_FOLD) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d[(d.role == "train") & (d.cv_fold == fold)].reset_index(drop=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = {
        "v1 旧模型(文件级训练)": config.MODEL_DIR / "ds_cnn_best_v1_filelevel.pth",
        "v2 新模型(录音级训练)": config.MODEL_DIR / "ds_cnn_best.pth",
    }
    tests = {
        "v1 文件级测试集": V1,
        "v2 录音级测试集": config.METADATA_CLEAN,
    }

    rows = []
    for cname, cpath in ckpts.items():
        if not cpath.exists():
            print(f"跳过（缺检查点）：{cname}")
            continue
        model = load_model(str(cpath), device)
        for tname, tpath in tests.items():
            test = split_test(tpath)
            probs = infer_probs(model, AudioSegmentDataset(test, augment=False), device)
            y = test.class_id.to_numpy()
            pred = probs.argmax(1)
            acc = float((pred == y).mean())
            # 危险类 macro recall（安全口径）
            from eval_emergency import SAFETY_RELEVANT
            recs = []
            for c in SAFETY_RELEVANT:
                i = config.CLASS_TO_IDX[c]
                m = y == i
                recs.append(float((pred[m] == i).mean()))
            rows.append({"模型": cname, "测试集": tname, "段数": len(test),
                         "准确率": acc, "危险类 macro 召回": float(np.mean(recs))})
            print(f"{cname} × {tname}: acc={acc:.4f}  危险类召回={np.mean(recs):.4f}")

    df = pd.DataFrame(rows)
    L = ["---", "tags: [project, SafeSoundAI, aed, protocol]", "created: 2026-10-17", "---", "",
         "# 划分协议对照：文件级 vs 录音级（2026-10-17）", "",
         "同一批数据，两种划分方式对指标的影响。全部在 fold 9 上评估：", "",
         "| 模型 | 测试集 | 段数 | 准确率 | 危险类 macro 召回 |", "|---|---|---|---|---|"]
    L += [f"| {r['模型']} | {r['测试集']} | {r['段数']} | {r['准确率']*100:.2f}% | {r['危险类 macro 召回']*100:.2f}% |"
          for _, r in df.iterrows()]
    L += ["", "## 怎么读这张表", "",
          "- v1 文件级划分下，**91.2% 的测试段所属录音在训练集出现过**（UrbanSound8K 的 0.5 s 平移切片），"
          "模型可以靠「认录音」拿分，因此旧口径数字偏乐观。",
          "- v2 录音级划分下测试录音完全未见，是**部署时真实的泛化条件**，应作为论文与验收的主口径。",
          "- 若两套划分差距很大，说明模型对录音身份过拟合，需要更强增强（音高/时长扰动、RIR、更大噪声池）"
          "而不是继续堆 epoch。"]
    (REP / "protocol_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    df.to_csv(REP / "protocol_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {REP/'protocol_compare.md'}")


if __name__ == "__main__":
    main()
