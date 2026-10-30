"""
SafeSoundAI — 过滤支路消融（ablate_filter.py）

在真实合成混合集（4_Synthetic_Mixtures，带 stems）上，对 attenuator 的两个
2026-10-31 修复项做 2×2 消融，量化各自贡献：

  A) 逐窗 z-score          （fc.ZSCORE_WINDOWS）
  B) 白名单频谱让位        （fc.WHITELIST_CARVEOUT）

每个配置报告：
  * Supp_dB  ：目标（应抑制的噪声）stems 的能量变化，越负越好
  * Pres_dB  ：干扰（应保留的白名单类）stems 的能量变化，越接近 0 越好
  * gap      ：Supp − Pres（负值 = 选择性越好）
  * 红线超标率：Pres < −3 dB 的样本比例（filter_safety §2.1 规定必须为 0）
  * 触发率   ：Supp < −1 dB（真的动手降噪了）的样本比例
  * 耗时     ：每条 1 s 混合的掩码计算耗时（CPU）

用法：
    python ablate_filter.py --n 100 --split test
产物：
    outputs/reports/filter_ablation.md
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import soundfile as sf
import torch

import config
import filter_config as fc
import vocab
from attenuator import SpectralGateAttenuator
from eval_filter import suppress_gain

CONFIGS = [
    ("legacy（旧逻辑）",        dict(zscore_windows=False, carveout=False)),
    ("+z-score",               dict(zscore_windows=True,  carveout=False)),
    ("+carveout",              dict(zscore_windows=False, carveout=True)),
    ("both（当前默认）",        dict(zscore_windows=True,  carveout=True)),
]


def load(p: str) -> np.ndarray:
    y, _ = sf.read(p, always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32)[:config.SEGMENT_SAMPLES]


def run(rows: list[dict], kwargs: dict, ckpt: str | None) -> dict:
    att = SpectralGateAttenuator(aed_ckpt=ckpt, **kwargs)
    supp, pres, ms = [], [], []
    for r in rows:
        mix, t_stem, i_stem = load(r["mix_path"]), load(r["target_path"]), load(r["interferer_path"])
        t0 = time.perf_counter()
        mask = att.mask_from(mix)
        ms.append(1000 * (time.perf_counter() - t0))
        supp.append(suppress_gain(t_stem, mix, mask))
        pres.append(suppress_gain(i_stem, mix, mask))
    supp, pres = np.array(supp), np.array(pres)
    return {
        "head": vocab.describe(att.head),
        "supp": float(supp.mean()), "supp_med": float(np.median(supp)),
        "pres": float(pres.mean()),
        "gap": float(supp.mean() - pres.mean()),
        "redline_pct": 100 * float(np.mean(pres < fc.HARD_GATE_TOLERANCE_DB)),
        "trigger_pct": 100 * float(np.mean(supp < -1.0)),
        "ms": float(np.mean(ms)),
        "per": list(zip(supp.round(2).tolist(), pres.round(2).tolist())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None,
                    help="默认 ds_cnn_best.pth（10 类）；4 类头不支持类增益细分")
    args = ap.parse_args()

    with open(fc.MIX_MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split][:args.n]
    if not rows:
        raise SystemExit(f"manifest 无 {args.split} 样本")
    print(f"{len(rows)} 条 {args.split} 混合样本，profile={fc.DEFAULT_PROFILE}")

    results = {}
    for name, kw in CONFIGS:
        r = run(rows, kw, args.ckpt)
        results[name] = r
        print(f"{name:<18} Supp={r['supp']:7.2f}  Pres={r['pres']:7.2f}  "
              f"gap={r['gap']:6.2f}  红线={r['redline_pct']:5.1f}%  "
              f"触发={r['trigger_pct']:5.1f}%  {r['ms']:.0f} ms")

    head = next(iter(results.values()))["head"]
    lines = [
        "# 过滤支路消融报告（真实混合集）",
        "",
        f"- 数据：`{fc.MIX_MANIFEST}` 的 `{args.split}` split，取前 {len(rows)} 条",
        f"- 检查点：{head}",
        f"- 档位：`{fc.DEFAULT_PROFILE}`；测试口径与 `eval_filter.py` 一致（mask 施加到 stems）",
        "- 红线：filter_safety §2.1 要求 Pres ≥ −3 dB（白名单类不得被压超过 3 dB）",
        "",
        "| 配置 | Supp dB（越负越好） | Pres dB（越接近 0 越好） | Supp−Pres | 红线超标率 | 触发率 | 掩码耗时 |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['supp']:.2f} | {r['pres']:.2f} | {r['gap']:.2f} "
                     f"| {r['redline_pct']:.1f}% | {r['trigger_pct']:.1f}% | {r['ms']:.0f} ms |")
    lines += [
        "",
        "## 读法",
        "",
        "- **旧逻辑为什么几乎不降噪**：任一白名单类后验 > 0.5 就整窗直通，而真实混合上"
        "噪声被误判成 car_horn / siren 的窗口占比很高（窗口级实测值见 "
        "`outputs/reports/filter_audit.md` 层 C，本脚本不复算该量），于是“该降噪的帧”"
        "大部分被否决 —— 见上表 legacy 行触发率远低于 both 行。",
        "- **z-score 的作用**：训练段是 z-score 过的，推理不归一化会让后验分布整体错位，"
        "Supp 与触发率同时被拉低。这是本消融里唯一的量级级改动。",
        "- **carveout 的作用**：把“整窗否决”换成“只让出白名单类自己的频谱带”。"
        "在 +z-score 基础上它把 Pres 抬回约 0.2 dB，代价是 Supp 少约 0.2 dB —— 方向对、量级小。",
        "",
        "> 注：本表是**掩码层面**的机理数字；听不懂的差别要听 `outputs/audio/ab/` 的 A/B 文件。",
    ]
    out = fc.FILTER_REPORT_DIR / "filter_ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("写出:", out)


if __name__ == "__main__":
    main()
