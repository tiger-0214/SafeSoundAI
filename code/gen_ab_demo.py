"""
SafeSoundAI — 降噪 A/B 试听样张生成（gen_ab_demo.py, 2026-10-31）

为什么需要它：`outputs/audio/0X_*.wav` 那批「demo 音频」实为 220 Hz 正弦占位文件
（时长 2.50 s、频谱峰值全在 220 Hz、`00_air_conditioner` 与 `08_siren` 内容完全一致），
在此之前的降噪试听结论全部不可信。本脚本改用**真实语料合成的混合**（4_Synthetic_Mixtures，
带 stems 可分离核算），直接产出可听的 A/B 对与客观指标。

产物（`outputs/audio/ab/`）：
  {tag}_mix.wav        处理前（原始混合）
  {tag}_aed.wav        处理后（AED 驱动谱增益门）
  {tag}_target.wav     应被抑制的噪声分轨（参考，用于听「该压的压掉了没」）
  {tag}_interf.wav     应被保留的白名单干扰分轨（参考，用于听「该留的留住了没」）
  index.md             逐条指标 + 试听顺序说明

用法：
    python gen_ab_demo.py               # 默认 6 条，覆盖不同噪声类型
    python gen_ab_demo.py --n 10 --profile comfort
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import soundfile as sf

import config
import filter_config as fc
import vocab
from attenuator import SpectralGateAttenuator
from eval_filter import suppress_gain

OUT = fc.FILTER_AUDIO_DIR / "ab"


def load(p: str) -> np.ndarray:
    y, _ = sf.read(p, always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32)[:config.SEGMENT_SAMPLES]


def pick(rows: list[dict], n: int) -> list[dict]:
    """每个噪声类型各挑一条，确保样张覆盖不同频谱形态。"""
    seen, out = set(), []
    for r in rows:
        if r["target_class"] in seen:
            continue
        seen.add(r["target_class"])
        out.append(r)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--profile", default=fc.DEFAULT_PROFILE)
    ap.add_argument("--ckpt", default=None, help="默认 ds_cnn_best.pth（10 类，能区分噪声类型）")
    args = ap.parse_args()

    with open(fc.MIX_MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    rows = pick(rows, args.n)
    if not rows:
        raise SystemExit("manifest 无可用样本")

    att = SpectralGateAttenuator(profile=args.profile, aed_ckpt=args.ckpt)
    OUT.mkdir(parents=True, exist_ok=True)
    print("AED 头:", vocab.describe(att.head))

    lines = [
        "# 降噪 A/B 试听样张",
        "",
        f"- 生成命令：`python gen_ab_demo.py --n {args.n} --profile {args.profile}`",
        f"- 检查点：{vocab.describe(att.head)}",
        f"- 档位：`{args.profile}`（白名单类目标增益 0 dB：siren / car_horn / gun_shot / dog_bark）",
        "- 每条样本：先听 `_mix`（原始混合），再听 `_aed`（降噪后），"
        "`_target` / `_interf` 分别是应抑制的噪声与应保留的白名单声分轨。",
        "",
        "| 标签 | 噪声(应抑制) | 白名单干扰(应保留) | SNR_target | 噪声分轨 Supp | 白名单分轨 Pres | 判定 |",
        "|:--|:--|:--|--:|--:|--:|:--|",
    ]

    recs = []
    for i, r in enumerate(rows, 1):
        mix, tgt, itf = load(r["mix_path"]), load(r["target_path"]), load(r["interferer_path"])
        y_out, info = att.process(mix)
        mask = att.mask_from(mix)
        supp = suppress_gain(tgt, mix, mask)
        pres = suppress_gain(itf, mix, mask)
        # 判定分四类，避免把「安全策略」误读成「降噪失败」
        if supp > -1.0:
            verdict = ("🛡️ 检出白名单类→整窗直通" if info["top_name"] in fc.WHITELIST_NAMES
                       else "➖ 置信不足→fail-open 直通")
        elif pres <= fc.HARD_GATE_TOLERANCE_DB:
            verdict = "⚠️ 有效但误伤白名单"
        else:
            verdict = "✅ 有效且安全"
        tag = f"{i:02d}_{r['target_class']}"
        sf.write(OUT / f"{tag}_mix.wav", mix, config.SAMPLE_RATE, subtype="PCM_16")
        sf.write(OUT / f"{tag}_aed.wav", y_out, config.SAMPLE_RATE, subtype="PCM_16")
        sf.write(OUT / f"{tag}_target.wav", tgt, config.SAMPLE_RATE, subtype="PCM_16")
        sf.write(OUT / f"{tag}_interf.wav", itf, config.SAMPLE_RATE, subtype="PCM_16")
        recs.append(dict(tag=tag, target=r["target_class"], interf=r["interferer_class"],
                         supp=supp, pres=pres))
        print(f"{tag:<24} top={info['top_name']}({info['top_prob']:.2f}) "
              f"Supp={supp:6.2f} Pres={pres:6.2f} Δ能量={info['energy_delta_db']:6.2f} {verdict}")
        lines.append(f"| `{tag}` | {r['target_class']} | {r['interferer_class']} "
                     f"| {float(r['snr_target_db']):.1f} dB | {supp:.2f} dB | {pres:.2f} dB | {verdict} |")

    supp = np.array([x["supp"] for x in recs])
    pres = np.array([x["pres"] for x in recs])
    lines += [
        "",
        f"**汇总**：Supp 均值 {supp.mean():.2f} dB，Pres 均值 {pres.mean():.2f} dB，"
        f"触发率 {100*np.mean(supp < -1):.0f}%，白名单红线（Pres < −3 dB）超标 "
        f"{100*np.mean(pres < fc.HARD_GATE_TOLERANCE_DB):.0f}%。",
        "",
        "> 说明：类增益谱门只能按「该类能量带」加权，压不掉与噪声同频段混在一起的白名单声，"
        "这是方法本身的边界（与 TSE 的 α 收缩结论一致），不是调参能修好的，见 "
        "`filter_ablation.md` 与 `filter_ab_report.md`。",
    ]
    (OUT / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n写出目录:", OUT)


if __name__ == "__main__":
    main()
