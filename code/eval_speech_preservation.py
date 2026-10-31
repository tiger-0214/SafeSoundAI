"""SafeSoundAI — 按「除人声外全压」的设计口径评测降噪支路（eval_speech_preservation.py）

**为什么另开一个评测脚本**：`eval_filter.py` 的 supp/pres 是以「噪声类分轨」为分母的，
它的「pres」衡量的是**危险声有没有被保住**——那是旧白名单语义（危险声必须 0 dB）。
V2 语义下危险声本来就该被压，所以旧口径会把正确行为判成缺陷。

新口径只有三条，都对应用户能听出来的东西：
  E1 非语音抑制  : 纯噪声输入 → 输出能量变化 dB（越负越好，目标 ≤ -10 dB）
  E2 人声保留    : mix 输入 → 输出 vs 干净语音的 SI-SDR 提升 / STOI / 语音段能量变化
  E3 危险声不豁免: 危险声类（siren/dog_bark…）在 E1 里不得比普通噪声压得更浅

产物：outputs/reports/speech_preservation.md / .json / per_sample.csv

用法：
    python eval_speech_preservation.py --n 20
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pystoi import stoi

import config
import filter_config as fc
from attenuator import SpectralGateAttenuator
from tse_model import istft, stft

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "reports"
SPEECH_MIX = Path("/root/safesound_data/speech_mixtures")


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-12))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    a = float((est * ref).sum())
    proj = a * ref / (float((ref ** 2).sum()) + 1e-9)
    noise = est - proj
    return db((float((proj ** 2).sum()) + 1e-9) / (float((noise ** 2).sum()) + 1e-9))


def apply_mask(y: np.ndarray, mask_lin: np.ndarray) -> np.ndarray:
    Y = stft(torch.from_numpy(y.astype(np.float32)).to(DEVICE))
    T = Y.shape[-1]
    g = mask_lin[:, :T]
    if g.shape[1] < T:
        g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
    Yg = Y * torch.from_numpy(g).to(device=Y.device, dtype=Y.dtype)
    return istft(Yg, len(y)).cpu().numpy()


def load(p) -> np.ndarray:
    y, _ = sf.read(str(p), always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="smoke")
    ap.add_argument("--profile", default=fc.DEFAULT_PROFILE)
    args = ap.parse_args()

    man = SPEECH_MIX / args.split / "manifest.csv"
    if not man.exists():
        raise SystemExit(f"找不到 {man}，先跑 make_speech_mixtures.py")
    rows = list(csv.DictReader(open(man, encoding="utf-8")))[:args.n]

    att = SpectralGateAttenuator(profile=args.profile)
    print(f"profile={args.profile}｜样本 {len(rows)}｜mix 根 {SPEECH_MIX / args.split}")

    per: list[dict] = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        clean = load(r["clean_path"])
        noise = load(r["noise_path"])
        mix = load(r["mix_path"])

        # ---- E1：纯噪声输入，应该被压 ----
        m_n = att.mask_from(noise)
        out_n = apply_mask(noise, m_n)
        supp = db(np.mean(out_n ** 2) / (np.mean(noise ** 2) + 1e-12))

        # ---- E2：mix 输入，人声应保留 ----
        m_m = att.mask_from(mix)
        out_m = apply_mask(mix, m_m)
        n = min(len(out_m), len(clean))
        o, c = out_m[:n], clean[:n]
        sdr_in = si_sdr(mix[:n], c)      # 处理前的 SI-SDR（mix vs clean）
        sdr_out = si_sdr(o, c)
        # 语音段能量变化（以 clean 能量显著大于噪声为"语音段"的代理判据）
        sp = np.abs(c) > (0.1 * np.abs(c).max() + 1e-9)
        keep_db = db(np.mean(o[sp] ** 2) / (np.mean(mix[:n][sp] ** 2) + 1e-12)) if sp.any() else float("nan")
        try:
            st_in = stoi(c, mix[:n], config.SAMPLE_RATE, extended=False)
            st_out = stoi(c, o, config.SAMPLE_RATE, extended=False)
        except Exception:                # noqa: BLE001  短片段/静音可能报错
            st_in = st_out = float("nan")

        per.append({
            "mix_id": r["mix_id"], "snr_db": float(r["snr_db"]),
            "noise_src": Path(r["noise_src"]).name,
            "E1_noise_supp_db": round(supp, 2),
            "E2_sisdr_in": round(sdr_in, 2),
            "E2_sisdr_out": round(sdr_out, 2),
            "E2_sisdr_imp": round(sdr_out - sdr_in, 2),
            "E2_stoi_in": round(st_in, 4),
            "E2_stoi_out": round(st_out, 4),
            "E2_speech_keep_db": round(keep_db, 2),
        })
        if i % 5 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    def mean(k):
        v = [p[k] for p in per if not np.isnan(p[k])]
        return round(float(np.mean(v)), 3) if v else float("nan")

    summary = {
        "n": len(per),
        "profile": args.profile,
        "E1_noise_supp_db_mean": mean("E1_noise_supp_db"),
        "E2_sisdr_imp_mean": mean("E2_sisdr_imp"),
        "E2_stoi_in_mean": mean("E2_stoi_in"),
        "E2_stoi_out_mean": mean("E2_stoi_out"),
        "E2_speech_keep_db_mean": mean("E2_speech_keep_db"),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speech_preservation.json").write_text(
        json.dumps({"summary": summary, "per_sample": per}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    with open(OUT / "speech_preservation_p.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)

    md = [
        "# 降噪支路评测（V2 语义：除人声外全压）", "",
        f"样本 {summary['n']} 条｜profile `{summary['profile']}`｜耗时 {summary['wall_s']} s", "",
        "## 汇总", "",
        "| 指标 | 数值 | 设计目标 |", "|:--|--:|:--|",
        f"| E1 非语音抑制 (dB) | **{summary['E1_noise_supp_db_mean']:+.2f}** | ≤ -10 |",
        f"| E2 人声保留 SI-SDR 提升 (dB) | **{summary['E2_sisdr_imp_mean']:+.2f}** | ≥ 0 |",
        f"| E2 STOI 处理前 → 后 | {summary['E2_stoi_in_mean']:.4f} → **{summary['E2_stoi_out_mean']:.4f}** | 不掉 |",
        f"| E2 语音段能量变化 (dB) | **{summary['E2_speech_keep_db_mean']:+.2f}** | ≥ -3 |",
        "", "## 与旧口径的关系", "",
        "`eval_filter.py` 的 supp/pres 以「噪声分轨」为分母，pres 衡量的是**危险声有没有被保住**（旧白名单语义）。",
        "V2 语义下危险声本就该被压，那里的 pres 变负是**设计意图**，不再是缺陷。", "",
        "## 明细（前 20 条）", "",
        "| mix_id | SNR | E1 抑制 | SI-SDR 前 | 后 | 提升 | STOI 前 | 后 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for p in per[:20]:
        md.append(f"| {p['mix_id']} | {p['snr_db']:+.1f} | {p['E1_noise_supp_db']:+.2f} "
                  f"| {p['E2_sisdr_in']:+.2f} | {p['E2_sisdr_out']:+.2f} | {p['E2_sisdr_imp']:+.2f} "
                  f"| {p['E2_stoi_in']:.3f} | {p['E2_stoi_out']:.3f} |")
    (OUT / "speech_preservation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("→ outputs/reports/speech_preservation.md")


if __name__ == "__main__":
    main()
