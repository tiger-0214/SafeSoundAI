"""
SafeSoundAI — 降噪支路**功能测试**（filter_functional_check.py）

和 `eval_filter.py` 的分工：
  * `eval_filter.py` 回答「降噪效果多好」（能力层：Supp / Pres / SI-SDR / 红线率）。
  * 本文件回答「降噪支路**是不是按设计在工作**」（功能层：信号链、增益映射、
    hard gate、carveout 底线、fail-open、anti-collapse、块拼接、实时预算、
    以及"降噪之后检测器还听得见警笛吗"）。

区别在真值来源：功能层用**我们自己造的、答案已知的信号**（合成 siren / 空调噪声 /
白噪 + 真实混合集分轨），所以能对每一项写出可证伪的判据；能力层只能给统计量。

判据分两档：
  * `hard`       —— 实现契约。挂了就是 bug，退出码 1，可进 CI。
  * `capability` —— 已知能力边界。只测量、不过门（例：白名单红线 43% 超标是
                    已定稿的负结论，功能测试不该把它伪装成绿的）。

用法：
    python filter_functional_check.py --n 20
产物：
    outputs/reports/filter_functional.json / filter_functional.md
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import filter_config as fc
import vocab
from attenuator import SpectralGateAttenuator, mel_to_linear_gain
from filter_pipeline import FilterPipeline

SR = config.SAMPLE_RATE
RESULTS: list[dict] = []


# ---------------------------------------------------------------------------
# 0) 工具
# ---------------------------------------------------------------------------
def rms_db(x: np.ndarray) -> float:
    return 10.0 * np.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def rel_db(a: np.ndarray, b: np.ndarray) -> float:
    """a 相对 b 的误差能量（dB）。-inf = 完全一致。"""
    e = float(np.mean((a - b) ** 2))
    r = float(np.mean(np.asarray(b, dtype=np.float64) ** 2)) + 1e-12
    return 10.0 * np.log10(e / r + 1e-12)


def band_db(x: np.ndarray, lo: float, hi: float) -> float:
    """频带内能量（dB）。"""
    X = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    m = (f >= lo) & (f < hi)
    return 10.0 * np.log10(float(X[m].sum()) + 1e-12)


def peak_norm(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.max(np.abs(x)) + 1e-9) * peak).astype(np.float32)


def record(cid: str, level: str, name: str, severity: str, passed: bool,
           measured, note: str = "") -> None:
    RESULTS.append({"id": cid, "level": level, "name": name, "severity": severity,
                    "pass": bool(passed), "measured": measured, "note": note})
    flag = "PASS" if passed else ("FAIL" if severity == "hard" else "WARN")
    print(f"  [{flag}] {cid} {name} | {measured}")


# ---------------------------------------------------------------------------
# 1) 合成真值信号（答案已知，判据可证伪）
# ---------------------------------------------------------------------------
def synth_siren(dur: float = 2.0, seed: int = 0) -> np.ndarray:
    """警笛：700–1200 Hz 慢扫频 + 二三次谐波（Euro 警笛的简化版）。"""
    t = np.arange(int(SR * dur)) / SR
    f = 700 + 500 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.6 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    return peak_norm(y)


def synth_ac(dur: float = 2.0, seed: int = 0) -> np.ndarray:
    """空调/风机：50 Hz 基波 + 谐波 + 低频为主的宽带噪声。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    hum = sum(1.0 / (k ** 1.5) * np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6.28))
              for k in range(1, 9))
    noise = rng.standard_normal(len(t)).astype(np.float64)
    # 一阶低通（把能量压到 1 kHz 以下，模拟风道噪声）
    a = 0.12
    lp = np.zeros_like(noise)
    acc = 0.0
    for i, v in enumerate(noise):
        acc = (1 - a) * acc + a * v
        lp[i] = acc
    return peak_norm(hum * 0.6 + lp / (np.max(np.abs(lp)) + 1e-9) * 0.5)


def synth_white(dur: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return peak_norm(rng.standard_normal(int(SR * dur)))


def synth_tone(f: float = 1000.0, dur: float = 2.0) -> np.ndarray:
    return peak_norm(np.sin(2 * np.pi * f * np.arange(int(SR * dur)) / SR))


def load_mix_rows(split: str = "test") -> list[dict]:
    with open(fc.MIX_MANIFEST, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def load_wav(p: str) -> np.ndarray:
    import soundfile as sf
    y, _ = sf.read(p, always_2d=False, dtype="float32")
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32)[: config.SEGMENT_SAMPLES]


# ---------------------------------------------------------------------------
# 2) L0 — 信号链正确性
# ---------------------------------------------------------------------------
def check_identity(att: SpectralGateAttenuator) -> None:
    """α=0（全部 0 dB 增益）时输出必须等于输入：STFT/iSTFT/重叠加窗无失真。"""
    y = synth_tone(1000.0, 2.0)
    quiet = SpectralGateAttenuator(strength=0.0)
    out, info = quiet.process(y)
    err = rel_db(out, y)
    record("L0-1", "L0 信号链", "恒等直通（strength=0 → 逐样本还原）", "hard",
           err < -60.0, f"重构误差 {err:.1f} dB",
           "钉住 STFT/iSTFT + mel→线性增益映射；挂了说明信号链本身有损")


def check_zscore_frontend(att: SpectralGateAttenuator, row: dict) -> None:
    """推理前端必须与训练前端逐窗 z-score 一致（2026-10-31 修过的 bug）。"""
    y = load_wav(row["mix_path"])
    probs_zs, starts = att.window_probs(y, hop=SR // 2)
    w = y[starts[0]:starts[0] + SR].astype(np.float32)
    w = (w - w.mean()) / (w.std() + 1e-8)
    mel = torch.log(att.mel_tf(torch.from_numpy(w).to(att.device)) + 1e-9)
    mel = mel[None, None, :, :att.frames]
    with torch.no_grad():
        p_manual = F.softmax(att.model(mel), dim=1).cpu().numpy()[0]

    raw = SpectralGateAttenuator(zscore_windows=False)
    probs_raw, _ = raw.window_probs(y, hop=SR // 2)

    d_manual = float(np.abs(p_manual - probs_zs[0]).max())
    d_raw = float(np.abs(probs_zs[0] - probs_raw[0]).max())
    ok = d_manual < 1e-4 and d_raw > 0.02
    record("L0-2", "L0 信号链", "逐窗 z-score 与训练口径一致（且开关真的生效）", "hard",
           ok,
           f"手工复算 max|Δp|={d_manual:.2e}；关掉 z-score 后 max|Δp|={d_raw:.3f}",
           "第二项是关键：若关掉也不变，说明归一化是死代码（旧 bug 会复发）")


def check_gain_table_translation(att: SpectralGateAttenuator) -> None:
    """类增益必须按**名字**翻译到当前头空间，且 band 空间与 STFT 频点数对齐。"""
    prof = fc._PROFILE_NAMES[fc.DEFAULT_PROFILE]["gains"]
    bad = [(n, att.gains[i]) for i, n in enumerate(att.classes)
           if abs(att.gains.get(i, 0.0) - prof.get(n, 0.0)) > 1e-9]
    wl_bad = [(n, att.gains[i]) for i, n in enumerate(att.classes)
              if n in fc.WHITELIST_NAMES and abs(att.gains.get(i, 0.0)) > 1e-9]
    n_freqs = config.N_FFT // 2 + 1
    band_space_ok = (att.head.mel["n_fft"] == fc.FILTER_N_FFT
                     and att.class_band.shape[1] == config.N_MELS
                     and n_freqs == fc.FILTER_N_FFT // 2 + 1)
    ok = not bad and not wl_bad and band_space_ok
    record("L0-3", "L0 信号链", "类增益按名字翻译 + 频带空间与 STFT 对齐", "hard", ok,
           f"错位 {bad or '无'}｜白名单非 0 {wl_bad or '无'}｜"
           f"head mel n_fft={att.head.mel['n_fft']} vs FILTER_N_FFT={fc.FILTER_N_FFT}",
           "频带空间错位会让 -12 dB 打在错误频段上，而听感上只是「降噪怪怪的」")


def check_mel_to_linear(att: SpectralGateAttenuator) -> None:
    """mel 域增益 → 线性频点增益：常数必须仍是常数，单 bin 不得外溢。"""
    const = np.full((config.N_MELS, 1), -12.0, dtype=np.float32)
    n_freqs = fc.FILTER_N_FFT // 2 + 1
    lin = mel_to_linear_gain(const, n_freqs=n_freqs)
    # f_min/f_max 之外的频点没有 mel 覆盖（权重 0）→ 必然 0 dB，属正确行为，只查覆盖区
    import torchaudio
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs, f_min=config.F_MIN, f_max=config.F_MAX, n_mels=config.N_MELS,
        sample_rate=config.SAMPLE_RATE, norm=None, mel_scale="htk").numpy()
    covered = fb.sum(axis=1) > 0
    const_ok = float(np.abs(lin[covered] + 12.0).max()) < 0.5
    uncovered = int((~covered).sum())

    band = np.zeros((config.N_MELS, 1), dtype=np.float32)
    band[40, 0] = -20.0
    lin_band = mel_to_linear_gain(band, n_freqs=fc.FILTER_N_FFT // 2 + 1)[:, 0]
    freqs = np.arange(len(lin_band)) * SR / fc.FILTER_N_FFT
    hit = freqs[np.argmin(lin_band)]
    outside = np.abs(lin_band[freqs > hit + 400]).max()
    ok = const_ok and outside > -3.0
    record("L0-4", "L0 信号链", "mel→线性增益映射（常数保形 / 单 bin 不外溢）", "hard", ok,
           f"覆盖区内常数偏差 {float(np.abs(lin[covered] + 12.0).max()):.2f} dB"
           f"（{uncovered} 个 f_min/f_max 外频点按 0 dB 直通）；"
           f"bin40 峰值在 {hit:.0f} Hz，其外 400 Hz 处最大 {outside:.2f} dB")


# ---------------------------------------------------------------------------
# 3) L1 — 行为正确性（给了真值，支路该做的事）
# ---------------------------------------------------------------------------
def check_oracle_suppress_noise(att: SpectralGateAttenuator) -> None:
    """知道是空调时，必须真的压下去（隔离掉 AED 误判这一层）。"""
    y = synth_ac(2.0, seed=7)
    ac_id = config.CLASS_TO_IDX["air_conditioner"]
    out, info = att.process(y, oracle_class=ac_id)
    delta = rms_db(out) - rms_db(y)
    low_band = band_db(out, 0, 800) - band_db(y, 0, 800)
    ok = delta < -2.0 and low_band < -3.0
    record("L1-1", "L1 行为", "oracle=air_conditioner → 低频带真被压", "hard", ok,
           f"总能量 {delta:+.2f} dB｜0–800 Hz 带内 {low_band:+.2f} dB",
           "上界能力：给对上界标签时增益链是否有效")


def check_oracle_preserve_siren(att: SpectralGateAttenuator) -> None:
    """知道是警笛时，必须一律 0 dB（R1：白名单不得被衰减）。"""
    y = synth_siren(2.0)
    out, info = att.process(y, oracle_class=config.CLASS_TO_IDX["siren"])
    delta = rms_db(out) - rms_db(y)
    ok = delta > -1.0
    record("L1-2", "L1 行为", "oracle=siren → 不得衰减（≤1 dB）", "hard", ok,
           f"能量变化 {delta:+.2f} dB")


def check_carveout_floor(att: SpectralGateAttenuator, rows: list[dict]) -> None:
    """carveout 硬底线：让位区（protect ≥ 0.5）的衰减不得深于 CARVEOUT_GUARD_DB。"""
    worst, viol, guarded, eligible = 0.0, 0, 0, 0
    for r in rows:
        y = load_wav(r["mix_path"])
        probs = att.frame_probs(y)
        g_db = att.gain_curve_db(probs)
        # 与 attenuator.gain_curve_db 同一套 protect 定义：band × 白名单原样后验
        protect = np.zeros_like(g_db)
        for cid in att.whitelist_ids:
            protect = np.maximum(protect, att.class_band[cid][:, None] * probs[:, cid][None, :])
        guard = protect >= 0.5
        guarded += int(guard.sum())
        eligible += int((protect >= 0.1).sum())
        if guard.any():
            sub = g_db[guard]
            worst = min(worst, float(sub.min()))
            viol += int((sub < fc.CARVEOUT_GUARD_DB - 0.01).sum())
    ok = viol == 0
    record("L1-3", "L1 行为", "carveout 底线：让位区衰减不深于 -1 dB", "hard", ok,
           f"受护 bin·帧 {guarded}（protect≥0.1 候选 {eligible}，覆盖率 "
           f"{100 * guarded / max(eligible, 1):.0f}%），越界 {viol}，最低 {worst:.2f} dB",
           "把 fc.CARVEOUT_GUARD_DB 变成可执行断言，并量出「受护覆盖多少」："
           "protect=band×p，p=0.5 且 band=0.5 时只有 0.25，够不到 0.5 的门 —— "
           "让位只在「高置信 × 强能量带」上生效，这是覆盖率的由来")


def check_fail_open(att: SpectralGateAttenuator) -> None:
    """后验全低于 apply_conf → 必须逐样本直通（fail-open 0 dB）。"""
    y = synth_white(2.0, seed=3)
    orig = att._resolve_probs
    try:
        att._resolve_probs = lambda yy, oracle=None: np.full(
            (1 + len(yy) // fc.FILTER_HOP, att.n_classes), 1.0 / att.n_classes, dtype=np.float32)
        out, info = att.process(y)
    finally:
        att._resolve_probs = orig
    err = rel_db(out, y)
    ok = err < -60.0 and abs(info["energy_delta_db"]) < 0.1
    record("L1-4", "L1 行为", "低置信 → fail-open 直通（不猜不压）", "hard", ok,
           f"重构误差 {err:.1f} dB｜记录能量变化 {info['energy_delta_db']:+.2f} dB")


def check_anti_collapse(att: SpectralGateAttenuator) -> None:
    """掩码把整段压塌（< -20 dB）时，硬门必须原样放行并计数。"""
    import attenuator as A
    y = synth_tone(1000.0, 1.0)
    before = att.stats["over_suppression_events"]
    orig = A.mel_to_linear_gain
    try:
        A.mel_to_linear_gain = lambda g, n_freqs=fc.FILTER_N_FFT // 2 + 1: np.full(
            (n_freqs, np.asarray(g).shape[1]), -50.0, dtype=np.float32)
        out, info = att.process(y, oracle_class=config.CLASS_TO_IDX["air_conditioner"])
    finally:
        A.mel_to_linear_gain = orig
    err = rel_db(out, y)
    fired = att.stats["over_suppression_events"] - before
    ok = err < -60.0 and fired >= 1 and info["applied"] is False
    record("L1-5", "L1 行为", "能量塌陷保护（-50 dB 掩码 → 原样放行）", "hard", ok,
           f"重构误差 {err:.1f} dB｜触发 {fired} 次｜applied={info['applied']}")


def check_temporal_smoothness(att: SpectralGateAttenuator, row: dict) -> None:
    """帧间增益跳变必须被限幅（否则会出现「抽气」式调制噪声）。"""
    y = load_wav(row["mix_path"])
    g = att.gain_curve_db(att.frame_probs(y))
    d = float(np.abs(np.diff(g, axis=1)).max()) if g.shape[1] > 1 else 0.0
    ok = d <= fc.MAX_DB_JUMP_PER_FRAME + 1e-3
    record("L1-6", "L1 行为", "帧间增益跳变 ≤ 6 dB（平滑器生效）", "hard", ok,
           f"实测最大跳变 {d:.2f} dB / 上限 {fc.MAX_DB_JUMP_PER_FRAME} dB")


# ---------------------------------------------------------------------------
# 4) L2 — 集成有效性（产品级：拼接、实时预算、检测器是否还听得见）
# ---------------------------------------------------------------------------
def check_pipeline_continuity() -> None:
    """流式分块 + 交叉淡化：块边界不得比块内更「跳」。"""
    mix = peak_norm(synth_siren(1.5) * 0.6 + synth_ac(1.5, seed=5) * 0.6)
    pipe = FilterPipeline(use_tse=False)
    out, stats = pipe.process_offline(mix)
    block = 1600
    interior, boundary = [], []
    for k in range(block, len(mix) - block, block):
        boundary.append(abs(float(out[k] - out[k - 1])))
    diffs = np.abs(np.diff(out))
    interior = diffs[len(diffs) // 10: -len(diffs) // 10]
    p999 = float(np.percentile(interior, 99.9))
    bmax = float(np.max(boundary)) if boundary else 0.0
    ok = (len(out) == len(mix) and np.isfinite(out).all()
          and bmax <= max(3 * p999, 1e-3) and np.max(np.abs(out)) <= 2 * np.max(np.abs(mix)))
    record("L2-1", "L2 集成", "分块拼接无爆音 / 无能量注入", "hard", ok,
           f"块边界最大阶跃 {bmax:.4f}｜块内 p99.9 {p999:.4f}｜"
           f"峰值比 {np.max(np.abs(out)) / np.max(np.abs(mix)):.2f}")


def check_latency_budget() -> None:
    """单块处理耗时必须落在 filter_constraints 的 100 ms 预算内。"""
    mix = peak_norm(synth_siren(3.0) * 0.6 + synth_ac(3.0, seed=6) * 0.6)
    pipe = FilterPipeline(use_tse=False)
    t0 = time.perf_counter()
    out, stats = pipe.process_offline(mix)
    wall = time.perf_counter() - t0
    rtf = wall / (len(mix) / SR)
    ok = stats["block_ms"] < fc.FILTER_LATENCY_BUDGET_MS and rtf < 1.0
    record("L2-2", "L2 集成", "实时预算：单块 < 100 ms、RTF < 1", "hard", ok,
           f"块均 {stats['block_ms']:.1f} ms｜RTF {rtf:.2f}｜fail-open {stats['fail_open']}"
           f"｜塌陷 {stats['over_suppression']}")


def check_detector_survives(rows: list[dict], n: int = 10) -> None:
    """降噪之后，4 类检测器还听得见警笛吗（「降噪把危险声吃掉」是产品级致命失效）。"""
    head4 = vocab.load_aed(config.MODEL_DIR / "dscnn_v2_4cls_fft1024.pt", device="cpu")
    mel_tf = head4.mel_transform()
    siren_i = head4.classes.index("siren")
    att = SpectralGateAttenuator()

    def p_siren(y: np.ndarray) -> float:
        w = y[: config.SEGMENT_SAMPLES].astype(np.float32)
        w = (w - w.mean()) / (w.std() + 1e-8)
        with torch.no_grad():
            mel = torch.log(mel_tf(torch.from_numpy(w)) + 1e-9)[None, None, :, :head4.mel["frames"]]
            return float(F.softmax(head4.model(mel), dim=1)[0, siren_i])

    picked = [r for r in rows if r["interferer_class"] == "siren"][:n]
    drops, kept, drops_ener = [], 0, []
    for r in picked:
        mix = load_wav(r["mix_path"])
        out, _ = att.process(mix)
        a, b = p_siren(mix), p_siren(out)
        drops.append(a - b)
        kept += int(a >= 0.5 and b >= 0.5)
        drops_ener.append(rms_db(out) - rms_db(mix))
    mean_drop = float(np.mean(drops)) if drops else 0.0
    worst = float(np.max(drops)) if drops else 0.0
    ok = mean_drop <= 0.10 and worst <= 0.30
    record("L2-3", "L2 集成", "降噪后 siren 检测分数不被吃掉", "hard", ok,
           f"n={len(picked)}｜p(siren) 平均降 {mean_drop:+.3f}、最坏 {worst:+.3f}｜"
           f"双侧≥0.5 保持 {kept}/{len(picked)}｜能量 {np.mean(drops_ener):+.2f} dB",
           "危险声被降噪误伤属于致命失效，这条卡的是它")


def check_efficacy_and_boundary(att: SpectralGateAttenuator, rows: list[dict]) -> None:
    """能力层：真的在抑制（hard 门槛），但选择性 gap 只报告不过门（已知负结论）。"""
    from eval_filter import suppress_gain
    supp, pres = [], []
    for r in rows:
        mix = load_wav(r["mix_path"])
        mask = att.mask_from(mix)
        supp.append(suppress_gain(load_wav(r["target_path"]), mix, mask))
        pres.append(suppress_gain(load_wav(r["interferer_path"]), mix, mask))
    supp_a, pres_a = np.array(supp), np.array(pres)
    redline = 100 * float(np.mean(pres_a < fc.HARD_GATE_TOLERANCE_DB))
    ok = float(supp_a.mean()) < -1.0
    record("L2-4", "L2 集成", "真实混合集：抑制确实发生（能力下限）", "hard", ok,
           f"n={len(rows)}｜Supp {supp_a.mean():+.2f} dB｜Pres {pres_a.mean():+.2f} dB"
           f"｜gap {supp_a.mean() - pres_a.mean():+.2f} dB｜触发 {100 * float(np.mean(supp_a < -1)):.0f}%")
    record("L2-5", "L2 集成", "白名单红线超标率（能力边界，不过门）", "capability",
           redline == 0.0,
           f"{redline:.0f}% 样本 Pres < {fc.HARD_GATE_TOLERANCE_DB} dB",
           "已定稿的负结论：抑制与保真互相拖累，gap 恒负 → 见 P1-9")


def check_passthrough_mode(att: SpectralGateAttenuator, rows: list[dict]) -> None:
    """默认档不能是「全局直通」：真实混合上必须有一部分样本被真的处理过。"""
    diffs = []
    for r in rows:
        y = load_wav(r["mix_path"])
        out, _ = att.process(y)
        diffs.append(rel_db(out, y))
    touched = sum(1 for d in diffs if d > -40.0)
    ok = touched >= max(1, int(0.3 * len(diffs)))
    record("L2-6", "L2 集成", "默认档不是全局直通（降噪真的在开）", "hard", ok,
           f"{touched}/{len(diffs)} 条被实际处理（其余直通）；差异中位数 {np.median(diffs):.1f} dB",
           "合成信号测这条会误判：AED 对合成音色置信度天然低 → 全走 fail-open。必须用真实混合")


# ---------------------------------------------------------------------------
# 5) 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="真实混合集抽样条数")
    ap.add_argument("--split", default="test")
    ap.add_argument("--md", default=str(fc.FILTER_REPORT_DIR / "filter_functional.md"))
    ap.add_argument("--json", default=str(fc.FILTER_REPORT_DIR / "filter_functional.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 78)
    print("降噪支路功能测试")
    print("=" * 78)
    all_rows = load_mix_rows(args.split)
    rows = all_rows[: max(args.n, 10)]
    att = SpectralGateAttenuator()
    print(f"头: {vocab.describe(att.head)}｜z-score={att.zscore} carveout={att.carveout} "
          f"α={att.strength}；真实样本 {len(rows)} 条\n")

    for fn in (check_identity, check_gain_table_translation, check_mel_to_linear,
               check_fail_open, check_anti_collapse):
        try:
            fn(att)
        except Exception as e:  # 功能测试自己崩了也要留证据
            record(fn.__name__, "?", fn.__name__, "hard", False, f"异常: {e!r}")

    for fn in (check_pipeline_continuity, check_latency_budget):
        try:
            fn()
        except Exception as e:
            record(fn.__name__, "?", fn.__name__, "hard", False, f"异常: {e!r}")

    for fn in (check_zscore_frontend, check_temporal_smoothness):
        try:
            fn(att, rows[0])
        except Exception as e:
            record(fn.__name__, "?", fn.__name__, "hard", False, f"异常: {e!r}")

    try:
        check_passthrough_mode(att, rows)
    except Exception as e:
        record("L2-6", "L2 集成", "默认档不是全局直通", "hard", False, f"异常: {e!r}")
    try:
        check_carveout_floor(att, rows)
    except Exception as e:
        record("L1-3", "L1 行为", "carveout 底线", "hard", False, f"异常: {e!r}")
    try:
        check_detector_survives(all_rows)
    except Exception as e:
        record("L2-3", "L2 集成", "降噪后 siren 检测", "hard", False, f"异常: {e!r}")
    try:
        check_efficacy_and_boundary(att, rows)
    except Exception as e:
        record("L2-4", "L2 集成", "真实混合集抑制", "hard", False, f"异常: {e!r}")

    hard = [r for r in RESULTS if r["severity"] == "hard"]
    soft = [r for r in RESULTS if r["severity"] != "hard"]
    fails = [r for r in hard if not r["pass"]]
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 78)
    print(f"hard {len(hard) - len(fails)}/{len(hard)} 通过｜capability 记录 {len(soft)} 项"
          f"｜耗时 {elapsed:.1f}s")
    if fails:
        print("失败项：")
        for r in fails:
            print(f"  - {r['id']} {r['name']}：{r['measured']}")
    print("=" * 78)

    payload = {"n": len(rows), "split": args.split,
               "head": vocab.describe(att.head),
               "config": {"zscore": att.zscore, "carveout": att.carveout,
                          "strength": att.strength, "profile": att.profile_name,
                          "fail_open_conf": __import__("filter_pipeline").FAIL_OPEN_CONF,
                          "anti_collapse_db": fc.ANTI_COLLAPSE_DB},
               "hard_pass": len(hard) - len(fails), "hard_total": len(hard),
               "elapsed_s": round(elapsed, 1), "results": RESULTS}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 降噪支路功能测试报告", "",
             f"- 头：`{vocab.describe(att.head)}`；档位 `{att.profile_name}`"
             f"；z-score={att.zscore}；carveout={att.carveout}；α={att.strength}",
             f"- 真实混合样本：{len(rows)} 条（`{args.split}` split）",
             f"- 结论：**hard {len(hard) - len(fails)}/{len(hard)} 通过**"
             f"，capability 记录 {len(soft)} 项，耗时 {elapsed:.1f} s",
             "",
             "判据分两档：`hard` = 实现契约（挂了就是 bug）；"
             "`capability` = 已知能力边界（只测不过门，避免把负结论刷成绿）。",
             "",
             "| ID | 层 | 检查项 | 实测 | 结果 |",
             "|:--|:--|:--|:--|:--|"]
    for r in RESULTS:
        res = "✅" if r["pass"] else ("❌" if r["severity"] == "hard" else "⚠️")
        lines.append(f"| {r['id']} | {r['level']} | {r['name']} | "
                     f"{r['measured']} | {res} |")
    notes = [r for r in RESULTS if r.get("note")]
    if notes:
        lines += ["", "## 每条为什么这么测", ""]
        for r in notes:
            lines.append(f"- **{r['id']} {r['name']}** — {r['note']}")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("产物：", args.json)
    print("     ", args.md)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
