"""降噪支路的功能契约（pytest 门禁用）。

和 `tests/test_filter_semantics.py` 的分工：
  * semantics —— 只查**查表**（类名 → 增益、background R1/R2/R3），不碰信号。
  * 本文件   —— 查**信号链**：增益真的按 dB 施加、映射不重复计数、fail-open /
                anti-collapse / 分块拼接 / 实时预算真的生效。

每个用例背后都对应一条历史上真实发生过的失效：
  * `test_mel_to_linear_is_weighted_average` ← 2026-10-31 mel→线性增益双倍施加
    （mel 域 −12 dB 作用到线性频点变成 −24 dB）；
  * `test_zscore_frontend_matches_training` ← 推理侧漏掉逐窗 z-score；
  * `test_identity_passthrough` ← STFT/iSTFT 链路本身有损。

依赖模型的用例在没有数据集/权重时自动 skip（`make test` 在裸机上也能跑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import config  # noqa: E402
import filter_config as fc  # noqa: E402

SR = config.SAMPLE_RATE


# --------------------------------------------------------------------------- 夹具
@pytest.fixture(scope="module")
def att():
    """10 类过滤头（权重缺失则 skip，不让整条门禁变红）。"""
    ckpt = config.MODEL_DIR / "ds_cnn_best.pth"
    if not Path(ckpt).exists():
        pytest.skip(f"缺少过滤头权重 {ckpt}")
    from attenuator import SpectralGateAttenuator
    return SpectralGateAttenuator()


@pytest.fixture(scope="module")
def mix_row():
    from filter_config import MIX_MANIFEST
    if not Path(MIX_MANIFEST).exists():
        pytest.skip("缺少合成混合集 manifest")
    import csv
    with open(MIX_MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not rows:
        pytest.skip("manifest 无 test 样本")
    return rows[0]


def _tone(f: float = 1000.0, dur: float = 1.5) -> np.ndarray:
    y = np.sin(2 * np.pi * f * np.arange(int(SR * dur)) / SR)
    return (y / np.max(np.abs(y)) * 0.5).astype(np.float32)


def _rel_db(a: np.ndarray, b: np.ndarray) -> float:
    return 10.0 * np.log10(float(np.mean((a - b) ** 2)) /
                           (float(np.mean(b.astype(np.float64) ** 2)) + 1e-12) + 1e-12)


# --------------------------------------------------------------------------- L0
def test_identity_passthrough(att):
    """α=0（全 0 dB）时输出逐样本还原 —— 信号链本身不能有损。"""
    from attenuator import SpectralGateAttenuator
    quiet = SpectralGateAttenuator(strength=0.0)
    y = _tone()
    out, _ = quiet.process(y)
    assert _rel_db(out, y) < -60.0


def test_mel_to_linear_is_weighted_average(att):
    """mel 域常数增益 → 线性频点仍是同一常数（回归：不得被叠加成 2 倍）。

    这条是 2026-10-31 的实现 bug 回归测试。一个线性频点被 2 个 mel 滤波器覆盖，
    只做列归一化会把 −12 dB 打成 −24 dB；正确做法是按频点行归一（加权平均）。
    """
    from attenuator import mel_to_linear_gain
    n_freqs = fc.FILTER_N_FFT // 2 + 1
    g = np.full((config.N_MELS, 1), -12.0, dtype=np.float32)
    lin = mel_to_linear_gain(g, n_freqs=n_freqs)[:, 0]

    import torchaudio
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs, f_min=config.F_MIN, f_max=config.F_MAX, n_mels=config.N_MELS,
        sample_rate=config.SAMPLE_RATE, norm=None, mel_scale="htk").numpy()
    covered = fb.sum(axis=1) > 0
    assert covered.sum() > 0

    # 覆盖区内必须是 −12 dB（不是 −24 dB），容差 0.5 dB
    assert np.abs(lin[covered] + 12.0).max() < 0.5, \
        f"mel 增益被重复计数：覆盖区最大偏差 {np.abs(lin[covered] + 12.0).max():.2f} dB"
    # f_min/f_max 之外没有 mel 覆盖 → 正确行为是 0 dB 直通
    assert np.abs(lin[~covered]).max() < 1e-6


def test_gain_table_matches_head_space(att):
    """类增益必须按名字落到当前头空间，白名单恒 0，频带空间与 STFT 对齐。"""
    prof = fc._PROFILE_NAMES[fc.DEFAULT_PROFILE]["gains"]
    for i, name in enumerate(att.classes):
        assert abs(att.gains.get(i, 0.0) - prof.get(name, 0.0)) < 1e-9, f"{name} 增益错位"
    assert all(abs(att.gains[i]) < 1e-9 for i in att.whitelist_ids)
    assert att.head.mel["n_fft"] == fc.FILTER_N_FFT, "mel 档位与过滤 STFT 不一致"
    assert att.class_band.shape == (att.n_classes, config.N_MELS)


def test_zscore_frontend_matches_training(att, mix_row):
    """推理前端必须与训练口径同样逐窗 z-score（回归：漏归一化会让后验整体错位）。"""
    import soundfile as sf
    import torch.nn.functional as F
    from attenuator import SpectralGateAttenuator

    y, _ = sf.read(mix_row["mix_path"], always_2d=False, dtype="float32")
    y = np.asarray(y, dtype=np.float32)[: config.SEGMENT_SAMPLES]

    probs, starts = att.window_probs(y, hop=SR // 2)
    w = y[starts[0]:starts[0] + SR]
    w = (w - w.mean()) / (w.std() + 1e-8)
    mel = torch.log(att.mel_tf(torch.from_numpy(w).to(att.device)) + 1e-9)
    with torch.no_grad():
        p_manual = F.softmax(att.model(mel[None, None, :, :att.frames]), dim=1).cpu().numpy()[0]
    assert np.abs(p_manual - probs[0]).max() < 1e-4, "推理前端与手算 z-score 口径不一致"

    # 关掉开关必须真的改变结果，否则说明归一化是死代码
    raw = SpectralGateAttenuator(zscore_windows=False)
    probs_raw, _ = raw.window_probs(y, hop=SR // 2)
    assert np.abs(probs[0] - probs_raw[0]).max() > 0.02, "z-score 开关没有生效"


# --------------------------------------------------------------------------- L1
def test_low_confidence_goes_fail_closed(att):
    """所有类后验都低于 apply_conf → **不再直通**，按语音门控 fallback 衰减（V2）。

    旧版这里是 fail-open 0 dB（"不猜不压"）。设计改成「除人声外全压」后，
    认不出的声音必须也压 —— 否则脚步/键盘/鸟叫/交通全部原样进耳朵，
    降噪等于没做。700 Hz 纯音会被 Silero VAD 判为非语音，因此吃满 fallback。
    """
    y = _tone(700.0, 1.0)
    orig = att._resolve_probs
    try:
        att._resolve_probs = lambda yy, oracle=None: np.full(
            (1 + len(yy) // fc.FILTER_HOP, att.n_classes), 1.0 / att.n_classes,
            dtype=np.float32)
        out, info = att.process(y)
    finally:
        att._resolve_probs = orig
    assert info["energy_delta_db"] < -6.0, \
        f"低置信帧仍被直通（V2 未生效）：Δ={info['energy_delta_db']} dB"
    assert att.stats["gate_fallback_frames"] > 0, "没有帧走 gate fallback 分支"


def test_anti_collapse_guard(att):
    """掩码压塌整段（< ANTI_COLLAPSE_DB）时必须原样放行并计数。"""
    import attenuator as A
    y = _tone(1000.0, 1.0)
    before = att.stats["over_suppression_events"]
    orig = A.mel_to_linear_gain
    try:
        A.mel_to_linear_gain = lambda g, n_freqs=fc.FILTER_N_FFT // 2 + 1: np.full(
            (n_freqs, np.asarray(g).shape[1]), -50.0, dtype=np.float32)
        out, info = att.process(y, oracle_class=config.CLASS_TO_IDX["air_conditioner"])
    finally:
        A.mel_to_linear_gain = orig
    assert _rel_db(out, y) < -60.0
    assert att.stats["over_suppression_events"] > before
    assert info["applied"] is False


def test_carveout_floor_respected(att, mix_row):
    """carveout 让位区（protect ≥ 0.5）的衰减不得深于 CARVEOUT_GUARD_DB。"""
    import soundfile as sf
    y, _ = sf.read(mix_row["mix_path"], always_2d=False, dtype="float32")
    y = np.asarray(y, dtype=np.float32)[: config.SEGMENT_SAMPLES]
    g_db = att.gain_curve_db(att.frame_probs(y))

    protect = np.zeros_like(g_db)
    for cid in att.whitelist_ids:
        protect = np.maximum(protect, att.class_band[cid][:, None] * att.frame_probs(y)[:, cid][None, :])
    guard = protect >= 0.5
    if guard.any():
        assert g_db[guard].min() >= fc.CARVEOUT_GUARD_DB - 0.01


def test_frame_gain_jump_limited(att, mix_row):
    """帧间增益跳变 ≤ MAX_DB_JUMP_PER_FRAME（避免调制噪声）。"""
    import soundfile as sf
    y, _ = sf.read(mix_row["mix_path"], always_2d=False, dtype="float32")
    y = np.asarray(y, dtype=np.float32)[: config.SEGMENT_SAMPLES]
    g = att.gain_curve_db(att.frame_probs(y))
    if g.shape[1] > 1:
        assert np.abs(np.diff(g, axis=1)).max() <= fc.MAX_DB_JUMP_PER_FRAME + 1e-3


# --------------------------------------------------------------------------- L2
def test_pipeline_block_continuity_and_budget():
    """分块流式：长度不变、无 NaN、块边界不比块内更跳、单块耗时在预算内。"""
    from filter_pipeline import FilterPipeline

    t = np.arange(int(SR * 1.5)) / SR
    y = (0.4 * np.sin(2 * np.pi * 900 * t) + 0.3 * np.sin(2 * np.pi * 120 * t))
    y = (y / np.max(np.abs(y)) * 0.5).astype(np.float32)

    pipe = FilterPipeline(use_tse=False)
    out, stats = pipe.process_offline(y)

    assert len(out) == len(y)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 2 * np.max(np.abs(y)) + 1e-6

    block = 1600
    boundary = [abs(float(out[k] - out[k - 1])) for k in range(block, len(y) - block, block)]
    diffs = np.abs(np.diff(out))
    interior = diffs[len(diffs) // 10: -len(diffs) // 10] if len(diffs) > 20 else diffs
    p999 = float(np.percentile(interior, 99.9)) if len(interior) else 0.0
    if boundary:
        assert max(boundary) <= max(3 * p999, 1e-3), "块边界出现爆音"

    assert stats["block_ms"] < fc.FILTER_LATENCY_BUDGET_MS, \
        f"单块 {stats['block_ms']:.1f} ms 超出 {fc.FILTER_LATENCY_BUDGET_MS} ms 预算"
