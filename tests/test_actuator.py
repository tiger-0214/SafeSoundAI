"""
P0-3 验收测试：执行器后端抽象 + 9 码表

跑法：cd code && python3 -m pytest ../tests/test_actuator.py -q
（或 python3 ../tests/test_actuator.py 直接跑）
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import actuator as A                                      # noqa: E402
import haptic_encoder as he                                # noqa: E402


# ---------------------------------------------------------------------------
# 码表
# ---------------------------------------------------------------------------
def test_code_space_is_9():
    table = A.code_table()
    assert len(table) == 9
    assert set(table) == {f"{c}:{u}" for c in he.V3_CLASSES for u in he.URGENCY_LEVELS}


def test_code_ids_are_stable_and_unique():
    ids = sorted(A.CODE_ID.values())
    assert ids == list(range(9)), "code_id 进固件即固化，必须从 0 连续且唯一"


def test_amplitude_comes_from_encoder():
    """幅度只由紧迫度决定，且必须与 haptic_encoder 完全一致（禁止第二处定义）。"""
    for code, spec in A.code_table().items():
        assert spec.amplitude == pytest.approx(he.URGENCY_PWM[spec.urgency])
    assert he.URGENCY_PWM["critical"] / he.URGENCY_PWM["alert"] >= 2.0   # ≥6 dB 跨度


def test_unit_length_matches_render_envelope():
    """CodeSpec.unit_ms 必须等于 haptic_render.unit_envelope 的实际长度，否则后端会对不齐。"""
    from haptic_render import SR, unit_envelope
    for cls in he.V3_CLASSES:
        spec = A.code_spec(cls, "alert")
        assert spec.unit_ms == pytest.approx(len(unit_envelope(cls)) * 1000 / SR, abs=0.1)


def test_pattern_type_is_the_class_axis():
    """类别轴 = 时间形状；三类必须互不相同，且紧迫度不改变形状。"""
    shapes = {cls: A.code_spec(cls, "alert").pattern_type for cls in he.V3_CLASSES}
    assert len(set(shapes.values())) == 3, f"三类时间形状不能重复：{shapes}"
    for cls in he.V3_CLASSES:
        assert {A.code_spec(cls, u).pattern_type for u in he.URGENCY_LEVELS} == {shapes[cls]}


def test_dual_keeps_intra_unit_gap():
    """car_horn 的双响内部静默 80 ms 必须保留（丢了就变成 300 ms 长鸣）。"""
    spec = A.code_spec("car_horn", "high")
    assert spec.on_ms == (150, 150)
    assert spec.gaps_ms == (80,)
    assert spec.unit_ms == 380
    assert spec.interval_ms == 700


def test_invalid_combination_raises():
    with pytest.raises(KeyError):
        A.code_spec("dog_bark", "alert")
    with pytest.raises(KeyError):
        A.code_spec("siren", "极端")


# ---------------------------------------------------------------------------
# 对齐验收：同一事件流喂三种后端，Δt ≤ 5 ms
# ---------------------------------------------------------------------------
ODD_TIMES = [0.5137, 1.2341, 2.0777, 3.3331, 4.9003, 6.1049, 7.7773, 8.4517, 9.1357]


def _timeline():
    pats = [he.generate_vibration(c, urgency=u, scheme="ear")
            for c in he.V3_CLASSES for u in he.URGENCY_LEVELS]
    return list(zip(ODD_TIMES, pats))


@pytest.mark.parametrize("kind,kwargs", [
    ("null", {}),
    ("wavefile", {"duration_s": 20.0}),
    ("ble_stub", {}),
    ("android", {}),
])
def test_alignment_within_5ms(tmp_path, kind, kwargs):
    if kind == "wavefile":
        kwargs = dict(kwargs, path=tmp_path / "hap.wav")
    elif kind in ("ble_stub", "android"):
        kwargs = dict(kwargs, log_path=tmp_path / f"{kind}.jsonl")

    b = A.build_backend(kind, **kwargs).open()
    for t, pat in _timeline():
        b.emit(A.HapticCommand.from_pattern(pat, t_sched=t))
    b.close()

    rep = b.latency_report()
    assert rep["n"] == 9, f"{kind} 回执条数应为 9（静默也要回执）"
    assert rep["align_err_ms"]["abs_max"] <= 5.0, f"{kind} 对齐超标：{rep}"


def test_wavefile_onset_lands_on_right_sample(tmp_path):
    """波形后端的 t_emit 必须是首样本时刻，而不是调度时刻（否则对齐是自证）。"""
    out = tmp_path / "one.wav"
    b = A.build_backend("wavefile", path=out, duration_s=3.0).open()
    pat = he.generate_vibration("siren", urgency="critical", scheme="ear")
    rec = b.emit(A.HapticCommand.from_pattern(pat, t_sched=1.0137))
    b.close()

    i0 = rec.detail["i0"]
    assert i0 == round(1.0137 * 16000)
    assert abs(rec.t_emit - i0 / 16000) < 1e-9
    assert abs(rec.align_err_ms) <= 1000 / 16000 / 2 + 1e-6     # ≤ 半个采样周期

    import soundfile as sf
    from haptic_render import render_code

    y, sr = sf.read(out, dtype="float32")
    assert sr == 16000

    # (a) 起振点之前必须**严格静音** —— 这是「没有提前下发」的硬证据。
    #     不能用 argmax(|y|>1e-3) 判起振：8 ms 攻击斜坡 × 175 Hz 载波相位
    #     会让首个越阈样本晚 1 ms 左右，那是波形本身，不是放置误差。
    assert float(np.abs(y[:i0]).max(initial=0.0)) < 1e-6

    # (b) 放置精度：最终波形必须逐样本等于「渲染器输出平移到 i0」
    #     （PCM_16 量化误差 1/32768，取 2 LSB 容差）。
    expect = render_code("siren", "critical", dur_s=rec.t_done - rec.t_emit, sr=sr)
    n = min(len(expect), len(y) - i0)
    assert n > 0
    assert np.allclose(y[i0:i0 + n], expect[:n], atol=2 / 32768), "波形未落在 i0"


def test_wavefile_preserves_amplitude_axis(tmp_path):
    """三档幅度不能被归一化抹平（haptic_render 的核心约束，这里回归一次）。"""
    peaks = {}
    for u in ("alert", "high", "critical"):
        b = A.build_backend("wavefile", path=tmp_path / f"{u}.wav", duration_s=2.0).open()
        pat = he.generate_vibration("gun_shot", urgency=u, scheme="ear")
        b.emit(A.HapticCommand.from_pattern(pat, t_sched=0.5))
        peaks[u] = b.close()["peak"]
    assert peaks["critical"] > peaks["high"] > peaks["alert"] > 0.3
    span_db = 20 * np.log10(peaks["critical"] / peaks["alert"])
    assert span_db >= 6.0, f"跨档动态范围只有 {span_db:.1f} dB，幅度轴不可辨"


# ---------------------------------------------------------------------------
# 各后端的协议正确性
# ---------------------------------------------------------------------------
def test_ble_frame_layout(tmp_path):
    b = A.build_backend("ble_stub", log_path=tmp_path / "ble.jsonl").open()
    pat = he.generate_vibration("car_horn", urgency="high", scheme="ear")
    rec = b.emit(A.HapticCommand.from_pattern(pat, t_sched=0.25))
    b.close()

    raw = bytes.fromhex(rec.detail["frame_hex"])
    assert len(raw) == 8 == b.FRAME.size
    code_id, flags, unit_ms, amp100, interval_ms, repeats = b.FRAME.unpack(raw)
    spec = A.code_spec("car_horn", "high")
    assert code_id == A.CODE_ID[spec.code]
    assert unit_ms == spec.unit_ms == 380
    assert interval_ms == spec.interval_ms == 700
    assert amp100 == 70                       # 0.70 × 100
    assert flags == 0x00                      # 非 ramp
    assert repeats == 1

    # 帧必须能塞进默认 ATT MTU 载荷（20 B），否则真机上要分片
    assert len(raw) <= 20


def test_ble_t_air_is_not_in_alignment():
    """设备传输延迟必须单列，不能混进对齐误差（否则调包口径）。"""
    b = A.build_backend("ble_stub", t_air_ms=7.5).open()
    pat = he.generate_vibration("siren", urgency="alert", scheme="ear")
    rec = b.emit(A.HapticCommand.from_pattern(pat, t_sched=1.0))
    assert rec.t_air == pytest.approx(0.0075)
    assert rec.align_err_ms == pytest.approx(0.0)      # 命令入队时刻 = 调度时刻
    b.close()


def test_android_alternates_on_off():
    b = A.build_backend("android").open()
    pat = he.generate_vibration("car_horn", urgency="critical", scheme="ear")
    rec = b.emit(A.HapticCommand.from_pattern(pat, t_sched=0.0))
    b.close()
    timings, amps = rec.detail["timings_ms"], rec.detail["amplitudes"]
    assert len(timings) == len(amps)
    # on/off 严格交替，且以 off 收尾（防止马达长鸣）
    assert amps[-1] == 0
    assert all((amps[i] > 0) != (amps[i + 1] > 0) for i in range(len(amps) - 1))
    assert amps[0] == 255 and timings[0] == 150


def test_silent_code_still_returns_receipt():
    """非编码类（如 dog_bark）必须返回 ok=True 的静默回执，便于对账。"""
    pat = he.generate_vibration("dog_bark", scheme="ear")
    cmd = A.HapticCommand.from_pattern(pat, t_sched=0.0)
    assert cmd.spec.is_silent
    b = A.build_backend("null").open()
    rec = b.emit(cmd)
    b.close()
    assert rec.ok and rec.detail["silent"]
    assert b.latency_report()["n_silent"] == 1


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        A.build_backend("usb")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
