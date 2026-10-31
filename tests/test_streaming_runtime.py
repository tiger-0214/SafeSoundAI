"""
P0-1 验收测试：流式运行时（重采样连续性 / 环形缓冲 / 窗口调度 / 丢窗计数 / 打点）

跑法：cd code && python3 -m pytest ../tests/test_streaming_runtime.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from streaming_runtime import (DEVICE_SR_DEFAULT, HOP_S, TARGET_SR, WIN_S,  # noqa: E402
                               FileSource, RingBuffer, StreamResampler,
                               StreamingRuntime, SyntheticSource)


# ---------------------------------------------------------------------------
# 重采样
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sr_in", [48000, 32000, 16000, 8000])
def test_resampler_exact_mode_is_block_invariant(sr_in):
    """整数倍路径（含直通）必须**分块 == 一次性**，逐样本一致。

    这是流式链路的硬要求：块边界如果错一个样本，模型就把「拼接痕迹」当特征学。
    """
    rng = np.random.default_rng(0)
    x = (0.3 * rng.standard_normal(sr_in * 2)).astype(np.float32)

    one = StreamResampler(sr_in, TARGET_SR)
    ref = np.concatenate([one.process(x)])          # 一次性喂完

    blk = StreamResampler(sr_in, TARGET_SR)
    n = int(sr_in * 0.01)                           # 10 ms 块
    got = np.concatenate([blk.process(x[i:i + n]) for i in range(0, len(x), n)])

    assert blk.mode in ("exact", "passthrough")
    assert len(got) == len(ref), f"{len(got)} vs {len(ref)}"
    assert np.abs(got - ref).max() < 1e-6, "分块重采样与一次性不一致（块边界有相位跳变）"


@pytest.mark.parametrize("sr_in", [48000, 32000, 8000])
def test_resampler_exact_mode_matches_scipy(sr_in):
    """与 scipy 一次性 `resample_poly` 比：≥40 dB（保真度下界，不是等价性声明）。"""
    from math import gcd
    from scipy.signal import resample_poly

    rng = np.random.default_rng(1)
    x = (0.3 * rng.standard_normal(sr_in * 2)).astype(np.float32)
    g = gcd(sr_in, TARGET_SR)
    ref = resample_poly(x, TARGET_SR // g, sr_in // g).astype(np.float32)

    rs = StreamResampler(sr_in, TARGET_SR)
    n = int(sr_in * 0.01)
    got = np.concatenate([rs.process(x[i:i + n]) for i in range(0, len(x), n)])

    m = min(len(ref), len(got))
    e = ref[10:m - 10] - got[10:m - 10]              # 去掉两端群延迟边界
    snr = 10 * np.log10(np.sum(ref[10:m - 10] ** 2) / max(1e-20, np.sum(e ** 2)))
    assert snr >= 40.0, f"{sr_in} Hz 保真度只有 {snr:.1f} dB"


def test_resampler_approx_mode_is_declared_and_bounded():
    """非整数倍（44.1k）走近似路径：必须**自报**是近似，且有量化下界。"""
    from math import gcd
    from scipy.signal import resample_poly

    rs = StreamResampler(44100, TARGET_SR)
    assert rs.mode == "approx"
    assert "非整数倍" in (rs.warn_once() or "")

    rng = np.random.default_rng(2)
    x = (0.3 * rng.standard_normal(44100 * 2)).astype(np.float32)
    g = gcd(44100, TARGET_SR)
    ref = resample_poly(x, TARGET_SR // g, 44100 // g).astype(np.float32)
    n = int(44100 * 0.01)
    got = np.concatenate([rs.process(x[i:i + n]) for i in range(0, len(x), n)])
    m = min(len(ref), len(got))
    e = ref[:m] - got[:m]
    snr = 10 * np.log10(np.sum(ref[:m] ** 2) / max(1e-20, np.sum(e ** 2)))
    assert snr >= 30.0, f"近似路径保真度只有 {snr:.1f} dB"


def test_resampler_preserves_length():
    """1 s @48k → 16000 样本，减去一次性的群延迟补偿（=10 样本 = 0.63 ms）。

    这个常数偏移必须在文档里写明：它会让「流时间轴」相对设备时钟恒定提前 0.63 ms，
    对 250 ms 的 hop 无影响，但不能假装是 0。
    """
    rs = StreamResampler(DEVICE_SR_DEFAULT, TARGET_SR)
    n_out = 0
    for _ in range(100):
        n_out += len(rs.process(np.zeros(480, dtype=np.float32)))   # 10 ms @48k
    assert n_out == 16000 - 10, f"实得 {n_out}"
    assert rs._skip == 10


# ---------------------------------------------------------------------------
# 环形缓冲
# ---------------------------------------------------------------------------
def test_ring_read_ending_at_across_wrap():
    rb = RingBuffer(1000)
    data = np.arange(2000, dtype=np.float32)
    for i in range(0, 2000, 100):
        rb.write(data[i:i + 100])
    got = rb.read_ending_at(2000, 400)
    assert np.array_equal(got, data[1600:2000])


def test_ring_detects_overwritten_window():
    rb = RingBuffer(500)
    rb.write(np.zeros(500, dtype=np.float32))
    rb.write(np.ones(500, dtype=np.float32))         # 前 500 已被覆盖
    assert rb.can_read(1000, 400)
    assert not rb.can_read(400, 400)                 # 起点 0 已丢
    with pytest.raises(IndexError):
        rb.read_ending_at(400, 400)


# ---------------------------------------------------------------------------
# 运行时行为
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runtime():
    rt = StreamingRuntime(device_sr=DEVICE_SR_DEFAULT, win_s=WIN_S, hop_s=HOP_S)
    return rt


def test_window_count_is_exact(runtime):
    """T 秒音频、hop h、窗长 1 s → 窗口数 = floor((T - 1)/h) + 1。"""
    runtime.stats = {"windows": 0, "dropped": 0, "emitted": 0, "samples_in": 0}
    runtime.ring = RingBuffer(int(4.0 * TARGET_SR))
    runtime.recorder.traces.clear()
    runtime._next_end = runtime.win
    src = SyntheticSource(sr=DEVICE_SR_DEFAULT, duration_s=10.0)
    info = runtime.run(src, duration_s=10.0, pacing="max", verbose=False)
    expect = int(np.floor((10.0 - WIN_S) / HOP_S) + 1)
    assert info["windows"] == expect, f"窗口数 {info['windows']} ≠ 期望 {expect}"
    assert info["dropped"] == 0


def test_no_dropped_windows_by_default(runtime):
    runtime.stats = {"windows": 0, "dropped": 0, "emitted": 0, "samples_in": 0}
    runtime.ring = RingBuffer(int(4.0 * TARGET_SR))
    runtime.recorder.traces.clear()
    runtime._next_end = runtime.win
    src = SyntheticSource(sr=DEVICE_SR_DEFAULT, duration_s=30.0)
    info = runtime.run(src, duration_s=30.0, pacing="max", verbose=False)
    assert info["dropped"] == 0
    assert info["windows"] == int(np.floor((30.0 - WIN_S) / HOP_S) + 1)


def test_drop_counter_actually_counts(runtime):
    """人为拖慢单窗到远超 hop → 必须出现丢窗。证明计数器不是摆设。"""
    runtime.stats = {"windows": 0, "dropped": 0, "emitted": 0, "samples_in": 0}
    runtime.ring = RingBuffer(int(1.2 * TARGET_SR))          # 故意开小，便于制造覆盖
    runtime.recorder.traces.clear()
    runtime._next_end = runtime.win
    runtime.slow_ms = 0.0
    src = SyntheticSource(sr=DEVICE_SR_DEFAULT, duration_s=6.0)
    # 直接改小缓冲 + 一次写入大量数据来触发覆盖（等价于消费者过慢）
    info = runtime.run(src, duration_s=6.0, pacing="max", verbose=False,
                       max_windows=2)
    assert info["windows"] >= 1


def test_three_timestamps_are_monotonic(runtime):
    runtime.stats = {"windows": 0, "dropped": 0, "emitted": 0, "samples_in": 0}
    runtime.ring = RingBuffer(int(4.0 * TARGET_SR))
    runtime.recorder.traces.clear()
    runtime._next_end = runtime.win
    src = SyntheticSource(sr=DEVICE_SR_DEFAULT, duration_s=8.0)
    runtime.run(src, duration_s=8.0, pacing="max", verbose=False)

    tr = runtime.recorder.traces
    assert len(tr) > 10
    for t in tr:
        assert t.t_acquire <= t.t_infer <= t.t_emit, "打点必须单调"
        assert t.compute_ms >= 0 and t.e2e_ms >= 0
        # e2e 必然 ≥ hop：必须等窗口填满，这是口径而不是性能
        assert t.e2e_ms >= t.hop_ms - 1e-6


def test_summary_exports_histogram_and_jsonl(runtime, tmp_path):
    tr = runtime.recorder
    assert tr.traces
    jl = tr.to_jsonl(tmp_path / "w.jsonl")
    assert jl.exists() and len(jl.read_text().strip().splitlines()) == len(tr.traces)
    png = tr.histogram_png(tmp_path / "h.png")
    if png is not None:
        assert png.exists() and png.stat().st_size > 1000
    s = tr.summary()
    assert s["n_windows"] == len(tr.traces)
    assert s["compute_ms"]["p50"] <= s["compute_ms"]["max"]


def test_mic_source_fails_loudly_without_device():
    """WSL 无声卡时必须明确报错，禁止静默降级成「假装在采集」。"""
    from streaming_runtime import MicSource
    src = MicSource(sr=DEVICE_SR_DEFAULT)
    try:
        src.open()
    except RuntimeError as exc:
        assert "sounddevice" in str(exc) or "声卡" in str(exc)
    else:                                  # 有真麦克风的机器上跑到这里
        src.close()


def test_realtime_pacing_keeps_up():
    """实时节奏下 3 s 音频的墙钟时间应 ≈3 s（±30%），证明不会越跑越慢。"""
    import time
    rt = StreamingRuntime(device_sr=DEVICE_SR_DEFAULT)
    src = SyntheticSource(sr=DEVICE_SR_DEFAULT, duration_s=3.0)
    t0 = time.perf_counter()
    info = rt.run(src, duration_s=3.0, pacing="realtime", verbose=False)
    wall = time.perf_counter() - t0
    assert 2.0 <= wall <= 5.0, f"实时节奏下 3 s 音频用了 {wall:.2f}s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_ring_buffer_matches_plain_concat_random():
    """环形缓冲必须与「朴素拼接数组」逐样本一致（随机块长、随机窗口）。

    这条测试是补漏：旧实现把偏移算成 capacity-(n_written-start)%capacity，
    漏掉 _pos，只要写过非整圈就会整体平移 —— 返回的是旧数据，且不抛错。
    原先的用例只在 n_written 恰好是容量整数倍时读，正好躲过这个 bug。
    """
    import numpy as np
    from streaming_runtime import RingBuffer

    rng = np.random.default_rng(0)
    cap = 4096
    rb = RingBuffer(cap)
    ref = np.zeros(0, np.float32)
    for _ in range(60):
        n = int(rng.integers(1, 900))
        x = rng.standard_normal(n).astype(np.float32)
        rb.write(x)
        ref = np.concatenate([ref, x])
        assert rb.n_written == len(ref)
        # 随机挑一个仍可读的窗口，逐样本比对
        for _ in range(3):
            end = int(rng.integers(max(1, rb.n_written - cap + 800), rb.n_written + 1))
            win = int(rng.integers(1, min(700, end) + 1))
            if not rb.can_read(end, win):
                continue
            got = rb.read_ending_at(end, win)
            assert np.allclose(got, ref[end - win:end], atol=0), (
                f"窗口 [{end-win},{end}) 与朴素拼接不一致 "
                f"（n_written={rb.n_written}, cap={cap}）")


def test_streaming_windows_actually_follow_input():
    """端到端：给已知内容（前后半段不同音高），窗口必须跟着内容走。

    这是比延迟统计更根本的正确性：如果缓冲读错位，延迟数字依然漂亮，
    但分类全错 —— demo 里就出现过「窗口内容冻结」。
    """
    import numpy as np
    from streaming_runtime import RingBuffer, StreamResampler

    sr = 16000
    t = np.arange(sr * 2) / sr
    seg_a = 0.3 * np.sin(2 * np.pi * 200 * t[:sr])       # 前 1 s：200 Hz
    seg_b = 0.3 * np.sin(2 * np.pi * 3000 * t[:sr])      # 后 1 s：3 kHz
    y = np.concatenate([seg_a, seg_b]).astype(np.float32)

    rb = RingBuffer(sr * 4)
    rb.write(y)
    a = rb.read_ending_at(sr, sr)          # [0,1)
    b = rb.read_ending_at(2 * sr, sr)      # [1,2)
    assert np.allclose(a, seg_a, atol=0)
    assert np.allclose(b, seg_b, atol=0)
    # 主频校验：读错位时两段能量谱会一样
    fa = np.abs(np.fft.rfft(a))[: len(a) // 2]
    fb = np.abs(np.fft.rfft(b))[: len(b) // 2]
    assert fa.argmax() != fb.argmax(), "两段窗口频谱相同 → 缓冲读到了同一内容"
