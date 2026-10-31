"""
SafeSoundAI — 流式运行时（streaming_runtime.py, P0-1）

【解决什么问题】
此前的 `realtime_demo.py` 只是「把 wav 按 1 s 窗切片、循环推理」——没有采集、没有环形
缓冲、没有窗口调度、没有时间戳，全仓 `threading / Queue / sounddevice` 零命中。
所以「延迟」只能靠离线跑一遍窗数 × 单窗耗时估出来，那不叫实测。

本模块补齐真实链路：

    设备(48 kHz) → 分块读取 → 重采样(→16 kHz) → 环形缓冲
                 → 窗口调度(hop=250 ms) → 电平/z-score → mel → AED → 仲裁
                 → 触觉调度(去重/抢占) → 执行器后端

【三处打点（P0-1 的硬指标）】
每个窗口记录三个时刻，全部来自 `time.perf_counter()`（单调时钟，不受系统时间调整影响）：

    t_acquire  环形缓冲攒满该窗样本的时刻（此后才开始算）
    t_infer    推理 + 仲裁完成的时刻
    t_emit     触觉命令交给后端的时刻

由此得到两条可报的量：
    compute_ms  = t_infer - t_acquire          单窗算力（与设备无关，可复现）
    e2e_ms      = t_emit  - t_acquire + hop_ms 端到端（含等待窗口填满，随 hop 变化）

⚠ 口径：`e2e_ms` 永远 ≥ hop_ms，因为必须等窗口填满。所以**报延迟必须同时给 hop**，
  只报 e2e 会把「攒窗时间」误当成「算法慢」。这也是为什么 <50 ms 在 1 s 窗下不可达。

【丢窗判定】
环形缓冲容量固定；若某窗的起始样本已被新数据覆盖 → 记 `dropped`，并在 `self.stats`
里累加。验收要求：正常模式下 dropped == 0。--simulate-slow-infer 可以人为制造丢窗，
用来证明「这个计数器不是摆设」。

用法：
    # 实时（有麦克风；WSL 无声卡时会明确报错而不是静默降级）
    python3 streaming_runtime.py --source mic --duration 60 --backend ble_stub

    # 无人值守跑 30 min 材料（尽量快，不按实时节奏）
    python3 streaming_runtime.py --source file --audio scene.wav --duration 1800 \
        --pacing max --backend wavefile --out-wav /tmp/hap.wav

    # 无音频文件时用合成场景自检
    python3 streaming_runtime.py --source synth --duration 120 --pacing max
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import config
import haptic_encoder as he
from actuator import HapticCommand, build_backend
from haptic_scheduler import HapticScheduler, SchedulePolicy

TARGET_SR = config.SAMPLE_RATE            # 16000
DEVICE_SR_DEFAULT = 48000                 # 桌面/手机常见采集率
CHUNK_MS = 10                             # 10 ms 一块：与 WebRTC/Android AudioRecord 的常用块一致
WIN_S = config.SEGMENT_S                  # 1.0 s（P0-2 要把它降到 0.5 s）
HOP_S = 0.25                              # 50% overlap 的 1/4 步进，沿用离线协议


# ---------------------------------------------------------------------------
# 重采样（流式，带状态）
# ---------------------------------------------------------------------------
class StreamResampler:
    """流式重采样：整数倍路径**逐样本精确**，非整数倍路径有量化上限并显式告警。

    三条路径（按可靠性排序，部署时优先走第一条）：

    1. `passthrough`  设备原生 16 kHz。零误差零开销。**应当优先向设备要 16 kHz**
      （Android AudioRecord / WASAPI / CoreAudio 都支持），而不是自己硬转。
    2. `exact`        整数倍升降采样（48k→16k、32k→16k、16k→48k…）。
      用「上采样（插零）→ FIR `lfilter`（带 zi 状态）→ 抽取（带相位计数）」实现，
      状态跨块保持，因此**分块结果与一次性处理逐样本相同**（误差 ~1e-7，浮点级）。
      绝大多数采集设备落在这一条。
    3. `approx`       非整数倍（如 44.1k→16k）。用 overlap-save + `resample_poly`，
      块边界受其内部零填充影响，实测 SNR ≈ 35 dB（白噪声输入），
      **不是逐样本相等**。仅用于离线素材，且 `warn_once()` 会明确记录。

    这个分级是刻意的：与其声称「统一高精度流式重采样」，不如把「哪条路径精确、
    哪条不精确」写清楚——否则延迟/保真度的结论会不可复现。
    """

    def __init__(self, sr_in: int, sr_out: int = TARGET_SR):
        from math import gcd
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.passthrough = (self.sr_in == self.sr_out)
        self.mode = "passthrough"
        self.quality_snr_db: float | None = None
        self._warned = False

        if self.passthrough:
            return
        g = gcd(self.sr_in, self.sr_out)
        self.up, self.down = self.sr_out // g, self.sr_in // g

        if self.up == 1 or self.down == 1:                 # 整数倍：可精确
            self.mode = "exact"
            self._l = self.up
            self._m = self.down
            n_taps = 2 * 10 * max(self.up, self.down) + 1
            cutoff = 1.0 / max(self.up, self.down)
            from scipy.signal import firwin
            self._h = (firwin(n_taps, cutoff) * self._l).astype(np.float64)
            self._zi = np.zeros(n_taps - 1, dtype=np.float64)
            self._phase = 0                                # 抽取相位（0..M-1）
            # FIR 群延迟补偿：`resample_poly` 会切掉开头 half/down 个输出样本，
            # 本实现必须做同样的事，否则输出整体滞后 ~10 个样本
            # （听感是「慢半拍」，但把它喂给模型就是错的输入）。
            self._skip = (n_taps - 1) // 2 // self._m
            self._skip_remaining = self._skip
        else:                                              # 非整数倍：近似
            self.mode = "approx"
            self._hlen = 2 * 10 * max(self.up, self.down) + 1
            need = max(512, self._hlen + 1)
            self.overlap = -(-need // self.down) * self.down
            self._tail = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    def _ratio_out(self, n_in: int) -> int:
        return int(round(n_in * self.sr_out / self.sr_in))

    def warn_once(self) -> str | None:
        if self.mode == "approx" and not self._warned:
            self._warned = True
            return (f"⚠ {self.sr_in}→{self.sr_out} Hz 是非整数倍，走近似重采样"
                    f"（SNR≈35 dB，块边界非逐样本精确）。"
                    f"可能的话请让设备直接采 {self.sr_out} Hz。")
        return None

    # ------------------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.passthrough:
            return x
        if self.mode == "exact":
            return self._process_exact(x)
        return self._process_approx(x)

    def _process_exact(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        xf = x.astype(np.float64)
        if self._l == 1:
            y, self._zi = lfilter(self._h, [1.0], xf, zi=self._zi)
        else:                                              # 插值：先插零再滤
            xu = np.zeros(len(xf) * self._l, dtype=np.float64)
            xu[::self._l] = xf
            y, self._zi = lfilter(self._h, [1.0], xu, zi=self._zi)
        if self._m == 1:
            out = y
        else:
            # 抽取：保持跨块相位连续
            idx = np.arange(self._phase, len(y), self._m)
            self._phase = (self._phase - len(y)) % self._m
            out = y[idx]
        # 群延迟补偿（只在流的最开头丢一次）
        if self._skip_remaining > 0:
            k = min(self._skip_remaining, len(out))
            out = out[k:]
            self._skip_remaining -= k
        return out.astype(np.float32)

    def _process_approx(self, x: np.ndarray) -> np.ndarray:
        if self._tail.size < self.overlap:
            self._tail = np.concatenate(
                [np.zeros(self.overlap - self._tail.size, dtype=np.float32), self._tail])
        from scipy.signal import resample_poly
        n_new = len(x)
        cat = np.concatenate([self._tail, x])
        y = resample_poly(cat, self.up, self.down).astype(np.float32)
        n_skip = self.overlap * self.up // self.down      # overlap 是 down 的整数倍
        y = y[n_skip:]
        want = self._ratio_out(n_new)
        if len(y) > want:
            y = y[:want]
        elif len(y) < want:
            y = np.pad(y, (0, want - len(y)))
        self._tail = cat[-self.overlap:]
        return y

    def reset(self) -> None:
        if self.mode == "exact":
            self._zi[:] = 0.0
            self._phase = 0
            self._skip_remaining = self._skip
        elif self.mode == "approx":
            self._tail = np.zeros(0, dtype=np.float32)

    def info(self) -> dict:
        d = {"sr_in": self.sr_in, "sr_out": self.sr_out, "mode": self.mode}
        if self.mode == "exact":
            d.update({"up": self._l, "down": self._m, "taps": int(len(self._h))})
        elif self.mode == "approx":
            d.update({"up": self.up, "down": self.down, "overlap": self.overlap})
        return d


# ---------------------------------------------------------------------------
# 环形缓冲
# ---------------------------------------------------------------------------
class RingBuffer:
    """定长环形缓冲。`n_written` 记录总写入样本数（单调），窗口定位靠它。"""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0
        self.n_written = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).ravel()
        n = len(x)
        if n >= self.capacity:                       # 超长块：只保留最后 capacity 个
            x = x[-self.capacity:]
            n = self.capacity
            self._buf[:] = x
            self._pos = 0
            self.n_written += n
            return
        end = self._pos + n
        if end <= self.capacity:
            self._buf[self._pos:end] = x
        else:
            head = self.capacity - self._pos
            self._buf[self._pos:] = x[:head]
            self._buf[:end - self.capacity] = x[head:]
        self._pos = end % self.capacity
        self.n_written += n

    def can_read(self, end_index: int, n: int) -> bool:
        """能否取出 [end_index-n, end_index) —— 要求这段仍未被覆盖。"""
        oldest = self.n_written - self.capacity
        return end_index - n >= oldest and end_index <= self.n_written

    def read_ending_at(self, end_index: int, n: int) -> np.ndarray:
        if not self.can_read(end_index, n):
            raise IndexError(f"窗口 [{end_index - n}, {end_index}) 已被覆盖"
                             f"（oldest={self.n_written - self.capacity}）")
        start = end_index - n
        out = np.empty(n, dtype=np.float32)
        # 样本 i 落在 _buf[i % capacity]，因此窗口起点 start 的偏移就是 start % capacity。
        # （旧实现写成 capacity - (n_written-start)%capacity，漏掉了 _pos，读出来的是
        #   整体平移过的旧数据 —— 表现为"窗口内容冻结/全零"，且不会报错，极难发现。）
        off = start % self.capacity
        if off + n <= self.capacity:
            out[:] = self._buf[off:off + n]
        else:
            head = self.capacity - off
            out[:head] = self._buf[off:]
            out[head:] = self._buf[:n - head]
        return out


# ---------------------------------------------------------------------------
# 音源
# ---------------------------------------------------------------------------
class AudioSource(ABC):
    name = "base"

    def __init__(self, sr: int):
        self.sr = int(sr)

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self, n_frames: int) -> np.ndarray | None:
        """返回 None 表示流结束（file 源）；mic/synth 永不返回 None。"""

    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FileSource(AudioSource):
    """文件源。按实时节奏或尽可能快地吐出 10 ms 块。"""

    name = "file"

    def __init__(self, path: str | Path, sr: int = DEVICE_SR_DEFAULT,
                 loop: bool = True, gain: float = 1.0):
        super().__init__(sr)
        self.path = Path(path)
        self.loop = loop
        self.gain = float(gain)
        self._y: np.ndarray | None = None
        self._i = 0

    def open(self) -> None:
        import librosa
        y, sr0 = librosa.load(str(self.path), sr=self.sr, mono=True)
        self._y = (y.astype(np.float32) * self.gain)
        self._i = 0

    def read(self, n_frames: int) -> np.ndarray | None:
        assert self._y is not None
        if self._i >= len(self._y):
            if not self.loop:
                return None
            self._i = 0
        seg = self._y[self._i:self._i + n_frames]
        if len(seg) < n_frames:                       # 末尾补零，保证块长恒定
            seg = np.pad(seg, (0, n_frames - len(seg)))
        self._i += n_frames
        return seg


class MicSource(AudioSource):
    """麦克风源（sounddevice）。WSL/无声卡环境会**明确报错**，不静默降级。"""

    name = "mic"

    def __init__(self, sr: int = DEVICE_SR_DEFAULT, device: int | None = None,
                 blocksize: int | None = None):
        super().__init__(sr)
        self.device = device
        self.blocksize = blocksize or int(sr * CHUNK_MS / 1000)
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._stream = None
        self._buf = np.zeros(0, dtype=np.float32)

    def open(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "sounddevice 不可用（WSL 通常没有声卡）。请改用 "
                "--source file / --source synth；真机部署时再开 mic。"
            ) from exc

        def _cb(indata, frames, t, status):            # noqa: ANN001
            if status:
                pass                                   # 缓冲区溢出等：丢块即可，不崩
            try:
                self._q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      blocksize=self.blocksize, device=self.device,
                                      callback=_cb)
        self._stream.start()

    def read(self, n_frames: int) -> np.ndarray:
        while len(self._buf) < n_frames:
            blk = self._q.get(timeout=2.0)
            self._buf = np.concatenate([self._buf, blk])
        out, self._buf = self._buf[:n_frames], self._buf[n_frames:]
        return out

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SyntheticSource(AudioSource):
    """合成场景源：底噪 + 可配置的警报事件。

    用途：在没有麦克风、也不想依赖真实音频文件时**反复**跑通整条链路与丢窗统计。
    它不做「模型一定识别得出来」的承诺——那要靠真实音频回归（见 P0-2 事件级重测）。
    """

    name = "synth"

    def __init__(self, sr: int = DEVICE_SR_DEFAULT, duration_s: float = 30.0,
                 events: list[tuple[float, str]] | None = None, seed: int = 0):
        super().__init__(sr)
        self.duration_s = float(duration_s)
        self.events = events or [(3.0, "siren"), (8.0, "car_horn"), (15.0, "gun_shot")]
        self.seed = int(seed)
        self._t = 0.0
        self._rng = np.random.default_rng(seed)

    def open(self) -> None:
        self._t = 0.0

    def _render(self, t0: float, n: int) -> np.ndarray:
        sr = self.sr
        t = t0 + np.arange(n) / sr
        y = 0.005 * self._rng.standard_normal(n).astype(np.float32)   # 底噪
        for t_ev, kind in self.events:
            if not (t_ev <= t0 + n / sr and t_ev + 1.5 >= t0):
                continue
            dt = t - t_ev
            m = (dt >= 0) & (dt < 1.5)
            if not m.any():
                continue
            if kind == "siren":                       # 600→1000 Hz 扫频
                f = 600 + 400 * (dt[m] / 1.5)
            elif kind == "car_horn":                  # 400 Hz 双响
                f = 400.0
                m &= (dt < 0.25) | ((dt > 0.4) & (dt < 0.65))
            else:                                     # gun_shot 60 ms 脉冲
                f = 900.0
                m &= dt < 0.06
            y[m] += (0.3 * np.sin(2 * np.pi * f * dt[m])).astype(np.float32)
        return y

    def read(self, n_frames: int) -> np.ndarray:
        y = self._render(self._t, n_frames)
        self._t += n_frames / self.sr
        return y


# ---------------------------------------------------------------------------
# 打点记录
# ---------------------------------------------------------------------------
@dataclass
class WindowTrace:
    seq: int
    win_end_s: float          # 窗口右端在流时间轴上的位置
    t_acquire: float          # perf_counter
    t_infer: float
    t_emit: float
    compute_ms: float
    e2e_ms: float
    hop_ms: float
    cls: str | None
    conf: float
    code: str | None
    urgency: str | None
    level_db: float
    accepted: bool
    reject_reason: str | None
    preempted: str | None
    dropped: bool = False
    note: str = ""


class LatencyRecorder:
    """窗口级延迟记录 + 直方图导出（JSON + 可选 PNG）。"""

    def __init__(self) -> None:
        self.traces: list[WindowTrace] = []

    def add(self, tr: WindowTrace) -> None:
        self.traces.append(tr)

    # -- 统计 ------------------------------------------------------------
    def summary(self) -> dict:
        ok = [t for t in self.traces if not t.dropped]
        if not ok:
            return {"n_windows": len(self.traces), "n_dropped": len(self.traces)}
        comp = np.array([t.compute_ms for t in ok])
        e2e = np.array([t.e2e_ms for t in ok])
        acc = [t for t in ok if t.accepted]

        def q(a: np.ndarray) -> dict:
            return {"mean": round(float(a.mean()), 3),
                    "p50": round(float(np.percentile(a, 50)), 3),
                    "p95": round(float(np.percentile(a, 95)), 3),
                    "p99": round(float(np.percentile(a, 99)), 3),
                    "max": round(float(a.max()), 3)}

        return {
            "n_windows": len(self.traces),
            "n_dropped": sum(1 for t in self.traces if t.dropped),
            "hop_ms": round(float(np.mean([t.hop_ms for t in ok])), 3),
            "compute_ms": q(comp),
            "e2e_ms": q(e2e),
            "n_accepted": len(acc),
            "n_rejected": len(ok) - len(acc),
            "classes_seen": sorted({t.cls for t in ok if t.cls}),
            "codes_seen": sorted({t.code for t in ok if t.code}),
            "infer_over_hop_pct": round(float(np.mean(comp > np.mean([t.hop_ms for t in ok])) * 100), 2),
        }

    # -- 导出 ------------------------------------------------------------
    def to_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for tr in self.traces:
                f.write(json.dumps(asdict(tr), ensure_ascii=False) + "\n")
        return p

    def histogram_png(self, path: str | Path) -> Path | None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:                                  # pragma: no cover
            return None
        ok = [t for t in self.traces if not t.dropped]
        if not ok:
            return None
        comp = np.array([t.compute_ms for t in ok])
        e2e = np.array([t.e2e_ms for t in ok])
        hop = np.mean([t.hop_ms for t in ok])

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=140)
        axes[0].hist(comp, bins=40, color="#3b6ea5")
        axes[0].axvline(hop, color="#c0392b", ls="--", lw=1.2,
                        label=f"hop = {hop:.0f} ms")
        axes[0].set_title("per-window compute (t_infer - t_acquire)")
        axes[0].set_xlabel("ms")
        axes[0].set_ylabel("windows")
        axes[0].legend(fontsize=7)

        axes[1].hist(e2e, bins=40, color="#5a9e6f")
        axes[1].set_title("end-to-end (t_emit - t_acquire + hop)")
        axes[1].set_xlabel("ms")
        fig.tight_layout()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
        plt.close(fig)
        return p


# ---------------------------------------------------------------------------
# 运行时
# ---------------------------------------------------------------------------
class StreamingRuntime:
    """采集 → 缓冲 → 调度 → 决策 → 触觉 的同步实现（单线程，无锁）。"""

    def __init__(
        self,
        *,
        backend=None,
        policy: SchedulePolicy | None = None,
        ckpt: str | Path | None = None,
        device_sr: int = DEVICE_SR_DEFAULT,
        win_s: float = WIN_S,
        hop_s: float = HOP_S,
        threshold: float = 0.8,
        ring_s: float = 4.0,
        simulate_slow_infer_ms: float = 0.0,
    ):
        self.device_sr = int(device_sr)
        self.win = int(round(win_s * TARGET_SR))
        self.hop = int(round(hop_s * TARGET_SR))
        self.threshold = float(threshold)
        self.backend = backend or build_backend("null")
        self.scheduler = HapticScheduler(self.backend, policy)
        self.resampler = StreamResampler(self.device_sr, TARGET_SR)
        _w = self.resampler.warn_once()
        if _w:
            print(_w, flush=True)
        self.ring = RingBuffer(int(ring_s * TARGET_SR))
        self.recorder = LatencyRecorder()
        self.slow_ms = float(simulate_slow_infer_ms)

        # 复用离线管线的状态机与分类前向，避免两套逻辑漂移
        from dual_path_pipeline import DualPathPipeline
        self.pipe = DualPathPipeline(ckpt=ckpt, threshold=threshold, use_filter=False)

        self._next_end = self.win          # 下一个窗口的右端（16 kHz 样本数）
        self.stats = {"windows": 0, "dropped": 0, "emitted": 0, "samples_in": 0}
        self._t0 = 0.0

    # -- 单窗 ------------------------------------------------------------
    def _process_window(self, end_index: int, t_acquire: float) -> WindowTrace | None:
        seq = self.stats["windows"]
        win = self.ring.read_ending_at(end_index, self.win)
        win_end_s = end_index / TARGET_SR

        from dual_path_pipeline import rms_db
        level_db = rms_db(win)                      # ← 电平必须在 z-score 之前

        probs = self.pipe.classify(win)
        t_infer = time.perf_counter()
        compute_ms = (t_infer - t_acquire) * 1e3
        if self.slow_ms:
            time.sleep(self.slow_ms / 1e3)           # 仅用于演示丢窗计数
            t_infer = time.perf_counter()

        cls = conf = code = urgency = None
        accepted = False
        reason = None
        preempted = None

        if probs is not None:
            from haptic_encoder import Detection
            dets = [Detection(int(i), float(p)) for i, p in enumerate(probs)]
            chosen = self.pipe.engine.arbitrate(dets)
            cls = self.pipe.classes[int(np.argmax(probs))]
            conf = float(np.max(probs))
            if chosen is not None:
                cls = self.pipe.classes[chosen.class_id]
                conf = float(chosen.confidence)
                self.pipe.state.level.push(cls, level_db)
                q = self.pipe.state.level.quantile(cls, level_db)
                dur, reps = self.pipe.state.touch(cls, win_end_s)
                pat = he.generate_vibration(cls, duration_s=dur, repeats_10s=reps,
                                            rel_quantile=q, scheme="ear")
                if pat.pwm_base > 0:
                    code, urgency = pat.code, pat.urgency
                    cmd = HapticCommand.from_pattern(pat, t_sched=win_end_s, source="stream")
                    dec = self.scheduler.propose(cmd, now=win_end_s)
                    accepted, reason, preempted = dec.accepted, dec.reason.value, dec.preempted
                    if accepted:
                        self.stats["emitted"] += 1

        t_emit = time.perf_counter()
        return WindowTrace(
            seq=seq, win_end_s=round(win_end_s, 4),
            t_acquire=round(t_acquire, 6), t_infer=round(t_infer, 6), t_emit=round(t_emit, 6),
            compute_ms=round(compute_ms, 4),
            e2e_ms=round((t_emit - t_acquire) * 1e3 + self.hop / TARGET_SR * 1e3, 4),
            hop_ms=round(self.hop / TARGET_SR * 1e3, 3),
            cls=cls, conf=round(conf, 4) if conf is not None else 0.0,
            code=code, urgency=urgency, level_db=round(level_db, 2),
            accepted=accepted, reject_reason=reason, preempted=preempted,
        )

    def warmup(self, n: int = 3) -> float:
        """预热：跑几个空窗，把「首次调用」的固定开销（懒加载/显存/cuDNN 算法选择/
        mel 滤波器缓存）挤出去。否则延迟分布里会出现一个 400+ ms 的假 max，
        对外报「max 延迟」时会严重失真。

        返回预热耗时（秒）——单独记录，不进最终延迟统计。
        """
        t0 = time.perf_counter()
        dummy = np.zeros(self.win, dtype=np.float32)
        for _ in range(max(1, n)):
            self.pipe.classify(dummy)
        return time.perf_counter() - t0

    # -- 主循环 ----------------------------------------------------------
    def run(
        self,
        source: AudioSource,
        *,
        duration_s: float | None = None,
        pacing: str = "realtime",
        max_windows: int | None = None,
        progress_every_s: float = 60.0,
        verbose: bool = True,
    ) -> dict:
        """`pacing`: realtime（按设备时钟）| max（尽可能快，用于 30 min 材料）。"""
        assert pacing in ("realtime", "max")
        n_chunk = int(self.device_sr * CHUNK_MS / 1000)
        warm = self.warmup()
        self.backend.open()
        source.open()
        self._t0 = time.perf_counter()
        t_audio = 0.0
        next_progress = progress_every_s
        dropped_this_loop = 0

        try:
            while True:
                if duration_s is not None and t_audio >= duration_s:
                    break
                if max_windows is not None and self.stats["windows"] >= max_windows:
                    break

                chunk = source.read(n_chunk)
                if chunk is None:
                    break
                if pacing == "realtime":
                    # 以音频时钟为准做补偿式休眠（不是固定 sleep，避免累积漂移）
                    target = self._t0 + t_audio + CHUNK_MS / 1000
                    dt = target - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)

                t_audio += len(chunk) / self.device_sr
                self.stats["samples_in"] += len(chunk)
                self.ring.write(self.resampler.process(chunk))

                # 所有已凑满的窗口依次处理（正常情况下每 250 ms 恰好一个）
                while self._next_end <= self.ring.n_written:
                    t_acq = time.perf_counter()
                    if not self.ring.can_read(self._next_end, self.win):
                        # 积压到窗口被覆盖：记丢窗，跳到可处理的最早窗口
                        self.stats["dropped"] += 1
                        dropped_this_loop += 1
                        oldest = self.ring.n_written - self.ring.capacity
                        self._next_end = oldest + self.win
                        if self._next_end > self.ring.n_written:
                            break
                        continue
                    tr = self._process_window(self._next_end, t_acq)
                    if tr is not None:
                        self.recorder.add(tr)
                        self.stats["windows"] += 1
                    self._next_end += self.hop

                if verbose and t_audio >= next_progress:
                    print(f"[{t_audio:7.1f}s] 窗口 {self.stats['windows']:5d} "
                          f"下发 {self.stats['emitted']:4d} 丢窗 {self.stats['dropped']}",
                          flush=True)
                    next_progress += progress_every_s
        finally:
            source.close()
            binfo = self.backend.close()

        wall = time.perf_counter() - self._t0
        return {
            "source": source.name,
            "device_sr": self.device_sr,
            "resampler": self.resampler.info(),
            "win_s": self.win / TARGET_SR,
            "hop_s": self.hop / TARGET_SR,
            "pacing": pacing,
            "warmup_s": round(warm, 3),
            "audio_s": round(t_audio, 3),
            "wall_s": round(wall, 3),
            "realtime_factor": round(t_audio / wall, 3) if wall > 0 else None,
            "windows": self.stats["windows"],
            "dropped": self.stats["dropped"],
            "emitted": self.stats["emitted"],
            "latency": self.recorder.summary(),
            "scheduler": self.scheduler.report(),
            "backend": binfo,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_source(args) -> AudioSource:
    if args.source == "mic":
        return MicSource(sr=args.device_sr, device=args.device)
    if args.source == "file":
        if not args.audio:
            raise SystemExit("--source file 需要 --audio")
        return FileSource(args.audio, sr=args.device_sr, loop=args.loop)
    return SyntheticSource(sr=args.device_sr, duration_s=args.duration or 60.0,
                           seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SafeSoundAI 流式运行时（P0-1）")
    ap.add_argument("--source", choices=["mic", "file", "synth"], default="synth")
    ap.add_argument("--audio", type=str, default=None)
    ap.add_argument("--loop", action="store_true", default=True)
    ap.add_argument("--device-sr", type=int, default=DEVICE_SR_DEFAULT)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--pacing", choices=["realtime", "max"], default="realtime")
    ap.add_argument("--win-s", type=float, default=WIN_S)
    ap.add_argument("--hop-s", type=float, default=HOP_S)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--ckpt", type=str, default=str(Path(config.MODEL_DIR) / "dscnn_v2_4cls_fft1024.pt"))
    ap.add_argument("--backend", choices=["null", "wavefile", "ble_stub", "android"], default="null")
    ap.add_argument("--out-wav", type=str, default=None)
    ap.add_argument("--out-log", type=str, default="outputs/reports/streaming")
    ap.add_argument("--cooldown", type=float, default=10.0, help="同类冷却（s）")
    ap.add_argument("--simulate-slow-infer", type=float, default=0.0,
                    help="人为拖慢单窗，用于验证丢窗计数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    bkw = {}
    if args.backend == "wavefile":
        bkw = {"path": args.out_wav or "outputs/audio/haptic_stream.wav",
               "duration_s": max(30.0, (args.duration or 60.0) + 5)}
    elif args.backend in ("ble_stub", "android"):
        bkw = {"log_path": Path(args.out_log) / f"{args.backend}.jsonl"}
    backend = build_backend(args.backend, **bkw)

    rt = StreamingRuntime(
        backend=backend,
        policy=SchedulePolicy(same_class_cooldown_s=args.cooldown),
        ckpt=args.ckpt, device_sr=args.device_sr,
        win_s=args.win_s, hop_s=args.hop_s, threshold=args.threshold,
        simulate_slow_infer_ms=args.simulate_slow_infer,
    )
    src = _build_source(args)

    print(f"源={src.name} 设备采样率={args.device_sr} 窗={args.win_s}s hop={args.hop_s}s "
          f"后端={args.backend} 时长={args.duration}s 节奏={args.pacing}")
    print(f"AED 头: {rt.pipe.classes or '（无权重，只跑调度链）'}")

    info = rt.run(src, duration_s=args.duration, pacing=args.pacing)

    out = Path(args.out_log)
    out.mkdir(parents=True, exist_ok=True)
    rt.recorder.to_jsonl(out / "windows.jsonl")
    png = rt.recorder.histogram_png(out / "latency_hist.png")
    (out / "streaming_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    lat = info["latency"]
    print("\n" + "=" * 78)
    print(f"预热 {info['warmup_s']}s（不计入延迟统计）")
    print(f"音频 {info['audio_s']}s / 墙钟 {info['wall_s']}s（加速比 {info['realtime_factor']}×）")
    print(f"窗口 {info['windows']}｜丢窗 **{info['dropped']}**｜触觉下发 {info['emitted']}")
    if "compute_ms" in lat:
        c, e = lat["compute_ms"], lat["e2e_ms"]
        print(f"单窗算力 compute  p50 {c['p50']} / p95 {c['p95']} / max {c['max']} ms")
        print(f"端到端 e2e(含 {lat['hop_ms']:.0f} ms 攒窗)  p50 {e['p50']} / p95 {e['p95']} ms")
        print(f"算力超过 hop 的窗口占比 {lat['infer_over_hop_pct']}%")
    print(f"调度: {info['scheduler']['by_reason']}｜抑制率 {info['scheduler']['suppression_ratio']}")
    print(f"产物: {out}/windows.jsonl, streaming_summary.json"
          + (f", {png.name}" if png else ""))
    print("=" * 78)

    if info["dropped"] > 0 and args.simulate_slow_infer == 0.0:
        print("❌ 出现丢窗（非人为），验收不通过", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
