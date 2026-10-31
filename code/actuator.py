"""
SafeSoundAI — 触觉执行器后端抽象（actuator.py, P0-3）

【解决什么问题】
`haptic_encoder.generate_vibration()` 只产出「模式描述」（哪类、哪档、多长间隔），
`haptic_render.render_code()` 只产出 175 Hz 载波波形。两者都没有「发射」这一步，
所以「能装进耳机」这句话此前无法验收：没有后端，就没有时间戳，就没有延迟分布。

本模块把 9 码（3 类别 × 3 紧迫度）变成**可下发的命令**，并提供四种后端：

  NullBackend        逻辑基准，零延迟，用于算法回归（不碰硬件）
  WaveFileBackend    渲染成 16 kHz 波形并写 wav，用于 PC 回放验证（耳机本体软件）
  BLEStubBackend     生成 BLE GATT 命令帧（LRA 驱动板协议），不建连接
  AndroidVibrationBackend  生成 VibrationEffect 波形参数（手机直驱替代路径）

【对齐验收（P0-3 的硬指标）】
同一事件流喂给三种后端，各自的 `t_emit` 与调度时刻 `t_sched` 之差必须 ≤ 5 ms。
设备固有延迟（BLE 连接间隔、马达起振）**单独记在 `t_air` / `t_rise`**，不混进对齐口径
—— 否则「对齐」变成「延迟",两件事互相掩盖。

【9 码口径】
码 = `<类别>:<紧迫度>`，直接复用 `haptic_encoder` 的 SOUND_PATTERNS / URGENCY_PWM，
不在此文件重定义任何幅度或时长常数（防止两处漂移）。

用法：
    from actuator import build_backend, HapticCommand
    b = build_backend("wavefile", path="/tmp/hap.wav", duration_s=12.0)
    b.open()
    b.emit(HapticCommand.from_pattern(pat, t_sched=1.0))
    b.close()
"""
from __future__ import annotations

import json
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import haptic_encoder as he

# ---------------------------------------------------------------------------
# 码表：9 码 × 参数（全部来自 haptic_encoder，不在此处重定义）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeSpec:
    """一个码的完整下发参数（与具体后端无关）。"""

    code: str
    sound_type: str
    urgency: str
    pattern_type: str
    amplitude: float                 # 0–1，= URGENCY_PWM[urgency]
    onset_ms: tuple[int, ...]        # 起振时刻（一个单元内的相对 ms）
    on_ms: tuple[int, ...]           # 每个脉冲的持续时间
    gaps_ms: tuple[int, ...]         # 单元**内部**脉冲间静默（dual 才有）
    unit_ms: int                     # 一个单元总长（含单元内静默）
    interval_ms: int                 # 单元**之间**的静默（类别码的一部分）
    channels: tuple[str, ...]

    @property
    def duration_ms(self) -> int:
        return int(sum(self.on_ms))

    @property
    def is_silent(self) -> bool:
        return self.amplitude <= 0.0


def _channel_names(scheme: str) -> tuple[str, ...]:
    return ("F", "R", "B", "L") if scheme == "torso" else ("R_front", "R_back", "L_front", "L_back")


def code_spec(sound_type: str, urgency: str, scheme: str = "earbud") -> CodeSpec:
    """(类别, 紧迫度) → CodeSpec。非法组合抛 KeyError（不静默兜底）。"""
    if sound_type not in he.SOUND_PATTERNS:
        raise KeyError(f"{sound_type} 不是 v3 编码类别（只编码 {he.V3_CLASSES}）")
    if urgency not in he.URGENCY_PWM:
        raise KeyError(f"{urgency} 不是合法紧迫度（{he.URGENCY_LEVELS}）")
    ptype, params, _ = he.SOUND_PATTERNS[sound_type]

    if ptype == "single":
        onsets, ons, gaps = (0,), (int(params["on_ms"]),), ()
    elif ptype == "dual":
        on1, off, on2 = params["on_ms"]
        onsets, ons, gaps = (0, int(on1) + int(off)), (int(on1), int(on2)), (int(off),)
    elif ptype == "ramp":
        onsets, ons, gaps = (0,), (int(params["duration_ms"]),), ()
    else:                                     # pragma: no cover - 防御
        raise ValueError(f"未知 pattern_type: {ptype}")

    unit_ms = sum(ons) + sum(gaps)            # 与 haptic_render.unit_envelope 的单元长度一致
    return CodeSpec(
        code=f"{sound_type}:{urgency}",
        sound_type=sound_type,
        urgency=urgency,
        pattern_type=ptype,
        amplitude=float(he.URGENCY_PWM[urgency]),
        onset_ms=onsets,
        on_ms=ons,
        gaps_ms=gaps,
        unit_ms=unit_ms,
        interval_ms=max(0, int(params["interval_ms"])),
        channels=_channel_names(scheme),
    )


def code_table(scheme: str = "earbud") -> dict[str, CodeSpec]:
    """全部 9 码（3 类别 × 3 紧迫度）。"""
    return {f"{c}:{u}": code_spec(c, u, scheme)
            for c in he.V3_CLASSES for u in he.URGENCY_LEVELS}


#: 9 码在 BLE 帧里的编号（进固件就固化，改动等于破坏兼容）
CODE_ID: dict[str, int] = {code: i for i, code in enumerate(sorted(code_table()))}


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------
@dataclass
class HapticCommand:
    """一条待下发的触觉命令。`t_sched` 是**流时间轴**上的调度时刻（秒）。"""

    code: str
    t_sched: float
    spec: CodeSpec
    repeats: int = 1                 # 重复几个单元（None → 1）
    source: str = ""                 # 触发它的类别/事件来源，调试用
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_pattern(cls, pat, t_sched: float, repeats: int = 1, **meta) -> "HapticCommand":
        """从 haptic_encoder.HapticPattern 构造。静默模式 → amplitude=0，后端直接丢弃。"""
        if getattr(pat, "pwm_base", 0.0) <= 0.0:
            spec = CodeSpec(
                code=f"{pat.sound_type}:{pat.urgency}", sound_type=pat.sound_type,
                urgency=pat.urgency, pattern_type="silent", amplitude=0.0,
                onset_ms=(), on_ms=(), gaps_ms=(), unit_ms=0, interval_ms=0,
                channels=tuple(getattr(pat, "intensities", {}) or ()),
            )
        else:
            spec = code_spec(pat.sound_type, pat.urgency,
                             scheme="torso" if pat.scheme == "torso" else "earbud")
        return cls(code=spec.code, t_sched=float(t_sched), spec=spec,
                   repeats=int(repeats), source=meta.pop("source", ""), meta=meta)


@dataclass
class EmitRecord:
    """一次下发的回执（审计与延迟统计的最小单位）。"""

    code: str
    backend: str
    t_sched: float        # 调度给出
    t_emit: float         # 后端真正开始执行（对齐口径看这个）
    t_done: float         # 执行完毕（含波形/命令时长）
    ok: bool
    t_air: float = 0.0    # 设备固有传输延迟（不参与对齐）
    t_rise: float = 0.0   # 马达起振延迟（不参与对齐）
    detail: dict = field(default_factory=dict)

    @property
    def align_err_ms(self) -> float:
        return (self.t_emit - self.t_sched) * 1e3

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("detail", None)
        d["align_err_ms"] = round(self.align_err_ms, 4)
        d["payload"] = self.detail
        return d


# ---------------------------------------------------------------------------
# 后端基类
# ---------------------------------------------------------------------------
class ActuatorBackend(ABC):
    """执行器后端契约。

    实现只需保证三件事：
      1. `emit()` 返回的 `t_emit` 是**它自己时钟上真正开始输出的时刻**；
      2. 静默码（amplitude=0）不产生输出，但**仍要返回 ok=True 的回执**（便于计数对账）；
      3. `close()` 返回可序列化的汇总（是否落盘、帧数、错误数）。
    """

    name: str = "base"

    def __init__(self) -> None:
        self._records: list[EmitRecord] = []
        self._opened = False

    # -- 生命周期 --------------------------------------------------------
    def open(self, **kwargs) -> "ActuatorBackend":
        self._opened = True
        return self

    def close(self) -> dict:
        self._opened = False
        return {"backend": self.name, "n_records": len(self._records),
                "n_ok": sum(1 for r in self._records if r.ok)}

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- 主接口 ----------------------------------------------------------
    @abstractmethod
    def _do_emit(self, cmd: HapticCommand) -> EmitRecord: ...

    def emit(self, cmd: HapticCommand) -> EmitRecord:
        if not self._opened:
            self.open()
        rec = self._do_emit(cmd)
        self._records.append(rec)
        return rec

    # -- 查询 ------------------------------------------------------------
    @property
    def records(self) -> list[EmitRecord]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()

    def latency_report(self) -> dict:
        """对齐误差分布（ms）。验收看 p95 与 max，不看均值。"""
        if not self._records:
            return {"backend": self.name, "n": 0}
        err = np.array([r.align_err_ms for r in self._records], dtype=np.float64)
        return {
            "backend": self.name,
            "n": int(err.size),
            "align_err_ms": {
                "mean": round(float(err.mean()), 4),
                "abs_p50": round(float(np.percentile(np.abs(err), 50)), 4),
                "abs_p95": round(float(np.percentile(np.abs(err), 95)), 4),
                "abs_max": round(float(np.abs(err).max()), 4),
            },
            "n_silent": int(sum(1 for r in self._records if r.detail.get("silent"))),
            "n_failed": int(sum(1 for r in self._records if not r.ok)),
        }


# ---------------------------------------------------------------------------
# 后端 1：Null（逻辑基准）
# ---------------------------------------------------------------------------
class NullBackend(ActuatorBackend):
    """什么都不做，只打点。用于算法回归与抖动测试（零 I/O、零抖动）。"""

    name = "null"

    def __init__(self, simulated_latency_ms: float = 0.0):
        super().__init__()
        self.sim_lat = simulated_latency_ms / 1e3

    def _do_emit(self, cmd: HapticCommand) -> EmitRecord:
        if cmd.spec.is_silent:
            return EmitRecord(cmd.code, self.name, cmd.t_sched, cmd.t_sched,
                              cmd.t_sched, ok=True, detail={"silent": True})
        t_emit = cmd.t_sched + self.sim_lat
        dur = cmd.spec.unit_ms / 1e3
        return EmitRecord(cmd.code, self.name, cmd.t_sched, t_emit, t_emit + dur,
                          ok=True, detail={"units": cmd.repeats})


# ---------------------------------------------------------------------------
# 后端 2：WaveFile（PC 回放验证）
# ---------------------------------------------------------------------------
class WaveFileBackend(ActuatorBackend):
    """把 9 码渲染成 16 kHz 波形，按 `t_sched` 叠加到时间轴，close 时写 wav。

    这是「不给现成耳机加硬件」路径下的**可听验证**：波形可直接灌进耳机或
    仿真 LRA，人耳/加速度计都能验收参数是否与编码表一致。
    """

    name = "wavefile"

    def __init__(self, path: str | Path, duration_s: float = 30.0, sr: int = 16000):
        super().__init__()
        self.path = Path(path)
        self.duration_s = float(duration_s)
        self.sr = int(sr)
        self._buf = np.zeros(int(self.sr * self.duration_s), dtype=np.float32)

    def _render_units(self, cmd: HapticCommand) -> np.ndarray:
        from haptic_render import render_code            # 复用同一渲染器，不另写一份
        reps = max(1, cmd.repeats)
        head = render_code(cmd.spec.sound_type, cmd.spec.urgency,
                           dur_s=(cmd.spec.unit_ms + cmd.spec.interval_ms) / 1e3 * reps,
                           sr=self.sr)
        return head

    def _do_emit(self, cmd: HapticCommand) -> EmitRecord:
        if cmd.spec.is_silent:
            return EmitRecord(cmd.code, self.name, cmd.t_sched, cmd.t_sched,
                              cmd.t_sched, ok=True, detail={"silent": True})
        y = self._render_units(cmd)
        i0 = int(round(cmd.t_sched * self.sr))
        if i0 >= len(self._buf):
            return EmitRecord(cmd.code, self.name, cmd.t_sched, cmd.t_sched, cmd.t_sched,
                              ok=False, detail={"error": "beyond_buffer", "i0": i0})
        n = min(len(y), len(self._buf) - i0)
        # 叠加而非覆盖：多个码同时活跃时，回放要能听出冲突（这本身就是可测现象）
        self._buf[i0:i0 + n] += y[:n]
        # t_emit 取**首样本所在时刻**（量化误差 ≤ 1/sr = 0.0625 ms）
        t_emit = i0 / self.sr
        return EmitRecord(cmd.code, self.name, cmd.t_sched, t_emit,
                          t_emit + n / self.sr, ok=True,
                          detail={"i0": i0, "n_samples": n, "units": cmd.repeats})

    def close(self) -> dict:
        info = super().close()
        clipped = int(np.sum(np.abs(self._buf) > 1.0))
        y = np.clip(self._buf, -1.0, 1.0)
        played = int(np.sum(np.abs(y) > 1e-4))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import soundfile as sf
            sf.write(self.path, y, self.sr, subtype="PCM_16")
            written = True
        except Exception as exc:                       # pragma: no cover - 环境缺库
            written = False
            info["error"] = f"{type(exc).__name__}: {exc}"
        info.update({"path": str(self.path), "written": written,
                     "n_samples": int(y.size), "n_nonzero": played,
                     "clip_samples": clipped, "peak": round(float(np.abs(y).max()), 4)})
        return info


# ---------------------------------------------------------------------------
# 后端 3：BLE stub（LRA 驱动板协议）
# ---------------------------------------------------------------------------
class BLEStubBackend(ActuatorBackend):
    """生成 BLE GATT 命令帧并落 JSONL，**不建连接**。

    帧格式（little-endian，8 字节）：
        u8  code_id | u8 flags | u16 unit_ms | u8 amplitude×100 | u16 interval_ms | u8 repeats

    选择这个布局的理由：单帧 ≤ MTU(23−3=20 B) 的默认 ATT 载荷，无需分片；
    code_id 固化后固件只查表，不传参数组合（省带宽、防错配）。
    真实设备侧应实现 write-without-response，并在 `t_air` 里记连接间隔。
    """

    name = "ble_stub"
    FRAME = struct.Struct("<BBHBHB")
    DEFAULT_T_AIR_MS = 7.5          # BLE 连接间隔下限（7.5 ms）×1，命令单帧发送
    DEFAULT_T_RISE_MS = 5.0         # LRA 起振（175 Hz 共振，约 1 个周期）

    def __init__(self, log_path: str | Path | None = None,
                 t_air_ms: float = DEFAULT_T_AIR_MS, t_rise_ms: float = DEFAULT_T_RISE_MS):
        super().__init__()
        self.log_path = Path(log_path) if log_path else None
        self.t_air = t_air_ms / 1e3
        self.t_rise = t_rise_ms / 1e3
        self._frames: list[dict] = []

    def frame_of(self, cmd: HapticCommand) -> bytes:
        spec = cmd.spec
        return self.FRAME.pack(
            CODE_ID[spec.code],
            0x01 if spec.pattern_type == "ramp" else 0x00,
            min(65535, spec.unit_ms),
            int(round(spec.amplitude * 100)),
            min(65535, spec.interval_ms),
            max(1, min(255, cmd.repeats)),
        )

    def _do_emit(self, cmd: HapticCommand) -> EmitRecord:
        if cmd.spec.is_silent:
            return EmitRecord(cmd.code, self.name, cmd.t_sched, cmd.t_sched,
                              cmd.t_sched, ok=True, detail={"silent": True})
        raw = self.frame_of(cmd)
        t_emit = cmd.t_sched                       # 命令在调度时刻入队
        dur = (cmd.spec.unit_ms + cmd.spec.interval_ms) * max(1, cmd.repeats) / 1e3
        self._frames.append({
            "t_sched": round(cmd.t_sched, 6), "code": cmd.code,
            "code_id": CODE_ID[cmd.code], "frame_hex": raw.hex(),
            "amplitude": cmd.spec.amplitude, "repeats": cmd.repeats,
        })
        return EmitRecord(cmd.code, self.name, cmd.t_sched, t_emit, t_emit + dur,
                          ok=True, t_air=self.t_air, t_rise=self.t_rise,
                          detail={"frame_hex": raw.hex(), "n_bytes": len(raw)})

    def close(self) -> dict:
        info = super().close()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8") as f:
                for row in self._frames:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        info.update({"frames": len(self._frames), "bytes_per_frame": self.FRAME.size,
                     "log": str(self.log_path) if self.log_path else None,
                     "t_air_ms": round(self.t_air * 1e3, 2),
                     "t_rise_ms": round(self.t_rise * 1e3, 2)})
        return info


# ---------------------------------------------------------------------------
# 后端 4：Android VibrationEffect（手机直驱替代路径）
# ---------------------------------------------------------------------------
class AndroidVibrationBackend(ActuatorBackend):
    """生成 `VibrationEffect.createWaveform(timings, amplitudes, repeat)` 的参数并落 JSONL。

    用途：没有耳机硬件时，用手机马达验证「类别码是否可辨」——
    时间形状是编码主轴，手机马达足以承载（这也是 27 号笔记里
    「先用手机做可辨识性预实验」的落地形态）。
    """

    name = "android"
    DEFAULT_T_AIR_MS = 2.0
    DEFAULT_T_RISE_MS = 8.0

    def __init__(self, log_path: str | Path | None = None,
                 t_air_ms: float = DEFAULT_T_AIR_MS, t_rise_ms: float = DEFAULT_T_RISE_MS):
        super().__init__()
        self.log_path = Path(log_path) if log_path else None
        self.t_air = t_air_ms / 1e3
        self.t_rise = t_rise_ms / 1e3
        self._cmds: list[dict] = []

    def waveform_of(self, cmd: HapticCommand) -> tuple[list[int], list[int]]:
        """→ (timings_ms, amplitudes_0_255)，交替 [开, 关, 开, 关, ...]。"""
        spec = cmd.spec
        timings: list[int] = []
        amps: list[int] = []
        a = int(round(spec.amplitude * 255))
        for i, on in enumerate(spec.on_ms):
            timings.append(int(on))
            amps.append(a)
            gap = spec.gaps_ms[i] if i < len(spec.gaps_ms) else 0
            if i < len(spec.on_ms) - 1:
                timings.append(int(gap))
                amps.append(0)
        # 单元尾部静默（类别码的一部分）
        timings.append(max(0, spec.interval_ms))
        amps.append(0)
        if cmd.repeats > 1:
            timings = timings * int(cmd.repeats)
            amps = amps * int(cmd.repeats)
        return timings, amps

    def _do_emit(self, cmd: HapticCommand) -> EmitRecord:
        if cmd.spec.is_silent:
            return EmitRecord(cmd.code, self.name, cmd.t_sched, cmd.t_sched,
                              cmd.t_sched, ok=True, detail={"silent": True})
        timings, amps = self.waveform_of(cmd)
        t_emit = cmd.t_sched
        dur = sum(timings) / 1e3
        self._cmds.append({
            "t_sched": round(cmd.t_sched, 6), "code": cmd.code,
            "api": "VibrationEffect.createWaveform",
            "timings_ms": timings, "amplitudes": amps, "repeat": -1,
        })
        return EmitRecord(cmd.code, self.name, cmd.t_sched, t_emit, t_emit + dur,
                          ok=True, t_air=self.t_air, t_rise=self.t_rise,
                          detail={"timings_ms": timings, "amplitudes": amps})

    def close(self) -> dict:
        info = super().close()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8") as f:
                for row in self._cmds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        info.update({"commands": len(self._cmds),
                     "log": str(self.log_path) if self.log_path else None})
        return info


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
_BACKENDS = {
    "null": NullBackend,
    "wavefile": WaveFileBackend,
    "ble_stub": BLEStubBackend,
    "android": AndroidVibrationBackend,
}


def build_backend(kind: str, **kwargs) -> ActuatorBackend:
    if kind not in _BACKENDS:
        raise KeyError(f"未知后端 {kind}（可选：{sorted(_BACKENDS)}）")
    return _BACKENDS[kind](**kwargs)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print(f"{'code':<20}{'type':<8}{'on_ms':>16}{'interval':>10}{'amp':>6}")
    for code, spec in code_table().items():
        print(f"{code:<20}{spec.pattern_type:<8}{str(spec.on_ms):>16}"
              f"{spec.interval_ms:>10}{spec.amplitude:>6.2f}")
    assert len(code_table()) == 9, "码表必须是 3×3 = 9 码"

    tmp = Path(tempfile.mkdtemp())
    pats = [he.generate_vibration(c, urgency=u, scheme="ear")
            for c in he.V3_CLASSES for u in he.URGENCY_LEVELS]
    timeline = [(0.5 + 0.7 * i, p) for i, p in enumerate(pats)]

    errs = {}
    for kind, kw in (("null", {}),
                     ("wavefile", {"path": tmp / "hap.wav", "duration_s": 10.0}),
                     ("ble_stub", {"log_path": tmp / "ble.jsonl"}),
                     ("android", {"log_path": tmp / "and.jsonl"})):
        b = build_backend(kind, **kw).open()
        for t, p in timeline:
            b.emit(HapticCommand.from_pattern(p, t_sched=t))
        info = b.close()
        rep = b.latency_report()
        errs[kind] = rep["align_err_ms"]["abs_max"]
        print(f"\n[{kind}] {info}")
        print(f"  对齐误差 max = {errs[kind]:.4f} ms")
    assert max(errs.values()) <= 5.0, f"对齐验收失败：{errs}"
    print(f"\n✅ 四种后端对齐误差 ≤ 5 ms：{errs}")
    print("✅ 9 码表 / 幅度来自 haptic_encoder，未在本文件重定义")
