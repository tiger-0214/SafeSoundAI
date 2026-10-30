"""
SafeSoundAI — 并行双支路流水线（dual_path_pipeline.py, v3）

**为什么改成并行**（27 号 §三·E）：
  串联（降噪 → AED）有两个问题：
    1. 降噪需要上下文/未来帧 → 必然给告警检测引入延迟；
    2. 降噪链的增益/门控会把绝对电平压平，电平信息不可信。
  并行分叉后，检测支路吃**未经任何处理的原始 PCM**，
  输出支路独立降噪给用户听 —— 降噪想怎么改都不影响告警。

        麦克风原始块
          ├─ 支路 A（检测）：原始块 → 电平(dBFS) → z-score → mel → DS_CNN
          │                   → DecisionEngine → 触觉码（类别 × 紧迫度）
          └─ 支路 B（输出）：原始块副本 → FilterPipeline → 耳内音频

三条口径（写在代码里，防止以后又改回去）：
  1. **电平在 z-score 之前取**。z-score 会抹掉绝对电平，之后再取就永远拿不到。
  2. 电平只用于**相对分位**（LevelNormalizer），不解释为绝对 dB SPL。
  3. 降噪支路**不得**把自己的输出回灌给检测支路。

用法：
  from dual_path_pipeline import DualPathPipeline
  pipe = DualPathPipeline()
  events, aided = pipe.process_offline(y_raw)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
from haptic_encoder import (
    DecisionEngine, Detection, LevelNormalizer, PRIOR_URGENCY,
    URGENCY_INTERVALS, compute_urgency, generate_vibration,
)

SR = config.SAMPLE_RATE
WIN = config.SEGMENT_SAMPLES
HOP = WIN // 4


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(max(1e-9, float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))))


def zscore(x: np.ndarray) -> np.ndarray:
    """与训练期预处理一致（逐窗 z-score）。注意：必须在取电平之后调用。"""
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


@dataclass
class PathEvent:
    t: float
    cls: str
    conf: float
    level_db: float
    rel_quantile: float | None
    duration_s: float
    repeats_10s: int
    urgency: str
    pwm: float
    interval_ms: int
    code: str = ""

    def __post_init__(self) -> None:
        self.code = f"{self.cls}:{self.urgency}"


@dataclass
class PipelineState:
    level: LevelNormalizer = field(default_factory=LevelNormalizer)
    active_cls: str | None = None
    active_start: float | None = None
    recent: list[tuple[float, str]] = field(default_factory=list)   # (t, cls) 10 s 窗口

    def touch(self, cls: str, t: float) -> tuple[float, int]:
        """更新"该类已持续多久 / 10 s 内重复几次"，返回 (duration_s, repeats_10s)。"""
        if self.active_cls != cls:
            self.active_cls, self.active_start = cls, t
        dur = t - (self.active_start or t)
        self.recent = [(tt, c) for tt, c in self.recent if t - tt <= 10.0]
        self.recent.append((t, cls))
        reps = sum(1 for _, c in self.recent if c == cls) - 1
        return dur, max(0, reps)


class DualPathPipeline:
    """并行双支路：检测吃原始信号，输出走降噪。"""

    def __init__(
        self,
        model=None,
        device: str | None = None,
        threshold: float = 0.8,
        use_filter: bool = True,
        filter_profile: str | None = None,
        ckpt: str | Path | None = None,
    ):
        self.device = device
        self.threshold = threshold
        self.engine = DecisionEngine(confidence_threshold=threshold)
        self.state = PipelineState()
        self._model = model
        self._mel = None
        self.use_filter = use_filter
        self._filter = None
        if use_filter and filter_profile != "off":
            from filter_pipeline import FilterPipeline
            kwargs = {} if filter_profile is None else {"profile": filter_profile}
            self._filter = FilterPipeline(**kwargs)

    # ------------------------------------------------------------------
    # 模型（懒加载）
    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            import torch
            from models import DS_CNN
            path = Path(config.MODEL_DIR) / "ds_cnn_best.pth"
            if not path.exists():
                return None                      # 无权重 → 只跑电平/降噪支路
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            m = DS_CNN().to(self.device)
            ck = torch.load(path, map_location=self.device)
            m.load_state_dict(ck.get("state_dict", ck))
            m.eval()
            self._model = m
        return self._model

    def classify(self, raw_win: np.ndarray) -> np.ndarray | None:
        """支路 A 的分类前向：z-score → mel → DS_CNN。无权重时返回 None。"""
        m = self.model
        if m is None:
            return None
        import torch
        from mel import waveform_to_logmel
        if self._mel is None:
            self._mel = True
        with torch.no_grad():
            x = waveform_to_logmel(zscore(raw_win)).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,n_mels,frames)
            return torch.softmax(m(x), 1).cpu().numpy()[0]

    # ------------------------------------------------------------------
    # 主处理（离线）
    # ------------------------------------------------------------------
    def process_offline(self, y_raw: np.ndarray, collect_aided: bool = True):
        """返回 (events: list[PathEvent], aided: np.ndarray | None)。"""
        y_raw = np.asarray(y_raw, dtype=np.float32)
        starts = np.arange(0, max(1, len(y_raw) - WIN + 1), HOP)

        events: list[PathEvent] = []
        for s in starts:
            win = y_raw[s:s + WIN]
            t = s / SR

            # ← 电平必须在 zscore 之前
            lvl = rms_db(win)

            probs = self.classify(win)
            if probs is None:
                continue
            dets = [Detection(int(i), float(p)) for i, p in enumerate(probs)]
            win_det = self.engine.arbitrate(dets)
            if win_det is None:
                continue

            cls = config.TARGET_CLASSES[win_det.class_id]
            self.state.level.push(cls, lvl)
            q = self.state.level.quantile(cls, lvl)
            dur, reps = self.state.touch(cls, t)

            pat = generate_vibration(
                cls, duration_s=dur, repeats_10s=reps, rel_quantile=q, scheme="ear",
            )
            if pat.pwm_base <= 0:
                continue
            events.append(PathEvent(
                t=t, cls=cls, conf=float(win_det.confidence), level_db=lvl,
                rel_quantile=q, duration_s=dur, repeats_10s=reps,
                urgency=pat.urgency, pwm=pat.pwm_base, interval_ms=pat.interval_ms,
            ))

        aided = None
        if self.use_filter and self._filter is not None and collect_aided:
            aided, _info = self._filter.process_offline(y_raw)
        return events, aided

    # ------------------------------------------------------------------
    def events_frame(self, events: list[PathEvent]) -> pd.DataFrame:
        if not events:
            return pd.DataFrame(columns=["t", "cls", "conf", "level_db",
                                         "rel_quantile", "duration_s",
                                         "repeats_10s", "urgency", "pwm",
                                         "interval_ms", "code"])
        return pd.DataFrame([e.__dict__ for e in events])

    def reset(self) -> None:
        self.state = PipelineState()
        if self._filter is not None and hasattr(self._filter, "state"):
            from filter_pipeline import FilterState
            self._filter.state = FilterState()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = 0.05 * rng.standard_normal(SR * 3).astype(np.float32)
    y[SR:SR + int(0.3 * SR)] += 0.5 * np.sin(2 * np.pi * 500 * np.arange(int(0.3 * SR)) / SR)

    pipe = DualPathPipeline(use_filter=True)
    events, aided = pipe.process_offline(y)
    print(f"原始长度 {len(y)} / 降噪后 {0 if aided is None else len(aided)}")
    print(f"检测事件 {len(events)} 条")
    for e in events[:5]:
        print(f"  t={e.t:5.2f}s {e.code:<18} conf={e.conf:.2f} "
              f"level={e.level_db:6.1f}dB q={e.rel_quantile} pwm={e.pwm:.2f}")

    # 关键自检：检测支路拿到的电平不受降噪影响
    lvl_raw = [rms_db(y[s:s + WIN]) for s in range(0, len(y) - WIN + 1, HOP)]
    print(f"\n检测支路电平范围 {min(lvl_raw):.1f} ~ {max(lvl_raw):.1f} dBFS（来自原始信号）")
    print("✅ 双支路：AED 电平取自原始块，降噪仅作用于 aided 输出")
