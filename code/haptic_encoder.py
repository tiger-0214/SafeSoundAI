"""
SafeSoundAI — 二维触觉编码 + 决策引擎（haptic_encoder.py, **v3 定稿 2026-10-22**）

【论文核心 novelty】把「环境声事件」映射为「可区分的触觉词表」：
  轴 1  时间 → 类别（名义量 nominal，3 码）
  轴 2  幅度 → 紧迫度（有序量 ordinal，3 档）
  位置（方向）本轮**不编码**，函数保留但标 P2。

编码空间 = 3 × 3 = 9 码，落在 Brown et al. 2006「两参数 ≈70%、
减少取值数可回升 81%」的区间。

三条硬约束（改动本文件前先读）：
  1. 类别只由**时间形状**承担，幅度**不得**按类别设固定值 —— 否则两轴共线，
     识别率必崩。语义先验只决定「默认档」，每类允许跨越 ≥2 档。
  2. 紧迫度 = 语义先验（主）+ 相对电平分位（次）。**禁止使用绝对 dB SPL**：
     麦克风灵敏度、佩戴遮挡、AGC 都会毁掉绝对电平（见 27 号 §三·E）。
  3. 幅度 3 档拉开 ≈8 dB（40 / 70 / 100%），个体佩戴差异用 LevelNormalizer
     做相对归一化（Luzhnica et al., ISWC 2017 证明通用标定不够）。

已废弃（不要复活）：11 类 SOUND_PATTERNS、DISTANCE_PWM 5 级距离映射、
5 级独立 URGENCY_INTERVALS、6-LRA/4-LRA/2-LRA 方向权重接入主路径。

文献依据见 ·重点节点总结笔记/27-触觉编码文献与设计依据（2026-10-22）.md
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import config
import vocab


# ---------------------------------------------------------------------------
# 轴 2：紧迫度（有序量，3 档）
# ---------------------------------------------------------------------------
URGENCY_LEVELS: list[str] = ["alert", "high", "critical"]        # 0 低 → 2 高
URGENCY_RANK: dict[str, int] = {u: i for i, u in enumerate(URGENCY_LEVELS)}

# 幅度 3 档：≈8 dB 间隔（Weber 定律下"拉开跨度"才分得开，见 Francisco 2008）
URGENCY_PWM: dict[str, float] = {
    "alert":    0.40,
    "high":     0.70,
    "critical": 1.00,
}

# 类别先验的默认档（只定默认，不锁死范围）
PRIOR_URGENCY: dict[str, str] = {
    "gun_shot": "critical",
    "car_horn": "high",
    "siren":    "alert",
}

# 每类允许的紧迫度范围。**全部放开 0–2**：先验只决定运行时的默认档，
# 不做硬钳制 —— 否则实验无法呈现"类别 × 紧迫度"的正交 9 码组合，
# 幅度轴的可辨识性也就无从测量。
URGENCY_RANGE: dict[str, tuple[int, int]] = {c: (0, 2) for c in PRIOR_URGENCY}


def get_urgency_pwm(urgency: str) -> float:
    """紧迫度 → PWM 基线（替代已废弃的 get_distance_pwm）。"""
    return URGENCY_PWM[urgency]


def clamp_urgency(sound_type: str, level: int) -> str:
    lo, hi = URGENCY_RANGE.get(sound_type, (0, 2))
    return URGENCY_LEVELS[max(lo, min(hi, level))]


# ---------------------------------------------------------------------------
# 轴 1：时间 → 类别（3 码）
# ---------------------------------------------------------------------------
# pattern_type 取值：single / dual / ramp
SOUND_PATTERNS: dict[str, tuple[str, dict, str]] = {
    "gun_shot": (
        "single",
        {"on_ms": 60, "interval_ms": 3000, "repeat_after_s": 3.0},
        PRIOR_URGENCY["gun_shot"],
    ),
    "car_horn": (
        "dual",
        {"on_ms": (150, 80, 150), "interval_ms": 700},
        PRIOR_URGENCY["car_horn"],
    ),
    "siren": (
        "ramp",
        {"duration_ms": 400, "direction": "up_down", "interval_ms": 400},
        PRIOR_URGENCY["siren"],
    ),
}

V3_CLASSES: list[str] = list(SOUND_PATTERNS)

# 重复间隔并入类别码（不再独立可变，避免与幅度轴抢"紧迫度"语义）
URGENCY_INTERVALS: dict[str, int] = {
    cls: int(params["interval_ms"]) for cls, (_, params, _) in SOUND_PATTERNS.items()
}

# 不编码的类别（有检测但不出触觉，或仅日志）
SILENT_CLASSES: set[str] = {
    "air_conditioner", "children_playing", "dog_bark", "drilling",
    "engine_idling", "jackhammer", "street_music", "unknown",
}


# ---------------------------------------------------------------------------
# 相对电平归一化（替代绝对 dB）
# ---------------------------------------------------------------------------
class LevelNormalizer:
    """按类别维护滚动窗口电平分位，把本次电平折算成 0–1 的相对分位。

    用途：约掉麦克风灵敏度 / 佩戴遮挡 / 增益常数造成的个体偏移
    （同一设备同一佩戴下这些是常数，做分位就消失了）。

    用法：
        nz = LevelNormalizer()
        for frame_db in stream:      # 由 AED 支路的**原始信号**算出
            nz.push("siren", frame_db)
        q = nz.quantile("siren", current_db)   # 0=历来最弱，1=历来最强
    """

    def __init__(self, window: int = 600, warmup: int = 30):
        self.window = window
        self.warmup = warmup            # 样本不足时返回 None（交给语义先验）
        self._buf: dict[str, deque[float]] = {}

    def push(self, sound_type: str, level_db: float) -> None:
        buf = self._buf.setdefault(sound_type, deque(maxlen=self.window))
        buf.append(float(level_db))

    def quantile(self, sound_type: str, level_db: float) -> float | None:
        buf = self._buf.get(sound_type)
        if buf is None or len(buf) < self.warmup:
            return None
        arr = sorted(buf)
        below = sum(1 for v in arr if v <= level_db)
        return below / len(arr)


# ---------------------------------------------------------------------------
# 紧迫度计算（语义先验为主，相对电平为辅）
# ---------------------------------------------------------------------------
def compute_urgency(
    sound_type: str,
    duration_s: float | None = None,      # 该类已持续多久（siren >5 s 升级）
    repeats_10s: int = 0,                 # 10 s 内同类重复触发次数
    rel_quantile: float | None = None,    # LevelNormalizer 输出，0–1；None=不用
) -> str:
    """紧迫度 = 语义先验 + 持续/重复 + 相对电平分位（❌ 不用绝对 dB、❌ 不用置信度）。"""
    if sound_type not in SOUND_PATTERNS:
        return "alert"

    level = URGENCY_RANK[PRIOR_URGENCY[sound_type]]          # 主：类别先验

    if duration_s is not None:
        if sound_type == "siren" and duration_s > 5.0:
            level += 1                                       # 警笛持续未消失 → 升级
        elif sound_type == "gun_shot" and duration_s > 3.0 and repeats_10s >= 2:
            level += 1                                       # 连续枪声
    if repeats_10s >= 3:
        level += 1                                           # 短时反复触发 → 升级

    if rel_quantile is not None:                             # 次：相对电平分位
        if rel_quantile >= 0.90:
            level += 1
        elif rel_quantile <= 0.30:
            level -= 1

    return clamp_urgency(sound_type, level)


# ---------------------------------------------------------------------------
# 震动模式数据结构
# ---------------------------------------------------------------------------
@dataclass
class HapticPattern:
    sound_type: str
    urgency: str
    pattern_type: str
    params: dict
    interval_ms: int
    pwm_base: float
    intensities: dict[str, float] = field(default_factory=dict)
    scheme: str = "earbud"

    @property
    def code(self) -> str:
        """9 码编号，如 'siren:critical'。"""
        return f"{self.sound_type}:{self.urgency}"

    def describe(self) -> str:
        return (
            f"[{self.sound_type}] {self.pattern_type} | 紧迫={self.urgency} "
            f"({self.interval_ms}ms) | 幅度={self.pwm_base:.2f} | {self.scheme}: {self.intensities}"
        )


# ---------------------------------------------------------------------------
# 主编码函数（v3）
# ---------------------------------------------------------------------------
def generate_vibration(
    sound_type: str,
    *,
    urgency: str | None = None,
    duration_s: float | None = None,
    repeats_10s: int = 0,
    rel_quantile: float | None = None,
    scheme: str = "earbud",
    distance_m: float | None = None,     # ⚠️ 已废弃：仅为兼容旧调用，不再影响输出
    azimuth_deg: float | None = None,    # ⚠️ P2，不参与编码
) -> HapticPattern:
    """生成 v3 触觉模式。

    sound_type : V3_CLASSES 之一；其它类别返回"静默模式"（pwm_base=0）
    urgency    : 显式指定紧迫度（实验用）；None 则按 compute_urgency() 推断
    scheme     : "earbug"/"ear"（4 通道等幅）| "torso"（4 通道等幅）
    """
    if sound_type not in SOUND_PATTERNS:
        # 非 3 正类：不震动（论文口径：只编码 gun_shot / car_horn / siren）
        return HapticPattern(
            sound_type=sound_type, urgency="alert", pattern_type="silent",
            params={}, interval_ms=0, pwm_base=0.0,
            intensities={}, scheme=scheme,
        )

    if distance_m is not None:
        import warnings
        warnings.warn(
            "distance_m 已废弃（v3 不做距离编码）；紧迫度请用 duration_s / "
            "repeats_10s / rel_quantile 传入。", DeprecationWarning, stacklevel=2)
    if azimuth_deg is not None:
        import warnings
        warnings.warn("azimuth_deg 为 P2，v3 不参与编码。", DeprecationWarning, stacklevel=2)

    pattern_type, params, _prior = SOUND_PATTERNS[sound_type]
    if urgency is None:
        urgency = compute_urgency(
            sound_type, duration_s=duration_s,
            repeats_10s=repeats_10s, rel_quantile=rel_quantile,
        )
    urgency = clamp_urgency(sound_type, URGENCY_RANK[urgency])

    pwm = get_urgency_pwm(urgency)                 # 幅度只由紧迫度决定
    interval_ms = URGENCY_INTERVALS[sound_type]    # 间隔并入类别码

    if scheme in ("ear", "earbug", "earbud"):
        intensities = {k: round(pwm, 2) for k in
                       ["R_front", "R_back", "L_front", "L_back"]}
    elif scheme == "torso":
        intensities = {k: round(pwm, 2) for k in ["F", "R", "B", "L"]}
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return HapticPattern(
        sound_type=sound_type, urgency=urgency, pattern_type=pattern_type,
        params=params, interval_ms=interval_ms, pwm_base=pwm,
        intensities=intensities, scheme=scheme,
    )


# ---------------------------------------------------------------------------
# P2：方向编码（本轮不接入主路径，保留备查）
# ---------------------------------------------------------------------------
def compute_torso_lra(azimuth_deg: float) -> dict[str, float]:
    """# P2, not used in v3 — 躯干 4-LRA 余弦正弦分解。"""
    theta = math.radians(azimuth_deg)
    return {
        "F": max(0.0, math.cos(theta)),
        "R": max(0.0, math.sin(theta)),
        "B": max(0.0, -math.cos(theta)),
        "L": max(0.0, -math.sin(theta)),
    }


def compute_ear_lra(azimuth_deg: float) -> dict[str, float]:
    """# P2, not used in v3 — 耳周 6-LRA（左右 sin 分解，前后 cos 比例分配）。"""
    theta = math.radians(azimuth_deg)
    rw = 0.5 * (1.0 + math.sin(theta))
    lw = 1.0 - rw
    fc, bc = max(0.0, math.cos(theta)), max(0.0, -math.cos(theta))
    total = fc + bc
    fr, br = (fc / total, bc / total) if total > 1e-6 else (0.5, 0.5)
    return {
        "R_front": round(rw * fr, 3), "R_back": round(rw * br, 3), "R_top": 0.0,
        "L_front": round(lw * fr, 3), "L_back": round(lw * br, 3), "L_top": 0.0,
    }


def compute_earbud_lra(azimuth_deg: float) -> dict[str, float]:
    """# P2, not used in v3 — 耳塞 2-LRA（左右 + 相位）。"""
    theta = math.radians(azimuth_deg)
    r = 0.5 * (1.0 + math.sin(theta))
    return {"R": round(r, 3), "L": round(1.0 - r, 3),
            "phase": "in" if math.cos(theta) >= 0 else "out"}


def get_top_activation(urgency: str) -> float:
    """# P2, not used in v3 — 头顶 LRA 辅助强度。"""
    return {"alert": 0.0, "high": 0.3, "critical": 0.6}[urgency]


# ---------------------------------------------------------------------------
# 决策引擎：多重警报优先级仲裁
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """分类器输出的一帧检测结果。"""
    class_id: int
    confidence: float


class DecisionEngine:
    """多声源冲突仲裁：紧迫度优先，同级取置信度更高者。

    注意：置信度只用于**仲裁与门限**，不参与紧迫度数值（语义反直觉，审稿人会问）。
    """

    def __init__(self, confidence_threshold: float = 0.8,
                 classes: list[str] | tuple[str, ...] | None = None):
        # classes = **产出 class_id 的那个模型**的类别表（vocab 铁律）。
        # 默认取 4 类产品头；传 10 类 legacy 头时请显式传入 config.TARGET_CLASSES。
        # 历史 bug：这里曾固定用 config.TARGET_CLASSES[class_id] 反查，
        # 于是 4 类头的 gun_shot(2) 被读成 children_playing、siren(0) 被读成
        # air_conditioner，触觉紧迫度整条链错位。
        self.threshold = confidence_threshold
        self.classes = list(classes) if classes is not None else list(vocab.AED4_CLASSES)

    def _urgency(self, class_id: int) -> str:
        name = self.classes[int(class_id)]
        return PRIOR_URGENCY.get(name, "alert")

    def name_of(self, class_id: int) -> str:
        return self.classes[int(class_id)]

    def arbitrate(self, detections: list[Detection]) -> Detection | None:
        valid = [d for d in detections if d.confidence >= self.threshold]
        if not valid:
            return None
        valid.sort(key=lambda d: (-URGENCY_RANK[self._urgency(d.class_id)],
                                  -d.confidence))
        return valid[0]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== v3 编码表：3 类 × 3 档 = 9 码 ===")
    shapes = {"single": "███", "dual": "███ ·· ███", "ramp": "▁▃▅█▅▃▁"}
    for cls in V3_CLASSES:
        for u in URGENCY_LEVELS:
            p = generate_vibration(cls, urgency=u)
            star = " ←先验默认" if u == PRIOR_URGENCY[cls] else ""
            print(f"  {p.code:<18} {shapes[p.pattern_type]:<12} "
                  f"间隔{p.interval_ms:>5}ms  幅度{p.pwm_base:.2f}{star}")

    print("\n=== 紧迫度推断（语义先验 + 持续 + 相对分位）===")
    cases = [
        dict(sound_type="siren", duration_s=1.0),
        dict(sound_type="siren", duration_s=8.0),
        dict(sound_type="car_horn", repeats_10s=4),
        dict(sound_type="gun_shot", rel_quantile=0.95),
        dict(sound_type="car_horn", rel_quantile=0.1),
    ]
    for c in cases:
        u = compute_urgency(**c)
        print(f"  {c} → {u} ({get_urgency_pwm(u):.2f})")

    print("\n=== LevelNormalizer（相对电平分位）===")
    nz = LevelNormalizer(warmup=10)
    for db in range(-60, -20, 2):
        nz.push("siren", db)
    print("  历史 -60..-24 dB 时，-30 dB 的分位 =", round(nz.quantile("siren", -30), 3))

    print("\n=== 决策引擎仲裁（gun_shot 应胜）===")
    # 用 4 类产品头空间自检（与 demo.py 实际部署口径一致）
    engine = DecisionEngine(confidence_threshold=0.8)
    c4 = {n: i for i, n in enumerate(vocab.AED4_CLASSES)}
    dets = [
        Detection(c4["siren"], 0.95),
        Detection(c4["gun_shot"], 0.85),
        Detection(c4["car_horn"], 0.90),
    ]
    chosen = engine.arbitrate(dets)
    print(f"  仲裁结果: {engine.name_of(chosen.class_id)}（4 类空间）")
    # 再用 10 类 legacy 空间自检，证明同一 id 在不同空间不会串味
    engine10 = DecisionEngine(confidence_threshold=0.8, classes=config.TARGET_CLASSES)
    dets10 = [Detection(config.CLASS_TO_IDX["siren"], 0.95),
              Detection(config.CLASS_TO_IDX["gun_shot"], 0.85),
              Detection(config.CLASS_TO_IDX["car_horn"], 0.90)]
    print(f"  仲裁结果: {engine10.name_of(engine10.arbitrate(dets10).class_id)}（10 类空间）")
