"""
SafeSoundAI — 声音过滤模块配置（A1/A2 共用）

把「过滤」相关的语义表、档位、信号链常量集中在此，
避免污染 config.py（检测/训练主链路）。

用法：
    import filter_config as fc
    fc.WHITELIST, fc.BLACKLIST, fc.PROFILE_PRESETS
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# 语义三表（filter_spec.md §二）—— class_id 与 config.TARGET_CLASSES 对齐
# 值是「默认目标增益 dB」（负值 = 抑制）
# ---------------------------------------------------------------------------
NAME_TO_ID = {name: i for i, name in enumerate(config.TARGET_CLASSES)}

# 白名单：永不抑制（安全关键）
WHITELIST = {8: 0.0, 1: 0.0, 6: 0.0, 3: 0.0}          # siren, car_horn, gun_shot, dog_bark
# 黑名单：默认抑制，用户可调
BLACKLIST = {0: -12.0, 9: -12.0, 5: -6.0, 7: -6.0, 4: -6.0}
# 中性：不主动处理
NEUTRAL = {2: 0.0}                                     # children_playing

ALL_IDS = set(range(config.NUM_CLASSES))
WHITELIST_IDS = frozenset(WHITELIST)
BLACKLIST_IDS = frozenset(BLACKLIST)
NEUTRAL_IDS = frozenset(NEUTRAL)

# 档位预设（filter_spec.md §2.4）
PROFILE_PRESETS = {
    "comfort": {
        "profile_id": "comfort",
        "gains": {**{k: 0.0 for k in WHITELIST},
                  **{k: -18.0 for k in BLACKLIST},
                  **{k: 0.0 for k in NEUTRAL}},
        "enhance_whitelist_db": 0.0,
        "max_total_attenuation_db": 24.0,
    },
    "ptsd_default": {
        "profile_id": "ptsd_default",
        "gains": {**WHITELIST, **BLACKLIST, **NEUTRAL},
        "enhance_whitelist_db": 0.0,
        "max_total_attenuation_db": 24.0,
    },
    "awareness": {
        "profile_id": "awareness",
        "gains": {**{k: 3.0 for k in WHITELIST},
                  **{0: -6.0, 9: -6.0, 5: 0.0, 7: 0.0, 4: 0.0},
                  **NEUTRAL},
        "enhance_whitelist_db": 3.0,
        "max_total_attenuation_db": 12.0,
    },
}
DEFAULT_PROFILE = "ptsd_default"


def get_profile(name: str | None = None) -> dict:
    """返回档位配置（未知名字回退默认档）。"""
    return PROFILE_PRESETS.get(name or DEFAULT_PROFILE, PROFILE_PRESETS[DEFAULT_PROFILE])


def class_gains(profile: str | None = None) -> dict[int, float]:
    """class_id → 目标增益 dB。"""
    return dict(get_profile(profile)["gains"])


# ---------------------------------------------------------------------------
# keep 集合四档（方案 B 的运行时常量，等价于 SoundBubble 的 distance_threshold）
# ---------------------------------------------------------------------------
KEEP_SETS = {
    "critical":     (8, 1),              # 只保留最核心两类
    "emergency":    (8, 1, 6),           # 三类紧急声
    "all_safety":   (8, 1, 6, 3),        # 全白名单
    "passthrough":  tuple(range(config.NUM_CLASSES)),   # 全通透
}
DEFAULT_KEEP_SET = "all_safety"


def keep_vector(keep_ids) -> list[float]:
    """keep 集合 → 10 维 0/1 条件向量（方案 B 的条件输入）。"""
    v = [0.0] * config.NUM_CLASSES
    for i in keep_ids:
        v[int(i)] = 1.0
    return v


# ---------------------------------------------------------------------------
# 信号链常量（filter_constraints.md §二/§六）
# ---------------------------------------------------------------------------
MIC_CHANNELS = {"left": 0, "right": 1}          # A3 DOA 只读，A2 输出单声道
FILTER_LATENCY_BUDGET_MS = 100.0
FILTER_CHUNK_SAMPLES = 512                      # 32 ms @16 kHz
FILTER_N_FFT = 512
FILTER_HOP = 160
FILTER_WIN = 512
SMOOTH_ALPHA = 0.5                              # 跨帧一阶平滑
MAX_DB_JUMP_PER_FRAME = 6.0                     # 帧间跳变上限
APPLY_CONF_THRESHOLD = 0.5                      # 施加增益所需的最小类别置信度（不足即 fail-open）
GAIN_CLAMP_DB = (-30.0, 6.0)
ANTI_COLLAPSE_DB = -20.0                        # 整体能量塌陷阈值（filter_safety §2.2）
HARD_GATE_TOLERANCE_DB = -3.0                   # 白名单允许的最大衰减（filter_safety §2.1）
SOFTMAX_TEMPERATURE = 0.7

# 检测层门控启用条件（未达标 → allow_gating 必须 False）
GATING_PREREQ_EVENT_RECALL = 0.95
GATING_PREREQ_FA_PER_HOUR = 5.0
ALLOW_GATING = False                            # 硬编码关闭，非配置项可改

# ---------------------------------------------------------------------------
# 合成混音集路径（filter_constraints §五：独立目录，不覆写 2_Preprocessed_Data）
# ---------------------------------------------------------------------------
MIX_DIR = Path(os.environ.get(
    "SAFESOUND_MIX_DIR",
    str(config.PREPROCESSED / "4_Synthetic_Mixtures"),
))
MIX_MANIFEST = MIX_DIR / "manifest.csv"
MIX_AUDIO = MIX_DIR / "audio"
MIX_STEMS = MIX_DIR / "stems"

# 过滤相关产物
FILTER_REPORT_DIR = config.OUTPUT_DIR / "reports"
FILTER_AUDIO_DIR = config.OUTPUT_DIR / "audio"
FILTER_FIG_DIR = config.OUTPUT_DIR / "figures"
TSE_CKPT = config.MODEL_DIR / "tse_best.pth"
TSE_ONNX = config.MODEL_DIR / "tse.onnx"

# ---------------------------------------------------------------------------
# 安全强度系数 α（2026-10-17 实测后新增）
#   g_eff = 1 - α·(1 - g)，α=1 = 模型原生掩码，α 越小越保守
# 实测（test 120 条，见 _diag_strength.py）：
#   α=1.00 → Supp -14.06 / Pres -9.55 dB，选择性 gap 4.51 dB（白名单红线 90% 样本超标）
#   α=0.35 → Supp  -2.88 / Pres -2.23 dB，gap 仅 0.66 dB（红线内，但几乎无净收益）
# 结论：α 收缩只是把抑制与保留等比例缩水，无法提升选择性。默认取 α=1.0 如实呈现
#       模型能力；生产若必须守住「白名单衰减 ≤ 3 dB」红线，应把 α 调到 0.35 并接受
#       几乎不降噪，或等更强模型（见 17-三阶段技术路线详解 §3.7 改进路线）。
# ---------------------------------------------------------------------------
TSE_STRENGTH = 1.0
TSE_STRENGTH_MIN = 0.0
TSE_STRENGTH_MAX = 1.0


def apply_strength(gain, alpha: float | None = None):
    """把 [0,1] 掩码按安全强度 α 向 1 收缩（α=0 → 直通，α=1 → 原生掩码）。"""
    a = TSE_STRENGTH if alpha is None else float(alpha)
    a = max(TSE_STRENGTH_MIN, min(TSE_STRENGTH_MAX, a))
    return 1.0 - a * (1.0 - gain)


def ensure_mix_dirs() -> None:
    for d in (MIX_DIR, MIX_AUDIO, MIX_STEMS):
        d.mkdir(parents=True, exist_ok=True)


def validate_profile(profile: dict) -> tuple[bool, str]:
    """用户配置校验（filter_safety §三）：白名单不可被抑制。"""
    for cid, g in profile["gains"].items():
        if int(cid) in WHITELIST_IDS and g < 0:
            return False, f"白名单类 {config.TARGET_CLASSES[int(cid)]} 不允许负增益（{g} dB）"
        if g < -24.0:
            return False, f"类 {config.TARGET_CLASSES[int(cid)]} 抑制超上限 -24 dB"
    return True, "ok"


if __name__ == "__main__":
    print("WHITELIST:", {config.TARGET_CLASSES[k]: v for k, v in WHITELIST.items()})
    print("BLACKLIST:", {config.TARGET_CLASSES[k]: v for k, v in BLACKLIST.items()})
    print("NEUTRAL  :", {config.TARGET_CLASSES[k]: v for k, v in NEUTRAL.items()})
    for name in PROFILE_PRESETS:
        ok, msg = validate_profile(get_profile(name))
        print(f"profile {name}: {'OK' if ok else 'FAIL - ' + msg}")
    print("MIX_DIR  :", MIX_DIR)
    print("KEEP_SETS:", {k: [config.TARGET_CLASSES[i] for i in v] for k, v in KEEP_SETS.items()})
