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
import warnings
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# 降噪目标定义 —— 2026-10-31 设计澄清后重写（推翻旧 R1/R2/R3 的前提）
# ---------------------------------------------------------------------------
# **项目设计事实**：耳机内降噪目标 = *除人声外全部衰减*。
#   危险声（警笛 / 车喇叭 / 枪声 / 狗叫）**不靠耳机听觉保留，靠触觉振动通道交付**。
#   因此「把危险声压掉」是**设计意图**，不是缺陷；旧版把它们列成「永不抑制」
#   与设计直接冲突（评测里那条 44% 的"白名单红线"随之作废）。
#
# 新三条规则（V1–V3）：
#
#   V1. **人声永不抑制**。10 类词表里只有 ``children_playing`` 含人声（儿童喊叫）
#       ⇒ 恒 0 dB。词表里没有 speech 类，一般成人语音会落到「未识别」，
#       由 V2 的语音门控兜住。
#   V2. **词表外的声音默认衰减（fail-closed），但必须过语音门控**。
#       取代旧版"未识别 → fail-open 0 dB"——那等于把脚步、键盘、鸟叫、交通
#       全部原样放进来，降噪等于没做。新逻辑：VAD 判为语音的帧抬回 0 dB，
#       其余帧按 ``SPEECH_GATE_FALLBACK_DB`` 衰减。开关 ``SPEECH_GATE``。
#   V3. **抑制必须有限幅**。total ≤ 档位 ``max_total_attenuation_db``（≤24 dB）
#       且受 ``ANTI_COLLAPSE_DB`` 兜底，避免"全压"压成一片死寂 ——
#       PTSD 次臂的"触觉不得惊跳 / 不得丧失环境感"约束也依赖这条。
#
# 4 类部署头里的 ``background`` 是"不是那三种危险声"的袋子（空调、电钻、狗叫、
# 儿童嬉闹、街道音乐都在里面），分不出来 ⇒ 正是 V2 语音门控的适用对象：
# 门控判为语音就放行（保护儿童喊叫），否则按 fallback 压。

# ---------------------------------------------------------------------------
# 语义三表（filter_spec.md §二）—— class_id 与 config.TARGET_CLASSES 对齐
# 值是「默认目标增益 dB」（负值 = 抑制）
# ---------------------------------------------------------------------------
NAME_TO_ID = {name: i for i, name in enumerate(config.TARGET_CLASSES)}

# --- 唯一真相：语义表用**类名**表达（2026-10-31 起） --------------------------
# 历史教训：旧版把白/黑名单写死成 10 类 class_id，一旦换成 4 类 AED 头
# （0=siren,1=car_horn,2=gun_shot,3=background）索引全部错位，
# 警笛会被打上 air_conditioner 的 -12 dB 增益。故所有语义一律按名字声明，
# 由 gains_for(classes) 在运行时翻译到**当前头自己的空间**。
WHITELIST_NAMES: dict[str, float] = {}                 # 清空（V1）：不再有"永不抑制"的类
ATTENUATE_NAMES = {                                    # 默认抑制，用户可调
    # 危险声 —— 耳机内压掉，改由触觉振动交付（2026-10-31 设计澄清）
    "siren": -12.0, "car_horn": -12.0, "gun_shot": -12.0, "dog_bark": -12.0,
    # 环境噪声
    "air_conditioner": -12.0, "street_music": -12.0,
    "drilling": -6.0, "jackhammer": -6.0, "engine_idling": -6.0,
}
VOICE_NAMES = {"children_playing": 0.0}                # 人声类：恒 0 dB（V1）

BLACKLIST_NAMES = ATTENUATE_NAMES                      # 向后兼容别名
NEUTRAL_NAMES = VOICE_NAMES                            # 向后兼容别名

# --- 语音门控（V2）：词表里分不清的声音，由 VAD 而不是"直通"来兜底 ----------
# Silero VAD 逐帧判定；语音帧抬回 SPEECH_GATE_HOLD_DB，非语音帧按 fallback 衰减。
# 前后各留 SPEECH_GATE_MARGIN_MS 保护带，否则字头字尾会被削掉（听感上是"断续"）。
SPEECH_GATE = True
SPEECH_GATE_FALLBACK_DB = -12.0                       # 非语音帧衰减量
SPEECH_GATE_HOLD_DB = 0.0                             # 语音帧增益
SPEECH_GATE_MARGIN_MS = 150.0                         # 语音判定前后保护带
SPEECH_GATE_MIN_SILENCE_MS = 100.0                    # 短于此的非语音空洞不压（防抖）
VAD_THRESHOLD = 0.5                                   # Silero 语音概率阈值
SPEECH_GATE_SMOOTH = 0.6                              # 门控一阶平滑（越大越软）
# 4 类头 background 显式声明为「门控类」（它分不出空调/电钻/狗叫/儿童喊叫）
SPEECH_GATED_NAMES = {"background": SPEECH_GATE_FALLBACK_DB}

# 源类空间（10 类）下的 id 版本 —— 过滤支路内部仍用 10 类坐标
WHITELIST = {NAME_TO_ID[n]: g for n, g in WHITELIST_NAMES.items() if n in NAME_TO_ID}
BLACKLIST = {NAME_TO_ID[n]: g for n, g in BLACKLIST_NAMES.items() if n in NAME_TO_ID}
NEUTRAL = {NAME_TO_ID[n]: g for n, g in NEUTRAL_NAMES.items() if n in NAME_TO_ID}

ALL_IDS = set(range(config.NUM_CLASSES))
WHITELIST_IDS = frozenset(WHITELIST)
BLACKLIST_IDS = frozenset(BLACKLIST)
NEUTRAL_IDS = frozenset(NEUTRAL)

# 档位预设（filter_spec.md §2.4）—— 同样按**类名**声明，再翻译成 10 类 id 版
_PROFILE_NAMES = {
    "comfort": {
        "profile_id": "comfort",
        "gains": {**WHITELIST_NAMES, **{n: -18.0 for n in BLACKLIST_NAMES}, **NEUTRAL_NAMES},
        "enhance_whitelist_db": 0.0,
        "max_total_attenuation_db": 24.0,
    },
    "ptsd_default": {
        "profile_id": "ptsd_default",
        "gains": {**WHITELIST_NAMES, **BLACKLIST_NAMES, **NEUTRAL_NAMES},
        "enhance_whitelist_db": 0.0,
        "max_total_attenuation_db": 24.0,
    },
    "awareness": {                                     # 减弱抑制：保留更多环境感知
        "profile_id": "awareness",
        "gains": {"siren": -6.0, "car_horn": -6.0, "gun_shot": -6.0, "dog_bark": -6.0,
                  "air_conditioner": -6.0, "street_music": -6.0,
                  "drilling": 0.0, "jackhammer": 0.0, "engine_idling": 0.0,
                  **VOICE_NAMES},
        "enhance_whitelist_db": 0.0,
        "max_total_attenuation_db": 12.0,
    },
}


def _to_ids(profile: dict) -> dict:
    """名字版档位 → 10 类源空间 id 版（保持向后兼容）。"""
    p = dict(profile)
    p["gains"] = {NAME_TO_ID[n]: g for n, g in profile["gains"].items() if n in NAME_TO_ID}
    return p


PROFILE_PRESETS = {k: _to_ids(v) for k, v in _PROFILE_NAMES.items()}
DEFAULT_PROFILE = "ptsd_default"


def get_profile(name: str | None = None) -> dict:
    """返回档位配置（未知名字回退默认档）。gain 的 key 为 10 类源空间 id。"""
    return PROFILE_PRESETS.get(name or DEFAULT_PROFILE, PROFILE_PRESETS[DEFAULT_PROFILE])


def get_profile_names(name: str | None = None) -> dict:
    """返回名字版档位配置（`gains` 的 key 是类名）。"""
    return _PROFILE_NAMES.get(name or DEFAULT_PROFILE, _PROFILE_NAMES[DEFAULT_PROFILE])


def class_gains(profile: str | None = None) -> dict[int, float]:
    """class_id → 目标增益 dB（10 类源空间）。"""
    return dict(get_profile(profile)["gains"])


_WARNED_HEADS: set[tuple[str, ...]] = set()


def gains_for(
    classes: list[str] | tuple[str, ...],
    profile: str | None = None,
    *,
    strict: bool = False,
    warn: bool = True,
) -> dict[int, float]:
    """把档位增益翻译到**任意 AED 头空间**：返回 {该头空间的 class_id: dB}。

    规则（与文件顶部的 R1–R3 一致）：

    * 语义表里有该类 → 取该档位增益；
    * 语义表里没有（最典型是 4 类头的 ``background``）→ **0 dB**，即
      「该头分不清的类别一律不衰减」（fail-open）。

    `strict=True` 时改为抛 `KeyError`：给 CI / 单元测试用，能立刻抓出
    "换了头却没换语义表"这种静默错配。运行时不要开 strict —— 部署现场
    因为出现一个新类名就崩掉，比少衰减一类糟得多。

    `warn=True` 时对每个新遇到的未知头**只告警一次**（`_WARNED_HEADS` 去重），
    否则流式循环里每 250 ms 刷一条，日志会被淹掉。
    """
    by_name = dict(get_profile_names(profile)["gains"])
    if SPEECH_GATE:                       # V2：门控类显式并入（background 等）
        by_name.update(SPEECH_GATED_NAMES)
    unknown = [n for n in classes if n not in by_name]
    if unknown:
        key = tuple(classes)
        msg = (f"类名 {unknown} 不在声学语义表里 → 这些类按 "
               f"{SPEECH_GATE_FALLBACK_DB if SPEECH_GATE else 0.0:+.1f} dB 处理"
               f"（SPEECH_GATE={SPEECH_GATE}）。若不是预期，说明语义表没跟上。")
        if strict:
            raise KeyError(msg)
        if warn and key not in _WARNED_HEADS:
            _WARNED_HEADS.add(key)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    fallback = SPEECH_GATE_FALLBACK_DB if SPEECH_GATE else 0.0
    return {i: float(by_name.get(n, fallback)) for i, n in enumerate(classes)}


def background_gain_guard(classes: list[str] | tuple[str, ...],
                          gains: dict[int, float]) -> None:
    """V1/V2 的运行时断言：门控类（background）的增益必须落在允许区间。

    旧版（R1）要求 background 恰好 0 dB —— 那时危险声还算"要保留"。
    新设计下 background 交给语音门控：**语音帧 0 dB、非语音帧 fallback**，
    所以静态增益表里它只能是这两个值之一。禁止的是：
      * 正增益 —— 抬底噪，助听器"放大一切"的老毛病；
      * 比 fallback 更深的衰减 —— 门控再准也救不回被压死的儿童喊叫。
    放在这里而不是测试里，是因为它同时保护部署路径。
    """
    lo = SPEECH_GATE_FALLBACK_DB if SPEECH_GATE else 0.0
    for i, n in enumerate(classes):
        if n not in SPEECH_GATED_NAMES:
            continue
        g = float(gains.get(i, 0.0))
        if g > 1e-9:
            raise AssertionError(
                f"{n} 不得给正增益（V1/V2），实测 {g:+.2f} dB；档位增益表被改坏了")
        if g < lo - 1e-9:
            raise AssertionError(
                f"{n} 的衰减不得深于门控下限 {lo:+.1f} dB，实测 {g:+.2f} dB；"
                f"会连儿童喊叫一起压掉，档位增益表被改坏了")


def whitelist_ids_in(classes: list[str] | tuple[str, ...]) -> frozenset[int]:
    """当前头空间里属于白名单的 class_id 集合。"""
    return frozenset(i for i, n in enumerate(classes) if n in WHITELIST_NAMES)


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

# --- 2026-10-31 修复项（真实混合集实测驱动，见 reports/filter_ab_report.md）------
# 1) 逐窗 z-score：训练用的段是 z-score 过的，推理不做同样归一化 → 分布错位。
#    实测（200 条真实混合）top-1：不归一化 0.17 vs 逐窗 z-score 0.53。
ZSCORE_WINDOWS = True
# 2) 白名单「频谱让位」而非「整窗否决」：旧逻辑只要某个白名单类 p>0.5 就整窗直通，
#    而误判恰好大量流向 car_horn / siren，导致该降噪时反而全程不降噪。
#    新逻辑：仍施加噪声增益，但把该白名单类的能量带抬回 0 dB（只让出频谱，不让出整帧）。
WHITELIST_CARVEOUT = True
CARVEOUT_GUARD_DB = -1.0                        # 让位后该类频带允许的最大衰减
WHITELIST_VETO_CONF = 0.85                      # 仅当白名单类极高置信时才整窗直通
# 3) 安全强度 α（与 TSE 同思路）：g_eff = α·g。α 越小越保守、白名单红线越容易守住。
#    实测（30 条真实混合）：α=1.0 → Supp -2.9 / Pres -2.4（红线 27% 样本超标）；
#    调小 α 会等比例缩水抑制与误伤，选择性 gap 不变（见 reports/filter_ablation.md）。
GAIN_STRENGTH = 1.0
# 4) 噪声子词表归一化：衰减权重在「非白名单类」内部归一化，白名单保护仍用**绝对**后验。
#    动机：37% 的真实混合窗口 argmax 落在白名单类（多数是噪声被误判成 car_horn），
#    若衰减权重也用绝对后验，这些窗口完全不动（Supp 0 dB）。改为子词表归一化后
#    触发率 40%→47%，同时该白名单类的能量带仍被让出，不牺牲保护语义。
NOISE_CONDITIONAL_GAIN = True

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
    # 跨空间自检：同一档位在 4 类头空间里必须仍然是「白名单 0 dB」
    from vocab import AED4_CLASSES
    g4 = gains_for(AED4_CLASSES)
    print("4 类空间增益:", {n: g4[i] for i, n in enumerate(AED4_CLASSES)})
    assert all(g4[i] == 0.0 for i in whitelist_ids_in(AED4_CLASSES)), "白名单在 4 类空间被误抑制"
    print("MIX_DIR  :", MIX_DIR)
    print("KEEP_SETS:", {k: [config.TARGET_CLASSES[i] for i in v] for k, v in KEEP_SETS.items()})
