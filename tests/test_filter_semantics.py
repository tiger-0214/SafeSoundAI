"""把「除人声外全压」的语义钉死成可执行的断言（V1–V3，2026-10-31 重写）。

**为什么整体重写**：旧版把 siren / car_horn / gun_shot / dog_bark 列为
「永不抑制」，理由是"聋人需要听见危险声"。2026-10-31 Tiger 澄清设计事实：
**危险声靠触觉振动通道交付，不靠耳机听觉保留** ⇒ 耳机内它们**就该被压掉**。
旧断言会把这个正确行为判成 bug，必须换掉，否则代码永远改不动。

现在钉死三条：

  V1 人声永不抑制 —— 10 类词表里只有 ``children_playing`` 含人声。
  V2 未识别 / 分不清的声音默认衰减（fail-closed），由帧级 VAD 把语音帧捞回来；
     4 类头的 ``background`` 是这种"分不清"的典型。
  V3 抑制必须有限幅（档位 ``max_total_attenuation_db`` + 能量塌陷兜底）。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import config  # noqa: E402
import filter_config as fc  # noqa: E402

HEAD4 = ["siren", "car_horn", "gun_shot", "background"]
HEAD10 = list(config.TARGET_CLASSES)
HAZARDS = ("siren", "car_horn", "gun_shot", "dog_bark")
PROFILES = ("comfort", "ptsd_default", "awareness")


# ---------------------------------------------------------------- V1：人声永不抑制
def test_whitelist_is_empty_by_design():
    """白名单必须为空 —— 保留任何一类都意味着"这类声音不进耳朵"，与降噪目标冲突。"""
    assert fc.WHITELIST_NAMES == {}, "白名单不为空：又有类被设成永不抑制了"
    assert fc.whitelist_ids_in(HEAD10) == frozenset()
    assert fc.whitelist_ids_in(HEAD4) == frozenset()


def test_voice_classes_never_attenuated_in_any_profile():
    """V1：人声类（children_playing）在任何档位下都不得为负。"""
    for prof in PROFILES:
        g = fc.gains_for(HEAD10, prof, warn=False)
        for name in fc.VOICE_NAMES:
            assert g[fc.NAME_TO_ID[name]] >= 0.0, f"{prof}: 人声类 {name} 被衰减了"


def test_all_non_voice_classes_attenuated_by_default():
    """10 类里除人声类外**全部**为负增益 —— 这就是「除人声外全压」。"""
    g = fc.gains_for(HEAD10, "ptsd_default", warn=False)
    for i, name in enumerate(HEAD10):
        if name in fc.VOICE_NAMES:
            continue
        assert g[i] < 0.0, f"{name} 未衰减（{g[i]:+.1f} dB），与「除人声外全压」冲突"


# ---------------------------------------------------------------- 危险声：该压
def test_hazard_classes_are_attenuated_not_protected():
    """危险声必须被抑制 —— 它们靠振动交付，不靠耳机听觉（旧版 0 dB 是错的）。"""
    for prof in PROFILES:
        g = fc.gains_for(HEAD10, prof, warn=False)
        for name in HAZARDS:
            assert g[fc.NAME_TO_ID[name]] < 0.0, f"{prof}: 危险声 {name} 未被抑制"
            assert name not in fc.WHITELIST_NAMES


# ---------------------------------------------------------------- V2：语音门控
def test_head4_background_follows_speech_gate():
    """4 类头 background 走门控 fallback，不是 0 dB 直通。"""
    g = fc.gains_for(HEAD4, "ptsd_default", warn=False)
    assert set(g) == {0, 1, 2, 3}
    assert g[3] == pytest.approx(fc.SPEECH_GATE_FALLBACK_DB), \
        "background 必须取语音门控 fallback（V2），不是 fail-open 0 dB"


def test_speech_gate_off_restores_passthrough(monkeypatch):
    """关掉 SPEECH_GATE 时退化为旧行为（0 dB 直通）—— 保证可回滚。"""
    monkeypatch.setattr(fc, "SPEECH_GATE", False)
    g = fc.gains_for(HEAD4, "ptsd_default", warn=False)
    assert g[3] == 0.0


def test_unknown_head_strict_raises_but_head4_does_not():
    """strict 仍要能抓出"换了头没换语义表"；4 类头现在是已知的，必须放行。"""
    fc.gains_for(HEAD4, "ptsd_default", strict=True, warn=False)      # 放行
    fc.gains_for(HEAD10, "ptsd_default", strict=True, warn=False)     # 放行
    with pytest.raises(KeyError):
        fc.gains_for(["foo_sound", "bar_sound"], "ptsd_default", strict=True, warn=False)


def test_unknown_head_warns_once_not_every_window():
    """未知头只告警一次 —— 否则流式循环 250 ms 刷一条，日志淹掉。"""
    fc._WARNED_HEADS.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fc.gains_for(["foo_sound"], "ptsd_default")
        fc.gains_for(["foo_sound"], "ptsd_default")
        fc.gains_for(["foo_sound"], "ptsd_default")
    assert len(w) == 1, f"应当只告警 1 次，实际 {len(w)} 次"


# ---------------------------------------------------------------- V3：限幅与 guard
def test_background_gain_guard_allows_fallback_rejects_boost_and_deepening():
    ok = fc.gains_for(HEAD4, "ptsd_default", warn=False)
    fc.background_gain_guard(HEAD4, ok)                    # fallback 放行
    fc.background_gain_guard(HEAD4, {**ok, 3: 0.0})        # 门控放行的语音帧也放行
    with pytest.raises(AssertionError):                    # 抬底噪
        fc.background_gain_guard(HEAD4, {**ok, 3: 3.0})
    with pytest.raises(AssertionError):                    # 深于门控下限会压死儿童喊叫
        fc.background_gain_guard(HEAD4, {**ok, 3: -30.0})


def test_no_profile_gives_background_positive_gain():
    """反查所有档位：不存在任何一条能把 background 抬起来的路径。"""
    for prof in fc._PROFILE_NAMES:
        assert fc._PROFILE_NAMES[prof]["gains"].get("background", 0.0) <= 0.0


def test_every_profile_respects_attenuation_ceiling():
    """V3：档位衰减上限 ≤ 24 dB，且不得出现"零增益"的假降噪档。"""
    for prof, cfg in fc._PROFILE_NAMES.items():
        assert cfg["max_total_attenuation_db"] <= 24.0, f"{prof}: 衰减上限过深"
        neg = [v for v in cfg["gains"].values() if v < 0]
        assert neg, f"{prof}: 没有任何衰减项，等于不做降噪"


# ---------------------------------------------------------------- 与 attenuator 的接线
def test_attenuator_binds_gains_to_its_own_head():
    """衰减器必须用自己的头的类名翻译增益（不是硬编码 10 类坐标）。"""
    src = (Path(__file__).resolve().parents[1] / "code" / "attenuator.py").read_text(
        encoding="utf-8")
    assert "fc.gains_for(self.classes" in src, "衰减器不再按自己的头翻译增益"
    assert "fc.whitelist_ids_in(self.classes)" in src


def test_attenuator_applies_frame_level_speech_gate():
    """V2 的落地位置：process 与 mask_from 都必须叠帧级语音门控。"""
    src = (Path(__file__).resolve().parents[1] / "code" / "attenuator.py").read_text(
        encoding="utf-8")
    assert "def voice_gate_frames(" in src, "缺少帧级 VAD 实现"
    assert src.count("self.apply_voice_gate(") >= 2, \
        "process / mask_from 必须都叠门控，否则评测口径与实际输出不一致"
