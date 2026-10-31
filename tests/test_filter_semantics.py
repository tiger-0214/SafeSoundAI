"""把「background 语义」钉死成可执行的断言（P1-3）。

为什么值得单开一个测试文件：
`background` 在 4 类头里不是"安静"，而是"不是那三种危险声" —— 空调、电钻、
狗叫、儿童嬉闹全挤在这一格。所以对 background 下负增益等于把狗叫和儿童喊叫
一起压掉（聋人恰恰需要知道这两个），下正增益等于抬底噪。
这三条规则（R1/R2/R3）以前只写在报告里，任何人改一行增益表都不会被拦住。
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


# ---------------------------------------------------------------- 4 类头
def test_head4_never_attenuates_and_never_touches_background():
    """4 类头的硬约束：**不许衰减任何类**（fail-open），且 background 恰好 0 dB。

    注意 awareness 档会给白名单 +3 dB —— 那是"增强警报"，属于允许项；
    被禁的是负增益（会误伤同一格里的狗叫/儿童），不是正增益。
    """
    for prof in ("comfort", "ptsd_default", "awareness"):
        g = fc.gains_for(HEAD4, prof, warn=False)
        assert set(g) == {0, 1, 2, 3}
        assert g[3] == 0.0, f"{prof}: background 必须 0 dB，实际 {g[3]}"
        assert all(v >= 0.0 for v in g.values()), f"{prof}: 4 类头出现负增益 {g}"


def test_head4_warns_once_not_every_window():
    """未知头只告警一次 —— 否则流式循环 250 ms 刷一条，日志淹掉。"""
    fc._WARNED_HEADS.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fc.gains_for(HEAD4, "ptsd_default")
        fc.gains_for(HEAD4, "ptsd_default")
        fc.gains_for(HEAD4, "ptsd_default")
    assert len(w) == 1, f"应当只告警 1 次，实际 {len(w)} 次"


def test_strict_mode_raises_for_ci():
    """strict=True 要能抓出"换了头没换语义表"，供 CI 使用。"""
    with pytest.raises(KeyError):
        fc.gains_for(HEAD4, "ptsd_default", strict=True)
    # 10 类头语义表齐全，strict 必须放行
    fc.gains_for(HEAD10, "ptsd_default", strict=True, warn=False)


# ---------------------------------------------------------------- 10 类头
def test_head10_gains_match_name_table():
    """10 类头的增益必须与按名字声明的表逐项相等（防止 id 错位回潮）。"""
    g = fc.gains_for(HEAD10, "ptsd_default", warn=False)
    for i, name in enumerate(HEAD10):
        want = fc._PROFILE_NAMES["ptsd_default"]["gains"][name]
        assert g[i] == want, f"{name}(id={i}) 期望 {want} dB，实际 {g[i]}"


def test_whitelist_ids_follow_names_not_numbers():
    """白名单必须按名字翻译：4 类头里 background 不在白名单，siren 在。"""
    assert fc.whitelist_ids_in(HEAD4) == frozenset({0, 1, 2})
    want10 = frozenset(fc.NAME_TO_ID[n] for n in fc.WHITELIST_NAMES)
    assert fc.whitelist_ids_in(HEAD10) == want10
    assert "background" not in fc.WHITELIST_NAMES, "R3：background 不进白名单"


def test_whitelist_never_attenuated_in_any_profile():
    """白名单类在任何档位下都不得为负（安全门）。"""
    for prof in ("comfort", "ptsd_default", "awareness"):
        g = fc.gains_for(HEAD10, prof, warn=False)
        for name in fc.WHITELIST_NAMES:
            assert g[fc.NAME_TO_ID[name]] >= 0.0, f"{prof}: {name} 被衰减了"


# ---------------------------------------------------------------- R1/R2 运行时断言
def test_background_gain_guard_catches_broken_table():
    ok = fc.gains_for(HEAD4, "ptsd_default", warn=False)
    fc.background_gain_guard(HEAD4, ok)                 # 0 dB 放行
    bad_atten = {**ok, 3: -12.0}
    with pytest.raises(AssertionError):
        fc.background_gain_guard(HEAD4, bad_atten)
    bad_boost = {**ok, 3: 3.0}
    with pytest.raises(AssertionError):
        fc.background_gain_guard(HEAD4, bad_boost)


def test_no_profile_ever_gives_background_positive_gain():
    """反查所有档位：不存在任何一条能把 background 抬起来的路径。"""
    for prof in fc._PROFILE_NAMES:
        assert fc._PROFILE_NAMES[prof]["gains"].get("background", 0.0) == 0.0


# ---------------------------------------------------------------- 与 attenuator 的接线
def test_attenuator_binds_gains_to_its_own_head():
    """衰减器必须用自己的头的类名翻译增益（不是硬编码 10 类坐标）。"""
    src = (Path(__file__).resolve().parents[1] / "code" / "attenuator.py").read_text(
        encoding="utf-8")
    assert "fc.gains_for(self.classes" in src, "衰减器不再按自己的头翻译增益"
    assert "fc.whitelist_ids_in(self.classes)" in src
