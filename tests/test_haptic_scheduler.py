"""
P1-4 验收测试：触觉调度器（去重 / 冷却 / 跨类抢占）+ 抖动测试

跑法：cd code && python3 -m pytest ../tests/test_haptic_scheduler.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import haptic_encoder as he                                   # noqa: E402
from actuator import NullBackend                               # noqa: E402
from haptic_scheduler import (Decision, HapticScheduler,        # noqa: E402
                              Reject, SchedulePolicy)


def _pat(cls: str, urgency: str | None = None):
    return he.generate_vibration(cls, urgency=urgency, scheme="ear")


def _cmd(cls: str, urgency: str, t: float):
    from actuator import HapticCommand
    return HapticCommand.from_pattern(_pat(cls, urgency), t_sched=t)


# ---------------------------------------------------------------------------
# 基本行为
# ---------------------------------------------------------------------------
def test_sustained_event_is_throttled_to_class_cooldown():
    """30 s 持续 siren（每 250 ms 一窗）→ 只应下发 3–4 次，而不是 120 次。"""
    s = HapticScheduler(NullBackend())
    t = 0.0
    while t < 30.0:
        s.propose(_cmd("siren", "alert", t))
        t = round(t + 0.25, 3)
    rep = s.report()
    assert rep["proposals"] == 120
    assert 3 <= rep["accepted"] <= 4, rep
    assert rep["suppression_ratio"] > 0.95
    # 拒绝原因必须是「同类冷却」而不是别的（说明节流真的生效了）
    assert rep["by_reason"].get("same_class_cooldown", 0) > 80


def test_higher_urgency_preempts_active():
    s = HapticScheduler(NullBackend())
    s.propose(_cmd("siren", "alert", 0.0))          # 单次播放 400+400 = 800 ms
    d = s.propose(_cmd("gun_shot", "critical", 0.40))   # 播放中段（剩 400 ms）
    assert d.accepted and d.preempted == "siren:alert"
    assert d.detail["rank_gap"] >= 1


def test_equal_urgency_does_not_preempt():
    s = HapticScheduler(NullBackend())
    s.propose(_cmd("siren", "alert", 0.0))
    d = s.propose(_cmd("car_horn", "alert", 0.40))
    assert not d.accepted
    assert d.reason is Reject.LOWER_PRIORITY


def test_lower_urgency_does_not_preempt_critical():
    s = HapticScheduler(NullBackend())
    s.propose(_cmd("gun_shot", "critical", 0.0))       # 60+3000 ms 长间隔，active 到 3.06 s
    d = s.propose(_cmd("siren", "alert", 1.0))
    assert not d.accepted and d.reason is Reject.LOWER_PRIORITY


def test_last_moment_preemption_is_blocked():
    """剩余不足 min_remaining_ms 时不打断 —— 否则末段会出现 10 ms 的碎震。"""
    s = HapticScheduler(NullBackend(), SchedulePolicy(min_remaining_ms=150))
    s.propose(_cmd("siren", "alert", 0.0))          # 单次播放 400+400=800 ms
    d = s.propose(_cmd("gun_shot", "critical", 0.78))   # 只剩 20 ms
    assert not d.accepted
    assert d.detail["reason"] == "almost_done"


def test_min_gap_blocks_cross_class_burst():
    s = HapticScheduler(NullBackend(), SchedulePolicy(min_gap_s=0.20))
    s.propose(_cmd("siren", "alert", 0.0))
    # 3 个不同类别在同一 100 ms 内连打 → 后两个必须被 min_gap 拦掉
    s.propose(_cmd("gun_shot", "critical", 0.05))
    d = s.propose(_cmd("car_horn", "high", 0.12))
    assert not d.accepted and d.reason is Reject.MIN_GAP


def test_silent_class_is_counted_but_not_emitted():
    s = HapticScheduler(NullBackend())
    d = s.propose(_cmd("dog_bark", "alert", 0.0))
    assert not d.accepted and d.reason is Reject.SILENT
    assert len(s.backend.records) == 0


def test_active_expires_after_playback():
    s = HapticScheduler(NullBackend())
    s.propose(_cmd("siren", "alert", 0.0))
    assert s.active == "siren:alert"
    s.tick(0.9)
    assert s.active is None


# ---------------------------------------------------------------------------
# 抖动测试：10 000 条随机命令，检查不变量
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jitter_invariants(seed):
    rng = np.random.default_rng(seed)
    policy = SchedulePolicy()
    s = HapticScheduler(NullBackend(), policy)

    classes = list(he.V3_CLASSES) + ["dog_bark", "jackhammer"]      # 含静默类
    t = 0.0
    n = 10_000
    for _ in range(n):
        t += float(rng.exponential(0.35))            # 泊松到达
        cls = classes[int(rng.integers(len(classes)))]
        u = he.URGENCY_LEVELS[int(rng.integers(3))]
        s.propose(_cmd(cls, u, t))

    rep = s.report()
    emits = [d for d in s.decisions if d.accepted]

    # 不变量 1：计数自洽
    assert rep["proposals"] == n
    assert rep["accepted"] + rep["rejected"] == n
    assert sum(rep["by_reason"].values()) == n

    # 不变量 2：同类冷却
    last_cls: dict[str, float] = {}
    for d in emits:
        cls = d.cmd.spec.sound_type
        prev = last_cls.get(cls)
        if prev is not None:
            assert d.cmd.t_sched - prev >= policy.same_class_cooldown_s - 1e-9, \
                f"{cls} 同类间隔 {d.cmd.t_sched - prev:.3f}s < 冷却 {policy.same_class_cooldown_s}s"
        last_cls[cls] = d.cmd.t_sched

    # 不变量 3：同码冷却
    last_code: dict[str, float] = {}
    for d in emits:
        prev = last_code.get(d.cmd.code)
        if prev is not None:
            assert d.cmd.t_sched - prev >= policy.same_code_cooldown_s - 1e-9
        last_code[d.cmd.code] = d.cmd.t_sched

    # 不变量 4：任意两次下发的最小间隔
    ts = [d.cmd.t_sched for d in emits]
    assert all(b - a >= policy.min_gap_s - 1e-9 for a, b in zip(ts, ts[1:]))

    # 不变量 5：抢占必须严格提升紧迫度
    for d in emits:
        if d.preempted:
            assert he.URGENCY_RANK[d.cmd.spec.urgency] > he.URGENCY_RANK[
                d.preempted.split(":")[1]]

    # 不变量 6：每条下发都有对应回执（下发与执行不脱节）
    assert len(s.backend.records) == len(emits)

    # 不变量 7：节流必须有实际效果（否则「去重」形同虚设）
    assert rep["suppression_ratio"] > 0.8, rep




if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
