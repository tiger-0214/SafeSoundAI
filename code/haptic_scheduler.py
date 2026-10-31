"""
SafeSoundAI — 触觉调度器：去重 / 抢占 / 节流（haptic_scheduler.py, P1-4）

【为什么单独一层】
`haptic_encoder.DecisionEngine` 解决的是「**同一帧内**多个声源谁优先」。
真实流式场景还有第二类冲突，是**跨帧**的：

  * 持续 30 s 的 siren 会每 250 ms 触发一次 → 不节流就是连续震 30 s，人会被震麻，
    而且「类别码」的时间形状完全消失（这正是 27 号笔记里「触觉疲劳」的具体形态）。
  * siren(alert) 正在播放时来一个 gun_shot(critical) → 必须打断，否则关键事件被淹没。
  * car_horn 的单元间隔是 700 ms，若每窗都重新下发，实际听到的是 250 ms 抖动包络，
    不是 700 ms 的「双响」——**类别码被采样率污染**。

所以本层只做三件事：**同类节流、同码冷却、高优先级抢占**，并把每次拒绝的原因
写成可统计的枚举（否则「为什么没震」无法回答，也就无法调参）。

【与 actuator 的分工】
本层决定「发不发、发哪一条、要不要打断正在播放的」，actuator 决定「怎么发」。
本层不 import 任何后端实现细节，只调用 `backend.emit()` / `backend.records`。

【验收（P1-4）】
`tests/test_haptic_scheduler.py` 用 10 000 条随机命令流做抖动测试，断言：
  1. 同类两次下发间隔 ≥ same_class_cooldown_s
  2. 同码两次下发间隔 ≥ same_code_cooldown_s
  3. 任意两次下发间隔 ≥ min_gap_s
  4. 抢占只在紧迫度高出 ≥ preempt_urgency_gap 档时发生
  5. 计数字段自洽（accept + reject 各原因之和 == 总提议数）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import haptic_encoder as he
from actuator import HapticCommand


class Reject(str, Enum):
    """拒绝原因枚举 —— 对外可统计，禁止用裸字符串。"""

    ACCEPTED = "accepted"
    SAME_CLASS_COOLDOWN = "same_class_cooldown"
    SAME_CODE_COOLDOWN = "same_code_cooldown"
    MIN_GAP = "min_gap"
    LOWER_PRIORITY = "lower_priority"
    SILENT = "silent"
    QUEUE_FULL = "queue_full"


@dataclass
class SchedulePolicy:
    """节流/抢占参数。默认值来自 27 号笔记的触觉疲劳约束，**改动需同步更新笔记**。"""

    same_class_cooldown_s: float = 10.0   # 同类 10 s 内只下发一次（对应「10 s 内不重复触发」）
    same_code_cooldown_s: float = 3.0     # 同码更快地允许重复（紧迫度升级属于同码）
    min_gap_s: float = 0.20               # 任意两次下发的最小间隔（双响间隔 700 ms 不受影响）
    preempt_urgency_gap: int = 1          # 高出 ≥1 档才允许打断
    min_remaining_ms: int = 150           # 剩余不足则不打断（避免末段抖动）
    max_pending: int = 4                  # 队列上限，超出丢弃最低优先级


@dataclass
class Decision:
    accepted: bool
    reason: Reject
    cmd: HapticCommand | None = None
    preempted: str | None = None          # 被打断的码
    detail: dict = field(default_factory=dict)


@dataclass
class _Active:
    code: str
    sound_type: str
    urgency_rank: int
    t_start: float
    t_end: float


class HapticScheduler:
    """把连续的事件流变成稀疏、可辨、不疲劳的触觉下发序列。"""

    def __init__(self, backend, policy: SchedulePolicy | None = None):
        self.backend = backend
        self.policy = policy or SchedulePolicy()
        self._active: _Active | None = None
        self._last_emit: dict[str, float] = {}     # code -> t
        self._last_class: dict[str, float] = {}    # cls  -> t
        self._last_any: float = -1e9
        self._pending: list[HapticCommand] = []
        self.stats: dict[str, int] = {r.value: 0 for r in Reject}
        self.decisions: list[Decision] = []

    # ------------------------------------------------------------------
    def _expire(self, now: float) -> None:
        if self._active is not None and now >= self._active.t_end:
            self._active = None

    def _remaining_ms(self, now: float) -> float:
        return 0.0 if self._active is None else (self._active.t_end - now) * 1e3

    # ------------------------------------------------------------------
    def propose(self, cmd: HapticCommand, now: float | None = None) -> Decision:
        """提交一条命令，返回是否下发。`now` 缺省用 cmd.t_sched。"""
        now = float(cmd.t_sched if now is None else now)
        self._expire(now)

        d = self._decide(cmd, now)
        d.cmd = cmd if d.accepted else None
        self.stats[d.reason.value] += 1
        self.decisions.append(d)

        if d.accepted:
            self._last_emit[cmd.code] = now
            self._last_class[cmd.spec.sound_type] = now
            self._last_any = now
            dur = (cmd.spec.unit_ms + cmd.spec.interval_ms) * max(1, cmd.repeats) / 1e3
            self._active = _Active(cmd.code, cmd.spec.sound_type,
                                   he.URGENCY_RANK[cmd.spec.urgency], now, now + dur)
            rec = self.backend.emit(cmd)
            d.detail["emit_ok"] = rec.ok
            d.detail["t_emit"] = rec.t_emit
        return d

    def _decide(self, cmd: HapticCommand, now: float) -> Decision:
        p = self.policy
        spec = cmd.spec

        if spec.is_silent:
            return Decision(False, Reject.SILENT)

        # 1) 同码冷却
        last_code = self._last_emit.get(cmd.code)
        if last_code is not None and now - last_code < p.same_code_cooldown_s:
            return Decision(False, Reject.SAME_CODE_COOLDOWN,
                            detail={"since_s": round(now - last_code, 3)})

        # 2) 同类冷却
        last_cls = self._last_class.get(spec.sound_type)
        if last_cls is not None and now - last_cls < p.same_class_cooldown_s:
            return Decision(False, Reject.SAME_CLASS_COOLDOWN,
                            detail={"since_s": round(now - last_cls, 3)})

        # 3) 最小间隔（跨类）
        if now - self._last_any < p.min_gap_s:
            return Decision(False, Reject.MIN_GAP,
                            detail={"since_s": round(now - self._last_any, 3)})

        # 4) 正在播放：只允许高优先级打断
        if self._active is not None:
            gap = he.URGENCY_RANK[spec.urgency] - self._active.urgency_rank
            if gap < p.preempt_urgency_gap:
                return Decision(False, Reject.LOWER_PRIORITY,
                                detail={"active": self._active.code, "rank_gap": gap})
            if self._remaining_ms(now) < p.min_remaining_ms:
                return Decision(False, Reject.LOWER_PRIORITY,
                                detail={"active": self._active.code, "reason": "almost_done"})
            return Decision(True, Reject.ACCEPTED, preempted=self._active.code,
                            detail={"preempt": True, "rank_gap": gap})

        # 5) 队列水位（仅用于诊断：本层是同步下发，无异步积压）
        if len(self._pending) >= p.max_pending:
            return Decision(False, Reject.QUEUE_FULL)
        return Decision(True, Reject.ACCEPTED)

    # ------------------------------------------------------------------
    def tick(self, now: float) -> None:
        """推进时钟（清理已播放完毕的 active）。"""
        self._expire(now)

    @property
    def active(self) -> str | None:
        return None if self._active is None else self._active.code

    def report(self) -> dict:
        total = sum(self.stats.values())
        return {
            "policy": {
                "same_class_cooldown_s": self.policy.same_class_cooldown_s,
                "same_code_cooldown_s": self.policy.same_code_cooldown_s,
                "min_gap_s": self.policy.min_gap_s,
                "preempt_urgency_gap": self.policy.preempt_urgency_gap,
            },
            "proposals": total,
            "accepted": self.stats[Reject.ACCEPTED.value],
            "rejected": total - self.stats[Reject.ACCEPTED.value],
            "by_reason": {k: v for k, v in self.stats.items() if v},
            "suppression_ratio": (round(1 - self.stats[Reject.ACCEPTED.value] / total, 4)
                                  if total else None),
            "preemptions": sum(1 for d in self.decisions if d.preempted),
        }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from actuator import NullBackend

    sched = HapticScheduler(NullBackend())
    # 30 s 持续 siren（每 250 ms 一个窗）+ 12 s 处插一个 gun_shot(critical)
    t, k = 0.0, 0
    while t < 30.0:
        pat = he.generate_vibration("siren", urgency="alert", scheme="ear")
        sched.propose(HapticCommand.from_pattern(pat, t_sched=t))
        if abs(t - 12.0) < 1e-6:
            pat2 = he.generate_vibration("gun_shot", urgency="critical", scheme="ear")
            d = sched.propose(HapticCommand.from_pattern(pat2, t_sched=t + 0.02))
            print(f"gun_shot @12.02s → {d.reason.value} preempted={d.preempted}")
        t = round(t + 0.25, 3)
        k += 1
    print(f"\n提议 {k} 次，下发 {sched.stats['accepted']} 次")
    print(sched.report())
