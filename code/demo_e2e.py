#!/usr/bin/env python3
"""一键端到端流式 demo（P1-2）：从"没有任何输入"到"看到延迟分布 + 事件级命中率"。

设计原则：
  1. **一条命令跑通** —— `make demo-e2e` / `python3 code/demo_e2e.py`，不需要记参数。
  2. **环境先体检，缺什么当场说清楚**，不要跑到一半抛栈。
  3. **有数据集用真音频，没有就退化到合成源**，并在报告里标明用的是哪种，
     绝不让合成源的数字冒充真音频结论。
  4. 输出的每个数字都能在产物文件里查到（summary.json / windows.jsonl）。

流程：
  ① 环境体检（python / torch / cuda / onnxruntime / 数据 / 检查点 / 元数据）
  ② 造测试场景（真音频 20 s，48 kHz，含 12 个人工标注事件）
  ③ 流式跑一遍（模型在环，wavefile 后端落 haptics.wav）
  ④ 事件级评分：每个真值事件有没有被对上？误报多少？首个触觉延迟多少？
  ⑤ 打印摘要 + 写产物

退出码：0 = 全流程通；1 = 有致命项（缺模型/缺依赖/零窗口）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import config  # noqa: E402

REPORT = Path(config.OUTPUT_DIR) / "reports" / "demo"
CKPT = Path(config.MODEL_DIR) / "dscnn_v2_4cls_fft1024.pt"
META = Path(config.PREPROCESSED) / "trigger_v2" / "metadata_v2.csv"

DURATION_S = 60.0
WIN_S = 1.0
DEVICE_SR = 48000


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
# ① 环境体检
# --------------------------------------------------------------------------
def health_check() -> dict:
    hr("① 环境体检")
    h: dict = {"fatal": [], "warn": []}

    print(f"  python      {sys.version.split()[0]}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  torch       {torch.__version__}  cuda={cuda}"
              f"{'  ' + torch.cuda.get_device_name(0) if cuda else ''}")
    except Exception as e:                              # noqa: BLE001
        h["fatal"].append(f"torch 不可用：{e}")
        print(f"  torch       ❌ {e}")

    for mod in ("numpy", "scipy", "soundfile", "pandas"):
        try:
            m = __import__(mod)
            print(f"  {mod:<11} {getattr(m, '__version__', 'ok')}")
        except Exception as e:                          # noqa: BLE001
            h["fatal"].append(f"缺 {mod}: {e}")
            print(f"  {mod:<11} ❌ {e}")

    try:
        import onnxruntime
        print(f"  onnxruntime {onnxruntime.__version__}  (真机部署路径)")
    except Exception:                                   # noqa: BLE001
        h["warn"].append("onnxruntime 缺失 → 只能跑 PyTorch 路径，真机部署另需装")

    data_dir = Path(config.PREPROCESSED)
    if data_dir.exists():
        print(f"  数据目录     {data_dir} ✓")
    else:
        h["warn"].append(f"数据目录 {data_dir} 不存在 → 场景退化为合成源")

    if CKPT.exists():
        print(f"  检查点       {CKPT.name}  {CKPT.stat().st_size / 1024:.0f} KB ✓")
    else:
        h["fatal"].append(f"缺检查点 {CKPT}（先跑 train_dscnn_v2.py）")
        print(f"  检查点       ❌ 缺 {CKPT}")

    if META.exists():
        print(f"  元数据       {META.name} ✓")
    else:
        h["warn"].append(f"缺 {META} → 无法用真音频，退化为合成源")
    return h


# --------------------------------------------------------------------------
# ② 造场景
# --------------------------------------------------------------------------
def build_scene() -> tuple[Path, Path, str]:
    hr("② 造测试场景")
    out = Path("/tmp/demo_scene48k.wav")
    if META.exists() and (Path(config.PREPROCESSED)).exists():
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, str(CODE / "_make_scene.py"),
             "--duration", str(DURATION_S), "--seed", "11", "--out", str(out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("  ⚠ 造景失败，退化到合成源：", r.stderr.strip().splitlines()[-1:])
        else:
            ev = out.with_name(out.stem + "_events.txt")
            print(f"  真音频场景 {out}  {DURATION_S:.0f}s @ {DEVICE_SR} Hz "
                  f"（{time.perf_counter() - t0:.1f}s 造完）")
            return out, ev, "real"
    # 退化：合成源（多音调 + 静音交替，只验证管道，不代表识别能力）
    import soundfile as sf
    sr = DEVICE_SR
    n = int(DURATION_S * sr)
    t = np.arange(n) / sr
    y = 0.2 * np.sin(2 * np.pi * 880 * t) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    sf.write(out, y.astype(np.float32), sr)
    ev = out.with_name(out.stem + "_events.txt")
    ev.write_text("", encoding="utf-8")
    print(f"  ⚠ 合成场景 {out}（无真值，事件级指标不可用）")
    return out, ev, "synth"


def read_events(path: Path) -> list[tuple[float, str, float]]:
    """读 <scene>_events.txt：每行 `起点秒 \t 标签 \t 时长秒`（旧格式 2 列也兼容）。"""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = line.split("\t")
        out.append((float(f[0]), f[1], float(f[2]) if len(f) > 2 else 1.0))
    return out


# --------------------------------------------------------------------------
# ③ 流式跑
# --------------------------------------------------------------------------
def run_stream(wav: Path) -> tuple[dict, list]:
    hr("③ 流式运行（模型在环）")
    from collections import Counter

    from streaming_runtime import FileSource, StreamingRuntime, build_backend

    rt = StreamingRuntime(
        backend=build_backend("wavefile", path="/tmp/demo_haptics.wav",
                              duration_s=DURATION_S + 5),
        ckpt=CKPT, device_sr=DEVICE_SR)
    print(f"  AED 头: {getattr(rt.pipe.head, 'classes', '未加载')}")
    warm = rt.warmup(3)
    print(f"  预热 {warm:.3f}s（不计入延迟统计）")
    t0 = time.perf_counter()
    rt.run(FileSource(str(wav), sr=DEVICE_SR, loop=False), duration_s=DURATION_S,
           pacing="max", verbose=False)
    wall = time.perf_counter() - t0

    summ = rt.recorder.summary()
    traces = list(rt.recorder.traces)
    # 把调度判定挂进汇总（recorder 只管延迟，不管决策）
    # "提议"= 真的生成了触觉命令（code 非空）。argmax 是危险类但 code 为空
    # 表示仲裁层没通过（置信度不足/背景拒识），那是另一类统计，不能混进抑制率。
    proposed = sum(1 for tr in traces if not tr.dropped and tr.code is not None)
    below = sum(1 for tr in traces
                if not tr.dropped and tr.code is None and tr.cls not in (None, "background"))
    sched = Counter(tr.reject_reason or "accepted"
                    for tr in traces if not tr.dropped and tr.code is not None)
    n_acc = summ.get("n_accepted", 0)
    summ["sched"] = dict(sched)
    summ["n_proposed"] = proposed
    summ["n_below_threshold"] = below
    summ["suppress_rate"] = round(1 - n_acc / proposed, 4) if proposed else None

    print(f"  音频 {DURATION_S:.1f}s / 墙钟 {wall:.2f}s（{DURATION_S / wall:.1f}×）")
    print(f"  窗口 {summ['n_windows']}｜丢窗 {summ['n_dropped']}｜触觉下发 {n_acc}"
          f"｜提议 {proposed}｜抑制率 {summ['suppress_rate']}")
    print(f"  危险类但未过仲裁 {below} 窗（置信度/拒识门槛拦下，不参与抑制率）")
    if "compute_ms" in summ:
        c, e = summ["compute_ms"], summ["e2e_ms"]
        print(f"  单窗算力 p50 {c['p50']} / p95 {c['p95']} / max {c['max']} ms")
        print(f"  端到端 e2e   p50 {e['p50']} / p95 {e['p95']} ms（含攒窗）")
    print(f"  调度判定 {summ['sched']}")

    out_dir = Path(config.OUTPUT_DIR) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rt.recorder.to_jsonl(out_dir / "demo_windows.jsonl")
    return summ, traces


# --------------------------------------------------------------------------
# ④ 事件级评分
# --------------------------------------------------------------------------
def score_events(events: list[tuple[float, str, float]], traces: list) -> dict:
    """事件级：一个事件算命中，当且仅当"落在事件区间内的窗口"判出了正确类别。

    为什么这样定：流式系统关心的是「声音响起后多久被认出来」和「会不会漏」，
    段级 acc 已由 audit_detection_numbers 负责。窗口必须整段落在事件内
    （[t+1.0, t+dur] 这段），否则窗口混进前一段静音，测的是「混音鲁棒性」而非识别能力 ——
    那是另一个实验，不该混进 demo 的命中率。
    """
    if not events:
        return {"available": False}
    hits, miss, lat = [], [], []
    for t_ev, label, dur in events:
        if label == "background":
            continue
        # 只取完全落在事件内部的窗口：win_end 在 [t+win, t+dur] 内
        cand = [tr for tr in traces
                if not tr.dropped and t_ev + WIN_S <= tr.win_end_s <= t_ev + dur + 0.01]
        got = next((tr for tr in cand if tr.cls == label), None)
        if got:
            hits.append(label)
            lat.append((got.win_end_s - t_ev) * 1000)
        else:
            miss.append({"t": t_ev, "label": label, "dur": dur,
                         "pred": [(round(tr.win_end_s, 2), tr.cls, tr.conf) for tr in cand][:4],
                         "note": "无窗口判成该类" if cand else "事件短于一个窗口，无法整段落窗"})

    # 误报：整段落在 background 事件内部的窗口被判成危险类
    fa = []
    for t_ev, label, dur in events:
        if label != "background":
            continue
        for tr in traces:
            if tr.dropped or tr.cls in (None, "background"):
                continue
            if t_ev + WIN_S <= tr.win_end_s <= t_ev + dur + 0.01 and tr.conf >= 0.8:
                fa.append({"t": round(tr.win_end_s, 2), "pred": tr.cls, "conf": tr.conf})
    # 首响延迟：窗口只要求与事件有重叠（win_end ≥ t+hop），更接近系统的实际反应时间
    lat_first = []
    for t_ev, label, dur in events:
        if label == "background":
            continue
        got = next((tr for tr in traces
                    if not tr.dropped and tr.cls == label
                    and t_ev + 0.25 <= tr.win_end_s <= t_ev + dur + 0.5), None)
        if got:
            lat_first.append((got.win_end_s - t_ev) * 1000)

    n = max(1, len(hits) + len(miss))
    return {"available": True,
            "n_events": len(hits) + len(miss),
            "recall": round(len(hits) / n, 4),
            "miss": miss[:10],
            "false_alarm_windows": fa[:10],
            "n_false_alarm": len(fa),
            "latency_ms_to_detect": {
                "p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
                "p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
                "n": len(lat)},
            "latency_ms_first_partial": {
                "p50": round(float(np.percentile(lat_first, 50)), 1) if lat_first else None,
                "p95": round(float(np.percentile(lat_first, 95)), 1) if lat_first else None,
                "n": len(lat_first)},
            "note": ("latency_ms_to_detect =「声音起点 → 判出该类的窗口右端」，"
                     "不含上报/振动起振；真机总延迟 = 该值 + 触觉 t_air/t_rise。"
                     "窗口必须整段落在事件内，混音鲁棒性不在此口径内。")}


# --------------------------------------------------------------------------
def main() -> int:
    print("SafeSoundAI 端到端 demo（检测 → 决策 → 触觉）")
    h = health_check()
    if h["fatal"]:
        hr("❌ 致命项，先修这些再跑")
        for f in h["fatal"]:
            print("  -", f)
        return 1

    wav, ev_path, kind = build_scene()
    summ, traces = run_stream(wav)
    events = read_events(ev_path)
    ev_score = score_events(events, traces)

    hr("④ 事件级评分")
    if ev_score["available"]:
        print(f"  真值事件 {ev_score['n_events']} 个（不含 background）｜"
              f"命中率 {ev_score['recall']:.2%}")
        if ev_score["latency_ms_to_detect"]["p50"] is not None:
            print(f"  整段落窗判出延迟 p50 {ev_score['latency_ms_to_detect']['p50']} ms "
                  f"/ p95 {ev_score['latency_ms_to_detect']['p95']} ms")
        lp = ev_score.get("latency_ms_first_partial", {})
        if lp.get("p50") is not None:
            print(f"  首响延迟（窗口与事件有重叠即算） p50 {lp['p50']} ms / p95 {lp['p95']} ms")
        if ev_score["miss"]:
            print(f"  漏报 {len(ev_score['miss'])} 个，前几个：")
            for m in ev_score["miss"][:4]:
                print(f"    {m['t']:6.2f}s 真值 {m['label']:<9} 时长 {m['dur']}s"
                      f" → {m['note']}")
                for tt, cc, cfd in m["pred"][:3]:
                    print(f"        窗口 {tt:6.2f}s 判成 {cc} ({cfd:.2f})")
        print(f"  整段落窗的误报 {ev_score['n_false_alarm']} 窗"
              f"（背景事件内被判成危险类且 conf≥0.8）")
    else:
        print("  合成源无真值，跳过（只验证了管道通不通）")

    REPORT.mkdir(parents=True, exist_ok=True)
    out = {"scene_kind": kind, "scene": str(wav), "duration_s": DURATION_S,
           "device_sr": DEVICE_SR, "ckpt": str(CKPT),
           "stream": summ, "event_level": ev_score,
           "health": {"fatal": h["fatal"], "warn": h["warn"]}}
    (REPORT / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    hr("产物")
    print(f"  {REPORT / 'summary.json'}")
    print(f"  {Path(config.OUTPUT_DIR) / 'reports' / 'demo_windows.jsonl'}")
    print("  /tmp/demo_haptics.wav   （触觉序列叠在音轨上，可直接听）")
    ok = summ["n_windows"] > 0 and summ["n_dropped"] == 0
    hr("✅ demo 通过" if ok else "❌ demo 有异常（窗口数 0 或有丢窗）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
