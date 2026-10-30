"""
SafeSoundAI — Demo 1：单文件滑窗推理 → 决策仲裁 → 触觉编码
==========================================================
把「音频进来 → 分类 → 决策 → 震动指令」这条链完整跑一遍并打印出来，
是 MVP 闭环的最小可运行证据。

用法：
  # 任意 wav（会重采样到 16 kHz）
  python3 demo_infer.py --wav path/to/street.wav
  # 直接指定测试集里的某类样本（自动从 fold 9 抽 1 条）
  python3 demo_infer.py --sample siren
  # 十大类各抽一条，批量跑成一张总表（答辩/汇报用）
  python3 demo_infer.py --demo-set

输出：终端表格 + outputs/reports/demo_infer_<源名>.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import viz_style as vs
from dataset import load_splits
from data_audit import md_table
from haptic_encoder import DecisionEngine, Detection, generate_vibration
from mel import waveform_to_logmel
from models import DS_CNN

WIN = config.SEGMENT_SAMPLES          # 1 s
HOP = WIN // 2                        # 0.5 s


# ---------------------------------------------------------------------------
def load_wav(path: str | Path) -> tuple[np.ndarray, str]:
    """读任意音频 → 16 kHz 单声道，并返回该段原始（未标准化）RMS 语义。"""
    import librosa
    y, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
    return y.astype(np.float32), "wav（含绝对响度）"


def load_npy(path: str | Path) -> tuple[np.ndarray, str]:
    """读预处理后的 .npy 片段（已 z-score，绝对响度不可用）。"""
    return np.load(path).astype(np.float32), "npy（z-score 标准化，响度已抹除）"


def windows(y: np.ndarray, win: int = WIN, hop: int = HOP) -> list[tuple[int, np.ndarray]]:
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    return [(i, y[i:i + win]) for i in range(0, len(y) - win + 1, hop)]


def normalize_window(x: np.ndarray) -> np.ndarray:
    """逐窗 z-score —— 与训练期预处理一致（wav 输入必须做这一步）。"""
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


@torch.no_grad()
def infer_windows(model, y: np.ndarray, device) -> tuple[list[int], np.ndarray]:
    wins = windows(y)
    mels = torch.stack([waveform_to_logmel(normalize_window(w)).unsqueeze(0) for _, w in wins]).to(device)
    probs = torch.softmax(model(mels), 1).cpu().numpy()
    return [i for i, _ in wins], probs


def rms_of(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg ** 2)))


def level_from_rms(rms: float) -> tuple[float | None, str]:
    """原始响度 → 距离档位。z-score 数据（rms≈1）不能用来估距。"""
    if rms > 0.9:                       # 已被标准化
        return None, "不可用（z-score）→ 默认 75%"
    if rms > 0.25:
        return 2.0, "极近 <3m（100%）"
    if rms > 0.10:
        return 8.0, "近 3-10m（75%）"
    if rms > 0.04:
        return 20.0, "中 10-30m（50%）"
    if rms > 0.015:
        return 40.0, "远 30-50m（30%）"
    return 60.0, "极远 >50m（15%）"


# ---------------------------------------------------------------------------
def run(model, y, device, threshold: float = 0.8, verbose: bool = True) -> dict:
    starts, probs = infer_windows(model, y, device)
    eng = DecisionEngine(confidence_threshold=threshold)

    rows, events = [], []
    for k, (s, p) in enumerate(zip(starts, probs)):
        top = int(p.argmax())
        seg = y[s:s + WIN]
        rms = rms_of(seg)
        dist, level = level_from_rms(rms)
        name = config.TARGET_CLASSES[top]

        # 决策引擎：把 top-3 中过阈值者一起交给仲裁
        dets = [Detection(class_id=int(i), confidence=float(p[i]))
                for i in np.argsort(-p)[:3] if p[i] >= threshold * 0.6]
        winner = eng.arbitrate(dets)

        rows.append({
            "窗口": k, "起点s": round(s / config.SAMPLE_RATE, 2),
            "Top1": name, "置信度": round(float(p[top]), 3),
            "Top2": f"{config.TARGET_CLASSES[int(np.argsort(-p)[1])]}"
                    f"({p[int(np.argsort(-p)[1])]:.2f})",
            "RMS": round(rms, 3), "响度档": level,
            "触发": "✅" if winner is not None else "—",
        })
        if winner is not None:
            pat = generate_vibration(name, distance_m=dist, scheme="ear", azimuth_deg=None)
            events.append({
                "时间s": round(s / config.SAMPLE_RATE, 2),
                "声音": name, "紧急度": pat.urgency,
                "震动模式": pat.pattern_type,
                "重复间隔ms": pat.interval_ms,
                "强度": pat.pwm_base,
                "角色": config.SAFETY_ROLE[name],
            })
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\n共 {len(rows)} 个滑窗（1 s 窗 / 0.5 s 步进），阈值 {threshold}")
        print(df.to_string(index=False))
        if events:
            print("\n=== 触觉警报事件（决策引擎仲裁后）===")
            ev = pd.DataFrame(events).drop_duplicates(subset=["声音", "时间s"])
            print(ev.to_string(index=False))
        else:
            print("\n无报警触发（全部低于阈值）。")
    return {"windows": df, "events": pd.DataFrame(events), "probs": probs, "starts": starts}


# ---------------------------------------------------------------------------
def write_md(name: str, res: dict, source: str, threshold: float) -> Path:
    md = ["---\ntags: [project, SafeSoundAI, demo, inference]\ncreated: 2026-10-17\n"
          "status: generated\n---\n",
          f"# Demo 1 · 滑窗推理报告 — `{name}`\n",
          f"> 源：`{source}` ｜ 窗口 1 s / 步进 0.5 s ｜ 决策阈值 {threshold} ｜ "
          f"模型 DS-CNN `ds_cnn_best.pth`\n",
          "## 一、逐窗分类结果\n", md_table(res["windows"]),
          "\n## 二、触觉警报事件（经决策引擎仲裁）\n",
          md_table(res["events"]) if len(res["events"]) else "_无触发_",
          "\n## 三、怎么读这张表\n",
          "- **Top1/置信度**：模型对该 1 秒片段的判断；置信度 < 阈值不发警报（防止打扰）。",
          "- **RMS / 响度档**：绝对响度 → 距离档位 → 震动强度（`pwm_base`）。"
          "若源是 `.npy`（预处理后 z-score），响度已被抹除，只能用默认 75%。",
          "- **震动模式**：类型→节奏、紧急度→重复间隔、响度→强度，三维合成一个 `HapticPattern`。",
          "- **触发**：决策引擎在 top-3 中按「紧急度优先，同级取高置信」仲裁，"
          "多个危险声同时出现时只输出一个警报。"]
    out = config.OUTPUT_DIR / "reports" / f"demo_infer_{name}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
def pick_sample(cls: str) -> tuple[str, str]:
    _, _, test = load_splits()
    sub = test[test["target_label"] == cls]
    row = sub.iloc[len(sub) // 2]
    return row["npy_path"], cls


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--wav", type=str)
    g.add_argument("--npy", type=str)
    g.add_argument("--sample", type=str, help="从 test fold 抽该类一条（10 类之一）")
    g.add_argument("--demo-set", action="store_true", help="10 类各抽一条批量跑")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DS_CNN().to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "ds_cnn_best.pth",
                                     map_location=device, weights_only=True))
    model.eval()
    print(f"设备 {device} ｜ 模型 ds_cnn_best.pth（{sum(p.numel() for p in model.parameters()):,} 参数）")

    if args.demo_set:
        summary = []
        for cls in config.TARGET_CLASSES:
            path, _ = pick_sample(cls)
            y, kind = load_npy(path)
            res = run(model, y, device, args.threshold, verbose=False)
            top = res["windows"].iloc[res["windows"]["置信度"].idxmax()]
            summary.append({
                "真值类": cls, "安全角色": config.SAFETY_ROLE[cls],
                "预测": top["Top1"], "置信度": top["置信度"],
                "是否正确": "✅" if top["Top1"] == cls else "❌",
                "触发": "🔔" if len(res["events"]) else "—",
                "震动模式": res["events"].iloc[0]["震动模式"] if len(res["events"]) else "—",
            })
        df = pd.DataFrame(summary)
        print("\n=== 十大类抽样推理（每条取最自信窗口）===")
        print(df.to_string(index=False))
        acc = (df["是否正确"] == "✅").mean()
        print(f"\n抽样命中率 {acc:.0%}（单条抽样，仅供链路验证，非精度评估）")
        out = config.OUTPUT_DIR / "reports" / "demo_infer_demo_set.md"
        out.write_text("\n".join([
            "---\ntags: [project, SafeSoundAI, demo]\ncreated: 2026-10-17\n---\n",
            "# Demo 1b · 十大类抽样推理\n",
            "> 每类从 test fold 9 取中位序号一条，跑完整链路（分类→决策→触觉编码）\n",
            md_table(df),
            "\n**说明**：这是链路连通性验证，不是精度评估；精度见 `aed_eval.md`（Acc 89.82%）。",
        ]), encoding="utf-8")
        print(f"报告 → {out}")
        return

    if args.wav:
        y, kind = load_wav(args.wav)
        src, name = args.wav, Path(args.wav).stem
    elif args.npy:
        y, kind = load_npy(args.npy)
        src, name = args.npy, Path(args.npy).stem
    elif args.sample:
        path, _ = pick_sample(args.sample)
        y, kind = load_npy(path)
        src, name = path, f"sample_{args.sample}"
    else:
        ap.error("需要 --wav / --npy / --sample / --demo-set 之一")

    dur = len(y) / config.SAMPLE_RATE
    print(f"音频：{kind} ｜ {dur:.2f} s ｜ RMS {rms_of(y):.3f}")
    res = run(model, y, device, args.threshold)
    out = write_md(name, res, src, args.threshold)
    print(f"\n报告 → {out}")


if __name__ == "__main__":
    main()
