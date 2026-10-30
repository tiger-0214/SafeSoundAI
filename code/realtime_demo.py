"""
SafeSoundAI — PC 端实时推理 Demo（realtime_demo.py）

把「检测 → 决策 → 触觉编码」整条链跑通（MVP：无真马达，输出触觉模式描述）。

运行（从音频文件模拟流式）：
  python realtime_demo.py --audio path/to/sound.wav --ckpt outputs/models/ds_cnn_best.pth

每 1s（50% overlap）推理一次，打印：
  时间窗口 | 检测结果(类别+置信度) | 音量强度档 | 触觉模式

加 --filter 时同时跑声音过滤（A2.4）：
  输出 filtered wav + 每窗过滤决策（执行器 / 能量变化 / 是否 fail-open）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from audio_utils import load_and_resample, rms_normalize, segment_audio
from haptic_encoder import generate_vibration, DecisionEngine, Detection
from mel import waveform_to_logmel
from models import DS_CNN


def rms_to_distance_level(rms: float) -> tuple[float, str]:
    """粗略把 RMS 映射到距离档位（MVP 启发式，未标定）。"""
    if rms > 0.25:
        return 2.0, "极近(100%)"
    elif rms > 0.10:
        return 8.0, "近(75%)"
    elif rms > 0.04:
        return 20.0, "中(50%)"
    elif rms > 0.015:
        return 40.0, "远(30%)"
    else:
        return 60.0, "极远(15%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=str(config.MODEL_DIR / "ds_cnn_best.pth"))
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--scheme", type=str, default="ear", choices=["ear", "torso", "earbud"])
    ap.add_argument("--filter", action="store_true", help="启用声音过滤管线（A2.4）")
    ap.add_argument("--filter-profile", type=str, default=None,
                    help="过滤档位：comfort / ptsd_default / awareness")
    ap.add_argument("--filter-keep", type=str, default=None,
                    help="keep 集合：critical / emergency / all_safety / passthrough")
    ap.add_argument("--no-tse", action="store_true", help="过滤时禁用方案 B（仅谱增益门）")
    ap.add_argument("--force-b", action="store_true",
                    help="过滤时跳过\"AED 置信度>0.8 才启用 B\"的门控（部署推荐：keep 来自 profile）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DS_CNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    engine = DecisionEngine(confidence_threshold=args.threshold)
    print(f"模型: {args.ckpt} | 设备: {device} | 阈值: {args.threshold}\n")

    # 读原始音频（未归一化，保留音量信息）
    y = load_and_resample(args.audio)
    segs = segment_audio(y)
    print(f"音频 {len(y) / config.SAMPLE_RATE:.1f}s → {len(segs)} 段\n")

    # ---- 可选：声音过滤（A2.4 集成）----
    filter_hist: dict[float, dict] = {}
    if args.filter:
        import filter_config as fc
        from filter_pipeline import FilterPipeline
        pipe = FilterPipeline(profile=args.filter_profile or fc.DEFAULT_PROFILE,
                              keep_set=args.filter_keep or fc.DEFAULT_KEEP_SET,
                              use_tse=not args.no_tse, force_b=args.force_b)
        y_f, fstats = pipe.process_offline(y)
        fc.FILTER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        fpath = fc.FILTER_AUDIO_DIR / f"realtime_filtered_{Path(args.audio).stem}.wav"
        import soundfile as sf
        sf.write(fpath, y_f, config.SAMPLE_RATE, subtype="PCM_16")
        filter_hist = {h["t"]: h for h in pipe.state.history}
        print(f"[过滤] 启用（TSE={'on' if pipe.tse is not None else 'off'}）"
              f" → {fpath.name} | {fstats}\n")

    for i, seg in enumerate(segs):
        raw_rms = float(np.sqrt(np.mean(seg ** 2)))
        dist_m, dist_label = rms_to_distance_level(raw_rms)

        # 归一化（与训练一致）后推理
        seg_norm = rms_normalize(seg)
        mel = waveform_to_logmel(seg_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,100)
        with torch.no_grad():
            logits = model(mel)
            probs = F.softmax(logits, dim=1)[0]

        topk = torch.topk(probs, 3)
        dets = [Detection(int(idx), float(probs[idx]))
                for idx in topk.indices.cpu().numpy()]

        chosen = engine.arbitrate(dets)
        if chosen is None:
            print(f"[{i * 0.5:5.1f}s] 无警报触发 (top1={config.TARGET_CLASSES[int(topk.indices[0])]} "
                  f"{topk.values[0]:.2f})")
            continue

        name = config.TARGET_CLASSES[chosen.class_id]
        pattern = generate_vibration(name, distance_m=dist_m, azimuth_deg=None,
                                     scheme=args.scheme)
        top1_name = config.TARGET_CLASSES[int(topk.indices[0])]
        f_info = filter_hist.get(round(i * 0.5, 2), {})
        f_tag = ""
        if args.filter and f_info:
            f_tag = (f" | 过滤={f_info.get('executor', '?')}"
                     f"{'' if f_info.get('applied', True) else '(直通)'}"
                     f" Δ={f_info.get('delta_db', 0):+.1f}dB")
        print(f"[{i * 0.5:5.1f}s] 检测={top1_name}({topk.values[0]:.2f}) "
              f"→ 仲裁={name}({chosen.confidence:.2f}) | 音量={dist_label} | "
              f"{pattern.describe()}{f_tag}")


if __name__ == "__main__":
    main()
