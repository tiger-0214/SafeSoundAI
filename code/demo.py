"""
SafeSoundAI — v3 end-to-end demo (detect → decide → haptic code)
================================================================
One command, no dataset required, no plotting. Runs the whole MVP loop:

    audio -> sliding window -> DS-CNN (4 cls: siren/car_horn/gun_shot/background)
          -> temperature-scaled probabilities -> threshold + debounce
          -> 2-axis haptic code (time axis = category, amplitude axis = urgency)

Usage
-----
    # 1) built-in synthetic street scene (12 s, assembled from the fold-9 test set)
    python3 demo.py

    # 2) your own recording (resampled to 16 kHz mono)
    python3 demo.py --wav street.wav

    # 3) a single preprocessed 1-s segment (.npy, already z-scored)
    python3 demo.py --npy seg.npy

Notes
-----
* Model input is 128 log-Mel x 100 frames at n_fft=1024 (640/160 win/hop, 16 kHz).
* The 1-s analysis window is the current latency bottleneck: compute is 2-4 ms,
  the cost is waiting for the buffer to fill (see reports/winlen_bench.md).
* Urgency uses the semantic prior + duration + repetition only. Audio level is
  deliberately NOT used (AGC and mic coupling destroy absolute level).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import config
from dataset import load_splits
from haptic_encoder import generate_vibration
from mel import waveform_to_logmel
from models import DS_CNN

WIN = config.SEGMENT_SAMPLES      # 1 s @ 16 kHz
HOP = WIN // 2                    # 0.5 s
N_FFT = 1024                      # the "fft1024" variant (removes degenerate filters)
HAZARD = ("siren", "car_horn", "gun_shot")


# --------------------------------------------------------------------------- io
def load_model(ckpt: Path, device: torch.device) -> tuple[DS_CNN, list[str]]:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    classes = blob["classes"]
    model = DS_CNN(n_classes=len(classes)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, classes


def load_wav(path: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def build_scene(seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Assemble a 12 s street scene from test-fold segments; return audio + timeline."""
    _, _, test = load_splits()
    rng = np.random.default_rng(seed)
    script = [("background", 2.0), ("siren", 3.0), ("background", 1.0),
              ("car_horn", 2.0), ("background", 1.0), ("gun_shot", 1.0),
              ("background", 2.0)]
    chunks, timeline, t = [], [], 0.0
    for cls, dur in script:
        if cls == "background":
            pool = test[~test["target_label"].isin(HAZARD)]
        else:
            pool = test[test["target_label"] == cls]
        n_seg = max(1, int(np.ceil(dur)))
        picked = rng.choice(len(pool), size=min(n_seg, len(pool)), replace=False)
        seg = np.concatenate([np.load(p) for p in pool.iloc[picked]["npy_path"]])[:int(dur * config.SAMPLE_RATE)]
        if len(seg) < int(dur * config.SAMPLE_RATE):
            seg = np.pad(seg, (0, int(dur * config.SAMPLE_RATE) - len(seg)))
        chunks.append(seg)
        if cls != "background":
            timeline.append((t, t + dur, cls))
        t += dur
    return np.concatenate(chunks).astype(np.float32), timeline


# ---------------------------------------------------------------------- inference
def infer(model: DS_CNN, classes: list[str], x: np.ndarray, device: torch.device,
          temperature: float = 1.0) -> np.ndarray:
    x = (x - x.mean()) / (x.std() + 1e-8)                  # per-window z-score, as in training
    mel = waveform_to_logmel(x.astype(np.float32), n_fft=N_FFT)   # (128, 100)
    with torch.no_grad():
        logits = model(mel.unsqueeze(0).unsqueeze(0).to(device))
        prob = torch.softmax(logits / temperature, dim=1).cpu().numpy()[0]
    return prob


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--wav", type=str, help="任意音频文件")
    g.add_argument("--npy", type=str, help="预处理后 1 s 片段（已 z-score）")
    ap.add_argument("--ckpt", type=str,
                    default=str(config.MODEL_DIR / "dscnn_v2_4cls_fft1024.pt"))
    ap.add_argument("--tau", type=float, default=0.60, help="hazard 触发阈值")
    ap.add_argument("--temperature", type=float, default=1.0, help="温度标定（fft1024 变体 T≈1.0 即已校准）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(Path(args.ckpt), device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device {device} | ckpt {Path(args.ckpt).name} | {n_par:,} params | classes {classes}")

    if args.npy:
        y, timeline = np.load(args.npy).astype(np.float32), []
    elif args.wav:
        y, timeline = load_wav(args.wav), []
    else:
        y, timeline = build_scene()
        print("source: built-in 12 s scene (fold-9 test segments)")

    n_win = max(1, 1 + (len(y) - WIN) // HOP)
    print(f"audio: {len(y)/config.SAMPLE_RATE:.2f} s | {n_win} windows (1 s / 0.5 s hop) | tau={args.tau}\n")
    print(f"{'t(s)':>5} {'top1':<11}{'p':>6} {'top2':<11}{'p2':>6}  trigger")

    events, prev = [], None
    for i in range(n_win):
        seg = y[i * HOP: i * HOP + WIN]
        if len(seg) < WIN:
            seg = np.pad(seg, (0, WIN - len(seg)))
        prob = infer(model, classes, seg, device, args.temperature)
        order = np.argsort(-prob)
        top1, p1 = classes[order[0]], float(prob[order[0]])
        top2, p2 = classes[order[1]], float(prob[order[1]])
        fire = top1 in HAZARD and p1 >= args.tau
        if fire and prev != top1:
            events.append({"t": i * HOP / config.SAMPLE_RATE, "cls": top1, "p": p1})
        prev = top1 if fire else None
        flag = "FIRE" if fire else ""
        print(f"{i*HOP/config.SAMPLE_RATE:5.1f} {top1:<11}{p1:6.3f} {top2:<11}{p2:6.3f}  {flag}")

    print("\n--- haptic alarm events (after arbitration) ---")
    if not events:
        print("none")
    print(f"{'t(s)':>5} {'class':<10}{'urgency':<10}{'pattern':<10}{'interval':>9} {'amplitude':>10}  code")
    seen: dict[str, float] = {}
    for ev in events:
        cls, t = ev["cls"], ev["t"]
        dur = t - seen.get(cls, t) if cls in seen else 0.0
        seen[cls] = t
        pat = generate_vibration(cls, duration_s=dur if dur > 0 else None,
                                 repeats_10s=sum(1 for e in events if e["cls"] == cls and abs(e["t"] - t) <= 10))
        print(f"{t:5.1f} {cls:<10}{pat.urgency:<10}{pat.pattern_type:<10}"
              f"{pat.interval_ms:>7}ms {pat.pwm_base:>10.2f}  {pat.code}")

    if timeline:
        print("\n--- ground truth in the built-in scene ---")
        for a, b, c in timeline:
            hit = [e for e in events if a - 1.0 <= e["t"] <= b + 1.0 and e["cls"] == c]
            if hit:
                status = "detected @ %.1fs" % hit[0]["t"]
            else:
                status = "MISSED"
            print(f"  {a:4.1f}-{b:4.1f} s  {c:<10} {status}")


if __name__ == "__main__":
    main()
