"""
SafeSoundAI — 混合过滤管线（filter_pipeline.py, A2.4）

方案 C：A（谱增益门，兜底）+ B（类别条件掩码，主用）+ 硬门校验 + fail-open。
全部逻辑按 docs/filter_safety.md 落地；门控级（整窗衰减）默认禁用（ALLOW_GATING=False）。

用法（离线整段模拟流式）：
    python filter_pipeline.py --audio outputs/audio/05_engine_idling.wav --out /tmp/o.wav
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import config
import filter_config as fc
from attenuator import SpectralGateAttenuator, mel_to_linear_gain
from audio_utils import load_and_resample
from tse_model import MaskNet, complex_mask, istft, stft

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK = 1600            # 100 ms 处理块（10 个 hop）
CTX = 16000             # 1 s 左侧上下文（覆盖网络感受野，无未来帧）
FADE = 80               # 5 ms 块间交叉淡化
FAIL_OPEN_CONF = 0.4    # 平均后验 top-1 低于此值即直通（filter_safety §2.3）
MODEL_HOP = 4           # 每 4 个块（400 ms）跑一次模型，中间块零阶保持掩码


@dataclass
class FilterState:
    history: list[dict] = field(default_factory=list)
    silent_suppression: int = 0
    over_suppression: int = 0
    fail_open: int = 0
    blocks: int = 0


class FilterPipeline:
    """混合执行器（A + B + 安全门）。"""

    def __init__(self, profile: str = fc.DEFAULT_PROFILE,
                 tse_ckpt: str | Path | None = None,
                 use_tse: bool = True,
                 device: torch.device = DEVICE,
                 keep_set: str = fc.DEFAULT_KEEP_SET,
                 force_b: bool = False,
                 strength: float | None = None):
        self.att = SpectralGateAttenuator(profile=profile, device=device)
        self.device = device
        self.keep_set_name = keep_set
        self.keep_ids = list(fc.KEEP_SETS[keep_set])
        self.use_tse = use_tse and Path(tse_ckpt or fc.TSE_CKPT).exists()
        self.tse: MaskNet | None = None
        if self.use_tse:
            ck = torch.load(tse_ckpt or fc.TSE_CKPT, map_location=device)
            self.tse = MaskNet().to(device)
            self.tse.load_state_dict(ck["model"] if "model" in ck else ck)
            self.tse.eval()
        # force_b：跳过"置信度 > 0.8 才启用 B"的前置条件。
        # 依据：方案 B 的保留集合来自用户 profile（配置级），**不需要 AED 判断类别**；
        # 实测 AED 偏置严重（11 条 demo 有 8 条判成 car_horn），若等它置信度 >0.8
        # 则 B 永远不会被触发。故部署形态应为：profile → keep 向量 → B 常开，A 仅兜底。
        self.force_b = bool(force_b)
        # 安全强度 α：g_eff = 1 - α(1 - g)，α=1 为模型原生掩码（默认）
        self.strength = fc.TSE_STRENGTH if strength is None else float(strength)
        self._gain_col: np.ndarray | None = None
        self._info_cache: dict = {}
        self.state = FilterState()

    # ------------------------------------------------------------------
    # 单块处理（流式核心）
    # ------------------------------------------------------------------
    def _mask_for_block(self, buf: np.ndarray) -> tuple[np.ndarray, dict]:
        """buf = 上下文 + 当前块（全长）；返回与 buf 等长的**线性**频域增益 (F, T) 及决策信息。

        唯一一次 AED 前向，避免旧版在 process_offline 里重复计算 frame_probs（耗时翻倍）。
        """
        probs = self.att.frame_probs(buf)               # (T,10) 帧级后验
        mean_p = probs.mean(axis=0)
        top = int(np.argmax(mean_p))
        conf = float(mean_p[top])
        F = fc.FILTER_N_FFT // 2 + 1

        info = {"top": config.TARGET_CLASSES[top], "conf": round(conf, 3),
                "executor": "A", "applied": True}

        # fail-open：置信度不足 → 直通
        if conf < FAIL_OPEN_CONF:
            self.state.fail_open += 1
            info.update(executor="fail-open", applied=False)
            return np.ones((F, probs.shape[0]), dtype=np.float32), info

        mel_gain = self.att.gain_curve_db(probs)
        gain_db = mel_to_linear_gain(mel_gain)          # (F,T) dB
        allow = self.att.hard_gate_mask(probs)
        gain_db[:, ~allow] = 0.0
        gain_lin = (10.0 ** (gain_db / 20.0)).astype(np.float32)

        # 方案 B：AED 高置信且 TSE 可用 → 用条件掩码，A 的增益作为上界保护
        # 注意：两者必须都在**线性**域比较，旧版把线性掩码与 dB 取 min 是错的
        if self.tse is not None and (self.force_b or conf > 0.8):
            with torch.no_grad():
                yb = torch.from_numpy(buf.astype(np.float32)).to(self.device)
                Y = stft(yb)
                mag = torch.log1p(Y.abs())[None, None]
                cond = torch.tensor([fc.keep_vector(self.keep_ids)],
                                    dtype=torch.float32, device=self.device)
                mm, pr = self.tse(mag, cond)
                M = complex_mask(mm, pr)[0].abs().cpu().numpy()      # (F,T)
                M = fc.apply_strength(np.clip(M, 0.0, 1.0), self.strength)
            T = min(M.shape[1], gain_lin.shape[1])
            gain_lin = np.minimum(M[:, :T], gain_lin[:, :T]).astype(np.float32)
            info["executor"] = "B"

        return gain_lin, info

    def process_block(self, buf: np.ndarray) -> tuple[np.ndarray, dict]:
        gain, info = self._mask_for_block(buf)
        Y = stft(torch.from_numpy(buf.astype(np.float32)).to(self.device))
        T = Y.shape[-1]
        g = gain[:, :T]
        if g.shape[1] < T:
            g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
        y_out = istft(Y * torch.from_numpy(g).to(device=Y.device, dtype=Y.dtype), len(buf)).cpu().numpy()

        # 硬门：能量塌陷 → 该块直通
        if info.get("executor") in ("A", "B"):
            e_in = float(np.mean(buf[-BLOCK:] ** 2) + 1e-12)
            e_out = float(np.mean(y_out[-BLOCK:] ** 2) + 1e-12)
            delta = 10 * np.log10(e_out / e_in)
            if delta < fc.ANTI_COLLAPSE_DB:
                self.state.over_suppression += 1
                info.update(applied=False, executor="anti-collapse")
                y_out = buf.copy()
            info["delta_db"] = round(float(delta), 2)
        return y_out, info

    # ------------------------------------------------------------------
    # 离线整段（模拟流式，逐块输出 + 交叉淡化）
    # ------------------------------------------------------------------
    def process_offline(self, y: np.ndarray, collect: bool = True) -> tuple[np.ndarray, dict]:
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        out = np.zeros(n, dtype=np.float32)
        prev_tail: np.ndarray | None = None
        t0 = time.perf_counter()

        for blk, start in enumerate(range(0, n, BLOCK)):
            end = min(start + BLOCK, n)
            buf = y[max(0, start - CTX):end]
            Y = stft(torch.from_numpy(buf.astype(np.float32)).to(self.device))
            T = Y.shape[-1]

            # 模型调用降频：每 MODEL_HOP 个块推一次，中间块沿用上一次掩码的末帧（零阶保持）。
            # 依据：AED 分析窗本来就该从 1 s 缩短到 250–400 ms（filter_constraints §三·L2），
            # 逐 100 ms 重跑 1.7 s 上下文属于 17× 冗余（实测 RTF 1.26 → 不可实时）。
            if blk % MODEL_HOP == 0 or self._gain_col is None:
                gain, info = self._mask_for_block(buf)
                self._gain_col = gain[:, -1:].copy()
                self._info_cache = dict(info)
            else:
                info = dict(self._info_cache)
                info["held"] = True
            info = {"t": round(start / config.SAMPLE_RATE, 2), **info}
            self._last_info = info

            if info.get("executor") == "fail-open":
                seg = buf[-BLOCK:].copy()
            else:
                if blk % MODEL_HOP == 0:
                    g = gain[:, :T]
                    if g.shape[1] < T:
                        g = np.pad(g, ((0, 0), (0, T - g.shape[1])), mode="edge")
                else:
                    g = np.repeat(self._gain_col, T, axis=1)
                seg = istft(Y * torch.from_numpy(g).to(device=Y.device, dtype=Y.dtype),
                            len(buf)).cpu().numpy()[-BLOCK:]

                e_in = float(np.mean(buf[-BLOCK:] ** 2) + 1e-12)
                e_out = float(np.mean(seg ** 2) + 1e-12)
                delta = 10 * np.log10(e_out / e_in)
                info["delta_db"] = round(float(delta), 2)
                if delta < fc.ANTI_COLLAPSE_DB:
                    self.state.over_suppression += 1
                    info.update(applied=False, executor="anti-collapse")
                    seg = buf[-BLOCK:].copy()
                else:
                    info.setdefault("executor", "A")
                    info["applied"] = True

            seg = seg[:end - start]
            if prev_tail is not None and len(seg) > FADE:
                w = np.linspace(0, 1, FADE, dtype=np.float32)
                seg[:FADE] = prev_tail[:FADE] * (1 - w) + seg[:FADE] * w
            out[start:end] = seg
            prev_tail = seg.copy()
            self.state.blocks += 1
            if collect:
                self.state.history.append(info)

        elapsed = time.perf_counter() - t0
        stats = {"blocks": self.state.blocks,
                 "fail_open": self.state.fail_open,
                 "over_suppression": self.state.over_suppression,
                 "silent_suppression": self.state.silent_suppression,
                 "rtf": round(elapsed / (n / config.SAMPLE_RATE + 1e-9), 3),
                 "block_ms": round(1000 * elapsed / max(self.state.blocks, 1), 2)}
        return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default=fc.DEFAULT_PROFILE)
    ap.add_argument("--keep-set", default=fc.DEFAULT_KEEP_SET, choices=list(fc.KEEP_SETS))
    ap.add_argument("--no-tse", action="store_true")
    ap.add_argument("--force-b", action="store_true",
                    help="不做 AED 置信度门控，直接用 B（profile→keep 向量→TSE）")
    ap.add_argument("--strength", type=float, default=None,
                    help=f"安全强度 α（默认 {fc.TSE_STRENGTH}；1=原生掩码，"
                         "0.35=守白名单 3 dB 红线的保守档）")
    args = ap.parse_args()

    y = load_and_resample(args.audio)
    pipe = FilterPipeline(profile=args.profile, keep_set=args.keep_set,
                          use_tse=not args.no_tse, force_b=args.force_b,
                          strength=args.strength)
    out, stats = pipe.process_offline(y)
    dst = Path(args.out or (fc.FILTER_AUDIO_DIR / f"pipeline_{Path(args.audio).stem}.wav"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, out, config.SAMPLE_RATE, subtype="PCM_16")
    print(f"TSE 启用: {pipe.tse is not None} | keep_set={args.keep_set}")
    for h in pipe.state.history[:6]:
        print(" ", h)
    print("stats:", json.dumps(stats, ensure_ascii=False))
    print("→", dst)


if __name__ == "__main__":
    main()
