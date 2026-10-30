"""SafeSoundAI v3 实验核心：波形级在线前端 + 多标签 sigmoid + 时域上下文 + 拒识评测。

设计要点（相对 v2 的改动，逐条对应已知缺陷）：
  D1 mel 退化带：v2 用 512/128 时有 1 个滤波器全零（死通道）→ v3 改 n_fft=1024/win=640，0 死带。
  D2 无能量/增益鲁棒性：训练期波形域 ±8 dB 增益抖动（v2 只有 SpecAugment）。
  D3 无重叠事件能力：波形域随机混入第二段（相对增益 U(-12,+3) dB）+ 多标签并集目标。
  D4 闭集互斥：sigmoid 多标签头，4 个告警 tag + 1 个 other（非告警声音）tag；拒识=全部告警 tag 低于阈值。
  D5 时域上下文只在推理侧 K=5 平滑：ctx 变体把 5 个连续段（2.5 s）堆成通道输入模型。
  D6 GAP 池化丢失时间定位：注意力池化（时间轴）+ 段内膨胀时间卷积。
  D7 未知类无评测集：把 15 个 ESC-50 类整类从训练池剔除，作为真正未见的 unknown 评测集
     （10 类 benign + 5 类 hazard），全部 5,400 段从未参与训练。
"""
from __future__ import annotations

import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import average_precision_score, roc_auc_score

SR = 16000
SEG = 16000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EXP = Path(os.environ.get("SAFESOUND_EXP_DIR", "outputs/exp"))
V3 = EXP / "v3"
V3.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("SAFESOUND_DATA_DIR", "preprocessed_data"))
META_V2 = DATA / "trigger_v2" / "metadata_v2.csv"
META_ALL = DATA / "metadata_clean.csv"

ALARMS = ["siren", "car_horn", "gun_shot", "fireworks"]
BASE5 = ALARMS + ["background"]
TAGS = ALARMS + ["other"]          # sigmoid 输出维度 = 5

# 整类剔除、作为未见 unknown 评测集（10 benign + 5 hazard，各 40 条录音 / 360 段）
UNSEEN_BENIGN = ["keyboard_typing", "mouse_click", "drinking_sipping", "brushing_teeth",
                 "clock_tick", "pouring_water", "toilet_flush", "laughing", "sneezing", "snoring"]
UNSEEN_HAZARD = ["glass_breaking", "crying_baby", "chainsaw", "helicopter", "thunderstorm"]
UNSEEN = UNSEEN_BENIGN + UNSEEN_HAZARD


# ------------------------------------------------------------------ 元数据
class WrappedWaves:
    """waves.npy 是按完整 metadata_v2 行序建的；剔除 unseen 类后用行映射保持对齐。"""

    def __init__(self, w, rows):
        self.w, self.rows = w, np.asarray(rows, dtype=np.int64)

    def __getitem__(self, i):
        return self.w[self.rows[i]]

    def __len__(self):
        return len(self.rows)


def load_pool(drop_unseen: bool = True) -> pd.DataFrame:
    """v2 pool；drop_unseen=True 时把 15 个 unseen 类整类删除（用于训练/验证/测试）。"""
    df = pd.read_csv(META_V2)
    df["wave_row"] = np.arange(len(df))
    if drop_unseen:
        df = df[~df.label_raw.isin(UNSEEN)].reset_index(drop=True)
    df["cls"] = df["label"].map({c: i for i, c in enumerate(BASE5)})
    return df


def build_waves(df: pd.DataFrame, cache) -> np.ndarray:
    cache = Path(cache)
    """把 df 的 1 s 波形写成 memmap；v2 池直接复用已有 waves.npy（行序与 metadata_v2 一致）。"""
    if cache.exists():
        w = np.load(cache, mmap_mode="r")
        if w.shape == (len(df), SEG) or w.shape[1] == SEG:
            return w
    out = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(len(df), SEG))
    for i, p in enumerate(df.npy_path.values):
        x = np.load(p).astype(np.float32)
        if len(x) != SEG:
            x = np.pad(x, (0, max(0, SEG - len(x))))[:SEG]
        out[i] = x
    out.flush()
    return np.load(cache, mmap_mode="r")


def load_unknown() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """未见 unknown 评测集：15 个整类剔除的 ESC-50 类。返回 (waves, hazard_flag, labels)。"""
    m = pd.read_csv(META_ALL)
    m = m[m.label.isin(UNSEEN)].reset_index(drop=True)
    m["rec_group"] = m.dataset + "|" + m.original_file
    w = build_waves(m, V3 / "waves_unknown.npy")
    haz = m.label.isin(UNSEEN_HAZARD).values.astype(np.int64)
    m[["label", "rec_group", "fold", "segment_idx"]].to_csv(V3 / "unknown_meta.csv", index=False)
    return w, haz, m.label.tolist()


def build_ctx_index(df: pd.DataFrame, k: int = 5) -> np.ndarray:
    """每个样本的 k 个连续段（同一录音内，按 segment_idx 排序，越界用边界段补齐）。"""
    idx = np.arange(len(df))
    out = np.empty((len(df), k), dtype=np.int64)
    half = k // 2
    for _, g in df.groupby("rec_group"):
        g_sorted = g.sort_values(["segment_idx"]).index.values
        n = len(g_sorted)
        for j, i in enumerate(g_sorted):
            for t in range(k):
                out[j if False else i, t] = g_sorted[min(max(j + t - half, 0), n - 1)]
    return out


# ------------------------------------------------------------------ 前端
def make_mel(n_fft=1024, win=640, n_mels=128, f_min=50.0, f_max=7500.0, hop=160):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0, normalized=False).to(DEV)


@torch.no_grad()
def wav_to_mel(mel, x: torch.Tensor, cmvn: str = "none") -> torch.Tensor:
    """x: (B, T) 波形 → (B,1,n_mels,100) log-Mel。cmvn: none / seg（逐段 z-score）。"""
    m = torch.log(mel(x) + 1e-8)
    if m.shape[-1] > 100:
        m = m[..., :100]
    elif m.shape[-1] < 100:
        m = F.pad(m, (0, 100 - m.shape[-1]))
    if cmvn == "seg":
        m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
    return m.unsqueeze(1)


def spec_augment(m: torch.Tensor, n_f=2, n_t=2, max_w=12) -> torch.Tensor:
    B, _, C, T = m.shape
    for b in range(B):
        for _ in range(n_f):
            w = int(torch.randint(0, max_w, (1,)))
            if w:
                f0 = int(torch.randint(0, C - w, (1,)))
                m[b, :, f0:f0 + w, :] = m[b, :, :1, :1].mean()
        for _ in range(n_t):
            w = int(torch.randint(0, max_w, (1,)))
            if w:
                t0 = int(torch.randint(0, T - w, (1,)))
                m[b, :, :, t0:t0 + w] = m[b, :, :1, :1].mean()
    return m


# ------------------------------------------------------------------ 批构造
def gather(waves, idx: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([waves[i] for i in idx])).to(DEV).float()


def labels_sigmoid(y_cls: np.ndarray) -> torch.Tensor:
    """5 类（4 告警 + background）→ (N,5) 多标签目标：告警 one-hot，background → other=1。"""
    n = len(y_cls)
    t = np.zeros((n, len(TAGS)), dtype=np.float32)
    for i, c in enumerate(y_cls):
        if c == 4:
            t[i, 4] = 1.0          # other
        else:
            t[i, c] = 1.0
    return torch.from_numpy(t)


# ------------------------------------------------------------------ 模型
class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        return F.relu(self.bn2(self.pw(x)), inplace=True)


class NetV3(nn.Module):
    """DS-CNN 主干 + 段内膨胀时间卷积 + 注意力池化 + 多头。

    mode: 'softmax'（5 类闭集，对照）/ 'sigmoid'（4 告警 + other 多标签）
    ctx:  输入通道数（1=单段，5=五个连续段）
    """

    def __init__(self, mode="sigmoid", ctx=1, attn=True, tdil=True, n_out=5, n_mels=128, frames=100):
        super().__init__()
        self.mode, self.attn, self.tdil = mode, attn, tdil
        self.stem = nn.Sequential(nn.Conv2d(ctx, 32, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.b1, self.b2 = DSBlock(32, 64, 2), DSBlock(64, 128, 2)
        self.b3, self.b4 = DSBlock(128, 128, 2), DSBlock(128, 256, 1)
        if tdil:
            self.t1 = nn.Conv1d(256, 256, 3, padding=1, dilation=1, groups=256, bias=False)
            self.t2 = nn.Conv1d(256, 256, 3, padding=2, dilation=2, groups=256, bias=False)
            self.tbn = nn.BatchNorm1d(256)
        if attn:
            self.att = nn.Conv1d(256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat = 256 * (2 if attn else 1)
        self.fc = nn.Linear(feat, n_out)

    def embed(self, x):
        h = self.b4(self.b3(self.b2(self.b1(self.stem(x)))))     # (B,256,H',T')
        if self.tdil:
            s = h.mean(dim=2)                                     # (B,256,T')
            s = F.relu(self.tbn(s + self.t2(F.relu(self.t1(s)))), inplace=True)
            h = h + s.unsqueeze(2)
        return h

    def forward(self, x, return_embed=False):
        h = self.embed(x)
        parts = [self.gap(h).flatten(1)]
        if self.attn:
            s = h.mean(dim=2)                                     # (B,256,T')
            a = torch.softmax(self.att(s), dim=2)                 # (B,1,T')
            parts.append((s * a).sum(2))
        z = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        logits = self.fc(z)
        return (logits, z) if return_embed else logits


# ------------------------------------------------------------------ 训练/推理
def forward_logits(model, waves, idx, mel, ctx_idx=None, cmvn="none", bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            if ctx_idx is None:
                x = wav_to_mel(mel, gather(waves, b), cmvn)
            else:
                xs = [wav_to_mel(mel, gather(waves, ctx_idx[b][:, t]), cmvn) for t in range(ctx_idx.shape[1])]
                x = torch.cat(xs, 1)
            outs.append(model(x))
    return torch.cat(outs).cpu()


def build_input(mel, waves, batch_idx, ctx_idx, cmvn, gain_db=0.0):
    """把一个 batch 变成网络输入 (B, ctx, n_mels, 100)。ctx=1 时就是单段。"""
    def prep(idx):
        g = gather(waves, idx)
        if gain_db:
            g = g * torch.pow(10.0, (torch.rand(len(idx), 1, device=DEV) * 2 - 1) * gain_db / 20)
        return wav_to_mel(mel, g, cmvn)

    if ctx_idx is None:
        return prep(batch_idx)
    return torch.cat([prep(ctx_idx[batch_idx][:, t]) for t in range(ctx_idx.shape[1])], 1)


def train_v3(waves, df, tr_idx, va_idx, cfg, seed=42, log=None, return_model=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mel = make_mel(cfg.get("n_fft", 1024), cfg.get("win", 640), cfg.get("n_mels", 128),
                   cfg.get("f_min", 50.0), cfg.get("f_max", 7500.0))
    ctx = cfg.get("ctx", 1)
    ctx_idx = build_ctx_index(df, ctx) if ctx > 1 else None
    mode = cfg["mode"]
    model = NetV3(mode=mode, ctx=ctx, attn=cfg.get("attn", True), tdil=cfg.get("tdil", True),
                  n_out=5, n_mels=cfg.get("n_mels", 128)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1.5e-3), weight_decay=1e-4)
    bs = cfg.get("bs", 64)
    epochs = cfg.get("epochs", 25)
    steps = math.ceil(len(tr_idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1.5e-3),
                                                total_steps=epochs * steps, pct_start=0.25)
    y_cls = df.cls.values
    yt_sig = labels_sigmoid(y_cls)
    lossf_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    lossf_bce = nn.BCEWithLogitsLoss()
    p_mix, gain_db, cmvn = cfg.get("p_mix", 0.0), cfg.get("gain_db", 0.0), cfg.get("cmvn", "none")
    best = (-1, None)
    best_state = None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            xb = gather(waves, b)
            if gain_db:
                xb = xb * torch.pow(10.0, (torch.rand(len(b), 1, device=DEV) * 2 - 1) * gain_db / 20)
            yb_sig = yt_sig[torch.from_numpy(b)].to(DEV)
            if p_mix and len(b) > 1:
                # 与另一段随机混叠（相对增益 U(-12,+3) dB），目标取并集 → 训练重叠事件能力
                b2 = np.random.choice(tr_idx, size=len(b))
                x2 = gather(waves, b2)
                lvl = torch.pow(10.0, (torch.rand(len(b), 1, device=DEV) * 15 - 12) / 20)
                m = (torch.rand(len(b), 1, device=DEV) < p_mix).float()
                xb = xb + m * lvl * x2
                yb_sig = torch.maximum(yb_sig, yt_sig[torch.from_numpy(b2)].to(DEV) * m)
            if ctx_idx is None:
                x = wav_to_mel(mel, xb, cmvn)
            else:
                xs = [wav_to_mel(mel, xb, cmvn)]
                for t in list(range(ctx_idx.shape[1] // 2)) + list(range(ctx_idx.shape[1] // 2 + 1,
                                                                        ctx_idx.shape[1])):
                    g = gather(waves, ctx_idx[b][:, t])
                    if gain_db:
                        g = g * torch.pow(10.0, (torch.rand(len(b), 1, device=DEV) * 2 - 1) * gain_db / 20)
                    xs.insert(t, wav_to_mel(mel, g, cmvn))
                x = torch.cat(xs, 1)
            x = spec_augment(x)
            yb = torch.from_numpy(y_cls[b]).long().to(DEV)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = lossf_ce(out, yb) if mode == "softmax" else lossf_bce(out, yb_sig)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        val = evaluate(model, waves, df, va_idx, mel, ctx_idx, cfg)
        score = val["headline"] + val["map"]
        if score > best[0]:
            best = (score, {k: v for k, v in val.items()})
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log:
            log(f"    ep{ep + 1:02d} loss {tot / len(perm):.4f} headline {val['headline']:.4f} "
                f"mAP {val['map']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, best[1]) if return_model else best[1]


# ------------------------------------------------------------------ 评测
def _to_multihot(logits, mode, tau):
    if mode == "softmax":
        p = torch.softmax(logits, 1).numpy()
        pred = p.argmax(1)
        out = np.zeros((len(p), 4), dtype=np.int8)
        for i, c in enumerate(pred):
            if c < 4:
                out[i, c] = 1
        return out, p
    p = torch.sigmoid(logits).numpy()                     # (N,5) 列：4 告警 + other
    return (p[:, :4] > tau).astype(np.int8), p


def evaluate(model, waves, df, idx, mel, ctx_idx, cfg, tau_grid=None, cmvn=None):
    """统一口径：多标签指标 + 可对照旧基线的单标签 headline。"""
    mode = cfg["mode"]
    cmvn = cmvn or cfg.get("cmvn", "none")
    logits = forward_logits(model, waves, idx, mel, ctx_idx, cmvn)
    y_cls = df.cls.values[idx]
    y_true = np.zeros((len(idx), 4), dtype=np.int8)
    for i, c in enumerate(y_cls):
        if c < 4:
            y_true[i, c] = 1
    taus = tau_grid or cfg.get("taus", [0.5])
    best = None
    for tau in taus:
        pred, p = _to_multihot(logits, mode, tau)
        # headline：与旧口径对齐 = 每段"最强的告警 tag 或 background"
        if mode == "softmax":
            pm = p.argmax(1)
        else:
            pm = np.where(p[:, :4].max(1) > tau, p[:, :4].argmax(1), 4)
        acc = float((pm == y_cls).mean())
        f1s = []
        for c in range(5):
            tp = np.sum((pm == c) & (y_cls == c))
            fp = np.sum((pm == c) & (y_cls != c))
            fn = np.sum((pm != c) & (y_cls == c))
            if tp + fp + fn:
                f1s.append(2 * tp / (2 * tp + fp + fn))
        rec = {"tau": tau, "headline": acc, "macro_f1": float(np.mean(f1s)),
               "exact_match": float((pred == y_true).all(1).mean())}
        if best is None or rec["headline"] > best["headline"]:
            best = rec
            best["pred_multi"], best["prob"] = pred, p
    # 与阈值无关的多标签指标
    p = best["prob"][:, :4] if mode == "sigmoid" else best["prob"][:, :4]
    aps = {}
    for c in range(4):
        if y_true[:, c].sum() > 0:
            aps[ALARMS[c]] = float(average_precision_score(y_true[:, c], p[:, c]))
    best["map"] = float(np.mean(list(aps.values())))
    best["ap"] = aps
    best["y_true"], best["y_cls"] = y_true, y_cls
    best["per_class_recall"] = {ALARMS[c]: round(float(((best["pred_multi"][y_true[:, c] == 1, c]).mean())), 4)
                                for c in range(4) if y_true[:, c].sum()}
    return best


def event_level(pred_multi, y_cls, rec_ids):
    """录音级：任一段命中即判该录音存在该事件（tau 已内含在 pred_multi）。"""
    out = {}
    for c in range(4):
        recs = {}
        for r, yy, pm in zip(rec_ids, y_cls, pred_multi[:, c]):
            recs.setdefault(r, [0, 0])
            recs[r][0] = max(recs[r][0], int(yy == c))
            recs[r][1] = max(recs[r][1], int(pm))
        tp = sum(1 for v in recs.values() if v[0] and v[1])
        pos = sum(1 for v in recs.values() if v[0])
        fp = sum(1 for v in recs.values() if v[1] and not v[0])
        out[ALARMS[c]] = {"recall": round(tp / pos, 4) if pos else None, "n_pos": pos,
                          "false_rec": fp}
    return out


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)
