"""A-3 时域上下文：学习式录音内聚合 vs 手调 K=5 平滑。

做法：冻结已训段级模型 → 取 512 维 embedding（GAP⊕注意力池化）+ 5 个 logit，
在每条录音的段序列上跑双向 GRU，输出每段的 5 维残差 logit（加到段级 logit 上）。
对照三档：段级直接判决 / 手调 K=5 平滑（旧方案）/ 学习式聚合。

用法：python3 agg_context.py --cfg ml_v3 --seeds 42,43,44
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from core import (ALARMS, DEV, V3, WrappedWaves, build_ctx_index, build_waves, event_level,
                  forward_logits, load_pool, make_mel)
from run import CFGS


def smooth_probs(p, rec_ids, K=5):
    out = p.copy()
    order = {}
    for i, r in enumerate(rec_ids):
        order.setdefault(r, []).append(i)
    for r, ids in order.items():
        ids = np.array(ids)
        if len(ids) < 3:
            continue
        q = p[ids]
        k = min(K, len(ids))
        ker = np.ones(k) / k
        sm = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, q)
        out[ids] = sm
    return out


class Agg(nn.Module):
    def __init__(self, din, nout=5, hid=64):
        super().__init__()
        self.rnn = nn.GRU(din, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hid, nout)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h)


def headline_from(p4, y_cls, tau):
    pm = np.where(p4.max(1) > tau, p4.argmax(1), 4)
    acc = float((pm == y_cls).mean())
    f1s = []
    for c in range(5):
        tp = np.sum((pm == c) & (y_cls == c)); fp = np.sum((pm == c) & (y_cls != c))
        fn = np.sum((pm != c) & (y_cls == c))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return acc, float(np.mean(f1s)), pm


@torch.no_grad()
def feats_and_logits(model, waves, idx, mel, ctx_idx, cmvn, bs=256):
    model.eval()
    Z, L = [], []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        if ctx_idx is None:
            from core import gather, wav_to_mel
            x = wav_to_mel(mel, gather(waves, b), cmvn)
        else:
            from core import gather, wav_to_mel
            x = torch.cat([wav_to_mel(mel, gather(waves, ctx_idx[b][:, t]), cmvn)
                           for t in range(ctx_idx.shape[1])], 1)
        lg, z = model(x, return_embed=True)
        Z.append(z.cpu().numpy()); L.append(lg.cpu().numpy())
    return np.concatenate(Z), np.concatenate(L)


def build_seq(df, idx):
    """按录音切段序列：返回 [(rec, 段在 idx 中的位置数组)]。"""
    pos = {}
    for j, i in enumerate(idx):
        pos.setdefault(df.rec_group.values[i], []).append(j)
    out = []
    for r, js in pos.items():
        js = np.array(js)
        js = js[np.argsort(df.segment_idx.values[idx[js]])]
        out.append((r, js))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="ml_v3")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    cfg = dict(CFGS[a.cfg]); cfg["epochs"] = 25
    seeds = [int(s) for s in a.seeds.split(",")]

    df = load_pool()
    waves = WrappedWaves(build_waves(df, "/root/clacky_workspace/safesound_exp/waves.npy"),
                         df.wave_row.values)
    folds = df.cv_fold.values
    tr, va, te = (np.where(np.isin(folds, range(8)))[0], np.where(folds == 8)[0],
                  np.where(folds == 9)[0])
    mel = make_mel(cfg.get("n_fft", 1024), cfg.get("win", 640))
    ctx_idx = build_ctx_index(df, cfg["ctx"]) if cfg.get("ctx", 1) > 1 else None
    res = {"cfg": a.cfg, "seeds": seeds}

    # ---- 三档对照：段级 / K=5 平滑 / 学习式聚合（多 seed 概率平均）
    Ztr, Ltr = None, None
    Zva_all, Lva_all, Zte_all, Lte_all = [], [], [], []
    for sd in seeds:
        from core import NetV3
        m = NetV3(mode=cfg["mode"], ctx=cfg.get("ctx", 1), attn=cfg.get("attn", True),
                  tdil=cfg.get("tdil", True), n_out=5).to(DEV)
        m.load_state_dict(torch.load(V3 / f"ckpt_{a.cfg}_s{sd}.pt", map_location=DEV))
        ztr, ltr = feats_and_logits(m, waves, tr, mel, ctx_idx, cfg.get("cmvn", "none"))
        zva, lva = feats_and_logits(m, waves, va, mel, ctx_idx, cfg.get("cmvn", "none"))
        zte, lte = feats_and_logits(m, waves, te, mel, ctx_idx, cfg.get("cmvn", "none"))
        if Ztr is None:
            Ztr, Ltr = ztr, ltr
        Zva_all.append(zva); Lva_all.append(lva); Zte_all.append(zte); Lte_all.append(lte)
    Zva, Lva = np.mean(Zva_all, 0), np.mean(Lva_all, 0)
    Zte, Lte = np.mean(Zte_all, 0), np.mean(Lte_all, 0)

    y = df.cls.values
    taus = np.arange(0.2, 0.81, 0.05)

    def evaluate_version(p4_va, p4_te, tag):
        """阈值只在 val 上选，原样搬到 test（与 final_v3 同纪律）。"""
        yv, yt = y[va], y[te]
        cands = [(headline_from(p4_va, yv, t)[:2], t) for t in taus]
        (acc_v, f1_v), tau = max(cands, key=lambda r: r[0][0] + r[0][1])
        acc_t, f1_t, _ = headline_from(p4_te, yt, tau)
        return {"variant": tag, "tau": round(float(tau), 2),
                "val_headline": round(acc_v, 4), "val_macro_f1": round(f1_v, 4),
                "headline": round(acc_t, 4), "macro_f1": round(f1_t, 4)}

    p_te_base = torch.sigmoid(torch.from_numpy(Lte).float()).numpy()[:, :4]
    p_va_base = torch.sigmoid(torch.from_numpy(Lva).float()).numpy()[:, :4]
    rec_te = df.rec_group.values[te]
    rec_va = df.rec_group.values[va]
    res["stage_segment"] = evaluate_version(p_va_base, p_te_base, "段级直接判决")
    res["stage_k5"] = evaluate_version(smooth_probs(p_va_base, rec_va, 5),
                                       smooth_probs(p_te_base, rec_te, 5), "手调 K=5 平滑")

    # ---- 学习式聚合：在 train 段序列上训练，val 早停
    din = Ztr.shape[1] + 5
    seq_tr = build_seq(df, tr)
    Ytr = []
    Xtr = []
    for _, js in seq_tr:
        Xtr.append(np.c_[Ztr[js], Ltr[js]][None])
        c = y[tr[js]]
        t = np.zeros((1, len(js), 5), dtype=np.float32)
        for k, cc in enumerate(c):
            t[0, k, 4 if cc == 4 else cc] = 1.0
        Ytr.append(t)
    agg = Agg(din).to(DEV)
    lossf = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(agg.parameters(), lr=2e-3, weight_decay=1e-4)
    best = (1e9, None)

    def val_loss():
        agg.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for r, js in build_seq(df, va):
                x = torch.from_numpy(np.c_[Zva[js], Lva[js]][None]).float().to(DEV)
                cc = y[va[js]]
                t = np.zeros((1, len(js), 5), dtype=np.float32)
                for k, c in enumerate(cc):
                    t[0, k, 4 if c == 4 else c] = 1.0
                tot += float(lossf(agg(x), torch.from_numpy(t).to(DEV))) * len(js)
                n += len(js)
        return tot / max(n, 1)

    for ep in range(a.epochs):
        agg.train()
        order = np.random.permutation(len(Xtr))
        for j in order:
            x = torch.from_numpy(Xtr[j]).float().to(DEV)
            t = torch.from_numpy(Ytr[j]).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = lossf(agg(x), t)
            loss.backward()
            opt.step()
        vl = val_loss()
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in agg.state_dict().items()})
        print(f"  agg ep{ep + 1:02d} val_loss {vl:.4f}", flush=True)
    agg.load_state_dict(best[1])
    agg.eval()

    def agg_logits(Z, L, idx):
        out = L.copy()
        with torch.no_grad():
            for r, js in build_seq(df, idx):
                x = torch.from_numpy(np.c_[Z[js], L[js]][None]).float().to(DEV)
                out[js] = L[js] + agg(x)[0].cpu().numpy()
        return out

    Lte_agg = agg_logits(Zte, Lte, te)
    Lva_agg = agg_logits(Zva, Lva, va)
    p_te_agg = torch.sigmoid(torch.from_numpy(Lte_agg).float()).numpy()[:, :4]
    p_va_agg = torch.sigmoid(torch.from_numpy(Lva_agg).float()).numpy()[:, :4]
    res["stage_learned_agg"] = evaluate_version(p_va_agg, p_te_agg, "学习式 BiGRU 聚合")
    res["stage_learned_agg_k5"] = evaluate_version(smooth_probs(p_va_agg, rec_va, 5),
                                                   smooth_probs(p_te_agg, rec_te, 5),
                                                   "学习式聚合 + K=5")
    res["event_level"] = {
        k: event_level((p > res[stage]["tau"]).astype(np.int8), y[te], rec_te)
        for k, p, stage in [("segment", p_te_base, "stage_segment"),
                            ("k5", smooth_probs(p_te_base, rec_te, 5), "stage_k5"),
                            ("learned_agg", p_te_agg, "stage_learned_agg")]}
    (V3 / f"agg_{a.cfg}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
