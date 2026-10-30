"""v3 终评：folds0-7 训练 → fold8 早停 → fold9 单次评测。

一次跑完四件事（都对应已知缺陷）：
  1) 单标签口径 headline / 宏 F1（与旧 88.90% 对齐）
  2) 多标签口径 mAP / 逐类 AP / 精确匹配 + 合成重叠事件的双 tag 召回（旧 softmax 结构上做不到）
  3) 拒识：未见 15 类 unknown（10 benign / 5 hazard）上的 AUROC 与误报率
  4) 标定：温度缩放前后 ECE

用法：python3 run.py --cfg ml_v3 [--seeds 42,43,44] [--epochs 25]
"""
import argparse
import json

import numpy as np
import torch

from core import (ALARMS, DEV, TAGS, UNSEEN_BENIGN, UNSEEN_HAZARD, V3, NetV3, WrappedWaves,
                  build_ctx_index, build_waves, evaluate, event_level, ece, forward_logits,
                  gather, load_pool, load_unknown, make_mel, train_v3, wav_to_mel)
from run import CFGS


def fit_temperature(logits, y_cls, mode):
    """在验证集上拟合单一温度 T（softmax: 5 类 CE；sigmoid: BCE）。"""
    z = torch.from_numpy(logits).float()
    if mode == "softmax":
        tgt = torch.from_numpy(y_cls).long()
        lossf = torch.nn.CrossEntropyLoss()
    else:
        t = np.zeros((len(y_cls), 5), dtype=np.float32)
        for i, c in enumerate(y_cls):
            t[i, 4 if c == 4 else c] = 1.0
        tgt = torch.from_numpy(t)
        lossf = torch.nn.BCEWithLogitsLoss()
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(3.0), 60)):
        loss = float(lossf(z / T, tgt))
        if loss < best[0]:
            best = (loss, float(T))
    return best[1]


def probs_of(logits, mode, T=1.0):
    z = torch.from_numpy(logits).float() / T
    return (torch.softmax(z, 1) if mode == "softmax" else torch.sigmoid(z)).numpy()


def overlap_test(model, waves, df, idx, mel, ctx_idx, cfg, n_pairs=600, seed=0):
    """合成重叠事件：两段不同录音不同类别线性叠加，评测双 tag 召回。"""
    rng = np.random.default_rng(seed)
    cls = df.cls.values[idx]
    rec = df.rec_group.values[idx]
    cand = [i for i in range(len(idx)) if cls[i] < 4]
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.choice(cand, 2, replace=False)
        if cls[a] != cls[b] and rec[a] != rec[b]:
            pairs.append((a, b))
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    lvl = 10 ** (rng.uniform(-10, 0, size=len(pa)) / 20)
    xa = gather(waves, idx[pa])
    xb = gather(waves, idx[pb]) * torch.from_numpy(lvl[:, None].astype(np.float32)).to(DEV)
    model.eval()
    with torch.no_grad():
        outs = []
        for s in range(0, len(pa), 256):
            x = wav_to_mel(mel, xa[s:s + 256] + xb[s:s + 256], cfg.get("cmvn", "none"))
            if cfg.get("ctx", 1) > 1:   # ctx 模型：把混合段复制到每个上下文通道（近似）
                x = x.expand(-1, cfg["ctx"], -1, -1).contiguous()
            outs.append(model(x))
        logits = torch.cat(outs).cpu().numpy()
    tau = cfg.get("tau_eval", 0.5)
    if cfg["mode"] == "softmax":
        p = torch.softmax(torch.from_numpy(logits), 1).numpy()
        hits_a = p.argmax(1) == cls[idx[pa]]
        hits_b = p.argmax(1) == cls[idx[pb]]
        both = hits_a & hits_b
        return {"mode": "softmax(单标签，结构上不可能同时命中)", "tau": tau,
                "single_recall": round(float(hits_a.mean()), 4),
                "both_recall": round(float(both.mean()), 4), "n_pairs": len(pa)}
    p = torch.sigmoid(torch.from_numpy(logits)).numpy()[:, :4]
    hits_a = p[np.arange(len(pa)), cls[idx[pa]]] > tau
    hits_b = p[np.arange(len(pa)), cls[idx[pb]]] > tau
    extra = p > tau
    return {"mode": "sigmoid(多标签)", "tau": tau,
            "single_recall": round(float(hits_a.mean()), 4),
            "both_recall": round(float((hits_a & hits_b).mean()), 4),
            "avg_tags_fired": round(float(extra.sum(1).mean()), 3),
            "n_pairs": len(pa)}


def mahalanobis_scores(model, waves, df, tr_idx, eval_sets, mel, ctx_idx, cfg, bs=256):
    """类内共享协方差的 Mahalanobis 距离（在 512 维 embedding 上），距离越大越"未知"。"""
    def emb(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                if ctx_idx is None:
                    x = wav_to_mel(mel, gather(waves, b), cfg.get("cmvn", "none"))
                else:
                    x = torch.cat([wav_to_mel(mel, gather(waves, ctx_idx[b][:, t]), cfg.get("cmvn", "none"))
                                   for t in range(ctx_idx.shape[1])], 1)
                outs.append(model(x, return_embed=True)[1].cpu().numpy())
        return np.concatenate(outs)

    Ztr = emb(tr_idx)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr = (Ztr - mu) / sd
    cls = df.cls.values[tr_idx]
    mus = np.stack([Ztr[cls == c].mean(0) for c in range(4)])
    R = Ztr - mus[cls.clip(0, 3)]                      # 残差
    cov = np.cov(R.T) + 1e-3 * np.eye(R.shape[1])
    P = np.linalg.pinv(cov)
    out = {}
    for name, (w, idx) in eval_sets.items():
        Z = (emb(idx, w) if False else None)
        out[name] = Z
    return out, mu, sd, mus, P


def run(cfg_name, seeds, epochs, cfg_over=None, eval_only=False):
    cfg = dict(CFGS[cfg_name])
    cfg["epochs"] = epochs
    if cfg_over:
        cfg.update(cfg_over)
    df = load_pool()
    waves = WrappedWaves(build_waves(df, "/root/clacky_workspace/safesound_exp/waves.npy"),
                         df.wave_row.values)
    folds = df.cv_fold.values
    tr = np.where(np.isin(folds, range(8)))[0]
    va = np.where(folds == 8)[0]
    te = np.where(folds == 9)[0]
    mel = make_mel(cfg.get("n_fft", 1024), cfg.get("win", 640))
    ctx_idx = build_ctx_index(df, cfg["ctx"]) if cfg.get("ctx", 1) > 1 else None
    mode = cfg["mode"]

    models, val_logits = [], []
    for sd in seeds:
        if eval_only:
            print(f"=== load {cfg_name} seed {sd}", flush=True)
            m = NetV3(mode=mode, ctx=cfg["ctx"], attn=cfg.get("attn", True), tdil=cfg.get("tdil", True),
                      n_out=5).to(DEV)
            m.load_state_dict(torch.load(V3 / f"ckpt_{cfg_name}_s{sd}.pt", map_location=DEV))
        else:
            print(f"=== train {cfg_name} seed {sd}", flush=True)
            m, _ = train_v3(waves, df, tr, va, cfg, seed=sd, log=lambda s: print(s, flush=True))
            torch.save(m.state_dict(), V3 / f"ckpt_{cfg_name}_s{sd}.pt")
        models.append(m)
        val_logits.append(forward_logits(m, waves, va, mel, ctx_idx, cfg.get("cmvn", "none")).numpy())
    V = np.mean(val_logits, 0)

    # 温度标定（在 val 上拟合，绝不用 test）
    T = fit_temperature(V, df.cls.values[va], mode)
    ece_val_before = ece(probs_of(V, mode), df.cls.values[va])
    ece_val_after = ece(probs_of(V, mode, T), df.cls.values[va])

    # fold 9 单次评测
    test_logits = [forward_logits(m, waves, te, mel, ctx_idx, cfg.get("cmvn", "none")).numpy()
                   for m in models]
    Lt = np.mean(test_logits, 0)
    y_te = df.cls.values[te]
    res = {"cfg": cfg_name, "seeds": seeds, "epochs": epochs, "N_fit": int(len(tr)),
           "N_val": int(len(va)), "N_test": int(len(te)),
           "temperature": round(T, 4),
           "ece_val_before": round(ece_val_before, 4), "ece_val_after": round(ece_val_after, 4),
           "ece_test_before": round(ece(probs_of(Lt, mode), y_te), 4),
           "ece_test_after": round(ece(probs_of(Lt, mode, T), y_te), 4)}

    # 统一评测（headline 用未标定的 τ 搜索，标定只影响概率不改变排序）
    class _M:
        def __init__(self, logits):
            self.logits = logits

        def __call__(self, x):
            raise RuntimeError

    class _FakeModel:
        def __init__(self, logits):
            self.logits = logits

    # 直接用 logits 算指标，绕过 forward
    def metrics_from_logits(logits, y_cls, idx, cfg, taus=(0.3, 0.4, 0.5, 0.6), fixed_tau=None):
        from core import _to_multihot
        y_true = np.zeros((len(idx), 4), dtype=np.int8)
        for i, c in enumerate(y_cls):
            if c < 4:
                y_true[i, c] = 1
        best = None
        for tau in ([fixed_tau] if fixed_tau is not None else taus):
            pred, p = _to_multihot(torch.from_numpy(logits).float(), mode, tau)
            pm = (p.argmax(1) if mode == "softmax"
                  else np.where(p[:, :4].max(1) > tau, p[:, :4].argmax(1), 4))
            acc = float((pm == y_cls).mean())
            f1s = []
            for c in range(5):
                tp = np.sum((pm == c) & (y_cls == c)); fp = np.sum((pm == c) & (y_cls != c))
                fn = np.sum((pm != c) & (y_cls == c))
                if tp + fp + fn:
                    f1s.append(2 * tp / (2 * tp + fp + fn))
            r = {"tau": tau, "headline": acc, "macro_f1": float(np.mean(f1s)),
                 "exact_match": float((pred == y_true).all(1).mean()),
                 "pred_multi": pred, "prob": p}
            if best is None or r["headline"] > best["headline"]:
                best = r
        from sklearn.metrics import average_precision_score
        aps = {ALARMS[c]: round(float(average_precision_score(y_true[:, c], best["prob"][:, c])), 4)
               for c in range(4) if y_true[:, c].sum()}
        best["map"] = round(float(np.mean(list(aps.values()))), 4)
        best["ap"] = aps
        best["per_class_recall"] = {ALARMS[c]: round(float(best["pred_multi"][y_true[:, c] == 1, c].mean()), 4)
                                    for c in range(4) if y_true[:, c].sum()}
        return best, y_true

    # 阈值只在 val 上选，再原样搬到 test（避免 test 调参）
    mv, _ = metrics_from_logits(V, df.cls.values[va], va, cfg)
    tau_val = mv["tau"]
    res["tau_from_val"] = tau_val
    mt, y_true = metrics_from_logits(Lt, y_te, te, cfg, fixed_tau=tau_val)
    res.update({k: mt[k] for k in ("tau", "headline", "macro_f1", "exact_match", "map", "ap",
                                   "per_class_recall")})
    res["event_level"] = event_level(mt["pred_multi"], y_te, df.rec_group.values[te])
    cfg["tau_eval"] = mt["tau"]
    res["overlap"] = overlap_test(models[0], waves, df, te, mel, ctx_idx, cfg)

    # 拒识：未见 unknown（15 类）+ 已知 test 上的分数分布
    wu, haz, lab = load_unknown()
    from sklearn.metrics import roc_auc_score

    def score_set(models, w, idx=None, ctxx=None):
        L = np.mean([forward_logits(m, w, np.arange(len(w)) if idx is None else idx, mel,
                                    ctxx, cfg.get("cmvn", "none")).numpy() for m in models], 0)
        if mode == "softmax":
            p = torch.softmax(torch.from_numpy(L).float(), 1).numpy()
            alarm = p[:, :4].max(1)
            energy = torch.logsumexp(torch.from_numpy(L).float(), 1).numpy()
        else:
            p = torch.sigmoid(torch.from_numpy(L).float()).numpy()
            alarm = p[:, :4].max(1)
            energy = torch.logsumexp(torch.from_numpy(L).float(), 1).numpy()
        return alarm, energy, p

    import pandas as _pd
    dfu = _pd.read_csv(V3 / "unknown_meta.csv")
    ctxu = build_ctx_index(dfu, cfg["ctx"]) if cfg.get("ctx", 1) > 1 else None
    a_unk_b, e_unk_b, p_unk_b = score_set(models, wu, None, ctxu)
    L_te_alarm = Lt[y_te < 4]
    a_known = (torch.softmax(torch.from_numpy(L_te_alarm).float(), 1).numpy()[:, :4].max(1)
               if mode == "softmax" else
               torch.sigmoid(torch.from_numpy(L_te_alarm).float()).numpy()[:, :4].max(1))
    a_te_all, e_te_all, p_te_all = score_set(models, waves, te, ctx_idx)
    bg = y_te == 4
    benign = haz == 0
    hazard = haz == 1
    res["reject"] = {
        "auroc_alarm_known_vs_benign": round(float(roc_auc_score(
            np.r_[np.ones(len(a_known)), np.zeros(benign.sum())],
            np.r_[a_known, a_unk_b[benign]])), 4),
        "auroc_alarm_hazard_vs_benign_unknown": round(float(roc_auc_score(
            np.r_[np.ones(hazard.sum()), np.zeros(benign.sum())],
            np.r_[a_unk_b[hazard], a_unk_b[benign]])), 4),
        "alarm_fp_rate_unseen_benign@tau": round(float((a_unk_b[benign] > mt["tau"]).mean()), 4),
        "alarm_fp_rate_unseen_hazard@tau": round(float((a_unk_b[hazard] > mt["tau"]).mean()), 4),
        "alarm_fp_rate_test_background@tau": round(float((a_te_all[bg] > mt["tau"]).mean()), 4),
        "mean_alarm_prob": {"test_alarm": round(float(a_known.mean()), 4),
                            "unseen_benign": round(float(a_unk_b[benign].mean()), 4),
                            "unseen_hazard": round(float(a_unk_b[hazard].mean()), 4),
                            "test_background": round(float(a_te_all[bg].mean()), 4)},
    }
    # 未知类被硬分到的具体类别（解释"垃圾桶"行为）
    if mode == "sigmoid":
        wrong = p_unk_b[:, :4].argmax(1)
        res["reject"]["unseen_top_alarm"] = {
            "benign": {ALARMS[c]: int(((wrong == c) & benign).sum()) for c in range(4)},
            "hazard": {ALARMS[c]: int(((wrong == c) & hazard).sum()) for c in range(4)},
        }
    else:
        wrong = p_unk_b[:, :4].argmax(1)
        res["reject"]["unseen_top_alarm"] = {
            "benign": {ALARMS[c]: int(((wrong == c) & benign).sum()) for c in range(4)},
            "hazard": {ALARMS[c]: int(((wrong == c) & hazard).sum()) for c in range(4)},
        }
    (V3 / f"final_{cfg_name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="ml_v3")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    run(a.cfg, [int(s) for s in a.seeds.split(",")], a.epochs, eval_only=a.eval_only)
