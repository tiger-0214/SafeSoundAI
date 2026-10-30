"""
SafeSoundAI — 方案 B 训练脚本（train_tse.py, A2.3）

两阶段训练（对齐 SoundBubble 的 pretrain→finetune 结构）：
  阶段 1（预训练）：理想比例掩码（IRM）MSE —— 让网络先学会"分得开"
  阶段 2（微调）：SI-SNR + 抑制惩罚（neg_weight=100 的漏抑重罚）+ 白名单保留约束

数据：docs/synth_mixer_spec.md 定义的合成混音集（4_Synthetic_Mixtures/manifest.csv）

用法：
    python train_tse.py --epochs-pre 6 --epochs-ft 14 --bs 16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config
import filter_config as fc
from tse_model import MaskNet, complex_mask, count_params, istft, stft

SR = config.SAMPLE_RATE


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class MixtureDataset(Dataset):
    """每条样本返回 (mix, s_keep, s_supp, cond)。"""

    def __init__(self, manifest: Path, split: str, cache: bool = True, max_items: int | None = None,
                 require_supp: bool = True):
        self.items: list[dict] = []
        skipped = 0
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                # 只在"确有可抑制对象"的样本上训练：
                # 本数据的语义是 target=待抑制的（黑名单/中性）事件声，
                # interferer=必须保留的白名单事件声。若 keep 集合把 target 也保留了
                # （passthrough），抑制侧只剩底噪，目标函数退化为恒等映射。
                if require_supp:
                    keep_ids = [int(x) for x in row["keep_set"].split(",") if x != ""]
                    if row["keep_set_name"] == "passthrough" or int(row["target_id"]) in keep_ids:
                        skipped += 1
                        continue
                self.items.append(row)
        self.skipped = skipped
        if max_items:
            self.items = self.items[:max_items]
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.items)

    def _load(self, path: str) -> torch.Tensor:
        if self.cache is not None and path in self.cache:
            return self.cache[path]
        y, _ = sf.read(path, always_2d=False, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        if len(y) < config.SEGMENT_SAMPLES:
            y = F.pad(y, (0, config.SEGMENT_SAMPLES - len(y)))
        y = y[:config.SEGMENT_SAMPLES]
        if self.cache is not None:
            self.cache[path] = y
        return y

    def __getitem__(self, i):
        r = self.items[i]
        keep_ids = [int(x) for x in r["keep_set"].split(",")]
        mix = self._load(r["mix_path"])
        t = self._load(r["target_path"])
        itf = self._load(r["interferer_path"])
        noise = self._load(r["noise_path"])

        keep_parts = []
        if int(r["interferer_id"]) in keep_ids:
            keep_parts.append(itf)
        if int(r["target_id"]) in keep_ids:
            keep_parts.append(t)
        if r.get("keep_set_name") == "passthrough":
            keep_parts.append(noise)
        s_keep = torch.stack(keep_parts).sum(0) if keep_parts else torch.zeros_like(mix)
        s_supp = mix - s_keep
        cond = torch.tensor(fc.keep_vector(keep_ids), dtype=torch.float32)
        return mix, s_keep, s_supp, cond


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8,
           clamp_db: float = 30.0) -> torch.Tensor:
    """批次 SI-SNR（逐样本）。

    clamp 到 ±30 dB：当 interferer 能量为 0（mix==keep）时理论值无穷，
    不截断会让均值被 70+ dB 的退化样本拉偏（实测 mean in=14.3 而真实值 ~7 dB）。
    """
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / (ref.pow(2).sum(-1, keepdim=True) + eps)
    noise = est - proj
    val = 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))
    return val.clamp(-clamp_db, clamp_db)


def band_weight(S: torch.Tensor) -> torch.Tensor:
    return S.abs().sum(1, keepdim=True)          # (B, F, T) 用于能量加权


def loss_pretrain(mag_mask: torch.Tensor, Y: torch.Tensor, S_keep: torch.Tensor,
                  S_supp: torch.Tensor) -> torch.Tensor:
    """阶段 1：IRM 目标 + 能量加权 MSE。"""
    mag = Y.abs()
    irm = (S_keep.abs() / (S_keep.abs() + S_supp.abs() + 1e-8)).clamp(0, 1)
    w = mag / (mag.mean() + 1e-8)
    return (((mag_mask[:, 0] - irm) ** 2) * w).mean()


def loss_finetune(y_out: torch.Tensor, s_keep: torch.Tensor, mag_out: torch.Tensor,
                  S_keep: torch.Tensor, S_supp: torch.Tensor,
                  neg_weight: float = 100.0, keep_penalty: float = 1.0) -> tuple[torch.Tensor, dict]:
    """阶段 2：SI-SNR + 保留约束 + 漏抑惩罚（严格监督 TSE 的经典三段式）。

    所有辅助项都做成**相对量**（无量纲、量级 ~1），否则能量绝对值大的样本
    会让 leak 项主导（旧版 loss≈170，SI 项被淹没 → mask 整体塌到 0.29）。
    """
    l_si = -si_snr(y_out, s_keep).mean()

    # 保留失真：相对幅度误差，按 keep 能量加权
    w_keep = band_weight(S_keep)
    num = ((mag_out - S_keep.abs()) ** 2 * w_keep).sum()
    l_keep = num / (S_keep.abs().pow(2) * w_keep).sum().clamp_min(1e-8)

    # 漏抑制：抑制带能量 / 保留带能量（比值，天然有界）
    w_supp = band_weight(S_supp)
    leak = (mag_out ** 2 * w_supp).sum()
    l_leak = leak / (mag_out ** 2 * w_keep).sum().clamp_min(1e-8)

    loss = l_si + keep_penalty * l_keep + (neg_weight / 100.0) * l_leak
    return loss, {"si_snr": float(-l_si.detach()), "keep": float(l_keep.detach()),
                  "leak": float(l_leak.detach())}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: MaskNet, loader: DataLoader, device) -> dict:
    """只在"确有可抑制对象"的批内样本上统计，避免退化样本污染均值。"""
    model.eval()
    si_in, si_out, imp = [], [], []
    for mix, s_keep, s_supp, cond in loader:
        mix, s_keep, s_supp, cond = [t.to(device) for t in (mix, s_keep, s_supp, cond)]
        Y = stft(mix)
        mag = torch.log1p(Y.abs())[:, None]
        mm, pr = model(mag, cond)
        M = complex_mask(mm, pr)
        y_out = istft(Y * M, mix.shape[-1])
        e_keep = s_keep.pow(2).sum(-1)
        e_supp = s_supp.pow(2).sum(-1)
        valid = (e_keep > 1e-4) & (e_supp > 1e-4)
        if not bool(valid.any()):
            continue
        a = si_snr(mix, s_keep)[valid]
        b = si_snr(y_out, s_keep)[valid]
        si_in += a.tolist(); si_out += b.tolist(); imp += (b - a).tolist()
    model.train()
    if not si_in:
        return {"si_snr_in": 0.0, "si_snr_out": 0.0, "imp": 0.0, "n": 0}
    return {"si_snr_in": float(np.mean(si_in)), "si_snr_out": float(np.mean(si_out)),
            "imp": float(np.mean(imp)), "n": len(si_in)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-pre", type=int, default=6)
    ap.add_argument("--epochs-ft", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neg-weight", type=float, default=100.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=str(fc.TSE_CKPT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = MixtureDataset(fc.MIX_MANIFEST, "train", max_items=args.max_train)
    ds_va = MixtureDataset(fc.MIX_MANIFEST, "val")
    if len(ds_tr) == 0:
        raise SystemExit(f"manifest 无 train 样本：{fc.MIX_MANIFEST}（先跑 synth_mix.py）")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=0)

    model = MaskNet().to(device)
    print(f"参数量 {count_params(model):,} | train {len(ds_tr)}(skip {ds_tr.skipped}) / "
          f"val {len(ds_va)}(skip {ds_va.skipped}) | device {device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5, min_lr=1e-6)

    hist: list[dict] = []
    best_imp = -1e9
    t0 = time.time()

    for phase, epochs in (("pretrain", args.epochs_pre), ("finetune", args.epochs_ft)):
        if phase == "finetune":
            # 微调阶段降 lr：IRM→SI-SNR 目标切换时 lr 过大容易一步崩掉
            for g in opt.param_groups:
                g["lr"] = min(args.lr, 5e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=3, factor=0.5, min_lr=1e-6)
        for ep in range(1, epochs + 1):
            model.train()
            agg = {"loss": 0.0, "si_snr": 0.0, "keep": 0.0, "leak": 0.0}
            n = 0
            for mix, s_keep, s_supp, cond in dl_tr:
                mix, s_keep, s_supp, cond = [t.to(device) for t in (mix, s_keep, s_supp, cond)]
                Y = stft(mix)
                S_keep = stft(s_keep)
                S_supp = stft(s_supp)
                mag = torch.log1p(Y.abs())[:, None]
                mm, pr = model(mag, cond)
                if phase == "pretrain":
                    loss = loss_pretrain(mm, Y, S_keep, S_supp)
                    metrics = {"si_snr": 0.0, "keep": 0.0, "leak": 0.0}
                else:
                    M = complex_mask(mm, pr)
                    Y_out = Y * M
                    y_out = istft(Y_out, mix.shape[-1])
                    loss, metrics = loss_finetune(y_out, s_keep, Y_out.abs(), S_keep, S_supp,
                                                  neg_weight=args.neg_weight)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                agg["loss"] += float(loss.detach())
                for k, v in metrics.items():
                    agg[k] += v
                n += 1

            ev = evaluate(model, dl_va, device)
            sched.step(-ev["imp"])
            rec = {"phase": phase, "epoch": ep, "loss": round(agg["loss"] / max(n, 1), 4),
                   "val_si_snr_in": round(ev["si_snr_in"], 2),
                   "val_si_snr_out": round(ev["si_snr_out"], 2),
                   "val_improve_db": round(ev["imp"], 2)}
            hist.append(rec)
            print(f"[{phase} {ep:02d}/{epochs}] loss={rec['loss']:.4f} "
                  f"val SI-SNR {rec['val_si_snr_in']:.2f} → {rec['val_si_snr_out']:.2f} dB "
                  f"(+{rec['val_improve_db']:.2f})")
            if ev["imp"] > best_imp:
                best_imp = ev["imp"]
                Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "best_improve_db": best_imp,
                            "epoch": ep, "phase": phase}, args.ckpt)
                print(f"  ↳ 保存最优权重（+{best_imp:.2f} dB）→ {args.ckpt}")

    sha = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()[:12]
    out = {"params": count_params(model), "best_improve_db": round(best_imp, 2),
           "history": hist, "ckpt": str(args.ckpt), "ckpt_sha12": sha,
           "elapsed_s": round(time.time() - t0, 1),
           "config": {"bs": args.bs, "lr": args.lr, "neg_weight": args.neg_weight,
                      "epochs_pre": args.epochs_pre, "epochs_ft": args.epochs_ft,
                      "n_train": len(ds_tr), "n_val": len(ds_va)}}
    (fc.FILTER_REPORT_DIR / "tse_train_log.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成（{out['elapsed_s']}s）| 最优 val 提升 {best_imp:+.2f} dB | sha256 {sha}")


if __name__ == "__main__":
    main()
