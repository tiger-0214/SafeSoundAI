"""临时诊断：掩码后处理（逐帧归一化/百分位重标定）能否提升选择性 gap。

注意 manifest 语义：target_path = 待抑制对象，interferer_path = 白名单须保留对象。
"""
import os, sys
import pandas as pd, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soundfile as sf
import filter_config as fc
from tse_model import MaskNet, complex_mask, stft
from eval_filter import suppress_gain

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(fc.MIX_MANIFEST)
df = df[df.split == "test"].head(60)
ck = torch.load(fc.TSE_CKPT, map_location=DEV)
model = MaskNet().to(DEV).eval(); model.load_state_dict(ck["model"])


def post_max(m, **_):
    """逐帧归一到最大值（保证每帧最强带完整保留）。"""
    return np.clip(m / (m.max(axis=0, keepdims=True) + 1e-6), 0, 1)


def post_pct(m, q=0.9):
    hi = np.percentile(m, q * 100, axis=0, keepdims=True) + 1e-6
    return np.clip(m / hi, 0, 1)


def post_soft_max(m, t=0.5):
    """逐帧 softmax 式重标定：把相对排序拉成指数对比。"""
    e = np.exp((m - m.max(axis=0, keepdims=True)) / t)
    return np.clip((m / (m.max(axis=0, keepdims=True) + 1e-6)) ** (1.0 / max(t, 1e-3)) * 0 + e / (e.max(axis=0, keepdims=True)), 0, 1)


VARIANTS = {
    "raw": lambda m: m,
    "frame_max": post_max,
    "frame_p90": post_pct,
}
rows = []
with torch.no_grad():
    for _, r in df.iterrows():
        mix = sf.read(r["mix_path"], dtype="float32")[0]
        tgt = sf.read(r["target_path"], dtype="float32")[0]      # 待抑制
        itf = sf.read(r["interferer_path"], dtype="float32")[0]  # 须保留
        Y = stft(torch.from_numpy(mix).to(DEV))
        cond = torch.tensor([fc.keep_vector([int(i) for i in str(r["keep_set"]).split(",")])],
                            dtype=torch.float32, device=DEV)
        mm, pr = model(torch.log1p(Y.abs())[None, None], cond)
        m = np.clip(complex_mask(mm, pr)[0].abs().cpu().numpy(), 0, 1)
        rec = {"mix_id": r["mix_id"]}
        for name, fn in VARIANTS.items():
            g = fn(m)
            rec[f"supp_{name}"] = suppress_gain(tgt, mix, g)
            rec[f"pres_{name}"] = suppress_gain(itf, mix, g)
        rows.append(rec)
d = pd.DataFrame(rows)
print("变体        Supp(mean)  Pres(mean)   gap")
for name in VARIANTS:
    s, p = d[f"supp_{name}"].mean(), d[f"pres_{name}"].mean()
    print(f"{name:10s} {s:9.2f}  {p:9.2f}  {p - s:6.2f}")
