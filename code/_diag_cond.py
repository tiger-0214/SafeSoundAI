"""临时诊断：条件注入是否真的起作用（keep 向量消融）+ 掩码频谱结构。"""
import os, sys
from pathlib import Path
import pandas as pd, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soundfile as sf
import filter_config as fc
from tse_model import MaskNet, complex_mask, stft
from eval_filter import suppress_gain

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(fc.MIX_MANIFEST)
df = df[df.split == "test"].head(20)
ck = torch.load(fc.TSE_CKPT, map_location=DEV)
model = MaskNet().to(DEV).eval(); model.load_state_dict(ck["model"])
NC = len(fc.CLASSES) if hasattr(fc, "CLASSES") else 10
print("classes:", getattr(fc, "CLASSES", "n/a"))

conds = {"true": None, "all_keep": np.ones(NC), "none_keep": np.zeros(NC)}
agg = {k: {"supp": [], "pres": [], "mmean": [], "mstd_over_f": []} for k in conds}
with torch.no_grad():
    for _, r in df.iterrows():
        mix = sf.read(r["mix_path"], dtype="float32")[0]
        tgt = sf.read(r["target_path"], dtype="float32")[0]
        itf = sf.read(r["interferer_path"], dtype="float32")[0]
        Y = stft(torch.from_numpy(mix).to(DEV))
        true_keep = fc.keep_vector([int(i) for i in str(r["keep_set"]).split(",")])
        for k, c in conds.items():
            cv = true_keep if c is None else c
            cond = torch.tensor([cv], dtype=torch.float32, device=DEV)
            mm, pr = model(torch.log1p(Y.abs())[None, None], cond)
            m = np.clip(complex_mask(mm, pr)[0].abs().cpu().numpy(), 0, 1)
            agg[k]["supp"].append(suppress_gain(tgt, mix, m))
            agg[k]["pres"].append(suppress_gain(itf, mix, m))
            agg[k]["mmean"].append(float(m.mean()))
            agg[k]["mstd_over_f"].append(float(m.std(axis=0).mean()))
for k in conds:
    a = agg[k]
    print(f"{k:10s} mask_mean={np.mean(a['mmean']):.3f}  mask_std_f={np.mean(a['mstd_over_f']):.3f}  "
          f"supp={np.mean(a['supp']):6.2f}  pres={np.mean(a['pres']):6.2f}")
