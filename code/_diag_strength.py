"""临时诊断：安全强度 α 的 Supp/Pres 权衡曲线（用于确定 filter_config.TSE_STRENGTH 默认值）。"""
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
df = df[df.split == "test"].head(120)
ck = torch.load(fc.TSE_CKPT, map_location=DEV)
model = MaskNet().to(DEV).eval(); model.load_state_dict(ck["model"])

rows = []
with torch.no_grad():
    for _, r in df.iterrows():
        mix = sf.read(r["mix_path"], dtype="float32")[0]
        tgt = sf.read(r["target_path"], dtype="float32")[0]
        itf = sf.read(r["interferer_path"], dtype="float32")[0]
        Y = stft(torch.from_numpy(mix).to(DEV))
        cond = torch.tensor([[float(x) for x in fc.keep_vector([int(i) for i in str(r["keep_set"]).split(",")])]],
                            dtype=torch.float32, device=DEV)
        mm, pr = model(torch.log1p(Y.abs())[None, None], cond)
        m = complex_mask(mm, pr)[0].abs().cpu().numpy()
        rec = {"mix_id": r["mix_id"]}
        for a in (0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0):
            g = fc.apply_strength(np.clip(m, 0, 1), a)
            rec[f"supp_a{a}"] = suppress_gain(tgt, mix, g)
            rec[f"pres_a{a}"] = suppress_gain(itf, mix, g)
        rows.append(rec)
d = pd.DataFrame(rows)
print("α  |  Supp(mean)  Pres(mean)   gap   静默压制率(>3dB)")
for a in (0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0):
    s = d[f"supp_a{a}"].mean(); p = d[f"pres_a{a}"].mean()
    viol = float((d[f"pres_a{a}"] < -3.0).mean() * 100)
    print(f"{a:4.2f} | {s:9.2f}  {p:9.2f}  {p-s:6.2f}   {viol:5.1f}%")
