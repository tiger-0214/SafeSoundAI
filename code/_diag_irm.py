"""临时诊断：预测掩码 vs IRM 标签的一致性（判断模型是否真在做频谱选择）。"""
import os, sys
import pandas as pd, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soundfile as sf
import filter_config as fc
from tse_model import MaskNet, complex_mask, stft

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(fc.MIX_MANIFEST)
df = df[df.split == "test"].head(30)
ck = torch.load(fc.TSE_CKPT, map_location=DEV)
model = MaskNet().to(DEV).eval(); model.load_state_dict(ck["model"])

def spec(x):
    return stft(torch.from_numpy(x).to(DEV))

corrs, m_hi, m_lo, irm_means = [], [], [], []
with torch.no_grad():
    for _, r in df.iterrows():
        mix = sf.read(r["mix_path"], dtype="float32")[0]
        tgt = sf.read(r["target_path"], dtype="float32")[0]
        itf = sf.read(r["interferer_path"], dtype="float32")[0]
        Y, S, N = spec(mix), spec(tgt), spec(itf)
        cond = torch.tensor([fc.keep_vector([int(i) for i in str(r["keep_set"]).split(",")])],
                            dtype=torch.float32, device=DEV)
        mm, pr = model(torch.log1p(Y.abs())[None, None], cond)
        m = np.clip(complex_mask(mm, pr)[0].abs().cpu().numpy(), 0, 1)
        irm = np.clip((S.abs() / (S.abs() + N.abs() + 1e-8)).cpu().numpy(), 0, 1)
        corrs.append(float(np.corrcoef(m.ravel(), irm.ravel())[0, 1]))
        m_hi.append(float(m[irm > 0.7].mean()) if (irm > 0.7).any() else np.nan)
        m_lo.append(float(m[irm < 0.3].mean()) if (irm < 0.3).any() else np.nan)
        irm_means.append(float(irm.mean()))
print(f"IRM 标签均值        = {np.nanmean(irm_means):.3f}")
print(f"预测掩码 vs IRM 相关 = {np.nanmean(corrs):.3f}")
print(f"IRM>0.7 处预测掩码   = {np.nanmean(m_hi):.3f}   (应接近 1)")
print(f"IRM<0.3 处预测掩码   = {np.nanmean(m_lo):.3f}   (应接近 0)")
