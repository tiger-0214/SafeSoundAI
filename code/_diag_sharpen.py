"""临时诊断：掩码锐化指数 p 对 选择性(Supp vs Pres) 的影响。不写报告，只打印。"""
import os, sys, pandas as pd, numpy as np, torch
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filter_config as fc
from tse_model import MaskNet, complex_mask, istft, stft
from eval_filter import suppress_gain, si_sdr

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIX = fc.MIX_DIR
df = pd.read_csv(fc.MIX_MANIFEST)
df = df[df.split == "test"].head(60)
ck = torch.load(Path("/tmp/tse_best_v2.pth") if True else fc.TSE_CKPT, map_location=DEV)
model = MaskNet().to(DEV).eval()
model.load_state_dict(ck["model"] if "model" in ck else ck)
print("加载权重 keys:", list(ck.keys())[:4] if isinstance(ck, dict) else type(ck))

def load(p):
    import soundfile as sf
    y, _ = sf.read(p, dtype="float32")
    return y

rows = []
with torch.no_grad():
    for _, r in df.iterrows():
        mix = load(r["mix_path"]); tgt = load(r["target_path"]); itf = load(r["interferer_path"])
        Y = stft(torch.from_numpy(mix).to(DEV))
        mag = torch.log1p(Y.abs())[None, None]
        keep_ids = [int(x) for x in str(r["keep_set"]).split(",")]
        cond = torch.tensor([fc.keep_vector(keep_ids)],
                            dtype=torch.float32, device=DEV)
        mm, pr = model(mag, cond)
        m = complex_mask(mm, pr)[0].abs().cpu().numpy()
        rec = {"mix_id": r["mix_id"]}
        for p in (1, 2, 3, 4):
            g = np.clip(m, 0, 1) ** p
            rec[f"supp_p{p}"] = suppress_gain(tgt, mix, g)
            rec[f"pres_p{p}"] = suppress_gain(itf, mix, g)
        rows.append(rec)
d = pd.DataFrame(rows)
print(d.describe().loc[["mean", "50%"]].round(2).to_string())
for p in (1, 2, 3, 4):
    print(f"p={p}: supp {d[f'supp_p{p}'].mean():6.2f} dB | pres {d[f'pres_p{p}'].mean():6.2f} dB | gap {d[f'pres_p{p}'].mean()-d[f'supp_p{p}'].mean():5.2f} dB")
