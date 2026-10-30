"""
SafeSoundAI — 过滤模型端侧导出与延迟基准（export_onnx.py, A2.6）

对应 WBS A2.6 / A5：
  1. 把方案 B（MaskNet）导出 ONNX（动态时间轴，固定频率 257）
  2. 用 ONNXRuntime CPU 基准：单 chunk（10 帧 = 100 ms）、单窗（100 帧 = 1 s）
  3. 校验导出前后数值一致性（max |Δ|）；不一致超阈值即视为掉点
  4. 若 onnx 包不可用 → 回退 torch.jit.trace，并明确标注

产物：outputs/models/tse.onnx（或 tse_torchscript.pt）+ outputs/reports/export_onnx.md

用法：
    python export_onnx.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import config
import filter_config as fc
from tse_model import MaskNet, count_params

DEVICE = torch.device("cpu")          # 导出与基准都在 CPU（对端侧最有意义）


def bench_ort(session, frames: int, n_iter: int = 20) -> float:
    """返回单次推理毫秒数。"""
    mag = np.random.randn(1, 1, config.N_FFT // 2 + 1, frames).astype(np.float32)
    cond = np.ones((1, config.NUM_CLASSES), dtype=np.float32)
    for _ in range(3):
        session.run(None, {"mag": mag, "cond": cond})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        session.run(None, {"mag": mag, "cond": cond})
    return 1000.0 * (time.perf_counter() - t0) / n_iter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(fc.TSE_CKPT))
    ap.add_argument("--out", default=str(fc.TSE_ONNX))
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        raise SystemExit(f"缺少权重 {args.ckpt}（先跑 train_tse.py）")

    ck = torch.load(args.ckpt, map_location="cpu")
    model = MaskNet().to(DEVICE)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    mag = torch.randn(1, 1, config.N_FFT // 2 + 1, args.frames)
    cond = torch.ones(1, config.NUM_CLASSES)

    with torch.no_grad():
        ref_mag, ref_ph = model(mag, cond)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported, note = None, ""

    try:
        import onnx  # noqa: F401
        torch.onnx.export(
            model, (mag, cond), str(out_path),
            input_names=["mag", "cond"], output_names=["mag_mask", "phase_res"],
            dynamic_axes={"mag": {3: "t"}, "mag_mask": {3: "t"}, "phase_res": {3: "t"}},
            opset_version=17, do_constant_folding=True,
        )
        exported = "onnx"
        note = f"ONNX opset 17，文件 {out_path.stat().st_size/1024:.0f} KB"
    except Exception as e:
        print(f"[warn] ONNX 导出失败（{e}），回退 torch.jit.trace")
        ts_path = out_path.with_suffix(".torchscript.pt")
        traced = torch.jit.trace(model, (mag, cond))
        traced.save(str(ts_path))
        out_path = ts_path
        exported = "torchscript"
        note = f"TorchScript，文件 {ts_path.stat().st_size/1024:.0f} KB（onnx 包缺失时的等价交付）"

    # 一致性校验
    if exported == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        mm, pr = sess.run(None, {"mag": mag.numpy(), "cond": cond.numpy()})
        d_mag = float(np.max(np.abs(mm - ref_mag.numpy())))
        d_ph = float(np.max(np.abs(pr - ref_ph.numpy())))
        lat_1s = bench_ort(sess, 100)
        lat_chunk = bench_ort(sess, 10)
    else:
        traced = torch.jit.load(str(out_path))
        with torch.no_grad():
            mm, pr = traced(mag, cond)
        d_mag = float((mm - ref_mag).abs().max())
        d_ph = float((pr - ref_ph).abs().max())
        sess = None
        with torch.no_grad():
            for _ in range(3):
                traced(torch.randn(1, 1, 257, 100), cond)
            t0 = time.perf_counter()
            for _ in range(20):
                traced(torch.randn(1, 1, 257, 100), cond)
            lat_1s = 1000 * (time.perf_counter() - t0) / 20
            for _ in range(3):
                traced(torch.randn(1, 1, 257, 10), cond)
            t0 = time.perf_counter()
            for _ in range(20):
                traced(torch.randn(1, 1, 257, 10), cond)
            lat_chunk = 1000 * (time.perf_counter() - t0) / 20

    params = count_params(model)
    ok_params = params <= 1_000_000
    ok_chunk = lat_chunk <= 10.0
    ok_parity = max(d_mag, d_ph) < 1e-3

    report = {
        "export_format": exported, "note": note, "path": str(out_path),
        "params": params, "size_kb": round(out_path.stat().st_size / 1024, 1),
        "latency_1s_ms": round(lat_1s, 2), "latency_chunk100ms_ms": round(lat_chunk, 2),
        "parity_max_abs_diff": {"mag_mask": d_mag, "phase_res": d_ph},
        "checks": {"params<=1M": bool(ok_params), "chunk<=10ms": bool(ok_chunk),
                   "parity<1e-3": bool(ok_parity)},
    }
    (fc.FILTER_REPORT_DIR / "export_onnx.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""---
tags: [project, SafeSoundAI, filter, deployment, A2.6]
created: 2026-10-17
status: generated
---

# 过滤模型导出与延迟基准

| 项 | 值 |
|---|---|
| 格式 | {exported} |
| 文件 | `{out_path.name}`（{report['size_kb']} KB）|
| 参数量 | {params:,}（上限 1,000,000）|
| 单窗 1 s（100 帧）推理 | {report['latency_1s_ms']} ms |
| 单 chunk 100 ms（10 帧）推理 | {report['latency_chunk100ms_ms']} ms（上限 10 ms）|
| 导出前后一致性 max|Δ| | mag {d_mag:.2e} / phase {d_ph:.2e} |

## 验收

- 参数量 ≤ 1 M：{'✅' if ok_params else '❌'}
- 单 chunk ≤ 10 ms：{'✅' if ok_chunk else '❌'}
- 数值一致性 < 1e-3：{'✅' if ok_parity else '❌'}

> 说明：{note}。基准为 {('ONNXRuntime CPU') if exported == 'onnx' else 'PyTorch JIT CPU'}，
> 未做 int8 量化（A5 阶段再量化，预期再降 2–3×）。
"""
    (fc.FILTER_REPORT_DIR / "export_onnx.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("→ outputs/reports/export_onnx.md")


if __name__ == "__main__":
    main()
