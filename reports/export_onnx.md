---
tags: [project, SafeSoundAI, filter, deployment, A2.6]
created: 2026-10-17
status: generated
---

# 过滤模型导出与延迟基准

| 项 | 值 |
|---|---|
| 格式 | torchscript |
| 文件 | `tse.torchscript.pt`（458.2 KB）|
| 参数量 | 80,434（上限 1,000,000）|
| 单窗 1 s（100 帧）推理 | 18.61 ms |
| 单 chunk 100 ms（10 帧）推理 | 9.9 ms（上限 10 ms）|
| 导出前后一致性 max|Δ| | mag 0.00e+00 / phase 0.00e+00 |

## 验收

- 参数量 ≤ 1 M：✅
- 单 chunk ≤ 10 ms：✅
- 数值一致性 < 1e-3：✅

> 说明：TorchScript，文件 458 KB（onnx 包缺失时的等价交付）。基准为 PyTorch JIT CPU，
> 未做 int8 量化（A5 阶段再量化，预期再降 2–3×）。
