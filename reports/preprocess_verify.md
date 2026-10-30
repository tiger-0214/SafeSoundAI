---
tags: [project, SafeSoundAI, data, audit]
created: 2026-10-17
---

# 预处理完整性二次核验（2026-10-17）

**结论：10/10 项 PASS。** 数据源 `/root/safesound_data/metadata_clean.csv`，train 58,161 段 / 9,172 文件，noise 14,040 段。

## 一、检查表

| 项 | 内容 | 结果 | 实测 | 预期 |
|---|---|:--:|---|---|
| P1 | 形状/精度/有限性（抽 400 段） | ✅ | 长度异常 0｜dtype 异常 0｜非有限 0 | 全 0 |
| P2 | segment_idx 从 0 连续无缺口 | ✅ | 起点≠0: 0｜终点≠n-1: 0｜序号重复: 0 | 全 0 |
| P3 | 原始 wav 全覆盖（无丢文件） | ✅ | urban 8732/8732｜esc50 映射类 440 ≤ 磁盘 2000 | 相等 |
| P4 | 10 类全覆盖 + 与 US8K 官方文件数对账 | ✅ | 缺类 0；少于官方：无 | 10 类且 ≥ 官方数 |
| P5 | 10 折齐全、每折 10 类且各类 ≥50 段 | ✅ | 折数 10｜最小类-折 115｜折规模 5763-6086 | 10 折, ≥50 |
| P6 | 泄漏：同文件 / 同录音 / 同波形跨折 | ✅ | 文件级 0｜录音级(fsID) 0｜波形级 0 | 全 0 |
| P7 | 噪声池独立（不与训练集共享原始文件） | ✅ | 14040 段 / 1560 文件；与训练集交集 0 | 交集 0 |
| P8 | 在线 Mel 输出 (128,100) 且有限 | ✅ | shape (128, 100)｜范围 [-20.7, 12.8] | (128,100) |
| P9 | 同一波形被标成不同类别（标签噪声） | ✅ | 0 组 | 0 组 |
| P10 | 类不平衡比 ≤10×（可不做重采样） | ✅ | 7.2×（engine_idling 8298 vs gun_shot 1157 段） | ≤10× |
| P11 | 对照：旧划分(fold9)中「同录音出现在训练集」的段占比 | 📝 | 5310/5825 = 91.2%（已修复为 0） | 0% |

## 二、细节

- 抽样长度唯一值：[16000]；std 中位 1.000（逐段 z-score 特征，std≈1）
- 每文件段数：min 1 / 中位 7 / max 9（1 s 窗 50% 重叠，4 s 文件应为 7 段）
- 段数分布：{7: 7333, 1: 786, 9: 440, 2: 196, 3: 171, 4: 112, 5: 73, 6: 61}
- 各折文件数（含 ESC-50 映射增量）：street_music +40｜street_music +40｜street_music +40｜street_music +160｜street_music +40｜street_music +40｜street_music +80
- 映射增量来源（ESC-50）：car_horn→car_horn，siren→siren，dog→dog_bark，engine→engine_idling，fireworks→gun_shot，glass_breaking→jackhammer，helicopter→engine_idling，airplane→engine_idling，train→engine_idling，chainsaw→drilling，clock_alarm→siren
- 折规模 CV（变异系数）= 0.017（<0.1 视为均衡）
- 波形 MD5 覆盖 58068 个唯一文件；完全重复波形组 0
- 噪声池按数据集：{'esc50': np.int64(14040)}
- 1.4 Mel 磁盘缓存：0 个文件 → 按 mel.py 设计**故意不缓存**（72k×128×100 float32 ≈ 14 GB 不划算），训练时在线计算，本次验证通过
- 旧划分的泄漏不止 87 组逐比特重复：**91.2% 的测试段来自训练集里出现过的同一录音**（UrbanSound8K 对同一录音做了 0.5 s 平移切片，1 s 窗重叠 50%），这是旧指标偏高的主因。

## 三、紧急类样本量（决定「能不能学会」）

| 类别 | 安全角色 | 段数 | 文件数 | 折内最少 |
|---|:--:|---|---|---|
| car_horn | 🔴 critical | 2136 | 469 | 173 |
| gun_shot | 🔴 critical | 1157 | 414 | 115 |
| siren | 🔴 critical | 7059 | 1009 | 689 |
| drilling | 🟠 danger | 6426 | 1040 | 549 |
| engine_idling | 🟠 danger | 8298 | 1160 | 826 |
| jackhammer | 🟠 danger | 6505 | 1040 | 553 |

## 四、历史缺陷修复记录（v1 文件级 → v2 录音级）

- **v1（原方案）**：按 `original_file` 划分。UrbanSound8K 对同一录音做 0.5 s 平移切片，导致 (a) 87 组逐比特相同的波形跨折，(b) 测试集大部分段与训练集共享同一录音 → 指标虚高。
- **实测泄漏**：旧 fold9 测试集中 91.2% 的段所属录音在训练集中出现过。
- **v2（现行）**：`recording_group()` 按 fsID 分组 + 贪心装箱分层，10 折每折 5,763–6,086 段、最小类-折 115 段，文件级/录音级/波形级泄漏全部为 0。
- 复现旧结果：`python3 build_metadata.py --file-level`。

## 五、与预期任务的差距（真实结论）

- `3_Mel_Spectrograms/` 为空**不是缺陷**（mel.py 明确选择在线计算），但笔记本/答辩若写「72,201 个 Mel 缓存」属于描述错误，应改为「在线计算」。
- `gun_shot` 仅 1157 段 / 414 文件（含 ESC-50 fireworks 40 文件），是唯一需要补数据的紧急类。
- ESC-50 映射存在语义风险：`fireworks→gun_shot`、`glass_breaking→jackhammer`、`clock_alarm→siren`、`chainsaw→drilling` 四类属于「近似替代」，会抬高紧急类召回但也引入标签噪声，论文里必须写明。
