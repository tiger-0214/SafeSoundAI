---
tags: [project, SafeSoundAI, filter, dataset, A1.6]
created: 2026-10-17
status: final-draft
task: A1.6
---

# 合成混音器规格（synth_mixer_spec）

> 目标：在没有真实多源录音的条件下，构造**带干净分轨（stems）**的混音数据，使过滤模块可被客观评测且可训练。
> 实现：`code/synth_mix.py`｜产物目录：`4_Synthetic_Mixtures/`

## 一、五元组定义

每条样本 = `(target, interferer, noise_bed, snr_target_db, keep_set)`：

| 分量 | 定义 | 来源 |
|---|---|---|
| **target** | 用户希望**抑制**的事件声（黑名单类 0/5/7/9/4 之一，或 neutral 2） | UrbanSound8K + ESC-50（映射后） |
| **interferer** | 用户希望**保留**的事件声（白名单类 1/8/6/3 之一） | 同上 |
| **noise_bed** | 无类别语音/环境底噪（不含任何黑名单/白名单事件），提供掩蔽与真实感 | ESC-50 非目标类（`ESC50_TO_TARGET=None`） |
| **snr_target_db** | target 相对 interferer 的输入 SNR（target 更强 = 干扰更大） | 采样自 `U(−5, 10)` dB |
| **keep_set** | 本次任务的保留集合（条件向量） | 采样自四档：{1,8} / {1,8,3} / {1,8,6,3} / 全 10 类 |

**关键约束**：单条样本内 `target` 与 `interferer` 必须来自**不同文件、不同类别**；同一 `rec_group`（同一原始录音）不得跨 train/val 划分（沿用文件级切分纪律）。

## 二、混音公式

```
y = a_t·s_target + a_i·s_interferer + a_n·s_noise
```

- 分轨先统一 16 kHz、单声道、裁剪/补零到 1 s（`SEGMENT_SAMPLES=16000`）。
- 分轨各自做峰值安全归一（`|s|∞ ≤ 0.9`），再按目标 SNR 定幅：
  ```
  E_t = Σ s_t²,  E_i = Σ s_i²
  a_i = 1,  a_t = sqrt(r · E_i / E_t),  r = 10^(snr_target_db/10)
  a_n = sqrt(r_n · (E_t·a_t² + E_i·a_i²) / E_n),  r_n = 10^(snr_noise_db/10), snr_noise_db ~ U(10, 20)
  ```
- 最终 `y` 做全局峰值归一（防削波），并记录 `scale` 以便对 stems 施加同一缩放（**评测必须对 stems 用同一 scale**，否则抑制量不可比）。

## 三、清单格式（`manifest.csv`）

| 列 | 含义 |
|---|---|
| `mix_id` | `mix_000001` |
| `mix_path` / `target_path` / `interferer_path` / `noise_path` | 四个 wav 的绝对路径（stems 必须保存，评测用） |
| `target_class` / `interferer_class` | 类名 |
| `target_id` / `interferer_id` | class_id |
| `snr_target_db` / `snr_noise_db` | 输入 SNR |
| `keep_set` | 逗号分隔的 class_id 列表（即条件向量真值） |
| `split` | `train` / `val` / `test` |
| `seed` | 随机种子（可复现） |
| `latency_ms` | 混音耗时（数据工程性能记录） |

## 四、划分与规模

| split | 条数 | 用途 |
|---|---:|---|
| train | 3,000 | 训练 TSE（方案 B） |
| val | 500 | 早停 / 超参 |
| test | 500 | 报告用（A2.5），**训练期不可见** |
| 合计 | 4,000 | ≈ 1.1 h 混音音频 + 3 倍 stems |

种子：`seed = 42 + index`，任何一条样本可用 `mix_id` 单条复现。

## 五、生成命令与产物

```bash
python code/synth_mix.py --n-train 3000 --n-val 500 --n-test 500 --workers 8
```

输出：

```
4_Synthetic_Mixtures/
├── manifest.csv
├── audio/mix_000001.wav          # 混音
├── stems/mix_000001_target.wav   # 黑名单分轨
├── stems/mix_000001_interf.wav   # 白名单分轨
├── stems/mix_000001_noise.wav    # 底噪
└── synth_stats.json              # 类分布 / SNR 分布 / 耗时
```

## 六、数据源现状（诚实口径）

| 数据集 | 路径 | 状态 | 用途 |
|---|---|---|---|
| UrbanSound8K | `1_Original_Database/UrbanSound8K/audio` | ✅ 8,732 文件（10 类齐全） | target/interferer 主力 |
| ESC-50 | `1_Original_Database/ESC-50-master/audio` | ✅ 2,000 文件 | 补充 + 噪声池 |
| FSD50K | `1_Original_Database/FSD50K` | ✅ 已下载 | 备用扩充（A2.3 若欠拟合） |
| AudioSet | `1_Original_Database/AudioSet`（47 G） | ⚠️ 结构未索引 | **未使用**，作为 AED 提召回的目标 |
| 真实多源录音 | — | ❌ 无 | 用 B4 主观评测兜底（自采 20–50 条） |

## 七、局限声明（必须写进报告）

1. 合成混音中的"保留/抑制"标签来自类别归属，**不含真实空间信息**（无 HRTF、无混响）。
2. 单声道混音 → 无法在数据层体现 A3 的 DOA 收益，过滤与定位的耦合只能靠实测。
3. 底噪取自 ESC-50 非目标类，仍可能包含与黑名单相似的频段 → 会有"类间泄漏"假阳性，需在报告中单列。
