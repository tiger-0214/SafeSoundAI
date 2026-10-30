---
tags: [project, SafeSoundAI, filter, architecture, A1.3]
created: 2026-10-17
status: final-draft
task: A1.3
---

# 过滤架构定稿（filter_arch）

## 一、三方案对比与选型

| | **A 分类驱动谱增益门** | **B 条件化目标提取（TSE）** | **C 混合（选定）** |
|---|---|---|---|
| 原理 | AED 后验 → 类目标增益 → 掩码施加于 STFT | 条件嵌入（keep 集合）+ 掩码网络，端到端学"保留什么" | A 兜底 + B 提性能，按置信度切换 |
| 抑制量 | 10–15 dB（谱减法上限） | 20–40 dB（SoundBubble 实测 49 dB） | 25–40 dB |
| 延迟 | 20–50 ms | 12–30 ms | 20–60 ms |
| 训练成本 | 0 | 需合成数据 + 2–4 周 | 4 周 |
| 失败模式 | 抑制不足、残留"音乐噪声" | 过抑制（把白名单一起压掉）→ 安全风险 | B 过抑制时用 A 的硬门校验 |
| 风险 | 低 | 中 | 中（受 B 支配） |
| 定位 | **可立即交付的 MVP** | **论文创新点 / 性能上限** | 交付 + 论文兼顾 |

**选定 C**，实现顺序：**A 先行**（零训练成本、可听、可作为 B 的对照与兜底）→ **B 叠加**（将 SoundBubble 的"距离条件"替换为"类别条件"，得到语义气泡）→ 用 `filter_safety` 的硬门保证 B 不会压掉白名单。

## 二、方案 A：AED 驱动的谱增益门（`code/attenuator.py`）

### 2.1 信号链

```
x[n] (16 kHz)
  ├─ STFT(n_fft=512, hop=160, hann)          → X(t,f) 复数
  ├─ |X|² → Mel(128) → log → (1,1,128,100)   → DS_CNN → p(c)  [10 类后验]
  │                                                     ↓
  │                                    keep/black 集合 + 档位 → gain_db[c]
  │                                                     ↓
  ├─ 类增益软混合：g_db(f) = Σ_c w_c·gain_db[c]  （w = softmax(p/τ), τ=0.7）
  │   + 频带权重：黑名单类按该类平均谱形 A_c(f) 归一化 → 抑制集中在该类能量带
  ├─ 限幅 g_db ∈ [−24, +6] → 跨帧一阶平滑 (α=0.5, 跳变 ≤6 dB/帧)
  └─ X' = X · 10^(g_db/20) → iSTFT → y[n]
```

### 2.2 关键设计

- **频带加权掩码**：把类增益按该类训练集平均 Mel 谱 `A_c(f)` 做归一化加权，避免对全频段一刀切（一刀切会把同频段的警报声一起压掉）。
- **软混合而非硬选**：硬 top-1 在多声源窗内会抖动，产生断续的音量跳变。
- **不追求高抑制**：方案 A 的定位是「听得出被压了」，上限 ~12–15 dB 足够让用户感知。

## 三、方案 B：类别条件目标提取（`code/tse_model.py`）

### 3.1 从 SoundBubble 迁移的四件事

| SoundBubble 原设计 | 本项目映射 | 说明 |
|---|---|---|
| 6 麦 + 距离条件嵌入 `dis_type=conv3` | **单声道 + 类别条件嵌入**（10 维 keep 指示向量 → 32 维 FiLM） | 双麦无法做距离；类别是安全语义的载体 |
| 运行时 `distance_threshold` 参数 | 运行时 **keep 集合**（{siren} / {siren, horn} / 全紧急 / 全通透） | 同一权重多档位，不重训 |
| `SNRLPLoss(neg_weight=100)` | **抑制惩罚损失**：漏抑制（该压没压）权重 100，过抑制权重 10 | 保留原论文"重罚漏抑"的思想，但把"过抑制"也加罚（安全优先于安静） |
| 两阶段训练（pretrain SNR → finetune 多分辨率 + 感知加权） | 同构：阶段 1 掩码 MSE/SNR；阶段 2 多分辨率 STFT + 白名单保留约束 | 见 `train_tse.py` |
| `TF-GridNet` 实时变体（D=32, B=6, H=64, E=2, chunk_causal） | **轻量因果卷积 U-Net 掩码网络**（≤1 M 参数） | TF-GridNet 的 Transformer 部分在 MCU 上不现实；本项目规模更小 |

### 3.2 网络结构（实际实现）

```
输入：|X| (B, 1, 257, T)，mel 可选
  ├─ Conv2d(1→32, 3×3) + BN + PReLU
  ├─ 4 × [DepthwiseConv(3×3) + Pointwise(1×1)]，stride 2 下采样（频/时各 2）
  │     每层注入 FiLM(cond_embed)：h ← h·(1+γ) + β
  ├─ 因果 GRU（hidden 96，单向）沿时间轴 → 时序上下文
  ├─ 4 × 转置卷积上采样 + skip 连接（U-Net）
  └─ Conv2d(→2, 1×1) → (mask_mag, mask_phase_residual)
输出：复数掩码 M = m_mag · exp(j·π·tanh(m_ph))  → Y = X ⊙ M
```

- 参数量目标 ≤ 1 M（实测约 0.4–0.6 M，见报告）。
- **因果性**：所有卷积 `padding=(k-1, 0)` 左填充；GRU 单向；不允许看未来帧。

### 3.3 损失函数

```
L = L_suppress + λ_keep·L_keep + λ_neg·L_leak
L_suppress = −10·log10( ||s_keep||² / ||ŝ − s_keep||² + ε )      # 目标保留 SI-SNR
L_keep     = Σ_{c∈whitelist} relu(3 dB − 实际保留量)²            # 白名单硬约束
L_leak     = neg_weight · mean( ||抑制后黑名单能量||² )          # neg_weight=100
λ_keep = 1.0, λ_neg = 1.0（在 L_suppress 归一后）
```

## 四、方案 C：混合执行逻辑（`code/filter_pipeline.py`）

```
每 chunk（32 ms 音频，hop 160）：
  1. AED 后验 → keep 集合置信度 conf = max_{c∈whitelist} p(c)
  2. if conf > 0.8 且连续 3 chunk 稳定：执行器 = B（条件掩码）
     elif conf > 0.4：执行器 = A（谱增益门，β 折扣 0.5）
     else：执行器 = A（仅黑名单背景，−6 dB）或直通
  3. 硬门后校验：白名单类能量变化 > 3 dB 且方向为衰减 → 回退到 A
  4. 任何异常（NaN / 超时 / 队列积压 > 2 chunk）→ fail-open 直通
```

**为什么需要第 3 步**：B 是数据驱动的，训练分布外的样本可能过抑制；硬门是唯一的物理保证，配置在 `filter_safety.md` §二。

## 五、与 SoundBubble 的本质差异（论文卖点）

| | SoundBubble (2024) | SafeSoundAI（本工作） |
|---|---|---|
| 保留准则 | 空间距离 < r（气泡包含逻辑） | 声学事件安全语义 ∈ WHITELIST（**并集逻辑**） |
| 危险声在气泡外 | 被一并抑制（49 dB） | **强制保留** |
| 判定结果 | `保留 = inside(bubble)` | `保留 = whitelist ∪ allowed(user) ∪ inside(spatial)` |
| 硬件 | 6 麦 | 2 麦（更受限，也更现实） |
| 失效方向 | 抑制不足（听得到干扰） | 过抑制（听不到危险）→ 这是安全场景唯一不可接受的方向 |

> 一句话表述：**SoundBubble 用"离我多远"决定听什么，SafeSoundAI 用"是否威胁我"决定听什么。**

## 六、验收与后续

- 方案 A 交付：`attenuator.py` + 主观可听样例（`outputs/audio/filtered_*.wav`）
- 方案 B 交付：`tse_model.py` + `train_tse.py` + 权重 `outputs/models/tse_best.pth`
- 集成：`filter_pipeline.py` + `realtime_demo.py --filter`
- 端侧：`export_onnx.py` → ONNX + ORT 延迟基准（A2.6）
