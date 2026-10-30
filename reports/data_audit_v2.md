# Trigger-AED v2 数据审计报告

数据源：`/root/safesound_data/trigger_v2/metadata_v2.csv`

## 0. 总览

- 总段数：**24,451**
- 独立来源录音：**3,972**（录音组 1,415）
- 类别：siren, car_horn, gun_shot, fireworks, background
- 总时长（1 s 段累加）：**6.79 小时**
- 类不平衡比（最大/最小）：**40.3×**

## A. 完整性

- 可加载且 shape 正确：24,451 / 24,451 （100.00%）
- NaN 段：0，Inf 段：0

## B. 信号质量

RMS 归一化后每段应有 `rms ≈ 1.0`；`peak` 反映峰均比（爆响类应明显 > 音调类）。

| 类 | rms 中位 | peak 中位 | max abs(dc) | 触顶平台比例均值 |
|---|---:|---:|---:|---:|
| siren | 1.000 | 3.794 | 0.9554 | 0.01% |
| car_horn | 1.000 | 4.315 | 0.6884 | 0.01% |
| gun_shot | 1.000 | 5.806 | 0.8627 | 0.01% |
| fireworks | 1.000 | 14.283 | 0.1589 | 0.01% |
| background | 1.000 | 4.486 | 0.9570 | 0.01% |

- 静音段（peak<1e-3）：**0**
- 疑似源端削波段（>1% 样本压在段内峰值上）：**0**

## C. 分布

| 类 | 1 s 段数 | 源文件 | 录音组 | 段/录音 | 段长中位 (s) |
|---|---:|---:|---:|---:|---:|
| siren | 6,664 | 964 | 114 | 6.9 | 1.00 |
| car_horn | 2,136 | 469 | 165 | 4.6 | 1.00 |
| gun_shot | 795 | 372 | 115 | 2.1 | 1.00 |
| fireworks | 360 | 40 | 40 | 9.0 | 1.00 |
| background | 14,496 | 2,127 | 982 | 6.8 | 1.00 |

按数据集：

| dataset | 段数 |
|---|---:|
| esc50 | 3,582 |
| urban | 20,869 |

按切分 × 类别：

| 类 | test | train | val |
|---|---:|---:|---:|
| siren | 1,486 | 4,514 | 664 |
| car_horn | 336 | 1,720 | 80 |
| gun_shot | 52 | 664 | 79 |
| fireworks | 27 | 306 | 27 |
| background | 1,659 | 11,600 | 1,237 |

## D. 泄漏检查

录音级分组切分的判据：同一个 `rec_group` 只能出现在一个 fold。

| dataset | 录音组 | 跨折组数 | 判定 |
|---|---:|---:|---|
| esc50 | 398 | 0 | ✅ 无泄漏 |
| urban | 1,017 | 0 | ✅ 无泄漏 |

- 已剔除的近似重复录音：0 段（跨库均值谱指纹命中，保留 urban > esc50 > fsd50k 优先级）

## E. 不平衡与训练建议

- 最大/最小 = **40.3×**（background vs fireworks）
- 若按独立录音看：最大/最小 = 53.2×
- 建议：训练用 `WeightedRandomSampler`（按类倒数频率）或 `CrossEntropyLoss(weight=...)`；评测一律报 **macro-F1** 与 **事件级召回/误报率**，不看总体准确率。

## 图

![fig_v2_class_dist](<workspace>/outputs/figs/v2/fig_v2_class_dist.png)
![fig_v2_mel_examples](<workspace>/outputs/figs/v2/fig_v2_mel_examples.png)
![fig_v2_seg_vs_file](<workspace>/outputs/figs/v2/fig_v2_seg_vs_file.png)
