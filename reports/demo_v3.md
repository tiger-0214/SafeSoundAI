---
tags: [report, demo, SafeSoundAI]
created: 2026-10-30
status: generated
---

# Demo v3 — detect → haptic code（一条命令跑通的闭环）

> 脚本：`code/demo_v3.py`（12 s 内置街景，测试集 fold 9 片段拼装，无需外部资产）
> 模型：`outputs/models/dscnn_v2_4cls_fft1024.pt`（4 类：siren / car_horn / gun_shot / background）
> 复现：`SAFESOUND_DATA_DIR=<data> python3 code/demo_v3.py`

```
device cuda | ckpt dscnn_v2_4cls_fft1024.pt | 65,796 params | classes ['siren', 'car_horn', 'gun_shot', 'background']
source: built-in 12 s scene (fold-9 test segments)
audio: 12.00 s | 23 windows (1 s / 0.5 s hop) | tau=0.6

 t(s) top1            p top2           p2  trigger
  0.0 background  0.958 car_horn    0.026  
  0.5 background  0.954 gun_shot    0.021  
  1.0 background  0.907 gun_shot    0.056  
  1.5 siren       0.591 background  0.292  
  2.0 siren       0.956 car_horn    0.019  FIRE
  2.5 siren       0.717 background  0.144  FIRE
  3.0 siren       0.751 background  0.142  FIRE
  3.5 siren       0.976 gun_shot    0.014  FIRE
  4.0 siren       0.998 car_horn    0.001  FIRE
  4.5 siren       0.832 background  0.123  FIRE
  5.0 background  0.918 car_horn    0.048  
  5.5 car_horn    0.640 background  0.140  FIRE
  6.0 car_horn    0.834 siren       0.138  FIRE
  6.5 car_horn    0.964 gun_shot    0.017  FIRE
  7.0 car_horn    0.994 gun_shot    0.003  FIRE
  7.5 car_horn    0.961 background  0.025  FIRE
  8.0 background  0.677 gun_shot    0.256  
  8.5 gun_shot    0.974 background  0.020  FIRE
  9.0 gun_shot    0.997 background  0.002  FIRE
  9.5 gun_shot    0.959 background  0.031  FIRE
 10.0 background  0.919 car_horn    0.044  
 10.5 background  0.904 gun_shot    0.049  
 11.0 background  0.946 gun_shot    0.039  

--- haptic alarm events (after arbitration) ---
 t(s) class     urgency   pattern    interval  amplitude  code
  2.0 siren     alert     ramp          400ms       0.40  siren:alert
  5.5 car_horn  high      dual          700ms       0.70  car_horn:high
  8.5 gun_shot  critical  single       3000ms       1.00  gun_shot:critical

--- ground truth in the built-in scene ---
   2.0- 5.0 s  siren      detected @ 2.0s
   6.0- 8.0 s  car_horn   detected @ 5.5s
   9.0-10.0 s  gun_shot   detected @ 8.5s
```
