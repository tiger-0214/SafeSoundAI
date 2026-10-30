"""
SafeSoundAI — 成果看板（build_dashboard.py）
===========================================
把 data_audit / eval_report / demo_stream / demo_haptic 的产物汇总成一个
自包含 HTML 看板（图片与音频用相对路径引用，双击即可在浏览器打开）。

产出：outputs/demo_dashboard.html
用法：python3 build_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config

OUT = config.OUTPUT_DIR
REP = OUT / "reports"
AUDIO = OUT / "audio" / "haptic"


def load_metrics() -> dict:
    p = REP / "aed_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    m = load_metrics()
    perc = m.get("per_class_f1", {})
    lat = m.get("latency_ms", {})
    foot = m.get("footprint", {})

    rows = sorted(perc.items(), key=lambda kv: kv[1])
    cls_rows = "\n".join(
        f"<tr><td>{c}</td><td>{'🔴 核心危险' if config.SAFETY_ROLE[c]=='critical' else ('🟠 高危' if config.SAFETY_ROLE[c]=='danger' else ('🟡 警示' if config.SAFETY_ROLE[c]=='warning' else '⚪ 背景'))}</td>"
        f"<td class='num'>{v:.3f}</td><td><div class='bar'><i style=\"width:{v*100:.0f}%\"></i></div></td></tr>"
        for c, v in rows)

    audio_items = "\n".join(
        f"""<div class="audio"><span class="tag">{config.SAFETY_ROLE[c]}</span>
        <span class="aname">{c.replace('_',' ')}</span>
        <audio controls preload="none" src="audio/haptic/{config.CLASS_TO_IDX[c]:02d}_{c}.wav"></audio></div>"""
        for c in config.TARGET_CLASSES)

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>SafeSoundAI · 数据工程 + AED 成果看板</title>
<style>
  :root {{ --ink:#0f1720; --sub:#6b7a8d; --line:#e3e8ee; --red:#c0392b;
           --org:#e67e22; --yel:#f1c40f; --grn:#27ae60; --blu:#2c6fb1; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
          color:var(--ink); background:#f6f8fa; }}
  header {{ background:linear-gradient(135deg,#0f1720,#1d2c3a 60%,#26425a); color:#fff;
            padding:38px 44px 26px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:.3px; }}
  header p {{ margin:0; color:#a9bccd; font-size:13.5px; }}
  .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:22px; }}
  .kpi {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14);
          border-radius:10px; padding:12px 16px; min-width:150px; }}
  .kpi b {{ display:block; font-size:22px; margin-top:4px; }}
  .kpi span {{ color:#9fb3c5; font-size:11.5px; letter-spacing:.6px; text-transform:uppercase; }}
  main {{ max-width:1180px; margin:0 auto; padding:26px 22px 60px; }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:14px;
             padding:22px 24px; margin-bottom:22px; }}
  h2 {{ font-size:17px; margin:0 0 4px; }}
  h2 small {{ font-weight:400; color:var(--sub); font-size:12.5px; margin-left:8px; }}
  .grid {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}
  img {{ width:100%; border:1px solid var(--line); border-radius:10px; display:block; }}
  .figcap {{ color:var(--sub); font-size:12px; margin-top:6px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--sub); font-weight:600; font-size:11.5px; text-transform:uppercase; letter-spacing:.5px; }}
  td.num {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  .bar {{ background:#eef2f6; border-radius:5px; height:8px; width:140px; overflow:hidden; }}
  .bar i {{ display:block; height:100%; background:var(--blu); }}
  .audio {{ display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px dashed var(--line); }}
  .audio .aname {{ min-width:150px; font-size:13px; }}
  .audio audio {{ height:28px; flex:1; }}
  .tag {{ font-size:10px; padding:2px 6px; border-radius:5px; color:#fff; background:var(--sub); }}
  .muted {{ color:var(--sub); font-size:12.5px; line-height:1.65; }}
  code {{ background:#f1f4f7; padding:1px 5px; border-radius:4px; font-size:12px; }}
  video {{ width:100%; border-radius:10px; border:1px solid var(--line); }}
  footer {{ color:var(--sub); font-size:12px; text-align:center; padding:10px 0 0; }}
</style></head>
<body>
<header>
  <h1>SafeSoundAI · 数据工程 + AED 模块成果</h1>
  <p>PTSD / 创伤后声音敏感人群的辅助性无障碍耳机 ｜ 更新于 2026-10-17 晚（录音级划分）｜
     模型 DS-CNN（<code>ds_cnn_best.pth</code>）｜ 指标口径：<b>fold 9，测试录音从未出现在训练集</b></p>
  <div class="kpis">
    <div class="kpi"><span>Test Accuracy（录音级）</span><b>{m.get('accuracy', 0)*100:.2f}%</b></div>
    <div class="kpi"><span>Macro F1</span><b>{m.get('macro_f1', 0):.3f}</b></div>
    <div class="kpi"><span>校准误差 ECE</span><b>{m.get('ece', 0):.4f}</b></div>
    <div class="kpi"><span>端到端延迟(CPU)</span><b>{lat.get('端到端(CPU)', 0):.2f} ms</b></div>
    <div class="kpi"><span>模型体积</span><b>{foot.get('checkpoint KB', 0):.0f} KB</b></div>
    <div class="kpi"><span>参数量</span><b>{foot.get('参数量', 0):,}</b></div>
  </div>
</header>
<main>

<section style="border-color:#e67e22;background:#fffaf3">
  <h2>⚠️ 口径修正（2026-10-17 晚）<small>旧数字作废，原因已定位</small></h2>
  <p class="muted" style="margin:8px 0 0">
    旧划分按「文件」切分，而 UrbanSound8K 对同一条录音做了 0.5 s 平移切片：
    <b>旧 fold 9 测试集中 91.2% 的段，其所属录音在训练集出现过</b>。模型因此可以「认录音」拿分。
    现已改为按 <b>录音（fsID）分组</b>切分，三类泄漏（文件/录音/波形）全部归零，模型重训。
  </p>
  <table style="margin-top:12px">
    <thead><tr><th>模型</th><th>测试集</th><th>准确率</th><th>危险类 macro 召回</th></tr></thead>
    <tbody>
      <tr><td>v1 旧模型</td><td>v1 文件级</td><td class="num">89.82%</td><td class="num">91.6%</td></tr>
      <tr><td>v1 旧模型</td><td>v2 录音级</td><td class="num">94.88%</td><td class="num">96.1%</td></tr>
      <tr><td>v2 新模型</td><td>v1 文件级</td><td class="num">86.35%</td><td class="num">90.4%</td></tr>
      <tr style="background:#fdf3e7"><td><b>v2 新模型</b></td><td><b>v2 录音级</b></td><td class="num"><b>64.11%</b></td><td class="num"><b>67.4%</b></td></tr>
    </tbody>
  </table>
  <p class="muted" style="margin-top:10px">
    最后一格才是真实泛化。紧急类事件召回（τ=0.5）核心 3 类 <b>83.3%</b> / 6 类 <b>72.4%</b>，
    误报 <b>123.8 次/小时</b> → <b>尚不满足验收（≥95% 召回、≤5 次/小时）</b>。
    详见 <code>reports/emergency_eval.md</code>、<code>reports/decision_policy.md</code>、<code>reports/protocol_compare.md</code>。
    Demo 3/4 的命中率与音频是在旧模型上生成的，仅作链路演示，不代表当前模型性能。
  </p>
</section>

<section>
  <h2>① 数据工程 <small>72,201 段 / 4 套数据集 / 10,732 个原始文件</small></h2>
  <div class="grid">
    <div><img src="figures/fig01_dataset_composition.png"><div class="figcap">数据集构成：UrbanSound8K 54,201 段为主训练集，ESC-50 映射 3,960 段，14,040 段进噪声池</div></div>
    <div><img src="figures/fig02_class_distribution.png"><div class="figcap">10 类分布：最大/最小 7.2×，三个核心危险类恰是样本最少的一批</div></div>
    <div><img src="figures/fig03_mel_samples.png"><div class="figcap">每类 log-Mel 样本（128×100，1 s @16 kHz）</div></div>
    <div><img src="figures/fig04_fold_and_leakage.png"><div class="figcap">10 折分层 + 质量核验：路径 100% 存在、零跨折泄漏、零静音/NaN</div></div>
  </div>
  <p class="muted" style="margin-top:14px">
    关键结论：<b>路径完整性 72,201/72,201</b>；<b>跨折泄漏 0</b>（按 <code>fsID</code> 录音级切分）；
    长度全部 16,000 点；样本为逐段 z-score（std≈1，峰值中位 4.4σ）。
    ⚠️ z-score 抹掉了绝对响度，所以「响度→震动强度」必须由原始波形单独估（已如此实现）。
  </p>
</section>

<section>
  <h2>② AED 模型评估 <small>从训练日志数字升级为可投稿的图与表</small></h2>
  <div class="grid">
    <div><img src="figures/fig05_confusion_matrix.png"><div class="figcap">混淆矩阵（计数 + 行归一化）</div></div>
    <div><img src="figures/fig06_per_class_prf.png"><div class="figcap">每类 P/R/F1，最弱项 gun_shot（114 帧）</div></div>
    <div><img src="figures/fig07_confidence_calibration.png"><div class="figcap">置信度分布 + 校准曲线（ECE 0.016，阈值可信）</div></div>
    <div><img src="figures/fig08_threshold_sweep.png"><div class="figcap">安全阈值扫描：误报（打扰）vs 漏报（危险）</div></div>
  </div>
  <h2 style="margin-top:22px">各类 F1（升序）</h2>
  <table><thead><tr><th>类别</th><th>安全角色</th><th>F1</th><th></th></tr></thead>
  <tbody>{cls_rows}</tbody></table>
</section>

<section>
  <h2>③ Demo 1 — 实时检测回放 <small>合成 24 s 城市街道，1 s 窗 / 0.25 s 步进</small></h2>
  <div class="grid" style="grid-template-columns:1.25fr .75fr">
    <div>
      <img src="figures/fig09_stream_timeline.png">
      <div class="figcap">真值区间 → 预测点 → 关键类概率 → 触觉强度（强度随距离变化）</div>
    </div>
    <div>
      <video controls loop muted playsinline src="demo_stream.mp4"></video>
      <div class="figcap">逐帧回放：波形游标 / 当前窗 Mel / 10 类概率条 / 触觉决策（mp4）</div>
      <p class="muted">全量命中率 <b>87.1%</b>，稳态窗（排除跨事件边界）<b>95.8%</b>；
      93 帧中 72 帧触发报警。场景音频：<a href="audio/demo_stream_scene.wav">demo_stream_scene.wav</a></p>
    </div>
  </div>
</section>

<section>
  <h2>④ Demo 2 — 触觉编码可听化 <small>175 Hz LRA 载波，四维编码全部可听</small></h2>
  <div class="grid" style="grid-template-columns:1fr 1fr">
    <div><img src="figures/fig10_haptic_envelopes.png"><div class="figcap">10 类震动包络：振幅=强度(D3)、周期=紧急度(D4)、形状=类型(D1)</div></div>
    <div>
      <p class="muted">把 <code>HapticPattern</code> 渲染成音频后可直接做主观盲测。
      修复了旧版 <code>subjective_eval.py</code> 丢维度的问题：<br>
      <code>air_conditioner</code> vs <code>engine_idling</code> 波形相关 <b>1.000 → 0.651</b>；
      10 类包络指纹唯一数 <b>10/10</b>。</p>
      <div class="audio"><span class="tag">对比</span><span class="aname">旧 bug vs 修复后</span>
        <audio controls preload="none" src="audio/haptic/_contrast_old_vs_new.wav"></audio></div>
      <div class="audio"><span class="tag">序列</span><span class="aname">喇叭 → 警笛 → 枪声</span>
        <audio controls preload="none" src="audio/haptic/_alert_sequence.wav"></audio></div>
      <h2 style="margin-top:16px;font-size:14px">十类震动试听</h2>
      {audio_items}
    </div>
  </div>
</section>

<section>
  <h2>⑤ 怎么复现</h2>
  <p class="muted">全部脚本在 <code>0-SafeSound AI workspace/code/</code>，先
  <code>export SAFESOUND_DATA_DIR=/root/safesound_data</code>（ext4 副本，绕开 /mnt/d 的 IO 瓶颈）：</p>
  <table>
    <thead><tr><th>脚本</th><th>产出</th></tr></thead>
    <tbody>
      <tr><td><code>python3 data_audit.py</code></td><td>数据工程审计：4 张图 + reports/data_audit.md</td></tr>
      <tr><td><code>python3 eval_report.py</code></td><td>AED 评估：4 张图 + reports/aed_eval.md + aed_metrics.json</td></tr>
      <tr><td><code>python3 demo_infer.py --wav X.wav</code></td><td>单文件滑窗推理 → 决策 → 触觉</td></tr>
      <tr><td><code>python3 demo_infer.py --demo-set</code></td><td>十类抽样链路验证</td></tr>
      <tr><td><code>python3 demo_stream.py</code></td><td>城市街道回放：时间线图 + mp4/gif + 场景 wav</td></tr>
      <tr><td><code>python3 demo_haptic_audio.py</code></td><td>触觉可听化：10 段 wav + 包络图</td></tr>
      <tr><td><code>python3 preprocess_verify.py</code></td><td>预处理 11 项核验 → 报告 + fig11</td></tr>
      <tr><td><code>python3 eval_emergency.py</code></td><td>紧急 6 类段级/事件级检出 + SNR 鲁棒性 → fig12-14</td></tr>
      <tr><td><code>python3 eval_decision_policy.py</code></td><td>温度标定 + 阈值权衡 → fig15</td></tr>
      <tr><td><code>python3 compare_protocols.py</code></td><td>两套划分 × 两个模型 对照表</td></tr>
      <tr><td><code>python3 build_dashboard.py</code></td><td>本页面</td></tr>
    </tbody>
  </table>
  <p class="muted" style="margin-top:12px">下一步（不在本 demo 范围）：A2 降噪 <code>attenuator.py</code>、
  A3 DOA <code>doa.py</code>、PANNs 迁移实验、用户主观盲测。</p>
</section>

<footer>SafeSoundAI · 数据工程与 AED 模块闭环 · 所有数字均为实测（fold 9，5,825 段）</footer>
</main></body></html>"""

    out = OUT / "demo_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"看板 → {out}（{out.stat().st_size / 1024:.0f} KB）")

    # 顺带生成一份纯文本索引，便于在 Obsidian 里引用
    idx = ["---\ntags: [project, SafeSoundAI, demo, index]\ncreated: 2026-10-17\n---\n",
           "# 成果产物索引（2026-10-17）\n",
           "## 看板\n", f"- `outputs/demo_dashboard.html` — 双击用浏览器打开（含全部图/视频/音频）\n",
           "## 数据工程\n",
           "| 文件 | 说明 |\n|---|---|\n"
           "| figures/fig01_dataset_composition.png | 数据集构成 |\n"
           "| figures/fig02_class_distribution.png | 10 类分布 + 类不平衡 |\n"
           "| figures/fig03_mel_samples.png | 各类 Mel 样本 |\n"
           "| figures/fig04_fold_and_leakage.png | 折分布 + 质量核验 |\n"
           "| reports/data_audit.md | 审计报告（含全部 json 核验结果）|\n",
           "## AED 评估\n",
           "| 文件 | 说明 |\n|---|---|\n"
           "| figures/fig05_confusion_matrix.png | 混淆矩阵 |\n"
           "| figures/fig06_per_class_prf.png | 每类 P/R/F1 |\n"
           "| figures/fig07_confidence_calibration.png | 置信度 + 校准 |\n"
           "| figures/fig08_threshold_sweep.png | 安全阈值扫描 |\n"
           "| reports/aed_eval.md | 评估报告 |\n"
           "| reports/aed_metrics.json | 机器可读指标快照 |\n",
           "## Demo\n",
           "| 文件 | 说明 |\n|---|---|\n"
           "| figures/fig09_stream_timeline.png | 城市街道检测时间线 |\n"
           "| demo_stream.mp4 / .gif | 逐帧回放动画 |\n"
           "| audio/demo_stream_scene.wav | 合成街景音频（24 s）|\n"
           "| figures/fig10_haptic_envelopes.png | 10 类触觉包络 |\n"
           "| audio/haptic/*.wav | 触觉可听化音频（10 类 + 对比 + 序列）|\n"
           "| reports/demo_infer_demo_set.md | 十类抽样链路验证 |\n"
           "| reports/demo_stream.md | 回放日志（报警事件表）|\n"
           "| reports/demo_haptic.md | 触觉编码说明 |\n"]
    (REP / "demo_index.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"索引 → {REP / 'demo_index.md'}")


if __name__ == "__main__":
    main()
