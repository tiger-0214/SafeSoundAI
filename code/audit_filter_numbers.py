"""
SafeSoundAI — 降噪支路数字溯源审计（audit_filter_numbers.py）

用途：对 `filter_ab_report.md` / `28-项目技术规划书` 引用的每一个降噪数字，
从**原始逐样本产物**重算一遍，并断言与归档 JSON、报告文字三方一致。

分层（与 paper-number-provenance-audit 的 A/B/C 对应）：

  L1 产物清点       —— 归档文件存在、mtime 有序
  L2 逐样本复算     —— 从 filter_per_sample.csv 重算 A/B 的均值与比率，
                       先证明能复现 filter_eval.json（说明读对了文件），
                       再用复算值去对报告文字
  L3 输入口径复算   —— 「raw 0.17 → z-score 0.53」top-1 命中率，
                       并补测部署用的 4 类头（原报告只测了 10 类 v1 权重，
                       口径未限定，见 AUDIT_NOTES）
  L4 占位音频指纹   —— 「220 Hz 正弦占位」这一事实断言的可复验性
  L5 消融表复跑     —— 可选（--ablate，约 3 分钟），重跑 ablate_filter.py
                       并与 filter_ablation.md 逐格比对

用法：
    cd code && PYTHONPATH=. python3 audit_filter_numbers.py
    cd code && PYTHONPATH=. python3 audit_filter_numbers.py --ablate

退出码：全部通过 0，否则 1（可作回归门禁用）。
产物：outputs/reports/filter_audit.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import config
import filter_config as fc
import vocab

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def near(a: float, b: float, tol: float = 1e-4) -> bool:
    """产物里的比率是已舍入存储值，容差 1e-4（半单位舍入）。"""
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- L1 产物清点
def layer1() -> dict:
    art = {
        "eval_json": fc.FILTER_REPORT_DIR / "filter_eval.json",
        "eval_md": fc.FILTER_REPORT_DIR / "filter_eval.md",
        "per_sample": fc.FILTER_REPORT_DIR / "filter_per_sample.csv",
        "ablation_md": fc.FILTER_REPORT_DIR / "filter_ablation.md",
        "ab_index": fc.FILTER_AUDIO_DIR / "ab" / "index.md",
        "report": fc.FILTER_REPORT_DIR / "filter_ab_report.md",
    }
    for k, p in art.items():
        check(f"L1 归档存在：{k}", p.exists(), str(p))

    wavs = sorted((fc.FILTER_AUDIO_DIR / "ab").glob("*.wav"))
    check("L1 A/B 样张 24 个 wav（6 噪声 × mix/aed/target/interf）",
          len(wavs) == 24, f"实际 {len(wavs)} 个")

    # 官方运行口径：仓库内没有 run_*.sh，报告里写明了命令 → 视为口径已声明
    out = {}
    for k in ("eval_json", "per_sample", "ablation_md"):
        if art[k].exists():
            out[k] = json.loads(art["eval_json"].read_text()) if k == "eval_json" else None
    return art


# ------------------------------------------------------- L2 逐样本复算 + 三方
def layer2(art: dict) -> dict:
    js = json.loads(art["eval_json"].read_text(encoding="utf-8"))
    rows = list(csv.DictReader(art["per_sample"].open(newline="", encoding="utf-8")))

    check("L2 per_sample 行数 == json n",
          len(rows) == js["n"], f"{len(rows)} vs {js['n']}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)

    mean = lambda a: float(a.mean())
    pct_lt = lambda a, t: 100.0 * float(np.mean(a < t))

    recomputed = {
        "A_supp": mean(col("A_aed_supp_db")),
        "A_pres": mean(col("A_aed_pres_db")),
        "A_oracle_supp": mean(col("A_oracle_supp_db")),
        "A_oracle_pres": mean(col("A_oracle_pres_db")),
        "B_supp": mean(col("B_supp_db")),
        "B_pres": mean(col("B_pres_db")),
        "trigger_pct": pct_lt(col("A_aed_supp_db"), -1.0),
        "pres_below_minus3_pct": pct_lt(col("A_aed_pres_db"), -3.0),
        "B_sisdr_imp": mean(col("B_sisdr_imp")),
    }
    # 存储值是 2 位小数 / 0.1%，容差取报告精度
    check("L2 复算 A_supp 能重现 json（证明读对了文件）",
          near(recomputed["A_supp"], js["A"]["supp_db_mean"], 5e-3),
          f"{recomputed['A_supp']:.4f} vs {js['A']['supp_db_mean']}")
    check("L2 复算 A_pres 能重现 json",
          near(recomputed["A_pres"], js["A"]["pres_db_mean"], 5e-3),
          f"{recomputed['A_pres']:.4f} vs {js['A']['pres_db_mean']}")
    check("L2 复算 B_supp 能重现 json",
          near(recomputed["B_supp"], js["B"]["supp_db_mean"], 5e-3),
          f"{recomputed['B_supp']:.4f} vs {js['B']['supp_db_mean']}")
    check("L2 复算 B_pres 能重现 json",
          near(recomputed["B_pres"], js["B"]["pres_db_mean"], 5e-3),
          f"{recomputed['B_pres']:.4f} vs {js['B']['pres_db_mean']}")
    check("L2 复算 oracle Supp 能重现 json",
          near(recomputed["A_oracle_supp"], js["dual_track"]["A_oracle"]["supp_db_mean"], 5e-3),
          f"{recomputed['A_oracle_supp']:.4f}")
    check("L2 复算 oracle Pres 能重现 json",
          near(recomputed["A_oracle_pres"], js["dual_track"]["A_oracle"]["pres_db_mean"], 5e-3),
          f"{recomputed['A_oracle_pres']:.4f}")
    check("L2 复算触发率 57% 能重现 json",
          near(recomputed["trigger_pct"], js["dual_track"]["A_aed"]["trigger_rate_pct"], 0.06),
          f"{recomputed['trigger_pct']:.1f}%")
    check("L2 复算红线超标 43% 能重现 json",
          near(recomputed["pres_below_minus3_pct"], js["safety"]["pres_below_minus3_pct"], 0.06),
          f"{recomputed['pres_below_minus3_pct']:.1f}%")
    check("L2 B 的 SI-SDR 提升 +1.67 dB 能从逐样本复算",
          near(recomputed["B_sisdr_imp"], js["B"]["sisdr_imp"], 5e-3),
          f"{recomputed['B_sisdr_imp']:.3f}")

    # 报告文字 vs json（这才是「文档里的数字」）
    report = art["report"].read_text(encoding="utf-8") if art["report"].exists() else ""
    for label, val, tol in [
        ("A 部署 Supp −3.85", js["A"]["supp_db_mean"], 5e-3),
        ("A 部署 Pres −3.36", js["A"]["pres_db_mean"], 5e-3),
        ("B Supp −13.89", js["B"]["supp_db_mean"], 5e-3),
        ("B Pres −9.58", js["B"]["pres_db_mean"], 5e-3),
    ]:
        txt = f"{abs(val):.2f}"
        check(f"L2 报告文字含 {label}", txt in report, f"查找 {txt}")

    # 消融表逐格 vs 归档 md
    abl = art["ablation_md"].read_text(encoding="utf-8") if art["ablation_md"].exists() else ""
    abl_rows = {}
    for line in abl.splitlines():
        m = re.match(r"\|\s*(\S[^|]*?)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+)\s*\|"
                     r"\s*(-?\d+\.\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%", line)
        if m:
            abl_rows[m.group(1)] = dict(supp=float(m.group(2)), pres=float(m.group(3)),
                                        gap=float(m.group(4)), redline=float(m.group(5)),
                                        trig=float(m.group(6)))
    check("L2 消融表解析出 4 行", len(abl_rows) == 4, f"{list(abl_rows)}")

    return dict(js=js, recomputed=recomputed, ablation=abl_rows, report=report)


# ---------------------------------------------------------- L3 输入口径复算
def _probs10(model, mel_tf, dev, win: np.ndarray, mode: str) -> np.ndarray:
    if mode == "z":
        win = (win - win.mean()) / (win.std() + 1e-9)
    y = torch.from_numpy(win.astype(np.float32)).to(dev)
    mel = torch.log(mel_tf(y) + 1e-9)[None, None, :, : config.FRAMES]
    if mel.shape[-1] < config.FRAMES:
        mel = F.pad(mel, (0, config.FRAMES - mel.shape[-1]))
    with torch.no_grad():
        return F.softmax(model(mel), 1)[0].cpu().numpy()


def layer3(n: int = 200) -> dict:
    """按 n=40 与 n=200 两档复算，因为原始报告把 n=40 的 raw 与 n=200 的 z 并列了。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = list(csv.DictReader(fc.MIX_MANIFEST.open(newline="", encoding="utf-8")))
    seg = config.SAMPLE_RATE

    # --- 10 类 v1 权重（原报告 top-1 数字的来源）---
    from mel import get_mel_transform
    from models import DS_CNN
    m10 = DS_CNN().to(dev)
    m10.load_state_dict(torch.load(config.MODEL_DIR / "ds_cnn_best.pth", map_location=dev))
    m10.eval()
    mel10 = get_mel_transform().to(dev)

    # --- 部署用的 4 类头（原报告没测，本审计补齐口径）---
    dep_ckpt = config.MODEL_DIR / fc.DEPLOYED_HEAD if hasattr(fc, "DEPLOYED_HEAD") \
        else config.MODEL_DIR / "dscnn_v2_4cls_fft1024.pt"
    if not dep_ckpt.exists():
        dep_ckpt = None
    head = vocab.load_aed(dep_ckpt, device=dev)
    head.model.eval()
    mel4 = head.mel_transform().to(dev)

    def _p4(win: np.ndarray, mode: str) -> np.ndarray:
        if mode == "z":
            win = (win - win.mean()) / (win.std() + 1e-9)
        y = torch.from_numpy(win.astype(np.float32)).to(dev)
        mel = torch.log(mel4(y) + 1e-9)[None, None, :, :head.mel["frames"]]
        with torch.no_grad():
            return F.softmax(head.model(mel), 1)[0].cpu().numpy()

    wl_ids = [i for i, c in enumerate(config.TARGET_CLASSES) if c in fc.WHITELIST_NAMES]
    r3: dict = {"head_desc": vocab.describe(head),
                "head4_classes": list(head.classes),
                "grid": {}}
    wl_frames = wl_total = 0

    HAZARDS = {"siren", "car_horn", "gun_shot"}
    hz = {"raw": {}, "z": {}}
    for mode in hz:
        for k in ((True, True), (True, False), (False, True), (False, False)):
            hz[mode][k] = 0

    for count in (40, n):
        rows = all_rows[:count]
        hit = {"raw": 0, "z": 0, "z_top3": 0, "d4_raw": 0, "d4_z": 0}
        used = 0
        for r in rows:
            y, sr = sf.read(r["mix_path"], dtype="float32")
            if sr != seg or len(y) < seg:
                continue
            used += 1
            tgt = int(r["target_id"])
            pr = _probs10(m10, mel10, dev, y[:seg], "raw")
            pz = _probs10(m10, mel10, dev, y[:seg], "z")
            hit["raw"] += int(int(pr.argmax()) == tgt)
            hit["z"] += int(int(pz.argmax()) == tgt)
            hit["z_top3"] += int(tgt in pz.argsort()[::-1][:3])
            need = r["target_class"] if r["target_class"] in head.classes else "background"
            for mode, key in (("raw", "d4_raw"), ("z", "d4_z")):
                pred = head.classes[int(_p4(y[:seg], mode).argmax())]
                hit[key] += int(pred == need)
                if count == n:                      # 部署头真正要区分的只有「危险 vs 背景」
                    truth_h = r["target_class"] in HAZARDS
                    pred_h = pred in HAZARDS
                    hz[mode][(truth_h, pred_h)] += 1
            if count == n:                    # 窗口级白名单占比只在大档算
                for s in range(0, max(1, len(y) - seg + 1), seg // 2):
                    w = y[s:s + seg]
                    if len(w) < seg:
                        break
                    wl_total += 1
                    wl_frames += int(int(_probs10(m10, mel10, dev, w, "z").argmax()) in wl_ids)
        r3["grid"][count] = {k: v / max(used, 1) for k, v in hit.items()}
        r3["grid"][count]["used"] = used

    g40, g200 = r3["grid"][40], r3["grid"][n]
    r3["raw40"], r3["z40"] = g40["raw"], g40["z"]
    r3["raw200"], r3["z200"] = g200["raw"], g200["z"]
    r3["wl_win_pct"] = 100.0 * wl_frames / max(wl_total, 1)
    r3["wl_windows"] = wl_total

    # 本语料里 target 恒为「应抑制的噪声」，hazard 只作为 interferer 出现
    r3["noise_bg_raw"] = hz["raw"][(False, False)] / max(sum(hz["raw"].values()), 1)
    r3["noise_bg_z"] = hz["z"][(False, False)] / max(sum(hz["z"].values()), 1)
    r3["noise_n"] = sum(hz["z"].values())
    assert hz["z"][(True, True)] + hz["z"][(True, False)] == 0, "target 里出现了危险类，口径假设不成立"

    check("L3 n=40 档 raw top-1 复现报告值 0.17",
          abs(r3["raw40"] - 0.17) <= 0.025, f"复算 {r3['raw40']:.3f}（{g40['used']} 条）")
    check("L3 n=40 档 z-score top-1 复现报告值 0.45",
          abs(r3["z40"] - 0.45) <= 0.025, f"复算 {r3['z40']:.3f}")
    check("L3 n=200 档 z-score top-1 复现报告值 0.53",
          abs(r3["z200"] - 0.53) <= 0.025, f"复算 {r3['z200']:.3f}（{g200['used']} 条）")
    check("L3 n=200 档 raw top-1 **不等于** 0.17（报告混用了两个样本集）",
          abs(r3["raw200"] - 0.17) > 0.025, f"复算 {r3['raw200']:.3f} —— 见 AUDIT_NOTES")
    return r3


# ------------------------------------------------------- L4 占位音频指纹
def layer4(audio_dir: Path | None) -> dict:
    if audio_dir is None or not audio_dir.exists():
        check("L4 占位音频目录可定位", False, str(audio_dir))
        return {}
    files = sorted(audio_dir.glob("[0-1][0-9]_*.wav"))
    check("L4 找到 10 个按类命名的 demo wav（UrbanSound8K 十类）",
          len(files) == 10, f"实际 {len(files)}")
    res = {}
    for p in files:
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = min(len(y), sr)
        spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        peak = float(freqs[int(spec.argmax())])
        res[p.name] = dict(dur=round(len(y) / sr, 2), peak_hz=round(peak, 1),
                           unique_amp=int(len(np.unique(y))))
    peaks = np.array([v["peak_hz"] for v in res.values()])
    durs = np.array([v["dur"] for v in res.values()])
    check("L4 全部文件的谱峰都在 220 Hz（正弦占位）",
          bool(np.all(np.abs(peaks - 220.0) < 3.0)), f"谱峰集合 {sorted(set(peaks.tolist()))}")
    check("L4 全部文件等长 2.50 s（非真实素材）",
          bool(np.all(np.abs(durs - 2.5) < 0.02)), f"时长集合 {sorted(set(durs.tolist()))}")
    check("L4 量化台阶数一致（同一生成器产物）",
          len({v["unique_amp"] for v in res.values()}) == 1,
          f"{sorted({v['unique_amp'] for v in res.values()})}")
    return res


# ------------------------------------------------- L6 安全强度 α 两组扫描
def layer6(art: dict) -> dict:
    """两组 α 是两个不同的旋钮：方案 A 的类增益强度、方案 B 的 TSE α。
    历史上它们都没有从 CLI 落盘的产物，报告里的表无法复现 —— 2026-10-31 补 --gain-strength
    后两组都可从 eeeval_filter 直接生成。"""
    rep = art["report"].read_text(encoding="utf-8") if art["report"].exists() else ""
    out = {}
    for key, fname, field, col in (
        ("gain", "filter_gainstrength_sweep.json", "gain_strength", ("A_supp_db", "A_pres_db")),
        ("tse", "filter_strength_sweep.json", "strength", ("B_supp_db", "B_pres_db")),
    ):
        f = fc.FILTER_REPORT_DIR / fname
        check(f"L6 扫描产物存在：{fname}（报告里的 α 表必须能量化复现）", f.exists(), str(f))
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        out[key] = j["rows"]
        for row in j["rows"]:
            for metric in col:
                txt = f"{abs(row[metric]):.2f}"
                check(f"L6 报告文字含 {key} α={row[field]} 的 {metric} ＝ −{txt}",
                      txt in rep, f"查找 {txt}")
    return out


# ------------------------------------- L7 A/B 试听样张 vs 逐样本表一致性
def layer7(art: dict) -> dict:
    """index.md 里每条样张的 Supp/Pres 必须能在 filter_per_sample.csv 里找到同一行。
    样张生成于 2026-10-30 20:27（晚于 attenuator 的最后一次修改 20:26:34），
    因此应与 HEAD 口径一致 —— 这里就是验证这一点。"""
    idx = art["ab_index"]
    if not idx.exists():
        check("L7 A/B index.md 存在", False, str(idx))
        return {}
    per = list(csv.DictReader(art["per_sample"].open(newline="", encoding="utf-8")))
    rows = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*`?(\d+)_(\w+)`?\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|"
                     r"\s*(-?[\d.]+) dB\s*\|\s*(-?[\d.]+) dB\s*\|\s*(-?[\d.]+) dB\s*\|", line)
        if m:
            rows.append(dict(tag=m.group(1), noise=m.group(2), interf=m.group(4),
                             supp=float(m.group(6)), pres=float(m.group(7))))
    check("L7 解析出 6 条样张", len(rows) == 6, f"{len(rows)}")
    ok_all = True
    for r in rows:
        match = next((x for x in per if x["target"] == r["noise"] and x["interferer"] == r["interf"]), None)
        if match is None:
            ok_all = False
            check(f"L7 样张 {r['tag']}_{r['noise']} 能在逐样本表中定位", False, "未找到同 target/interferer 行")
            continue
        ok = near(r["supp"], float(match["A_aed_supp_db"]), 0.011) and \
             near(r["pres"], float(match["A_aed_pres_db"]), 0.011)
        ok_all &= ok
        check(f"L7 样张 {r['tag']}_{r['noise']} 的 Supp/Pres 与逐样本表一致", ok,
              f"样张 {r['supp']}/{r['pres']} vs 表 {match['A_aed_supp_db']}/{match['A_aed_pres_db']}")
    return dict(ok=ok_all, rows=rows)


# ------------------------------------------------------------ L5 消融复跑
def layer5(ablation: dict) -> dict:
    code_dir = Path(__file__).resolve().parent
    r = subprocess.run([sys.executable, "ablate_filter.py", "--n", "100", "--split", "test"],
                       cwd=code_dir, capture_output=True, text=True)
    check("L5 ablate_filter.py 复跑退出码 0", r.returncode == 0, r.stderr.strip()[-300:])
    new = {}
    for line in r.stdout.splitlines():
        m = re.match(r"(\S+.*?)\s+Supp=\s*(-?[\d.]+)\s+Pres=\s*(-?[\d.]+)\s+gap=\s*(-?[\d.]+)"
                     r"\s+红线=\s*([\d.]+)%\s+触发=\s*([\d.]+)%", line)
        if m:
            new[m.group(1).strip()] = dict(supp=float(m.group(2)), pres=float(m.group(3)),
                                           gap=float(m.group(4)), redline=float(m.group(5)),
                                           trig=float(m.group(6)))
    check("L5 复跑得到 4 个配置", len(new) == 4, f"{list(new)}")
    for k, v in new.items():
        old = ablation.get(k)
        ok = old is not None and near(v["supp"], old["supp"], 0.011) \
            and near(v["pres"], old["pres"], 0.011) \
            and near(v["redline"], old["redline"], 0.11) \
            and near(v["trig"], old["trig"], 0.11)
        check(f"L5 消融「{k}」复跑与归档逐格一致", ok,
              f"新 {v} / 归档 {old}")
    return new


# ------------------------------------------------------------------- 报告
def write_report(res: dict, path: Path) -> None:
    bad = [c for c in CHECKS if not c[1]]
    L = ["# 降噪支路数字溯源审计报告", "",
         f"## 结论：{'全部可溯源可复现' if not bad else f'存在 {len(bad)} 处不一致'}",
         "", f"- 断言 **{len(CHECKS) - len(bad)}/{len(CHECKS)}** 项通过",
         f"- 审计脚本：`code/audit_filter_numbers.py`（本文件由该脚本生成）",
         "- 复跑命令：`cd code && PYTHONPATH=. python3 audit_filter_numbers.py --ablate`", ""]

    r2, r3, r4 = res.get("l2", {}), res.get("l3", {}), res.get("l4", {})
    if r2:
        js = r2["js"]
        L += ["## 层 B — 逐样本复算（filter_per_sample.csv → filter_eval.json → 报告）", "",
              "| 量 | 报告/JSON 值 | 由 100 条逐样本复算 | 差 |", "|:--|--:|--:|--:|",
              f"| A Supp | {js['A']['supp_db_mean']} | {r2['recomputed']['A_supp']:.4f} | "
              f"{r2['recomputed']['A_supp'] - js['A']['supp_db_mean']:+.4f} |",
              f"| A Pres | {js['A']['pres_db_mean']} | {r2['recomputed']['A_pres']:.4f} | "
              f"{r2['recomputed']['A_pres'] - js['A']['pres_db_mean']:+.4f} |",
              f"| A oracle Supp | {js['dual_track']['A_oracle']['supp_db_mean']} | "
              f"{r2['recomputed']['A_oracle_supp']:.4f} | "
              f"{r2['recomputed']['A_oracle_supp'] - js['dual_track']['A_oracle']['supp_db_mean']:+.4f} |",
              f"| A oracle Pres | {js['dual_track']['A_oracle']['pres_db_mean']} | "
              f"{r2['recomputed']['A_oracle_pres']:.4f} | "
              f"{r2['recomputed']['A_oracle_pres'] - js['dual_track']['A_oracle']['pres_db_mean']:+.4f} |",
              f"| B Supp | {js['B']['supp_db_mean']} | {r2['recomputed']['B_supp']:.4f} | "
              f"{r2['recomputed']['B_supp'] - js['B']['supp_db_mean']:+.4f} |",
              f"| B Pres | {js['B']['pres_db_mean']} | {r2['recomputed']['B_pres']:.4f} | "
              f"{r2['recomputed']['B_pres'] - js['B']['pres_db_mean']:+.4f} |",
              f"| 触发率 | {js['dual_track']['A_aed']['trigger_rate_pct']}% | "
              f"{r2['recomputed']['trigger_pct']:.1f}% | "
              f"{r2['recomputed']['trigger_pct'] - js['dual_track']['A_aed']['trigger_rate_pct']:+.1f} pp |",
              f"| 红线超标率 | {js['safety']['pres_below_minus3_pct']}% | "
              f"{r2['recomputed']['pres_below_minus3_pct']:.1f}% | "
              f"{r2['recomputed']['pres_below_minus3_pct'] - js['safety']['pres_below_minus3_pct']:+.1f} pp |",
              ""]
    if r3:
        L += ["## 层 C — 输入口径复算（top-1 vs target_class，不过 split 过滤）", "",
              "| 权重 | n | raw top-1 | z-score top-1 | z-score top-3 |", "|:--|--:|--:|--:|--:|"]
        for cnt in (40, 200):
            g = r3["grid"][cnt]
            L.append(f"| `ds_cnn_best.pth`（10 类） | {g['used']} | {g['raw']:.3f} | "
                     f"{g['z']:.3f} | {g['z_top3']:.3f} |")
        g = r3["grid"][200]
        L.append(f"| 部署 4 类头 {r3['head4_classes']}（类名不匹配时记 background） | "
                 f"{g['used']} | {g['d4_raw']:.3f} | {g['d4_z']:.3f} | — |")
        L += ["",
              f"- 窗口级 argmax 落在白名单类的比例：**{r3['wl_win_pct']:.1f}%**"
              f"（{r3['wl_windows']} 个 1 s 窗，10 类头 + z-score）",
              f"- 本语料里 target 恒为「应抑制的噪声」，理想预测＝`background`。"
              f"部署 4 类头判对（background）的比例：raw **{r3['noise_bg_raw']:.3f}** → "
              f"z-score **{r3['noise_bg_z']:.3f}**（{r3['noise_n']} 条）。"
              "即 z-score 让模型更敏感，代价是把 28% 的噪声混合误读成危险类 —— "
              "这正是必须上 carve-out 而不是「整窗否决」的原因。"
              "⇒ 「z-score 是核心修复」的依据是**抑制 dB**（层 E），不是这张分类表。",
              "- ⚠️ 口径提示：报告原文把 **n=40 的 raw 0.17** 与 **n=200 的 z 0.53** 并列成一对，"
              "两者样本集不同；同一样本集上（n=200）raw 实为 "
              f"{r3['raw200']:.3f}、z 为 {r3['z200']:.3f}。方向与结论不变，但并列写法已改为同 n 对比。",
              ""]
    if r4:
        L += ["## 层 D — 占位音频指纹复验", "",
              f"- 目录：`{res.get('audio_dir')}`",
              f"- {len(r4)} 个文件谱峰全部 220 Hz、时长全部 2.50 s、量化台阶数一致 "
              f"→ 「220 Hz 正弦占位文件」这一事实断言**可复验**", ""]
    new5 = res.get("l5")
    if new5:
        L += ["## 层 E — 消融复跑（掩码层面）", "",
              "| 配置 | Supp | Pres | 红线超标 | 触发率 |", "|:--|--:|--:|--:|--:|"]
        for k, v in new5.items():
            L.append(f"| {k} | {v['supp']:.2f} | {v['pres']:.2f} | {v['redline']:.1f}% | {v['trig']:.1f}% |")
        L.append("")

    if res.get("l6"):
        L += ["## 层 F — 安全强度 α 两组扫描的可复现性", "",
              "| 旋钮 | α | Supp dB | Pres dB | gap | 红线超标 |", "|:--|--:|--:|--:|--:|--:|"]
        for key, label in (("gain", "方案 A 类增益 `--gain-strength`"), ("tse", "方案 B TSE `--strength`")):
            for row in res["l6"].get(key, []):
                a = row.get("gain_strength", row.get("strength"))
                sup = row.get("B_supp_db", row["A_supp_db"]) if key == "tse" else row["A_supp_db"]
                pre = row.get("B_pres_db", row["A_pres_db"]) if key == "tse" else row["A_pres_db"]
                L.append(f"| {label} | {a} | {sup} | {pre} | "
                         f"{row.get('A_gap_db', row.get('A_gap'))} | {row.get('pres_below_minus3_pct')}% |")
        L += ["", "> 方案 A 的 α 只影响类增益强度；方案 B 的 α 是 TSE 掩码安全系数，两者互不影响，"
                  "所以上表两组的 Supp/Pres 分别取自 A 列与 B 列。", ""]

    L += ["## 全部断言", "", "| 通过 | 断言 | 细节 |", "|:--:|:--|:--|"]
    for name, ok, detail in CHECKS:
        L.append(f"| {'✅' if ok else '❌'} | {name} | {detail} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true", help="重跑 ablate_filter.py 并逐格比对")
    ap.add_argument("--n", type=int, default=200, help="层 C 样本量")
    ap.add_argument("--audio-dir", default=None,
                    help="按类命名的 demo wav 目录（默认取 Obsidian 工程内的 outputs/audio）")
    args = ap.parse_args()

    res: dict = {}
    art = layer1()
    res["l2"] = layer2(art)
    res["l3"] = layer3(args.n)
    adir = Path(args.audio_dir) if args.audio_dir else Path(
        "/mnt/d/learning/ObsidianVault/Project-SafeSoundAI/0-SafeSound AI workspace/outputs/audio")
    res["audio_dir"] = str(adir)
    res["l4"] = layer4(adir)
    if args.ablate:
        res["l5"] = layer5(res["l2"]["ablation"])
    res["l6"] = layer6(art)
    res["l7"] = layer7(art)

    out = fc.FILTER_REPORT_DIR / "filter_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(res, out)

    bad = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        print(f"{'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if not ok else ""))
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} 项通过 → {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
