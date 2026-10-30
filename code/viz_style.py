"""
SafeSoundAI — 绘图统一样式（viz_style.py）

所有审计/评估/demo 脚本共用：CJK 字体、配色、保存函数。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
OUTPUT_DIR = config.OUTPUT_DIR
FIG_DIR = config.OUTPUT_DIR / "figures"
REPORT_DIR = config.OUTPUT_DIR / "reports"
AUDIO_DIR = config.OUTPUT_DIR / "audio"

for _d in (FIG_DIR, REPORT_DIR, AUDIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 字体（WSL 下只有 Noto CJK，统一用 JP 变体，它能覆盖简体中文）
# ---------------------------------------------------------------------------
CJK_FONTS = ["Noto Sans CJK JP", "Noto Serif CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": CJK_FONTS,
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.frameon": False,
})

# ---------------------------------------------------------------------------
# 配色
# ---------------------------------------------------------------------------
INK = "#1b2430"
MUTED = "#6b7a8d"
ACCENT = "#c0392b"      # 危险/警报
ACCENT2 = "#e67e22"     # 高危
WARN = "#f1c40f"        # 警示
OK = "#27ae60"          # 背景/正常
BLUE = "#2c6fb1"

ROLE_COLOR = {
    "critical": "#c0392b",
    "danger": "#e67e22",
    "warning": "#f1c40f",
    "background": "#7f8c8d",
}
ROLE_LABEL = {
    "critical": "核心危险",
    "danger": "高危",
    "warning": "警示",
    "background": "背景",
}

DATASET_COLOR = {"urban": BLUE, "esc50": "#8e44ad", "noise": MUTED}


def class_colors() -> list[str]:
    """按安全角色给 10 个目标类上色。"""
    return [ROLE_COLOR.get(config.SAFETY_ROLE[c], MUTED) for c in config.TARGET_CLASSES]


def save(fig, name: str, subdir: Path | None = None) -> Path:
    """保存图片并返回路径。"""
    out = (subdir or FIG_DIR) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
