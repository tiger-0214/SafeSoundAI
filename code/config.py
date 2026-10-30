"""
SafeSoundAI — 全局配置

所有模块共享的路径、常量、目标类定义、超参数。
运行任意脚本前，请先确认本文件的路径解析正确（见 README）。

用法：
    from config import *
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径（基于本文件位置动态解析，不依赖硬编码绝对路径）
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent                 # code/
WORKSPACE = CODE_DIR.parent                                # 0-SafeSound AI workspace/
RAW_DATA = WORKSPACE / "1_Original_Database"

# 数据目录可被环境变量覆盖（用于把数据放到 WSL 本地盘加速 IO）：
#   export SAFESOUND_DATA_DIR=/root/safesound_data
PREPROCESSED = Path(os.environ.get("SAFESOUND_DATA_DIR", WORKSPACE / "2_Preprocessed_Data"))

# 原始 metadata（1.1 已产出，但 npy_path 为旧绝对路径，需 build_metadata.py 修复）
METADATA_RAW = PREPROCESSED / "metadata.csv"
# 修复 + 映射 + 划分后的干净 metadata（训练唯一数据源）
METADATA_CLEAN = PREPROCESSED / "metadata_clean.csv"

# 派生目录（运行时自动创建）
MEL_DIR = PREPROCESSED / "3_Mel_Spectrograms"              # 1.4 Mel 缓存
OUTPUT_DIR = WORKSPACE / "outputs"                         # 模型权重 / 日志 / 报告
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# 10 目标类（= UrbanSound8K 的 10 类，class_id 直接复用 0-9）
# ---------------------------------------------------------------------------
TARGET_CLASSES = [
    "air_conditioner",   # 0  🟢 背景（常被误判）
    "car_horn",          # 1  🔴 核心危险
    "children_playing",  # 2  🟡 潜在危险
    "dog_bark",          # 3  🟡 警示
    "drilling",          # 4  🔴 危险工具
    "engine_idling",     # 5  🔴 车辆靠近
    "gun_shot",          # 6  🔴 核心危险
    "jackhammer",        # 7  🔴 施工破坏
    "siren",             # 8  🔴 核心危险
    "street_music",      # 9  🟢 背景
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

# 安全角色（用于触觉编码的紧急度分级）
SAFETY_ROLE = {
    "air_conditioner": "background",   # 🟢 背景
    "car_horn":        "critical",     # 🔴 核心危险
    "children_playing":"warning",      # 🟡 潜在危险
    "dog_bark":        "warning",      # 🟡 警示
    "drilling":        "danger",       # 🔴 危险工具
    "engine_idling":   "danger",       # 🔴 车辆靠近
    "gun_shot":        "critical",     # 🔴 核心危险
    "jackhammer":      "danger",       # 🔴 施工破坏
    "siren":           "critical",     # 🔴 核心危险
    "street_music":    "background",   # 🟢 背景
}

# ---------------------------------------------------------------------------
# 1.3 标签映射：ESC-50 → 10 目标类
#   表中出现的 ESC-50 类别 → 映射到对应目标类；未出现的 → None（进入噪声池，供 1.2 增强用）
# ---------------------------------------------------------------------------
ESC50_TO_TARGET = {
    "car_horn":       "car_horn",
    "siren":          "siren",
    "dog":            "dog_bark",
    "engine":         "engine_idling",
    "fireworks":      "gun_shot",
    "glass_breaking": "jackhammer",
    "helicopter":     "engine_idling",
    "airplane":       "engine_idling",
    "train":          "engine_idling",
    "chainsaw":       "drilling",
    "clock_alarm":    "siren",
}

# ---------------------------------------------------------------------------
# 音频参数（1.1 预处理，与已产出的 72,201 段保持一致）
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
SEGMENT_S = 1.0
OVERLAP = 0.5
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_S)   # 16000

# ---------------------------------------------------------------------------
# Mel 参数（1.4，PANNs 标准）
# ---------------------------------------------------------------------------
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 512
F_MIN = 50.0
F_MAX = 7500.0         # 略低于 16kHz Nyquist(8000)，避免最高频 mel bin 全零
FRAMES = 100            # 1s @ 16kHz / hop 160 ≈ 100 帧

# ---------------------------------------------------------------------------
# 训练参数
# ---------------------------------------------------------------------------
N_FOLDS = 10
TRAIN_FOLDS = [0, 1, 2, 3, 4, 5, 6, 7]          # 8 折训练
VAL_FOLD = 8                                     # 1 折验证
TEST_FOLD = 9                                    # 1 折测试（最终报告用）
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = "cuda"  # 训练脚本运行时会自动 fallback 到 cpu


def ensure_dirs() -> None:
    """创建所有派生目录。"""
    for d in (MEL_DIR, MODEL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("CODE_DIR  =", CODE_DIR)
    print("WORKSPACE =", WORKSPACE)
    print("RAW_DATA  =", RAW_DATA)
    print("PREPROCESSED =", PREPROCESSED)
    print("目标类数量 =", NUM_CLASSES)
    print("目标类 =", TARGET_CLASSES)
