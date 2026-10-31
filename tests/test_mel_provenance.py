"""mel 参数必须来自检查点，不能来自文件名（P0-2 前置）。

历史缺陷：`infer_mel_profile` 靠**文件名里有没有 fft1024** 判断 mel 档位。
于是 `dscnn_v2_4cls_fft1024.pt` 改名成 `model.pt` 之后，推理按 fft512 变换，
前向照样输出一个 softmax、照样过阈值 —— 全错但不抛异常。短窗模型（0.5 s）
把这个问题放大成硬伤：窗长变了，帧数从 100 变 50，任何靠名字猜的路径都会
用补零把半个窗填成静音。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import config  # noqa: E402
import vocab  # noqa: E402
from models import DS_CNN  # noqa: E402

CKPT = Path(config.MODEL_DIR) / "dscnn_v2_4cls_fft1024.pt"


def _make_ckpt(tmp_path: Path, name: str, mel: dict | None) -> Path:
    """造一个 4 类检查点（结构对齐真实权重，不依赖磁盘上的大数据）。"""
    m = DS_CNN(n_classes=4)
    blob = {"state_dict": m.state_dict(), "classes": list(vocab.AED4_CLASSES),
            "arch": "DS_CNN", "n_classes": 4}
    if mel is not None:
        blob["mel"] = mel
    p = tmp_path / name
    torch.save(blob, p)
    return p


def test_metadata_wins_over_filename(tmp_path):
    """文件名说 fft512（没标记），元数据说 n_fft=1024 → 必须信元数据。"""
    mel = {"n_fft": 1024, "win_length": 640, "hop_length": 160, "frames": 100}
    p = _make_ckpt(tmp_path, "plain_name.pt", mel)
    head = vocab.load_aed(p)
    assert head.mel["n_fft"] == 1024 and head.mel["frames"] == 100
    assert head.mel_profile == "fft1024"


def test_short_window_frames_are_honoured(tmp_path):
    """0.5 s 模型：frames=50 必须原样传到 AEDHead（下游一律读它）。"""
    mel = {"n_fft": 1024, "win_length": 640, "hop_length": 160, "frames": 50}
    p = _make_ckpt(tmp_path, "short05.pt", mel)
    head = vocab.load_aed(p)
    assert head.mel["frames"] == 50, "帧数被档位表覆盖了 —— 短窗模型会被补零成半窗静音"
    assert head.mel_profile == "fft1024f50"


def test_custom_mel_is_not_snapped_to_a_profile(tmp_path):
    """表里没有的组合（frames=37）也要照用，不许被就近吸附。"""
    mel = {"n_fft": 1024, "win_length": 640, "hop_length": 160, "frames": 37}
    p = _make_ckpt(tmp_path, "weird.pt", mel)
    head = vocab.load_aed(p)
    assert head.mel["frames"] == 37 and head.mel_profile == "custom"


def test_missing_metadata_warns_about_filename_inference(tmp_path):
    """没有元数据 → 用文件名兜底，但必须告警（改名会静默改输入分布）。"""
    p = _make_ckpt(tmp_path, "legacy_fft1024.pt", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        head = vocab.load_aed(p)
    assert head.mel["n_fft"] == 1024          # 文件名兜底仍生效
    assert any("没有 mel 元数据" in str(x.message) for x in w), "兜底路径必须告警"


def test_real_deployment_ckpt_still_resolves_fft1024():
    """线上 4 类头是老检查点（无元数据）—— 行为不许变。"""
    if not CKPT.exists():
        pytest.skip("部署检查点不在本机")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        head = vocab.load_aed(CKPT)
    assert head.mel == {"n_fft": 1024, "win_length": 640,
                        "hop_length": 160, "frames": 100}


def test_trainer_writes_mel_metadata():
    """训练脚本必须把 mel 字典写进检查点，否则命名即契约这件事会回潮。"""
    src = (Path(__file__).resolve().parents[1] / "code" / "train_dscnn_v2.py").read_text(
        encoding="utf-8")
    assert '"mel": mel_meta' in src, "训练脚本没写 mel 元数据"
    assert "frames=self.frames" in src, "数据集没有按 crop_frac 缩帧"
