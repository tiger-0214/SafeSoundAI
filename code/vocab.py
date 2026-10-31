"""
SafeSoundAI — 唯一词表层（vocab.py）

【为什么需要这个文件】
项目里同时存在**两个合法的类别空间**，历史上因为混用索引导致类名错位：

  1. 源类空间（10 类 UrbanSound8K，`config.TARGET_CLASSES`）
     使用者：过滤支路（类增益表、TSE 条件向量 `keep_vector`、`eval_filter` 的
     `oracle_class = target_id`、混音 manifest 的 `target_id / interferer_id`）。
     **必须保持 10 类**，改维度会让 manifest 与既有评测协议全部失配。

  2. AED 头空间（当前产品模型 4 类：siren / car_horn / gun_shot / background）
     使用者：触觉/AED 支路（`demo.py`、`haptic_encoder`、实时环回）。
     类名**只能从检查点 blob["classes"] 读取**，禁止用 10 类表反查。

【铁律】
  * 任何 `classes[class_id]` 的索引，必须使用**产出该 class_id 的那个模型的 classes**。
  * 两个空间之间只允许通过本文件的 `bridge_to_source()` 显式转换。
  * mel 参数（n_fft / win_length）不写死在 config 里，按检查点档位取
    （`fft512` = 10 类 legacy；`fft1024` = 4 类产品模型）。

用法：
    from vocab import load_aed, role_of, bridge_to_source
    aed = load_aed(config.MODEL_DIR / "dscnn_v2_4cls_fft1024.pt")
    aed.classes, aed.mel.n_fft, aed.n_classes
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import warnings

import torch

import config
from mel import get_mel_transform
from models import DS_CNN

# --------------------------------------------------------------------------- 1) 空间定义
#: 源类空间（10 类）—— 过滤支路的内部坐标，**不要改**
SOURCE_CLASSES: tuple[str, ...] = tuple(config.TARGET_CLASSES)

#: 4 类产品模型的类别（仅作为无检查点时的兜底；正常路径从 blob["classes"] 取）
AED4_CLASSES: tuple[str, ...] = ("siren", "car_horn", "gun_shot", "background")

#: 安全相关类（出现在任一模型空间时的保险名单）
HAZARD_CLASSES: frozenset[str] = frozenset({"siren", "car_horn", "gun_shot"})
WARNING_CLASSES: frozenset[str] = frozenset({"dog_bark", "children_playing"})
NOISE_CLASSES: frozenset[str] = frozenset(
    {"air_conditioner", "drilling", "engine_idling", "jackhammer",
     "street_music", "background"}
)

# --------------------------------------------------------------------------- 2) mel 参数档位
#: 文件名/检查点里出现的档位标记 → mel 参数（hop 固定 160，帧数固定 100）
MEL_PROFILES: dict[str, dict[str, int]] = {
    "fft512":  {"n_fft": 512,  "win_length": 512, "hop_length": 160, "frames": 100},
    "fft1024": {"n_fft": 1024, "win_length": 640, "hop_length": 160, "frames": 100},
    # 0.5 s 短窗（P0-2）：帧数随窗长减半，n_fft/win 与 fft1024 一致
    "fft1024f50": {"n_fft": 1024, "win_length": 640, "hop_length": 160, "frames": 50},
}


def _mel_from_blob(blob: dict | None) -> dict[str, int] | None:
    """从检查点里取出显式 mel 参数（有就信它，没有就 None）。"""
    if not isinstance(blob, dict):
        return None
    raw = blob.get("mel")
    if not isinstance(raw, dict) or "n_fft" not in raw:
        return None
    keys = ("n_fft", "win_length", "hop_length", "frames")
    return {k: int(raw[k]) for k in keys if k in raw}


def infer_mel_profile(ckpt: str | Path, blob: dict | None = None) -> str:
    """从检查点推断 mel 档位：优先 blob 显式字段，其次文件名标记，兜底 fft512。

    .. warning::
       文件名兜底是**静默危险**的：把 ``xxx_fft1024.pt`` 改名成 ``xxx.pt``，
       推理就会按 fft512 变换，输出照样有个 softmax，只是全错。所以
       ``load_aed`` 会在此路径上告警，训练脚本必须把 mel 字典写进检查点。
    """
    if blob:
        raw = _mel_from_blob(blob)
        if raw:
            for name, prof in MEL_PROFILES.items():
                if all(prof.get(k) == v for k, v in raw.items()):
                    return name
            return "custom"
        for key in ("mel_profile", "n_fft"):
            v = blob.get(key)
            if isinstance(v, str) and v in MEL_PROFILES:
                return v
            if isinstance(v, int):
                return "fft1024" if v >= 1024 else "fft512"
    name = Path(ckpt).name.lower()
    if "fft1024" in name or "nfft1024" in name:
        return "fft1024"
    return "fft512"


def resolve_mel(ckpt: str | Path, blob: dict | None = None) -> tuple[str, dict[str, int]]:
    """返回 (档位名, 生效的 mel 参数)。

    显式元数据 > MEL_PROFILES 档位 > config 默认。``custom`` 档位表示检查点
    自带一套表里没有的参数，此时完全用检查点里的值。
    """
    profile = infer_mel_profile(ckpt, blob)
    if profile == "custom":
        base = dict(MEL_PROFILES["fft1024"])
    else:
        base = dict(MEL_PROFILES[profile])
    base.setdefault("hop_length", 160)
    base.setdefault("frames", 100)
    raw = _mel_from_blob(blob) or {}
    base.update(raw)                       # 检查点里的值覆盖档位默认
    return profile, base


# --------------------------------------------------------------------------- 3) AED 加载
@dataclass
class AEDHead:
    """一个已加载的 AED 头：模型 + 它自己的类别表 + 它自己的 mel 参数。"""
    model: torch.nn.Module
    classes: list[str]
    mel_profile: str
    mel: dict[str, int]
    ckpt: Path
    macro_f1: float | None = None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def mel_transform(self):
        return get_mel_transform(n_fft=self.mel["n_fft"],
                                 win_length=self.mel["win_length"],
                                 hop_length=self.mel["hop_length"])

    def idx(self, name: str) -> int | None:
        return self.classes.index(name) if name in self.classes else None

    def role(self, class_id: int) -> str:
        return role_of(self.classes[int(class_id)])


def load_aed(ckpt: str | Path | None = None, device: str | torch.device = "cpu") -> AEDHead:
    """加载 AED 头。兼容两种检查点格式：

    * dict 带 ``classes`` 键（4 类产品模型）
    * 纯 ``state_dict``（10 类 legacy，类名回退到 ``config.TARGET_CLASSES``）
    """
    ckpt = Path(ckpt or (config.MODEL_DIR / "ds_cnn_best.pth"))
    blob = torch.load(ckpt, map_location=device, weights_only=False)

    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
        classes = list(blob.get("classes") or (AED4_CLASSES if len(sd) else []))
        f1 = blob.get("best_val_macro_f1")
    else:                                     # 纯 state_dict
        sd = blob
        classes = list(SOURCE_CLASSES)        # 10 类 legacy 头
        f1 = None
    if not classes:
        raise ValueError(f"{ckpt} 无法确定类别表")

    blob_d = blob if isinstance(blob, dict) else None
    profile, mel_params = resolve_mel(ckpt, blob_d)
    if blob_d is None or _mel_from_blob(blob_d) is None:
        warnings.warn(
            f"检查点 {Path(ckpt).name} 没有 mel 元数据，按文件名推断档位 {profile} "
            f"(n_fft={mel_params['n_fft']}, frames={mel_params['frames']})。"
            "改名或换窗长都会静默改变输入分布 —— 训练脚本应写入 'mel' 字典。",
            RuntimeWarning, stacklevel=2)
    model = DS_CNN(n_classes=len(classes))
    model.load_state_dict(sd)
    model.to(device).eval()
    return AEDHead(model=model, classes=classes, mel_profile=profile,
                   mel=mel_params, ckpt=ckpt,
                   macro_f1=float(f1) if f1 is not None else None)


# --------------------------------------------------------------------------- 4) 角色与桥接
def role_of(name: str) -> str:
    """类名 → 角色（hazard / warning / noise）。未知类按 noise 处理（安全侧：可衰减）。"""
    if name in HAZARD_CLASSES:
        return "hazard"
    if name in WARNING_CLASSES:
        return "warning"
    return "noise"


def bridge_to_source(class_id: int, classes: list[str]) -> int | None:
    """AED 头空间的 class_id → 源类空间（10 类）的 class_id。

    background 等源空间没有的类返回 ``None``（调用方必须显式处理，不允许瞎猜索引）。
    """
    name = classes[int(class_id)]
    return config.CLASS_TO_IDX.get(name)


def source_to_bridge(source_id: int, classes: list[str]) -> int | None:
    """源类空间 → AED 头空间；头里没有该类（如 background 之外的噪声）返回 None。"""
    name = SOURCE_CLASSES[int(source_id)]
    return classes.index(name) if name in classes else None


def is_legacy_10class(head: AEDHead) -> bool:
    return head.n_classes == len(SOURCE_CLASSES) and list(head.classes) == list(SOURCE_CLASSES)


def describe(head: AEDHead) -> str:
    f1 = f"{head.macro_f1:.3f}" if head.macro_f1 is not None else "n/a"
    return (f"AED({head.n_classes}cls, {head.mel_profile}, macroF1={f1}) "
            f"{head.classes}")


if __name__ == "__main__":
    for ck in ("dscnn_v2_4cls_fft1024.pt", "ds_cnn_best.pth"):
        p = config.MODEL_DIR / ck
        if p.exists():
            print(describe(load_aed(p)))
        else:
            print(f"{p} 不存在，跳过")
