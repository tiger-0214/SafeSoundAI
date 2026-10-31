"""负向测试：篡改产物后，`audit_detection_numbers.py --check` 必须失败（退出码 1）。

正向的 29/29 只说明"当下一致"；真正证明门禁有效的是**篡改后必须红**。
本测试在临时副本上改 CSV / JSON，确认每一类篡改都能被拦住。

覆盖的篡改类型：
  A. JSON 里的汇总数字被手改（段级 acc 虚高）
  B. CSV 里某条记录的概率被手改（但仍能重算出不同汇总）
  C. CSV 行数被删（完整性对不上）
  D. ckpt md5 被改（模型被换）
  E. 混淆矩阵被手改
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import audit_detection_numbers as adn  # noqa: E402

SRC_CSV = Path(adn.CSV)
SRC_JSON = Path(adn.JSON)

pytestmark = pytest.mark.skipif(
    not (SRC_CSV.exists() and SRC_JSON.exists()),
    reason="需要先跑一次 --eval 生成产物",
)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把产物拷到 tmp，并把模块里的路径常量指过去。"""
    d = tmp_path / "reports"
    d.mkdir()
    shutil.copy(SRC_CSV, d / "detection_per_sample.csv")
    shutil.copy(SRC_JSON, d / "detection_eval.json")
    monkeypatch.setattr(adn, "REPORT_DIR", d)
    monkeypatch.setattr(adn, "CSV", d / "detection_per_sample.csv")
    monkeypatch.setattr(adn, "JSON", d / "detection_eval.json")
    return d


def _run_check() -> int:
    return adn.main(["--check"])


def _bump_json(d: Path, key_path: list[str], new_value) -> None:
    data = json.loads((d / "detection_eval.json").read_text(encoding="utf-8"))
    cur = data
    for k in key_path[:-1]:
        cur = cur[k]
    cur[key_path[-1]] = new_value
    (d / "detection_eval.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_baseline_passes(sandbox):
    """先确认未篡改时是绿的，否则后面的负向测试没有意义。"""
    assert _run_check() == 0


def test_tampered_segment_accuracy_fails(sandbox):
    # A. 把段级 acc 手改成 0.99（论文式"美化"）
    data = json.loads((sandbox / "detection_eval.json").read_text(encoding="utf-8"))
    real = data["segment"]["accuracy"]
    _bump_json(sandbox, ["segment", "accuracy"], 0.99)
    assert _run_check() == 1, f"把 acc 从 {real} 改成 0.99 竟然没被拦住"


def test_tampered_confusion_matrix_fails(sandbox):
    # E. 混淆矩阵挪一格
    data = json.loads((sandbox / "detection_eval.json").read_text(encoding="utf-8"))
    cm = [row[:] for row in data["segment"]["confusion"]]
    cm[0][3] += 1
    cm[0][0] -= 1
    _bump_json(sandbox, ["segment", "confusion"], cm)
    assert _run_check() == 1


def test_tampered_probability_in_csv_fails(sandbox):
    # B. 改一条记录的预测（CSV 变了，汇总却没重算）
    csv_path = sandbox / "detection_per_sample.csv"
    df = pd.read_csv(csv_path)
    # 交换两列概率 → argmax 与 pred 脱钩（等于手改预测结果）
    a, b = "p_siren", "p_background"
    assert {a, b} <= set(df.columns), f"CSV 缺概率列，实际列: {list(df.columns)}"
    df[[a, b]] = df[[b, a]].to_numpy()
    df.to_csv(csv_path, index=False)
    assert _run_check() == 1


def test_dropped_csv_row_fails(sandbox):
    # C. 删 5 行，汇总还是老数字
    csv_path = sandbox / "detection_per_sample.csv"
    df = pd.read_csv(csv_path)
    df.iloc[5:].to_csv(csv_path, index=False)
    assert _run_check() == 1


def test_tampered_ckpt_md5_fails(sandbox):
    # D. 模型被替换（md5 变了）但汇总没重跑
    _bump_json(sandbox, ["ckpt_md5"], "0" * 32)
    assert _run_check() == 1


def test_missing_probability_from_pipeline_is_explicit():
    """DualPathPipeline.classify 必须给出完整概率向量，否则录音级聚合无法审计。"""
    from dual_path_pipeline import DualPathPipeline  # noqa: F401
    # 只做接口契约检查，不加载模型
    import inspect
    src = inspect.getsource(DualPathPipeline.classify)
    assert "softmax" in src, "classify() 不再返回概率分布，录音级聚合会失效"
