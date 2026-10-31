# SafeSoundAI —— 一把梭入口
#
# 设计原则：每个 target 都能在干净环境里独立跑；需要数据集的 target 会先体检，
# 缺东西就明确报错，不跑到一半抛栈。
#
#   make            # 等价 make help
#   make demo       # 离线 v3 demo，**不需要数据集**（最快看到触觉编码长什么样）
#   make demo-e2e   # 流式端到端：造场景 → 检测 → 决策 → 触觉 → 事件级评分
#   make test       # 全部 pytest（80 项）
#   make check      # 门禁：过滤侧功能契约 + 两侧数字溯源（秒级，可直接进 CI）
#   make onnx       # 导出 fp32 / int8 并做精度回归
#   make stream     # 30 min 流式压测模板
#   make clean      # 清临时产物（不动权重与报告）

PY ?= python3
export SAFESOUND_DATA_DIR ?= /root/safesound_data
SCENE30 := /tmp/scene30min.wav

.PHONY: help all demo demo-e2e test check check-filter check-detection functional onnx stream scene30 clean

help:
	@echo "make demo        离线 v3 demo（无数据集也能跑）"
	@echo "make demo-e2e    流式端到端 demo（真音频场景 + 事件级评分）"
	@echo "make test        pytest 全量"
	@echo "make check       数字门禁（过滤 + 检测）"
	@echo "make onnx        ONNX 导出 + int8 精度回归"
	@echo "make stream      30 min 流式压测"
	@echo ""
	@echo "数据集：export SAFESOUND_DATA_DIR=/path/to/preprocessed_data（当前 $(SAFESOUND_DATA_DIR)）"

all: check test

# ---------------------------------------------------------------- 演示
demo:
	$(PY) code/demo.py

demo-e2e:
	$(PY) code/demo_e2e.py

# ---------------------------------------------------------------- 测试
test:
	@$(PY) -m pytest tests/ -q

# ---------------------------------------------------------------- 门禁
check: check-filter check-detection

# 先跑功能契约（信号链是否按设计工作），再跑数字溯源（报告里的数是否可复算）。
# 顺序有意义：功能挂了三分钟就能定位是实现 bug 还是能力边界，不用等审计跑完。
functional:
	@$(PY) code/filter_functional_check.py --n 20 | tail -4

check-filter: functional
	@echo "=== 过滤侧数字门禁 ==="
	@$(PY) code/audit_filter_numbers.py --ablate | tail -3

check-detection:
	@echo "=== 检测侧数字门禁 ==="
	@$(PY) code/audit_detection_numbers.py --check | tail -3

# ---------------------------------------------------------------- 训练（可溯源）
# 4 类部署头（论文里报的那个）。检查点会写入 mel 元数据，别再靠文件名猜档位。
train-4cls:
	$(PY) code/train_dscnn_v2.py --exclude-class fireworks --n-fft 1024 \
		--epochs 25 --tag 4cls_fft1024

# 0.5 s 短窗（P0-2 可行性）：帧数随窗长减半，不补零
train-short05:
	$(PY) code/train_dscnn_v2.py --exclude-class fireworks --n-fft 1024 \
		--crop-frac 0.5 --epochs 25 --tag short05

# 1 s 权重 vs 0.5 s 权重 同集对比
shortwin:
	$(PY) code/eval_short_window.py

# ---------------------------------------------------------------- 部署
onnx:
	$(PY) code/export_onnx.py

# ---------------------------------------------------------------- 压测
stream:
	@test -f $(SCENE30) || $(PY) code/_make_scene.py --duration 1800 --seed 21 --out $(SCENE30)
	$(PY) code/streaming_runtime.py --source file --audio $(SCENE30) \
		--device-sr 48000 --duration 1800 --pacing max --backend wavefile \
		--out-wav /tmp/hap30min.wav --out-log outputs/reports/streaming_30min \
		--ckpt outputs/models/dscnn_v2_4cls_fft1024.pt

scene30:
	$(PY) code/_make_scene.py --duration 1800 --seed 21 --out $(SCENE30)

clean:
	rm -rf /tmp/scene*.wav /tmp/hap*.wav /tmp/demo_haptics.wav /tmp/demo_scene48k*.wav
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
