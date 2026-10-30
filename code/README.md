# code/ — module map

Python 3.10+, flat imports (`import config`). Run scripts from inside `code/`, or set `PYTHONPATH=code`.

Paths resolve from `config.py`; override the preprocessed-data root with `SAFESOUND_DATA_DIR`.

## Configuration and data

| File | Role |
|:--|:--|
| `config.py` | paths, sample rate, segment length, Mel front end, class definitions, safety roles |
| `build_metadata.py` | repair stale absolute paths, build `metadata_clean.csv` (run first) |
| `label_mapper.py` | label mapping + recording-level fold assignment (leakage control) |
| `preprocess_v2.py` / `preprocess_verify.py` | segmentation into `.npy` + 11-point verification |
| `audio_utils.py` / `augmentation.py` / `mel.py` | I/O, noise mixing + SpecAugment, log-Mel |
| `dataset.py` | PyTorch dataset and split loader |
| `data_audit.py`, `data_audit_v2.py` | composition, integrity, silence and leakage audits |

## Model

| File | Role |
|:--|:--|
| `models.py` | DS-CNN (depthwise separable, 67 K params at 4 classes) |
| `train_baseline.py` | MFCC + SVM lower bound |
| `train_dscnn.py` / `train_dscnn_v2.py` | training; `--exclude-class fireworks` produces the 4-class model |
| `tse_model.py` / `train_tse.py` | conditional target-speech-enhancement branch |
| `evaluate.py` | accuracy and end-to-end latency |
| `eval_report.py` | confusion matrix, per-class PRF, calibration, threshold sweep, size |
| `eval_emergency.py` | hazard-class focus: segment/event level, latency, false alarms, SNR robustness |
| `eval_decision_policy.py` | temperature scaling and threshold trade-off |
| `tune_threshold_v2.py` | threshold policy for the 4-class model |
| `compare_protocols.py` | file-level vs recording-level split, side by side |

## Haptic delivery

| File | Role |
|:--|:--|
| `haptic_encoder.py` | **core novelty** — 2-axis vocabulary (category × urgency), urgency inference, decision engine |
| `haptic_render.py` | code → 175 Hz audible rendering, amplitude preserved (no peak normalisation) |
| `dual_path_pipeline.py` | parallel branches: AED on raw PCM, denoiser on a copy |
| `attenuator.py` | spectral-gate attenuation with a safety gate |
| `filter_pipeline.py` | conditional TSE + safety gate |
| `haptic_test.py` | 3AFC haptic test harness (category axis and amplitude axis reported separately) |
| `s1_amplitude_pretest.py` | S1 amplitude-axis pre-test with multi-participant pooling |

## Demo

| File | Role |
|:--|:--|
| `demo_v3.py` | **start here** — audio → sliding window → model → arbitration → haptic code |
| `demo_infer.py` | legacy single-file inference (10-class model, deprecated distance encoding) |
| `demo_haptic_v3.py` | renders the 9 codes to audio for listening |
| `demo_haptic_audio_v11_legacy.py` | legacy 11-class rendering, kept for provenance |
| `demo_stream.py` | 24 s synthesised street scene playback with timeline figure |
| `realtime_demo.py` | live microphone inference on a PC |
| `build_dashboard.py` | collects generated figures/reports into one HTML page |
| `export_onnx.py` | ONNX export (deployment path, not yet measured on device) |
| `_bench_winlen.py`, `_bench_winlen2.py` | short-window accuracy/latency benchmark |
| `_diag_*.py` | one-off diagnostics from the TSE debugging sessions, kept for provenance |

## Deprecated, kept for provenance

`demo_haptic_audio_v11_legacy.py` (11 categories), `subjective_eval.py` (10-class era),
`_diag_*.py`. The v1 file-level 10-class model (`ds_cnn_best.pth`) is superseded by the
4-class recording-level model; `demo_infer.py` still points at the old checkpoint and
should not be used to quote numbers.

Encoding history: distance-into-amplitude, five discrete urgency levels, and LRA-layout
direction encoding were all removed on 2026-10-22. Do not reintroduce them; the reasoning
is recorded in `haptic_encoder.py`'s module docstring.
