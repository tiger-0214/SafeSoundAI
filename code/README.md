# `code/` — module map

Python 3.10+. Flat imports (`import config`): run scripts from inside `code/`, or set `PYTHONPATH=code`.

Every path resolves from `config.py`; override the preprocessed-data root with `SAFESOUND_DATA_DIR`.

## Configuration and data

| File | Role |
|:--|:--|
| `config.py` | paths, sample rate, segment length, Mel front end, class vocabulary, safety roles |
| `preprocess.py` | corpus → 1 s segments: class mapping, background sampling, dedup, recording-level splits |
| `data_audit.py` | composition, integrity, silence and leakage audits |
| `preprocess_verify.py` | 11-point verification of the segment pipeline |
| `audio_utils.py` | resampling, peak/RMS normalisation, z-score |
| `label_mapper.py` | `(dataset, label)` → target class; recording-level stratified splits |
| `augmentation.py` | gain jitter, noise mixing, online overlay |

## Model and training

| File | Role |
|:--|:--|
| `mel.py` | waveform → log-Mel (128 × 100) |
| `dataset.py` | segment dataset, on-the-fly Mel, fold-aware sampling |
| `models.py` | DS-CNN (67 K parameters) |
| `train_baseline.py` | MFCC + linear SVM lower bound |
| `train_dscnn.py` | DS-CNN training; `--exclude-class` builds the 4-class model |
| `evaluate.py` | accuracy, F1, confusion matrix, latency |
| `eval_report.py` | per-class PRF, calibration, latency tables and figures |
| `eval_emergency.py` | event-level recall and false-alarm rate across a threshold sweep |
| `tune_threshold.py` | operating-point selection on the validation fold |
| `export_onnx.py` | ONNX export |

## Haptic encoding

| File | Role |
|:--|:--|
| `haptic_encoder.py` | 2-axis vocabulary: category → temporal pattern, urgency → amplitude; arbitration between competing detections |
| `haptic_render.py` | pattern → audible 175 Hz LRA rendering |
| `haptic_test.py` | forced-choice harness, separate confusion matrices per axis (`--no-audio` for design checks) |
| `s1_amplitude_pretest.py` | amplitude-axis discriminability pre-test (`--dry-run`) |

## Attenuation branch

| File | Role |
|:--|:--|
| `filter_config.py` | per-class gain tables **declared by class name**, user profiles, hard safety constraints |
| `attenuator.py` | spectral-gate attenuation (per-window z-score + whitelist spectral carve-out) |
| `tse_model.py` / `train_tse.py` | conditional target-speech-enhancement denoiser |
| `filter_pipeline.py` | streaming attenuation with fail-open on low confidence |
| `synth_mix.py` | signal-level mix builder for reproducible tests |
| `eval_filter.py` | measured attenuation vs whitelist preservation |
| `ablate_filter.py` | 2x2 ablation (z-score / carve-out) on the real mixture set → `filter_ablation.md` |
| `gen_ab_demo.py` | audible before/after pairs with separate target / interferer stems |

## Vocabulary

| File | Role |
|:--|:--|
| `vocab.py` | **single source of truth** for class spaces: the 10-class source space (filter branch) and the AED head space (4-class product model). Loads a checkpoint together with its own class list and mel profile; nothing may index a class list it did not produce. |

## Demos

| File | Role |
|:--|:--|
| `demo.py` | **start here** — audio → sliding window → model → arbitration → haptic code |
| `realtime_demo.py` | PC real-time loop, mic in, haptic pattern out |
| `dual_path_pipeline.py` | both branches over one stream |
| `viz_style.py` | shared plot style for report figures |

## Multi-label research code

`experiments/aed/` holds the training and evaluation code for the multi-label variant (overlapping events, rejection head, learned aggregation). It reads a cached waveform pool instead of the recording-level dataset, so it lives outside the flat import namespace. `run.py` is the entry point, `core.py` the shared machinery, `agg_context.py` the aggregation study.
