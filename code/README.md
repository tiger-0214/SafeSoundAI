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
| `mel.py` | waveform → log-Mel (128 × 100); frame count is a parameter, never a padded constant |
| `dataset.py` | segment dataset, on-the-fly Mel, fold-aware sampling |
| `models.py` | DS-CNN (67 K parameters) |
| `train_baseline.py` | MFCC + linear SVM lower bound |
| `train_dscnn.py` | DS-CNN training (v1, 10-class source space); `--exclude-class` builds the 4-class model |
| `train_dscnn_v2.py` | **the script that produced the deployment head** (trigger_v2, recording-level split, temperature calibration). Writes a `mel` dict into the checkpoint so mel params travel with the weights instead of being guessed from the filename; `--crop-frac 0.5` trains the 0.5 s short-window variant, halving the frame count (no zero-padding) |
| `eval_short_window.py` | P0-2 feasibility: same test segments scored with the 1 s head at 1.0 s / 0.5 s windows and the 0.5 s head at 0.5 s, so the latency-vs-accuracy trade-off is one JSON instead of folklore |
| `evaluate.py` | accuracy, F1, confusion matrix, latency |
| `eval_report.py` | per-class PRF, calibration, latency tables and figures |
| `eval_emergency.py` | event-level recall and false-alarm rate across a threshold sweep |
| `tune_threshold.py` | operating-point selection on the validation fold |
| `export_onnx.py` | ONNX export + int8 quantisation with an accuracy gate: exports fp32 and static-QDQ int8, then re-scores **all** test segments on both and refuses to pass if top-1 agreement or macro-F1 drifts beyond tolerance |

## Haptic encoding

| File | Role |
|:--|:--|
| `actuator.py` | **actuator abstraction** — 9 codes = 3 categories × 3 urgencies, amplitudes and unit lengths taken from `haptic_encoder` rather than redefined. Four backends: `null` (logic baseline), `wavefile` (audible LRA rendering), `ble_stub` (8-byte little-endian frame, ≤20 B so it never fragments), `android` (`VibrationEffect.createWaveform`). Emit-time alignment is asserted at ≤5 ms; device-intrinsic delay (`t_air`, `t_rise`) is reported separately so it never contaminates the alignment figure |
| `haptic_scheduler.py` | **de-duplication and pre-emption** — every proposal is adjudicated to exactly one of `accepted / same_code_cooldown / same_class_cooldown / min_gap / lower_priority / silent / queue_full`, so rejection reasons are countable rather than free-text. Sustained sirens are suppressed ~97 % over 30 s |
| `haptic_encoder.py` | 2-axis vocabulary: category → temporal pattern, urgency → amplitude; arbitration between competing detections |
| `haptic_render.py` | pattern → audible 175 Hz LRA rendering |
| `haptic_test.py` | forced-choice harness, separate confusion matrices per axis (`--no-audio` for design checks) |
| `s1_amplitude_pretest.py` | amplitude-axis discriminability pre-test (`--dry-run`) |

## Attenuation branch

| File | Role |
|:--|:--|
| `filter_config.py` | per-class gain tables **declared by class name**, user profiles, hard safety constraints. `background` semantics: it means *"none of the three hazards"*, **not** "quiet" — it absorbs AC hum, drills, barking and children playing, so it is never attenuated (R1), never boosted (R2) and never whitelisted (R3); `background_gain_guard()` enforces this at runtime, `gains_for(..., strict=True)` fails CI on an unknown head |
| `attenuator.py` | spectral-gate attenuation (per-window z-score + whitelist spectral carve-out) |
| `tse_model.py` / `train_tse.py` | conditional target-speech-enhancement denoiser |
| `filter_pipeline.py` | streaming attenuation with fail-open on low confidence |
| `synth_mix.py` | signal-level mix builder for reproducible tests |
| `eval_filter.py` | measured attenuation vs whitelist preservation |
| `ablate_filter.py` | 2x2 ablation (z-score / carve-out) on the real mixture set → `filter_ablation.md` |
| `gen_ab_demo.py` | audible before/after pairs with separate target / interferer stems |
| `audit_detection_numbers.py` | provenance audit for the **detection** side: writes one CSV of per-segment predictions (the only raw layer), derives every summary number from it with a single `summarize()` function, and `--check` runs 29 assertions across completeness / segment / recording / weight-anchor layers. Exit code is CI-usable. Scope is the 4-class deployment head only — the v3 multi-label numbers are audited elsewhere and the two must never be cross-quoted |
| `audit_filter_numbers.py` | provenance audit: recomputes every number in `filter_ab_report.md` from `filter_per_sample.csv` and the sweep JSONs, re-runs the ablation, and exits non-zero on any mismatch (`--ablate` for the slow layer) |

## Vocabulary

| File | Role |
|:--|:--|
| `vocab.py` | **single source of truth** for class spaces: the 10-class source space (filter branch) and the AED head space (4-class product model). Loads a checkpoint together with its own class list and mel params (checkpoint metadata wins over the filename, which is only a fallback and warns when used); nothing may index a class list it did not produce. |

## Demos

| File | Role |
|:--|:--|
| `demo.py` | **start here** — audio → sliding window → model → arbitration → haptic code (no dataset required) |
| `demo_e2e.py` | streaming end-to-end: health check → real-audio scene → detection → decision → actuation → **event-level scoring** (recall, false alarms, onset-to-detection latency). Degrades to a synthetic source and says so if no dataset is present |
| `streaming_runtime.py` | **the runtime** — resampler (passthrough / exact integer-ratio / declared-approximate), ring buffer, window scheduler, three-stage timing (`t_acquire → t_infer → t_emit`), JSONL + histogram artefacts. Reuses `DualPathPipeline`'s classifier and `DecisionEngine` so the streaming path cannot drift from the offline one |
| `_make_scene.py` | builds a real-audio test scene (2 s events, 1 s gaps) plus its ground-truth file; helpers prefixed `_` are not part of the product surface |
| `_diag_stream_mismatch.py` | one-off diagnostic: compares raw segment vs resampled vs peak-normalised on the same checkpoint (used to prove a streaming-vs-offline gap was a scene-design artefact, not a signal-path bug) |
| `realtime_demo.py` | PC real-time loop, mic in, haptic pattern out |
| `dual_path_pipeline.py` | both branches over one stream |
| `viz_style.py` | shared plot style for report figures |

## Tests

`tests/` (pytest, 56 cases) covers the parts where a silent bug is invisible in the output
numbers: actuator code-table consistency with the encoder, backend emit-time alignment,
scheduler invariants under 3 seeds × 10 000 random commands, resampler chunk-vs-whole
equivalence, ring-buffer random-window equivalence, **and the audit gates themselves** —
every tampering of the derived summaries must turn `--check` red.

```bash
make test          # all of it
make check         # the two provenance gates only (seconds, CI-usable)
```

## Multi-label research code

`experiments/aed/` holds the training and evaluation code for the multi-label variant (overlapping events, rejection head, learned aggregation). It reads a cached waveform pool instead of the recording-level dataset, so it lives outside the flat import namespace. `run.py` is the entry point, `core.py` the shared machinery, `agg_context.py` the aggregation study.
