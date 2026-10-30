# SafeSoundAI

**Real-time detection of safety-critical sounds, delivered as haptic patterns.**

A Deaf or hard-of-hearing person misses the siren, the horn, the reversing alarm. Most assistive apps respond by amplifying everything, which raises the noise floor rather than delivering the one thing that mattered. SafeSoundAI detects the three hazard classes that carry most of the risk in street traffic and encodes them into a **2-axis haptic vocabulary**, with an optional attenuation branch for users who would rather have the environment quietened than the alert louder.

The repository contains the detection model, the haptic encoder, the parallel two-branch pipeline, the evaluation protocol, and a demo that runs the whole loop from a single command. **No audio data, no model weights, and no build artefacts are committed** — see [Data](#data-not-included).

---

## Status

| Component | State | Notes |
|:--|:--|:--|
| Data pipeline | ✅ done | 72,201 source segments → 23,605-segment training pool; leakage fixed (file-level → recording-level split) |
| AED model (v3) | ✅ done | DS-CNN, 67 K params, 4 classes (3 hazards + background) |
| Haptic encoder (v3) | ✅ done | 2 axes × 3 values = 9 codes; amplitude span ≈ 8 dB |
| Parallel two-branch pipeline | ✅ done | detector reads the raw signal, denoiser reads a copy — 6/6 real clips correct |
| Real-time demo (PC) | ✅ runs | `code/demo_v3.py`, sliding 1 s window |
| Attenuation branch | 🟡 partial | spectral gate + TSE implemented, not validated on device |
| Direction of arrival | ⛔ 0% | not implemented, so position is not encoded |
| On-device deployment | ⛔ 0% | ONNX export path exists, no phone measurement |
| Human validation | ⛔ 0% | test harness written, no participants, no ethics submission |

---

## Headline numbers

All numbers are on fold 9, three seeds (42 / 43 / 44), recording-level split with no leakage.

| Metric | Value | Config |
|:--|:--|:--|
| Segment-level accuracy | **91.19%** | `ml_v3`, 1 s window, τ = 0.6 |
| Clip-level accuracy | **93.11%** | `ml_v3` + learned BiGRU aggregation + K=5 smoothing |
| Macro F1 (clip level) | **0.9259** | same |
| mAP | **0.9113** | `ml_v3` |
| Rejection: known alarm vs unseen benign | **AUROC 0.9762** | same |
| Rejection: unseen hazard vs unseen benign | **AUROC 0.5849** | ≈ chance — see limitations |
| Multi-label `both_recall` | **0.2417** | overlapping-event probe, 600 pairs |
| Model size | 67 K params · 295 KB | float32 |

Two numbers deserve context rather than a footnote. **0.5849** means the rejection channel only works for hazard classes the model has seen; a novel hazard is not reliably flagged as one. **0.2417** means that when two hazards overlap, roughly one quarter of the overlapping cases are caught by both tags — the multi-label head is open but weak. Both are reported here rather than hidden because they define what the system is currently allowed to claim.

---

## Repository layout

```
code/                     detection, haptic encoding, pipeline, evaluation
  demo_v3.py              ★ one-command demo: audio → model → haptic code
  haptic_encoder.py       ★ 2-axis haptic vocabulary (the core novelty)
  haptic_render.py        code → 175 Hz audible rendering for listening tests
  dual_path_pipeline.py   parallel branch: AED on raw PCM, denoiser on a copy
  filter_pipeline.py      conditional TSE + safety gate
  attenuator.py           spectral-gate attenuation
  haptic_test.py          3AFC haptic test harness
  s1_amplitude_pretest.py amplitude-axis pre-test (S1)
  models.py               DS-CNN
  train_dscnn_v2.py       training (supports --exclude-class)
  eval_report.py          confusion matrix, per-class PRF, calibration, latency
  _bench_winlen.py        short-window accuracy/latency benchmark
experiments/v3/           multi-label v3 experiments (needs the data pool)
docs/                     filter-branch design specs (spec → constraints → arch → eval → safety)
reports/                  generated reports, kept as evidence
```

## Quick start

```bash
pip install -r code/requirements.txt

# Full loop on a built-in 12 s scene assembled from the test fold
export SAFESOUND_DATA_DIR=/path/to/preprocessed_data
python3 code/demo_v3.py

# Or on your own recording
python3 code/demo_v3.py --wav street.wav
```

Expected output (fold-9 scene, τ = 0.6):

```
 t(s) class     urgency   pattern    interval  amplitude  code
  2.0 siren     alert     ramp          400ms       0.40  siren:alert
  5.5 car_horn  high      dual          700ms       0.70  car_horn:high
  8.5 gun_shot  critical  single       3000ms       1.00  gun_shot:critical
```

`reports/demo_v3.md` holds the full transcript, `reports/` holds the rest of the evidence.

## Haptic encoding (v3)

| Axis | Carries | Values |
|:--|:--|:--|
| Time | category (nominal) | siren / car horn / gunshot — three temporal shapes |
| Amplitude | urgency (ordinal) | 40% / 70% / 100%, ≈ 8 dB span |
| Position | — | not encoded (no direction of arrival yet) |

| Category | Temporal pattern | Repetition |
|:--|:--|:--|
| gunshot | single steep pulse, 60 ms | one-off, once more after 3 s |
| car horn | double pulse, 150 / 80 / 150 ms | every 700 ms |
| siren | rising–falling sweep, 400 ms | every 400 ms |

Three constraints hold the design together:

1. **Category is carried by time shape only.** Amplitude must not be a per-category constant, or the two axes become collinear and discriminability collapses. The semantic prior only sets the default level; every category may span at least two levels.
2. **Urgency = semantic prior (primary) + duration/repetition + relative level quantile (secondary).** Absolute dB SPL is banned: microphone coupling, occlusion and AGC destroy it.
3. **Amplitude levels are spread ≈ 8 dB** so that individual fitting differences survive; a `LevelNormalizer` handles relative calibration.

Position and distance are deliberately not encoded. Earlier versions encoded distance into amplitude and direction into a specific LRA layout; both were removed from the main path.

## Latency

Compute is not the bottleneck: 2–4 ms per 1 s window on CPU. The bottleneck is **filling the buffer** — the model expects a 1 s context, so mic→haptic latency cannot meet the 50 ms target without a short-window streaming front end. `reports/winlen_bench.md` quantifies the accuracy cost of shrinking the window:

| Window | Acc | siren R | car horn R | gunshot R |
|:--:|:--:|:--:|:--:|:--:|
| 1.0 s | .8865 | .733 | .850 | 1.000 |
| 0.5 s | .7192 | .342 | .558 | .850 |
| 0.3 s | .5615 | .125 | .325 | .537 |

The cliff at 0.5 s is a **training-distribution problem, not an information problem**: every training segment is fully occupied by its class, so the model never saw "part event, part background". Gunshot (60 ms transient) survives 0.3 s at .537 while siren (steady state) collapses to .125. The fix is partial-event augmentation and a retrain, not a bigger model.

## Limitations

- Novel hazard classes are not rejected (AUROC 0.5849).
- Overlapping events are only weakly decoded (both_recall 0.2417).
- Fireworks was evaluated as a fourth hazard and dropped: 360 segments total, recall .444, single-fold variance too large to select on.
- No direction of arrival, so no spatial encoding.
- No human participants; the amplitude axis and the 9-code vocabulary are unvalidated with users.
- Operational false-alarm rate was measured on the v2 model (49.1/h at τ = 0.8) and has not been re-measured for v3.

## Data (not included)

Training used UrbanSound8K and ESC-50, preprocessed into three hazard classes plus a background pool. Per the licences of both corpora, no audio is redistributed here, and no checkpoints are committed. `code/preprocess_v2.py`, `code/build_metadata.py` and `code/data_audit.py` rebuild the pipeline from the original corpora; `reports/data_audit_v2.md` documents the expected composition and the leakage checks.

Point `SAFESOUND_DATA_DIR` at the preprocessed directory to run anything that needs data. Scripts resolve paths from `code/config.py`, so no absolute path is hardcoded.

## Contact

Qingming Li · MSc Data Science and Machine Learning, UCL · qingming.li@ucl.ac.uk

## Licence

MIT — see `LICENSE`. Third-party corpora retain their own licences and are not redistributed here.
