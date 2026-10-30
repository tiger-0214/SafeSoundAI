# SafeSoundAI

**Real-time detection of safety-critical sounds, delivered as haptic patterns.**

A Deaf or hard-of-hearing person misses the siren, the horn, the reversing alarm. Most assistive apps respond by amplifying everything, which raises the noise floor rather than delivering the one thing that mattered. SafeSoundAI detects the three hazard classes that carry most of the risk in street traffic and encodes them into a **2-axis haptic vocabulary**, with an optional attenuation branch for users who would rather have the environment quietened than the alert louder.

This repository is the **code**: the detection model, the haptic encoder, the parallel two-branch pipeline, and the evaluation and measurement scripts. No audio, no model weights and no build artefacts are committed.

---

## Quick start

```bash
pip install -r code/requirements.txt

# data root for anything that reads segments (not needed by the demo)
export SAFESOUND_DATA_DIR=/path/to/preprocessed_data

python3 code/demo.py --ckpt /path/to/dscnn_4cls.pt         # built-in scene
python3 code/demo.py --ckpt /path/to/dscnn_4cls.pt --wav street.wav
```

`code/demo.py` prints one row per detected event — time, class, urgency, haptic pattern, interval, amplitude, code:

```
 t(s) class     urgency   pattern    interval  amplitude  code
  2.0 siren     alert     ramp          400ms       0.40  siren:alert
  5.5 car_horn  high      dual          700ms       0.70  car_horn:high
  8.5 gun_shot  critical  single       3000ms       1.00  gun_shot:critical
```

Everything else is a single script with a `--help`; `code/README.md` maps each file to its role.

---

## Haptic encoding

| Axis | Carries | Values |
|:--|:--|:--|
| Time | category (nominal) | siren / car horn / gunshot — three temporal shapes |
| Amplitude | urgency (ordinal) | 40% / 70% / 100%, ≈ 8 dB span |
| Position | — | not encoded |

| Category | Temporal pattern | Repetition |
|:--|:--|:--|
| gunshot | single steep pulse, 60 ms | one-off, once again after 3 s |
| car horn | double pulse, 150 / 80 / 150 ms | every 700 ms |
| siren | rising–falling sweep, 400 ms | every 400 ms |

Three constraints hold the design together:

1. **Category is carried by time shape only.** Amplitude must not be a per-category constant, or the two axes become collinear and discriminability collapses. The semantic prior only sets the default level; every category may span at least two levels.
2. **Urgency = semantic prior (primary) + duration/repetition + relative level quantile (secondary).** Absolute dB SPL is banned: microphone coupling, occlusion and AGC destroy it.
3. **Amplitude levels are spread ≈ 8 dB** so that individual fitting differences survive; `LevelNormalizer` in `code/haptic_encoder.py` handles relative calibration.

`code/haptic_render.py` turns any haptic pattern into an audible 175 Hz LRA rendering, so the vocabulary can be checked by ear before any hardware exists.

---

## Model

DS-CNN over log-Mel (128 × 100 from a 1 s, 16 kHz window), 67 K parameters, ≈ 295 KB in float32. Four output classes: `siren`, `car_horn`, `gun_shot`, `background`. Trained on recording-level folds so that no source recording appears in both training and test.

| Script | Role |
|:--|:--|
| `code/preprocess.py` | corpus → 1 s `.npy` segments + `metadata.csv` (class mapping, background sampling, dedup, splits) |
| `code/train_dscnn.py` | training; `--exclude-class` produces the 4-class model from the 5-class pool |
| `code/train_baseline.py` | MFCC + linear SVM lower bound on the same split |
| `code/evaluate.py` | accuracy, F1, confusion matrix, latency |
| `code/eval_report.py` | per-class PRF, calibration, latency tables and figures |
| `code/eval_emergency.py` | event-level recall and false-alarm rate across a threshold sweep |
| `code/tune_threshold.py` | operating-point selection on the validation fold |
| `code/export_onnx.py` | ONNX export |

---

## Pipeline

Two branches read the same microphone stream:

```
mic ──┬─► AED  (detect → decide → haptic code)        safety branch, always on
      └─► denoiser (conditional TSE + safety gate)    comfort branch, opt-in
```

The detection branch never waits on the denoiser, and the denoiser can never remove a whitelisted hazard class — the safety gate is hard-coded, not tuned.

| Script | Role |
|:--|:--|
| `code/filter_config.py` | per-class gain tables, user profiles, hard safety constraints |
| `code/attenuator.py` | spectral-gate attenuation driven by instance segmentation |
| `code/tse_model.py` / `code/train_tse.py` | conditional target-speech-enhancement denoiser |
| `code/filter_pipeline.py` | streaming attenuation with fail-open on low confidence |
| `code/synth_mix.py` | signal-level mix builder for reproducible filter tests |
| `code/dual_path_pipeline.py` | both branches over one stream |
| `code/realtime_demo.py` | PC real-time loop, mic in, haptic pattern out |

---

## Measurement

| Script | Role |
|:--|:--|
| `code/data_audit.py` | composition, integrity, silence and leakage audits |
| `code/preprocess_verify.py` | 11-point verification of the segment pipeline |
| `code/eval_filter.py` | attenuation effect: SNR change vs whitelist preservation |

---

## Tests and human-subject harness

| Script | Role |
|:--|:--|
| `code/haptic_test.py` | forced-choice test over the two axes, separate confusion matrices |
| `code/s1_amplitude_pretest.py` | amplitude-axis discriminability pre-test (3AFC) |

Both print a session transcript and write JSON/CSV per subject; `haptic_test.py --no-audio` and `s1_amplitude_pretest.py --dry-run` validate the stimulus design without an LRA driver.

---

## Data and weights

Training used UrbanSound8K and ESC-50, preprocessed into three hazard classes plus a background pool. Per the licences of both corpora, **no audio is redistributed here**, and no checkpoints are committed.

Point `SAFESOUND_DATA_DIR` at a preprocessed directory to run anything that needs data; `code/config.py` resolves every other path relative to the repository, so nothing is hardcoded.

---

## Known limitations

- **Novel hazard classes are not rejected** — the rejection channel is calibrated on seen hazards and does not generalise to unseen ones.
- **Overlapping events are weakly decoded** — when two hazards overlap, both are rarely tagged.
- **No direction of arrival**, so position is not encoded; the vocabulary has two axes, not three.
- **No on-device measurement** — the model exports to ONNX, but latency on a phone is unmeasured.
- **No human participants yet** — the vocabulary is validated for internal consistency only. Both axes are separately testable with `haptic_test.py`; the amplitude axis needs the pre-test first.
- **Latency is bounded by the 1 s analysis window**, not by compute (2–4 ms per window on CPU). Shrinking the window costs accuracy because training segments are fully occupied by their class; fixing that needs partial-event augmentation, not a bigger model.
- **The attenuation branch is present but not validated** — its gain tables and gate are exercised by `eval_filter.py`, with no listening test behind them.

---

## Contact

Qingming Li · MSc Data Science and Machine Learning, UCL · qingming.li@ucl.ac.uk

## Licence

MIT — see `LICENSE`. Third-party corpora retain their own licences and are not redistributed here.
