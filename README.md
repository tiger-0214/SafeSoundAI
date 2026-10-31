# SafeSoundAI

**Real-time detection of safety-critical sounds, delivered as haptic patterns.**

A Deaf or hard-of-hearing person misses the siren, the horn, the reversing alarm. Most assistive apps respond by amplifying everything, which raises the noise floor rather than delivering the one thing that mattered. SafeSoundAI detects the three hazard classes that carry most of the risk in street traffic and encodes them into a **2-axis haptic vocabulary**, with an optional attenuation branch for users who would rather have the environment quietened than the alert louder.

This repository is the **code**: the detection model, the haptic encoder, the parallel two-branch pipeline, and the evaluation and measurement scripts. No audio, no model weights and no build artefacts are committed.

---

## Quick start

```bash
pip install -r code/requirements.txt
export SAFESOUND_DATA_DIR=/path/to/preprocessed_data   # only for the data-dependent targets

make demo        # offline: audio → window → model → haptic code, no dataset needed
make demo-e2e    # streaming: health check → real-audio scene → haptics → event-level score
make test        # 80 pytest cases
make check       # provenance gates for the filter and detection numbers (CI-usable)
make onnx        # fp32 + int8 export with an accuracy gate
make stream      # 30 min streaming soak test
```

`make demo` needs nothing but the checkpoint; `make demo-e2e` checks the environment first and
tells you exactly what is missing instead of failing halfway.

`make demo` prints one row per detected event — time, class, urgency, haptic pattern, interval, amplitude, code:

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

The detection branch never waits on the denoiser, and the denoiser never relies on a class list to decide what is safe — the safety gate is hard-coded, not tuned.

> **Semantics changed in V2 (current).** The denoiser used to *protect* hazards (whitelist, 0 dB) and let unrecognised sound through (fail-open). It now does the opposite: **everything that is not the wearer's voice is attenuated**, hazards included, and unrecognised sound is **fail-closed**. Safety awareness is carried by the haptic channel, which is why the raw siren can be 18 dB down and still be reported. A frame-level Silero VAD gate lifts the gain back to 0 dB wherever speech is present, so the only class guaranteed 0 dB is `children_playing` (the one that carries speech).
>
> **Two layers, both current (2026-11-01):**
> 1. **V2.1 — VAD veto.** The gate used to be one-directional (lift speech frames, leave every other frame on its class gain). A frame the AED labelled with a speech-carrying class therefore became exempt from attenuation even when nobody was talking. The gate now **only** protects frames where the wearer is actually speaking, and independently caps leak-through on the rest (`SPEECH_GATE_FALLBACK_DB = −18 dB`, `SPEECH_GATE_LEAK_EPS_DB = −1 dB`). VAD failure is fail-open for the veto — a dead VAD never makes the device louder.
> 2. **A-route — VAD-modulated spectral subtraction.** Semantic gating is binary at 10 ms granularity: inside a speech frame the steady-state noise rides along untouched, which is why SI-SDR stalled at +0.13 dB. `spectral_subtraction.py` estimates per-bin noise with minimum statistics (Martin 2001, causal sliding window) and subtracts asynchronously — α = 8 on non-speech frames, α = 4 inside speech. The two masks are merged pointwise (min), so semantics sets the floor per window and spectral subtraction refines within it.
>
> Measured on 20 speech+noise mixtures (`eval_speech_preservation.py`, `ptsd_default`): non-speech suppression **−18.36 dB**, SI-SDR **+1.550 dB** (was +0.08 in V2), STOI **0.822 → 0.820**, speech-band energy **−1.05 dB**. Full backend × profile matrix in `outputs/reports/speech_backend_matrix.md`; parameter sweep in `spectral_sweep.md`.
>
> **Detection must not eat the denoiser's output.** With spectral subtraction in the chain, the z-score-normalised siren detector loses score on individual clips (worst −0.49) because its input is spectrum-shape sensitive, not energy sensitive. That is why the detection branch reads raw PCM by construction — enforced by a structural check (L2-7) that poisons the denoiser output with zeros and asserts the detection events do not change by a single sample.

| Script | Role |
|:--|:--|
| `code/filter_config.py` | per-class gain tables, user profiles, hard safety constraints |
| `code/attenuator.py` | spectral-gate attenuation driven by instance segmentation, plus VAD speech gate |
| `code/tse_model.py` / `code/train_tse.py` | conditional target-speech-enhancement denoiser |
| `code/filter_pipeline.py` | streaming attenuation with gate-fallback on low confidence |
| `code/synth_mix.py` | signal-level mix builder for reproducible filter tests |
| `code/make_speech_mixtures.py` | builds the speech + noise training set (clean speech vs everything non-speech) |
| `code/dual_path_pipeline.py` | both branches over one stream |
| `code/realtime_demo.py` | PC real-time loop, mic in, haptic pattern out |

---

## Measurement

| Script | Role |
|:--|:--|
| `code/data_audit.py` | composition, integrity, silence and leakage audits |
| `code/preprocess_verify.py` | 11-point verification of the segment pipeline |
| `code/eval_filter.py` | attenuation effect: class-stem SNR change vs hazardous-sound preservation (legacy framing, kept for number provenance) |
| `code/eval_speech_preservation.py` | the V2 scoreboard: non-speech suppression + speech preservation (SI-SDR gain / STOI / speech-band energy) |
| `code/ablate_filter.py` | 2 x 2 ablation (input normalisation x whitelist protection) |
| `code/gen_ab_demo.py` | audible before/after pairs, with target and interferer stems written separately |

### Denoising branch, measured

The attenuation branch is a class-conditioned spectral gate: an AED posterior selects a per-class gain, the gain is weighted by that class's measured energy envelope, and whitelisted hazards are carved out of the attenuation rather than vetoing it.

Two bugs made the branch look like it did nothing, and both are now fixed (`vocab.py` documents the class-space rule behind one of them):

1. **No per-window normalisation at inference.** The segments the model was trained on are z-scored; the gate fed raw waveforms to the mel transform, so the posterior was systematically wrong. Top-1 against the true noise class, same 200 real mixtures: 0.120 raw -> **0.525** z-scored (top-3 0.800). On the deployed 4-class head the effect is visible as sensitivity rather than accuracy: it labels the noise mixture `background` in 90.5 % of cases raw, 72.0 % after z-scoring, i.e. 28 % of mixtures get misread as a hazard - which is exactly why the carve-out below is needed instead of a veto.
2. **Whitelist veto instead of whitelist carve-out.** Any whitelist class above p = 0.5 used to let the whole 1 s window through untouched. Since noise is frequently misread as `car_horn`, the gate sat idle exactly when it was needed. It now attenuates and gives back only the whitelist class's own energy band.

Ablation on 100 test mixtures (`ablate_filter.py`, 1 s, target = noise to suppress, interferer = whitelisted hazard to keep):

| Configuration | Suppression (dB) | Preservation (dB) | Trigger rate | Red line violated |
|:--|--:|--:|--:|--:|
| previous logic | -1.52 | -1.59 | 26 % | 22 % |
| + per-window z-score | -4.28 | -3.51 | 54 % | 46 % |
| + carve-out only | -1.25 | -1.30 | 28 % | 21 % |
| + both (current) | **-4.06** | **-3.32** | 57 % | 43 % |

The honest reading: attenuation is now real (-4.06 dB instead of -1.52 dB), but **selectivity is not** (-0.74 dB against +0.07 dB), and that is a property of the method rather than of the tuning. A gain applied to a whole frequency band cannot separate a jackhammer from a horn overlapping it in time and frequency; shrinking the safety strength only scales both columns down proportionally (class-gain alpha = 0.4 gives -1.86 / -1.33 dB, so the Supp-Pres ratio stays at 0.25-0.28; the TSE branch behaves the same way, -3.39 / -2.68 dB at its own alpha = 0.4). Recovering selectivity needs a separation model, not a gate, which is why the hard gate stays disabled (`filter_config.ALLOW_GATING = False`).

`eval_filter.py --n 100` on the same split, both branches: AED gate -4.06 dB suppression / -3.32 dB preservation (57 % of mixtures trigger); conditional TSE -13.89 / -9.58 dB, whose selectivity is likewise about 4 dB at 3.4x the suppression.

Warning about the listening material: before this round the only demo audio was the ten files `outputs/audio/0X_*.wav`, which turned out to be **220 Hz sine placeholders** (2.50 s each, spectral peak at 220 Hz, quantisation step identical in all ten, and `00_air_conditioner` bit-identical to `08_siren`). Every earlier "the detector cannot tell these apart" conclusion drawn from them was an artefact of the fake audio. `gen_ab_demo.py` now builds sample pairs from the real mixture corpus with stems, so each pair can be checked by ear and by the numbers at once.

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

Qingming Li · MSc Data Science and Machine Learning, UCL · ucabq07@ucl.ac.uk

## Licence

MIT — see `LICENSE`. Third-party corpora retain their own licences and are not redistributed here.
