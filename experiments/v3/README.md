# experiments/v3 — multi-label v3 experiments

These three files produced the headline numbers (91.19% segment-level, 93.11% clip-level,
AUROC 0.9762 / 0.5849, both_recall 0.2417). They are kept separate from `code/` because they
depend on a different input contract: they read a cached waveform pool and train a
multi-label head, rather than the recording-level single-label model used by the demos.

| File | Role |
|:--|:--|
| `core.py` | pool loading, Mel front end, `NetV3` (4 hazard logits, sigmoid), training and evaluation loops |
| `final_v3.py` | temperature scaling, multi-label overlap probe, Mahalanobis rejection scores, experiment driver |
| `agg_context.py` | clip-level aggregation: segment / K=5 smoothing / learned BiGRU / learned + K=5 |

## What each configuration is

| Config | Meaning | Segment acc | Macro F1 | mAP |
|:--|:--|:--:|:--:|:--:|
| `v2repro` | v2 recipe re-run on fold 9 | 84.42 | .8315 | .9110 |
| `ml_v3` | multi-label + mix augmentation + fixed Mel filter bank | **91.19** | .8269 | .9113 |
| `ml_v3_512` | as above, 512/512 FFT/window | 88.73 | .8267 | .9012 |
| `ml_ctx5` | `ml_v3` + 5 adjacent segments as channels | 88.25 | .8349 | **.9476** |
| `melfix` | Mel fix only, no multi-label | 78.72 | .7960 | .9038 |

`ml_ctx5` wins on mAP but loses ~3 points after thresholding, so it is kept as a ranking /
event-level candidate and not promoted to the main line. `ml_v3` is the main line.

Clip-level aggregation over `ml_v3`:

| Variant | τ | Clip acc | Macro F1 |
|:--|:--:|:--:|:--:|
| segment-level decision | 0.7 | 91.16 | .8278 |
| hand-tuned K=5 smoothing | 0.5 | 91.36 | .8401 |
| learned BiGRU | 0.25 | 91.96 | .8657 |
| learned BiGRU + K=5 | 0.2 | **93.11** | **.9259** |

## Not runnable without the data pool

Neither the waveform cache nor the checkpoint is committed. `core.py` expects the
preprocessed pool and a `metadata_clean.csv` from `code/`; `agg_context.py` expects the
per-segment logits dumped by `final_v3.py`. See the top-level README for how to rebuild
the pool from the original corpora.
