# Corrected offline neural diagnostics

Completed on 8 November 2026. Exactly 30 offline fits: two neural models, five diagnostic instances and three GroupKFold folds by run. No solver run was started.

## Correction and frozen settings

The invalid validation EarlyStopping callback was removed. Each fit trains for the previously specified 100 epochs, using all outer-training rows, Adam with learning rate 0.01, two 64-unit hidden layers, BatchNorm and dropout 0.1. No hyperparameter search, model selection, test-dependent stopping or additional fit was performed. The online training entry points use the same fixed-epoch safeguard (their existing 100/200-epoch requests are retained); they were not executed in this study.

Standardisation uses outer-training rows only. Whole runs remain outside training in each outer fold. The legacy concordance calculation is retained: comparable pairs can cross held-out runs when a fold contains multiple runs. This is not the separate within-time diagnostic. DeepSurv log-risk and MTLR last-bin risk are unchanged.

## Results

| Instance | DeepSurv | MTLR | Records |
|---|---:|---:|---:|
| concentricCircles4 | 0.538295 | 0.497999 | 3739 |
| car_door_50 | 0.532166 | 0.492450 | 3733 |
| bonus1000 | 0.546914 | 0.487078 | 3751 |
| kroD100_or2 | 0.589970 | 0.498865 | 3179 |
| rat195_or30 | 0.568913 | 0.490827 | 3713 |
| MEAN | 0.555252 | 0.493444 | 18115 |

Total fitting time recorded across the 30 fits: 57.87 seconds. This excludes data preparation and verification and is not an end-to-end solver runtime.

## Comparability and limits

The historical table reports 17,568 records. Its exact sampled path list was not retained. New records were added to some raw folders after the original per-instance analyses. Applying the original sampling rule to the current folders yields 18,115 usable records. The exact new file selection, hashes and folds are frozen. Matching the selection method does not establish an identical historical sample. Changes from the historical scores combine corrected training, different samples and controlled new neural seeds; they are not a paired estimate of the training-fix effect alone.

The old reported mean concordances were DeepSurv 0.5006 and MTLR 0.4949. The corrected means are 0.555252 and 0.493444. These are descriptive means across five diagnostic instances, without a test of differences or an end-to-end filtering-utility claim. The retained positive-lifetime eviction sample omits unadmitted candidates, zero-lifetime records and terminal survivors.

MTLR retains the deployed 1 - survival at the last time bin as its risk score. Its current scores cluster near one (across folds: 0.9928835 to 1.00000036; the small overshoot is floating-point error). The average within-fold fraction of comparable pairs tied under the concordance routine is 18.15%. Its near-chance concordance therefore describes this scoring rule and trained configuration, not all possible rankings from the learned survival curve. No alternative scoring rule was tuned or substituted.

DeepSurv is no longer near 0.500 under this corrected evaluation. This warrants revising the neural rows and associated narrative after resolving sample comparability; it does not demonstrate superiority to the seven historical model rows on their different samples.

## Verification

All 30 fits completed 100 epochs with finite training losses and model states. Every fit changed trainable parameters and updated both BatchNorm counters. Initial and trained states are archived. Reloading each of the 30 saved models reproduced every recorded held-out risk exactly; concordances were independently recomputed from the prediction files. All 19,397 selected raw-file hashes were rechecked without mismatch. The historical model table and existing submission ZIP were not overwritten.

## Files

- protocol.json: fixed settings, environment versions and source/input identities.
- source/: exact code used; before/: previous sources and historical result table.
- input_selection.csv, inputs/, fold_assignments.csv, fit_manifest.csv: frozen input and execution selection.
- fits/: initial/trained weights, training history, scaler, MTLR cuts, warnings and held-out predictions for each fit.
- corrected_neural_cindex.csv, fold_results.csv, comparison.csv: new results and descriptive historical comparison.
- verification.json, verification_fits.csv, input_verification.json: independent checks.
- SHA256.json: file checksums.

The main manuscript, figures and packaged submission have not been updated. Historical online DeepSurv/MTLR cost and runtime rows remain unresolved by these offline fits.
