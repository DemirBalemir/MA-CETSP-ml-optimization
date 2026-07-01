# MA-CETSP — Session Context (updated 2026-06-29)

> Handoff doc for a fresh session. Read this first, then the files in
> **"Files to read to reason on the codebase"** below.

---

## 1. What the project is

**MA-CETSP** = Memetic Algorithm for the Clustered Euclidean Travelling Salesman
Problem. Core idea: in an evolutionary algorithm, each generation produces
offspring that are normally sent to an expensive local search (**VND**). Before
running VND, use **survival-analysis ML models** to predict which offspring are
low-quality and **reject them pre-VND**, saving VND time.

- **Prior work**: LA-CETSP (Balemir et al.) — 3 models (COX, RSF, GBSA), 4
  instances. This is the same first-author's **follow-up** paper.
- **This work**: 9 models (COX, RSF, GBSA, DEEPSURV, SSVM, WEIBULLAFT, KNN,
  ELASTICNET, MTLR), 68 instances, **island parallelism** (10 islands run
  simultaneously; islands 0–8 = one ML model each, island 9 = BASELINE / no ML).
  Per run, GLOBAL BEST is taken across islands.
- Author is a **fresh bachelor's graduate**; advisor wants to submit to
  **Expert Systems with Applications (ESWA, Elsevier, Q1)**. NOT a thesis.

Executable: `build/Release/MA-CETSP.exe` (links `gurobi120.dll`, calls LKH).

---

## 2. Architecture / execution flow

- `src/main.cpp` → `src/Algo.cpp::run()` is the per-island GA loop.
- Each iteration: `Population::nextPopulation(patience)` makes an offspring
  (crossover + mutation), optionally **ML-filters** it, then runs VND, then
  inserts it.
- ML prediction: **COX is native C++** (`SurvivalModel::predict_cox_score`); the
  other 8 models go over a **persistent Python pipe** (`predict_via_process`,
  one blocking round-trip per offspring, byte-by-byte read).
- ML training fires **once per run mid-search** via `system(train.bat)` →
  spawns a fresh Python that imports sklearn/torch and trains, blocking.
- Thresholds are calibrated on a held-out val split (`ml/scripts/threshold_utils.py`).

**One `patience` counter does two jobs**: (a) adaptive mutation rate
(`Population.cpp` ~line 236: `randomInt(1000) < patience` → higher patience =
more mutation) and (b) early-stop (`Algo.cpp`: stop when `patience >=
patience_threshold`). `patience_threshold` = `max(iteration/10, 50)`
(`src/Utils/Parameters.cpp`). For 5000 iters → 500.

---

## 3. What we changed this session (all committed, `main` @ bdfda9d, pushed)

Goal: make the **per-model comparison fair** and fix systemic over-rejection.
All validated on bonus1000 (1000 & 5000 iter) + dev/test sets.

1. **Decoupled training from patience** (`src/Algo.cpp`, training-trigger block).
   Training no longer reads/resets `patience`. It fires on budget-floor
   (`ML_TRAIN_FRAC_MIN`=20%) + event-count (`ML_MIN_EVENTS`=100), with a hard
   fallback at `ML_TRAIN_FRAC_HARD`=50%. Removed the `patience = 0` reset after
   training. → all islands (baseline + 9 ML) now early-stop on the *same* rule.
   Removed constants `ML_TRAIN_FRAC_MAX`, `ML_PATIENCE_FRACTION` from `Defs.hpp`.

2. **ML-rejected iterations are NEUTRAL to patience** (`src/Algo.cpp`, the
   `reject_before` / `ml_rejected` logic around the `improved`/`++patience`
   branch). A rejected offspring never reached VND, so it neither resets nor
   advances patience. Prevents an aggressive filter from triggering premature
   early-stop. **CAVEAT — this has a downside (see §4).**

3. **Soft-cap threshold objective** (`ml/scripts/threshold_utils.py`). Replaced
   `objective = rejection_rate * survival_gap` (which on the val set is
   mechanically `(100-pct)/100 * gap` → structural bias toward over-rejection)
   with `_soft_cap_objective`: reward discrimination (survival_gap), ramp up to
   `TARGET_REJECT_RATE`=0.20, penalise beyond it (`OVER_PENALTY`=2.0). Selected
   percentiles moved from ~54–72 to ~80; rejection dropped ~35-50% → ~15%; the
   5000-iter drift to ~50% disappeared.

4. **parse_logs.py fixes**: (a) `RE_TRAIN` updated to the new training-log
   format (`enough_data=` instead of `stagnation/soft_ceil`); (b) instance name
   now derived from the **folder index via `FILENAMES`**, NOT the corruptible
   `[SUMMARY] instance=` field (thread-unsafe cout was creating phantom
   instances like `80.012kroD100_or30`).

5. **analyze_development.py rewritten**: `group_of()` derives group from name
   (old hard-coded 15-instance dict only matched dev set); **active/inactive
   split** (primary = ML-active instances); **win/tie/loss** columns; **Wilcoxon
   on BOTH quality (best) and speed (time)**; artifact cleaning (drop rows with
   iter > budget); empty-subset guards.

---

## 4. Key experimental findings (test set: 53 instances × 5 seeds × 5000 iter)

Ran across 2 PCs (Demir + melis), merged into
`solutions/experiment_results/`. All 53 complete (2650 summary rows). Analysis
outputs in `analysis_test/results/`.

**Honest headline: the ML filtering does NOT provide a clear net benefit.**

- **Quality**: neutral. Mean degradation ~0 for every model (−0.01% … +0.06%).
  Wilcoxon vs baseline: **all p > 0.05** (no significant quality difference).
- **Speed (unweighted mean per-instance)**: 1.01–1.10x (SSVM 1.097, ELASTIC
  1.064, WEIBULL 1.059 best). Wilcoxon on time: **all p > 0.05** (NOT significant).
- **Speed (compute-weighted / total wall-clock)**: **< 1.0 for most models —
  ML is slightly SLOWER overall.** Reason: the **neutral-reject change (§3.2)
  backfires on the heaviest instances**. On `dsj1000_or2` (baseline 6158s), all
  ML models are 0.55–0.91x because they don't early-stop (median iter 2680–5001
  vs baseline stopping earlier) — they skip VND per-iter but run far more iters.
- **6 ML-INACTIVE instances** (rejects=0): bubbles1, bubbles2, rotatingDiamonds2,
  rat195rdmRad, team1_100rdmRad, team3_300rdmRad. They converge & early-stop
  **before** the training trigger (iter 1000 = 20%), so ML never activates → all
  models = baseline by construction. Driver is **convergence speed** (a function
  of node count AND geometry), not node count alone (e.g. bubbles1(36) inactive
  but concentricCircles2(36) active). Handle: primary analysis on 47 active;
  report the 6 as a "safe/no-harm" result, not a model difference.
- **The quality/speed KNOB** (important for framing): there are two operating
  points — `count-reject` (original: ML ~1.2–1.5x faster on hard instances but
  −2…−5% quality via premature stop) vs `neutral-reject` (our change: quality
  preserved but no/negative speedup). The apparent speedup in prior work is
  **confounded with premature stopping**.

Group nuance: `or2` (low overlap, hard) → best speedup (SSVM 1.21x), small +deg;
`or10/or30/rdmRad` → slightly negative deg (ML a touch better), mixed speed.

---

## 5. Publication plan (ESWA)

ESWA = applied AI/ML to real-world problems; high desk-rejection rate; wants a
clear **positive application contribution**, not a null result. To be viable:

1. **Reframe contribution away from "ML vs baseline"** onto the **adaptive
   memetic algorithm** (adaptive training trigger + soft-cap threshold +
   convergence-based stopping) + the **systematic 9-model comparison**.
2. **Add SOTA / best-known CETSP comparison** — the single most important
   missing piece. Compare our best-known objective values to published CETSP
   solvers (Mennell; Carrabs et al.; Lei & Hao). If competitive → paper's spine.
3. **Lead with the real-world angle**: `car_door_*` (automotive welding) instances.
4. **Present the speed/quality knob as a positive, actionable contribution** +
   the "fair-evaluation / confound" story as a methodological contribution.
5. Strong novelty statement + highlights (ESWA requires them).

**NEXT CONCRETE STEP**: build the SOTA/best-known comparison table from published
CETSP results vs our data. (Not yet started.)

---

## 6. Analysis pipeline (how to reproduce)

```powershell
# 1. run experiment (direct call in a PS session; comma-separated instances;
#    NOTE: `powershell -File ... -Instances 1,2,3` MANGLES the array — call directly)
.\run_experiment.ps1 -Instances 1,2,4,6,7 -Iterations 5000    # 5 seeds hardcoded

# 2. parse (point --results_dir at a folder containing only the wanted runs;
#    we stage the newest folder per instance via junctions to avoid old runs)
python analysis_first_try/parse_logs.py --results_dir <dir> --out_dir analysis_test/parsed

# 3. analyze
python analysis_first_try/analyze_development.py --parsed_dir analysis_test/parsed --out_dir analysis_test/results --budget 5000
```

Python with sksurv+lifelines+torch+pycox required:
`C:\Users\Demir\AppData\Local\Programs\Python\Python310\python.exe`.

Result CSVs (`analysis_test/results/`): `model_comparison_active.csv` (primary),
`model_comparison_full.csv`, `ml_activation.csv`, `group_breakdown.csv`,
`wilcoxon_test.csv` (quality+speed), `best_iter_analysis.csv`,
`threshold_stability.csv`, `training_events_summary.csv`.

---

## 7. Files to read to reason on the codebase

| File | Why |
|---|---|
| `src/Algo.cpp` | GA loop, training trigger, early-stop, patience/neutral-reject |
| `include/Defs.hpp` | ML constants (ML_MIN_EVENTS, ML_TRAIN_FRAC_*), `FILENAMES` (idx→name) |
| `src/Genetic/Population.cpp` | offspring gen, ML filter/rejection, mutation via patience (~L236) |
| `src/ML/SurvivalModel.cpp` | Python-pipe prediction, native COX, server lifecycle |
| `src/Utils/Parameters.cpp` | `patience_threshold = max(iteration/10, 50)` |
| `src/Utils/Data.cpp` | `write()` per-improvement snapshot; **has create_directory bug (§8)** |
| `ml/scripts/threshold_utils.py` | soft-cap threshold objective |
| `ml/scripts/train_*.py`, `predict_*.py` | per-model training + prediction servers |
| `analysis_first_try/parse_logs.py` | log → CSV parser (FILENAMES-based naming) |
| `analysis_first_try/analyze_development.py` | active/inactive analysis, wilcoxon, groups |
| `run_experiment.ps1` | experiment runner (5 seeds hardcoded, auto-detects python) |

---

## 8. Known issues / gotchas

- **Thread-unsafe `std::cout`**: parallel islands write without a mutex → some
  `[LOG]`/`[SUMMARY]` lines are corrupted (garbled iters like `iter=96009`,
  phantom instance names). Mitigations: parse uses folder-index naming +
  `RE_PARALLEL_ROW` fallback + summary; analyze drops iters > budget. `summary.txt`
  (best/time/rejected) is reliable. A proper fix (mutex around cout) is pending.
- **`Data::write` dir bug** (`src/Utils/Data.cpp` ~L81): uses
  `create_directory` (singular) for `solutions/normal/<...>/`; if the parent
  `solutions/normal/` doesn't exist it throws → "ERROR : write result" (cosmetic:
  only per-improvement tour snapshots are lost; analysis unaffected). Fix =
  `create_directories`. Not done (would require rebuild during live runs).
- **PowerShell `*>` redirect writes UTF-16**; the exe's own logs are UTF-8 (BOM).
  Decode accordingly when grepping.
- **`powershell -File script.ps1 -Instances 1,2,3` mangles the int[] array** into
  one number — call the script directly from a PS session instead.
- Log files are large (heavy instances 50–90 MB); parsing 265 of them takes
  minutes — run in background.

---

## 9. Instance reference

Index→name in `include/Defs.hpp` `FILENAMES` (0–67). Groups (from name):
`_or2/_or10/_or30`, `rdmRad`, `car_door`→real_world, else `varied`.

- **Dev set (15, already run @1000 iter)**: 0,5,11,16,24,27,36,39,44,50,51,61,62,64,66
- **Test set (53, run @5000 iter)**: 1,2,3,4,6,7,8,9,10,12,13,14,15,17,18,19,20,
  21,22,23,25,26,28,29,30,31,32,33,34,35,37,38,40,41,42,43,45,46,47,48,49,52,53,
  54,55,56,57,58,59,60,63,65,67
- **Heavy (999–1000 nodes, dominate wall-clock)**: 28 dsj1000_or2, 35 dsj1000_or10,
  42 dsj1000_or30, 48 bonus1000rdmRad. dsj1000_or2 is the single most expensive
  (~2.5–3 h/seed; low overlap = hard).
- **ML-inactive on test set (6)**: 1,2,17,54,56,58.
