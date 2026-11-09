# Single-island filter controls

`--filter_mode` defaults to `legacy`, preserving the existing CLI and ten-island
model cycle. Explicit `none`, `learned`, and `random` modes require `--islands 1`.
Explicit modes reject `--ml_enable` rather than assigning it precedence. The
legacy multi-island control remains unfiltered. An explicitly provided
`--ml_model MTLR` is honored, even though it equals the default model name.
Explicit `none` and `random` reject an explicitly supplied `--ml_model` because
it has no effect. `learned` accepts exactly `COX`, `RSF`, `GBSA`, `DEEPSURV`,
`SSVM`, `WEIBULLAFT`, `KNN`, `ELASTICNET`, or `MTLR` (case sensitive).

Every explicit-mode invocation reserves a fresh directory under `--output_root`
(default `experiment_runs` relative to the launch directory). An optional
`--run_id` names one child directory; an existing name is an error. Without
it, a unique name is allocated atomically, including for simultaneous starts.
The startup record prints the full run directory. Its `model`, `solutions`,
and `lkh` children hold the model and training files, solution and survival
logs, and temporary LKH files. Training reads the same run's survival logs;
prediction reads the same run's model files. Legacy paths are unchanged.

`none` bypasses training and prediction. `learned` enables the selected model
(for example `--ml_model COX`). `random` bypasses all model training, prediction,
and Cox threshold reads. It requires `--prune_seed` and exactly one of
`--random_reject_rate` or `--prune_schedule`. The rate must be in [0,1]. With
the fixed-rate form, filtering applies only when `current_iter >
random_start_iter`; the default start is zero. The random generator is separate
from the search generator, and zero or one probabilities do not draw from it.

Schedule CSV columns are exactly `start_iter,end_iter,reject_probability`.
Intervals are closed and must cover iterations 1 through the full `-r` budget
without gaps or overlaps. A zero-probability row can express an inactive
prefix. The schedule is loaded and validated before the solver starts. The
startup record prints its absolute path and FNV-1a 64-bit content hash.

Training options are `--train_frac_min` (default .20), `--train_frac_hard`
(default .50), and `--min_events` (default 100). Fractions require
`0 <= min <= hard <= 1`; minimum events must be positive. These settings
affect learned training only. Existing integer truncation of trigger iterations
is retained.

Examples launched from the repository's existing `build/Release` directory
(illustrative commands, not a completed main run):

```powershell
..\..\build-task1-validation\Release\MA-CETSP.exe -i 1 -s 11 -r 200 --max_time 30 --islands 1 --filter_mode none --output_root ..\..\validation-runs --run_id control-11
..\..\build-task1-validation\Release\MA-CETSP.exe -i 1 -s 11 -r 200 --max_time 30 --islands 1 --filter_mode learned --ml_model COX --output_root ..\..\validation-runs --run_id cox-11
..\..\build-task1-validation\Release\MA-CETSP.exe -i 1 -s 11 -r 200 --max_time 30 --islands 1 --filter_mode random --random_reject_rate 0.08 --random_start_iter 40 --prune_seed 917 --output_root ..\..\validation-runs --run_id random-11
```

The printed `ml_rejects` / `rejected` count includes random or learned filter
rejections. `Population::last_filter_decision` exposes `rejected` and
`reached_vnd` for the most recently processed candidate. Rejected candidates
skip VND and insertion, are deleted once, and leave patience unchanged.
For explicit modes the `[FILTER]` line exposes this outcome per iteration.

Random filtering uses the stated Bernoulli probabilities without the learned
filter's 50-candidate rolling cap. Observed rejection counts need not equal
their expected values. A schedule alone is not a calibrated matched-rate
experiment. The stopping rule can end runs before the schedule's budget, and
later Cox rates may then be unknown. Select calibration inputs and the rule
for constructing a schedule before the main experiment.

## Candidate and lifetime records (explicit modes)

`solutions/research/candidates.csv` has one row per processed offspring:
`candidate_id,iteration,outcome`. `outcome` is `filter_rejected`,
`not_admitted`, or `admitted`. An admitted offspring removed in its birth
iteration remains `admitted`; its lifetime row has zero survival iterations.
The initial population is excluded and has `candidate_id=0` internally.
One offspring is processed per search iteration, so the run-local candidate ID
equals its iteration number.

`solutions/research/lifetimes.csv` has exactly one final row per admitted
offspring: `candidate_id,birth_iter,death_iter,survival_iters,censored`.
`censored=0` means population eviction; `censored=1` means the offspring was
still resident after the final processed iteration. Training snapshots remain
under `solutions/ml_logs` and are *not* final lifetime rows. They contain only
information available at the training trigger, while the research file tracks
subsequent evictions and final survivors. Explicit-mode training refuses to
fall back to historical logs. Cox's intermediate dataset is placed in this
run's `model` directory.

The `[CANDIDATES]` line reports `generated`, `filter_rejected`,
`vnd_evaluated`, `admitted`, and `not_admitted`. It enforces
`generated = filter_rejected + vnd_evaluated` and
`vnd_evaluated = admitted + not_admitted`. It also prints
`admitted/generated` and `admitted/vnd_evaluated`; a zero denominator is
`undefined`. For explicit modes, `effective_iter` is the actual processed
offspring count. Legacy summary semantics remain unchanged. CSV writing and
final survivor recording are included in the measured `total_t`; the cost is
not subtracted. Per-run output overhead depends on storage and cohort size.

The same run-local `ResearchLedger` used by the solver is exercised by a
deterministic CTest case. It constructs a VND-evaluated but unadmitted candidate,
a same-iteration eviction, a death after a separate training-time snapshot,
and a final survivor. It checks the counters, CSV rows, uniqueness of final
lifetimes, and rejection of a duplicate final row. From the repository root:

```powershell
cmake --build build-task2-validation --config Release --target MA-CETSP prune_policy_test research_ledger_test
ctest --test-dir build-task2-validation -C Release --output-on-failure
```
