# Experiment Runner

The completed November revision and saved-result verification are documented in
[reproducibility/revision_20261109/README.md](reproducibility/revision_20261109/README.md).
The historical runner/task notes below retain their original context. Use the
saved-result script to check the published tables; no campaign execution is
needed. Current code checks are described in [tests/README.md](tests/README.md).


Runs the filtered memetic algorithm across benchmark instances with fixed seeds, launching 10 concurrent threads (islands) per run: nine carrying distinct survival-analysis filtering models and the tenth running with filtering disabled as an in-run control.

---

## Evaluation Protocol

As described in Section 4 of the paper:
- **Ten Parallel Islands:**
  - Islands 0–8: Survival models (`COX`, `RSF`, `GBSA`, `DEEPSURV`, `SSVM`, `WEIBULLAFT`, `KNN`, `ELASTICNET`, `MTLR`)
  - Island 9: Unfiltered control baseline (`--ml_enable 0`)
- **Seeding Scheme:** Island $i$ is seeded with base seed $\sigma + i$. For each configuration, five base seeds are evaluated: $\sigma \in \{1, 11, 21, 31, 41\}$.
- **Stopping Rule:** Early stopping terminates an island after $\rho = \max(T / 10, 50)$ non-improving iterations (e.g. 500 iterations for $T = 5000$). Rejections skip VND and are neutral for the stagnation counter.
- **Adaptive Trigger:** Models are trained when $\text{iter} \ge 0.20 \cdot T$ and at least 100 evictions have been observed, with a hard fallback at $0.50 \cdot T$.
- **Runtime Guard:** A rolling rejection-rate cap suspends filtering if rejections exceed 30% over a 50-candidate window.

---

## Files

| File | Purpose |
|---|---|
| `run_experiment.ps1` | Primary automated experiment runner (PowerShell) |
| `run_experiment.bat` | Wrapper batch script |
| `solutions/experiment_results/` | Target directory for logs and summary files |

---

## Usage

### PowerShell Runner
```powershell
# Run on bonus1000 (instance index 0) across all 5 seeds with 10 parallel islands:
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instances 0 -Islands 10 -Iterations 5000

# Run across multiple instances:
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instances 0,1,2,3 -Islands 10 -Iterations 5000
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `-Instances` | `0` | Array of instance indices into `FILENAMES` in `include/Defs.hpp` |
| `-Islands` | `10` | Number of parallel islands (9 ML models + 1 baseline control) |
| `-Iterations` | `200` | Iteration budget $T$ (use `5000` for full paper benchmark) |

---

## Output Structure

Output is written to `solutions/experiment_results/`:

```
solutions/experiment_results/
├── instance0_summary.txt          ← Aggregate results table
├── instance0_run1_seed1.log       ← Console output for seed 1
├── instance0_run2_seed11.log      ← Console output for seed 11
├── instance0_run3_seed21.log      ← Console output for seed 21
├── instance0_run4_seed31.log      ← Console output for seed 31
└── instance0_run5_seed41.log      ← Console output for seed 41
```

### Summary File Layout

```text
------------------------------------------------------------
Run 1  |  seed=1
------------------------------------------------------------
  Island 0  model=COX        best=   448.042  time=  107.509s  rejected=18
  Island 1  model=RSF        best=   442.311  time=  111.697s  rejected=12
  ...
  Island 9  model=BASELINE   best=   444.120  time=  109.821s  rejected=0
  --
  GLOBAL BEST  model=RSF  value=442.311  time=111.697s  wall-clock=117.644s
```

---

## Instance Index Reference

| Index | Instance | Index | Instance | Index | Instance |
|---|---|---|---|---|---|
| 0 | `bonus1000` | 1 | `bubbles1` | 2 | `bubbles2` |
| 3 | `bubbles3` | 4 | `bubbles4` | 5 | `bubbles5` |
| 6 | `bubbles6` | 10 | `chaoSingleDep` | 21 | `team1_100` |
| 22 | `team2_200` | 23 | `team3_300` | 24 | `team4_400` |
| 25 | `team5_499` | 26 | `team6_500` | 27 | `d493_or2` |
| 28 | `dsj1000_or2` | 34 | `d493_or10` | 35 | `dsj1000_or10` |
| 41 | `d493_or30` | 42 | `dsj1000_or30`| 48 | `bonus1000rdmRad` |
| 62 | `car_door_25` | 63 | `car_door_30` | 64 | `car_door_35` |
| 65 | `car_door_40` | 66 | `car_door_45` | 67 | `car_door_50` |

*Complete 68-instance list is defined in `include/Defs.hpp` $\to$ `FILENAMES`.*


## Task 3 review fixes (30 October 2026; review pending)

`tools/experiment_control.py` is the guarded sequential Task 3 runner. The
concurrent ten-island runner described above is the historical experiment.
Main/draft execution remains blocked. No new pilot, calibration or timing run
is authorized by these fixes.

Resume now requires a version-2 full configuration record and matching hash,
command, current input identities, valid outputs and complete startup record.
Legacy statuses are inspection-only and are never upgraded or rewritten.
Existing output directories are preserved, including mismatching successes.
Inspection with `validate_run` checks the fields available in historical logs;
it does not certify historical provenance or make a run resumable.

The source hash now covers the sorted per-file SHA-256 map for all C++ sources
and headers under `src/` and `include/`, all Python under `ml/scripts/`, root
CMake definitions and historical runner definitions, and the Task 3 runner.
`TASK3_SOURCE_IDENTITY_V2.json` is the review-time snapshot; each new status
also embeds its own full source map. Generated models, logs, datasets and build
outputs are excluded from this source hash. Runtime identity separately records
the selected dataset, executable, schedule, LKH executable, adjacent DLLs and
configured Gurobi runtime DLL when present. Python identity includes executable
bytes, version/implementation, installed distribution names/versions and RECORD
metadata digests. It does not hash every installed package payload or resolve
the complete system/transitive DLL closure. Environment variables are bound by
a digest without saving their values; platform, machine and working directory
are recorded. CPU contention, licenses and transient machine state are outside
this identity. This is scoped provenance, not a hermetic reproduction claim.

Old manifests deliberately retain their old source and binary identities. The
new source algorithm and rebuilt startup logger make them incompatible with new
execution. Do not edit them in place. A future separately authorized technical
run must use an explicit new manifest filename and new output root, for example
`--kind pilot --manifest pilot_manifest_task3_v2.csv --output-root
solutions/technical_pilot_task3_v2`. Without `--execute` this only prepares a
manifest/dry-run. Review the new manifest before any separately authorized run.
The default old-manifest command now fails clearly on different contents.

Cox score/threshold lines and JSON models are parsed and checked, including
feature agreement and usable standard deviations. The producer's low-variance
feature removal permits a nonempty subset of its ten actual input features.
The small-sample `objective=NaN` exception requires n<30, percentile 80 and the
exact producer fallback message. Used coefficients, means, standard deviations
and threshold must always be finite. Training success, observed prediction use,
convergence warnings and fallback remain separate diagnostics. Warnings do not
silently become a new scientific exclusion policy. Candidate/iteration IDs must
be exactly 1..generated; admitted IDs and final lifetimes must match one-to-one,
with valid zero durations and censoring only at the final processed iteration.
