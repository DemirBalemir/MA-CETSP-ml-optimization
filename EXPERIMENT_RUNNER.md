# Experiment Runner

Runs the MA-CETSP algorithm 10 times on a given instance, each time with a different fixed seed, and collects per-model results across all runs. Designed for fair model comparison in research.

---

## Why fixed seeds?

The algorithm uses randomness (parent selection, crossover, mutation). With a random seed every run, you cannot tell whether one model beats another because it is genuinely better or because it got a luckier starting point. By fixing the seed for each run and using the same set of seeds for every experiment, all models face identical random conditions and results become reproducible and comparable.

---

## Files

| File | Purpose |
|------|---------|
| `run_experiment.ps1` | Main experiment runner (PowerShell) |
| `run_experiment.bat` | Thin wrapper — alternative entry point |
| `solutions/experiment_results/` | All output lands here |

---

## How it works

1. Runs `MA-CETSP.exe` 10 times on the chosen instance with seeds `1, 11, 21, 31, 41, 51, 61, 71, 81, 91`.
2. Each run launches **4 parallel islands**, one per model (COX → RSF → GBSA → DEEPSURV), all sharing the same base seed (island *i* gets `seed + i`).
3. The full console output of each run is saved to an individual log file.
4. After all 10 runs, a single summary file is written containing:
   - Per-run table: every island's best value, runtime, and rejection count
   - Aggregate statistics per model: mean, std, min, max best value — average runtime — number of global wins

---

## Usage

```powershell
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instance <index>
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Instance` | `0` | Instance index (see list below) |
| `-Islands` | `4` | Number of parallel islands |
| `-Iterations` | `200` | Max iterations per run |

### Examples

```powershell
# Run on bonus1000 (index 0)
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instance 0

# Run on bubbles1 (index 1) with custom iteration budget
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instance 1 -Iterations 300
```

---

## Output files

All output is written to `solutions/experiment_results/`.

```
experiment_results/
├── instance0_summary.txt          ← main results file (open this)
├── instance0_run1_seed1.log       ← full console output, run 1
├── instance0_run2_seed11.log      ← full console output, run 2
├── ...
└── instance0_run10_seed91.log
```

### Summary file structure

```
------------------------------------------------------------
Run 1  |  seed=1
------------------------------------------------------------
  Island 0  model=COX        best=   448.042  time=  107.509s  rejected=2
  Island 1  model=RSF        best=   442.311  time=  111.697s  rejected=4
  Island 2  model=GBSA       best=   433.252  time=  102.246s  rejected=19
  Island 3  model=DEEPSURV   best=   458.619  time=  117.644s  rejected=2
  --
  GLOBAL BEST  model=GBSA  value=433.252  time=102.246s  wall-clock=117.644s

...

============================================================
 Aggregate Statistics (across 10 runs)
============================================================

  Model: COX
    Best solution  — mean: 449.100  std: 3.241  min: 443.200  max: 455.300
    Avg time (s)   — 108.500s
    Global wins    — 2 / 10

  Model: RSF
    ...
```

---

## Instance index reference

| Index | Instance | Index | Instance |
|-------|----------|-------|----------|
| 0 | bonus1000 | 27 | d493_or2 |
| 1 | bubbles1 | 28 | dsj1000_or2 |
| 2 | bubbles2 | 34 | d493_or10 |
| 3 | bubbles3 | 35 | dsj1000_or10 |
| 4 | bubbles4 | 41 | d493_or30 |
| 5 | bubbles5 | 42 | dsj1000_or30 |
| 6 | bubbles6 | 48 | bonus1000rdmRad |
| 10 | chaoSingleDep | 62 | car_door_25 |
| 21 | team1_100 | 63 | car_door_30 |
| 22 | team2_200 | 64 | car_door_35 |
| 23 | team3_300 | 65 | car_door_40 |
| 24 | team4_400 | 66 | car_door_45 |
| 25 | team5_499 | 67 | car_door_50 |
| 26 | team6_500 | | |

Full list in `include/Defs.hpp` → `FILENAMES`.

---

## Interpreting results for the thesis

- **Mean best value** — primary metric; lower is better (tour length)
- **Std** — consistency; lower means the model is more stable across different random runs
- **Global wins** — how often each model found the best solution across all 4 islands in a run
- **Wall-clock time** — actual elapsed time of the parallel run (~time of the slowest island)

Report as: *mean ± std* over 10 runs, consistent with standard metaheuristics benchmarking practice.
