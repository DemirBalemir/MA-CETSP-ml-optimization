# Assessing the Viability of Pre-Local-Search Surrogate Filtering: A Diagnostic Study of Survival-Based Offspring Pruning in Memetic CETSP

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gurobi](https://img.shields.io/badge/Solver-Gurobi%2012.0-orange.svg)](https://www.gurobi.com/)
[![LKH](https://img.shields.io/badge/TSP-LKH%202.0.11-lightgrey.svg)](http://akira.ruc.dk/~keld/research/LKH/)

This repository provides the official implementation, experimental benchmarks, and diagnostic reproducibility suite for the research paper:

> **Assessing the Viability of Pre-Local-Search Surrogate Filtering: A Diagnostic Study of Survival-Based Offspring Pruning in Memetic CETSP**  
> *Demir Balemir and Deniz Cantürk*  
> Department of Computer Engineering, TED University, Ankara, Türkiye  
> Submitted to *Expert Systems with Applications* (Elsevier)

The study investigates whether an offspring's pre-local-search features can support useful pruning decisions before an expensive local-search stage in a memetic algorithm for the **Close-Enough Traveling Salesman Problem (CETSP)**. Evaluating nine survival models against a concurrent in-run no-filter control on 53 benchmark instances with five seeds per configuration (2,650 island-runs), the experiment finds **no statistically detectable difference in solution quality or total runtime**.

To explain this outcome, the paper presents a set of mechanistic diagnostic analyses connecting target predictability, rank transmission through local search, and discrimination beyond temporal drift, formalising a reusable **one-sided viability gate** to screen surrogate-filtering designs before deployment.

---

## Key Empirical & Diagnostic Findings

1. **Controlled End-to-End Evaluation (53 Instances $\times$ 5 Seeds):**
   - Solution quality across all nine survival models differs from the concurrent no-filter control by only $-0.009\%$ to $+0.049\%$. Under two-sided Wilcoxon signed-rank tests over per-instance means, no model is statistically distinguishable from the control ($p \ge 0.31$).
   - Total runtime speedups range from $1.008$ to $1.084$, none of which are statistically significant ($p \ge 0.15$).
   - The filter was active in 1,533 runs with a median active window of 1,083 iterations (52% of run length), removing tens of thousands of local-search calls. Limited filter activity alone does not account for the null result.

2. **Limited Feature & Model Discrimination:**
   - Offline cross-validated concordance ($C$) across all nine survival model families lands between $0.495$ and $0.544$ (near chance $0.500$). Model flexibility (from linear Cox to deep networks and tree ensembles) provides no consistent gain.
   - Deployed geometry descriptors are effectively constant within an instance. Scale-free population-relative cost features reach $C \approx 0.53 - 0.54$. Directed Chamfer distance diversity proxies score $C \approx 0.48$.
   - While lineage features score $C = 0.598$, the birth iteration alone on the same runs scores $C = 0.613$. Apparent feature discrimination is driven by temporal search progress rather than contemporaneous candidate differentiation.

3. **Empirical Oracle Reference:**
   - An oracle given post-local-search outcome information reaches $C = 0.687$, demonstrating that even with the exact cost produced by local search, future population survival is not deterministically predictable.
   - Adding all pre-local-search features to the oracle increases concordance by only $0.001$ (from $0.687$ to $0.688$).

4. **Rank Attenuation & The Contraction Mechanism:**
   - Local search (VND) acts as a non-linear contraction: lower-ranked pre-VND candidates receive substantially larger relative improvements ($\kappa_{\mathrm{VND}} = 0.889$ across 25 diagnostic instances).
   - This compensatory improvement scrambles the entering cost order, resulting in an observed direct cost-rank concordance of only $C_{\mathrm{rank}} = 0.560$.

---

## The Three-Check Viability Gate

The paper formulates these diagnostic measurements into a reusable, low-cost screening procedure that can be evaluated on instrumented search logs *before* deploying an online surrogate filter:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        THE THREE-CHECK VIABILITY GATE                  │
├────────────────────────────────────────────────────────────────────────┤
│ Check 1: Is the label reachable?                                      │
│          Fit an empirical oracle model using recorded post-step data.  │
│          ► CETSP Oracle: C = 0.687 (far from deterministic survival)   │
├────────────────────────────────────────────────────────────────────────┤
│ Check 2: Does the pipeline preserve ordering?                         │
│          Compute contraction coefficient κ = Spearman(r_in, gain)     │
│          and input-output cost-rank concordance C_rank.                │
│          ► CETSP Full-VND: κ = 0.889, C_rank = 0.560                   │
│            (compensatory gains attenuate entering cost order)          │
├────────────────────────────────────────────────────────────────────────┤
│ Check 3: Does the feature set beat the clock?                          │
│          Evaluate features against a baseline of the iteration index. │
│          ► CETSP Features vs Clock: C = 0.605 vs C = 0.613             │
│            (pooled discrimination reflects temporal drift)             │
└────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **One-Sided Interpretation:**  
> The gate is one-sided. Adverse evidence across the checks (weak label reachability, strong rank contraction, failure to beat the temporal drift clock) justifies a decision **not to proceed** with the tested filtering design. Conversely, passing the gate merely justifies further evaluation without establishing that filtering will yield a net benefit.

---

## System Architecture

The solver embeds surrogate filtering into the state-of-the-art MA-CETSP algorithm (Lei & Hao, 2024):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MEMETIC SEARCH LOOP                             │
│                                                                        │
│  Population ──► Select Parents ──► Crossover ──► Mutation             │
│   (μ = 20)                                           │                 │
│      ▲                                               ▼                 │
│      │                                         ┌───────────┐           │
│      │               ┌─── [Reject: M(x) > θ] ──┤ ML Filter │           │
│      │               │   (Bypasses VND;        └─────┬─────┘           │
│      │               │    neutral stagnation)        │                 │
│      │               │                               ▼ [Accept]        │
│      │               │                         ┌───────────┐           │
│      │               ▼                         │    VND    │           │
│      └────── Insert & Manage ◄─────────────────┤ Pipeline  │           │
│             (Fitness = Rank_val + β·Rank_div)  └───────────┘           │
└───────────────────────┬────────────────────────────────────────────────┘
                        │ features, birth/death
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        ONLINE LEARNING LOOP                            │
│                                                                        │
│  Survival Dataset ──► Training Trigger ──► Fit Model & Calibrate θ    │
│                       (iter ≥ 0.20T &      (Held-out 20% validation;   │
│                        evictions ≥ 100)     target rejection r* = 0.20)│
└────────────────────────────────────────────────────────────────────────┘
```

### Components
- **Pre-Local-Search Features (10):** Pre-VND tour cost, edge length mean and variance, bounding box width, height, and area, centroid coordinates $(x, y)$, sum of point-to-centroid distances, and interior turning angle variance.
- **Nine Survival Model Families:**
  - *Cox Proportional Hazards (Cox PH)*: Semi-parametric linear baseline, implemented natively in C++.
  - *ElasticNet Cox*: Regularised linear Cox model for collinear feature sets.
  - *Random Survival Forests (RSF)*: Ensembled survival trees via bagging.
  - *Gradient Boosting Survival Analysis (GBSA)*: Stagewise Cox partial-likelihood boosting.
  - *DeepSurv*: Multilayer neural network (2 hidden layers, 64 units, batch norm, dropout).
  - *Multi-Task Logistic Regression (MTLR)*: Neural network discretising the time axis into logistic models.
  - *Survival Support Vector Machines (SSVM)*: Structured ranking formulation for censored pairs.
  - *Weibull Accelerated Failure Time (Weibull AFT)*: Parametric log-survival regression.
  - *$k$-NN Survival*: Non-parametric risk estimator using the 15 nearest neighbours.
- **Resident Process IPC:** Non-Cox models are served by a resident Python process communicating over standard anonymous I/O pipes via single-line JSON requests, removing interpreter start-up overhead.
- **Adaptive Training Trigger:** Fires when both $\text{iter} \ge 0.20 \cdot T$ and $\#\{\text{observed evictions}\} \ge 100$, with a hard fallback at $0.50 \cdot T$. The trigger does not alter or reset the stagnation counter.
- **Threshold Calibration:** Selected on a held-out 20% validation split by maximising regularised survival gap:
  $$J(r, g) = g \cdot \min\left(1, \frac{r}{r^*}\right) - \lambda \max(0, r - r^*)$$
  with target rejection rate $r^* = 0.20$ and penalty $\lambda = 2$.
- **Rolling Rejection-Rate Cap:** Suspends the filter for 50 iterations if the empirical rejection rate exceeds 30% over a sliding window of 50 candidates.
- **Parallel Multi-Island Portfolio:** 10 concurrent threads (9 ML models + 1 in-run unfiltered control), seeded with $\sigma + i$. No migration occurs between islands, providing an unpolluted in-run control under identical hardware conditions.
- **Stagnation Neutrality:** Rejected offspring bypass VND entirely and neither increment nor reset the non-improving stagnation counter $\pi$.

---

## Experimental Setup & Full Results

### Execution Environment
- **CPU:** Intel Core i7-12700H (14 cores / 20 threads, 2.7 GHz, 32 GB DDR4 RAM)
- **OS:** Windows 11 Pro (26200)
- **Compiler:** MSVC 2022 (C++17, `/O2`), CMake 4.2
- **Solvers:** Gurobi 12.0.3, LKH 2.0.11
- **Python:** 3.10.9 (scikit-survival 0.25.0, lifelines 0.30.0, scikit-learn 1.7.2, PyTorch 2.11 CPU, pycox 0.3.0)

### Benchmark Results (53 Instances $\times$ 5 Seeds = 2,650 Island-Runs)

| Model | Class | Mean Best Cost | Degradation (%) | Speedup | Reject (%) | Wilcoxon Quality ($p$) | Wilcoxon Speed ($p$) |
|---|---|---|---|---|---|---|---|
| **Baseline (no ML)** | Control | 905.956 | +0.000% | 1.000 | 0.00% | — | — |
| **RSF** | Tree ensemble | 906.253 | -0.009% | 1.008 | 2.01% | 0.944 ($n=39$) | 0.150 ($n=47$) |
| **KNN** | Non-parametric | 905.867 | -0.003% | 1.040 | 5.03% | 0.313 ($n=38$) | 0.949 ($n=47$) |
| **ElasticNet** | Penalised linear | 906.214 | +0.001% | 1.055 | 5.38% | 0.640 ($n=39$) | 0.841 ($n=47$) |
| **WeibullAFT** | Parametric AFT | 906.025 | +0.002% | 1.051 | 4.85% | 0.557 ($n=38$) | 0.657 ($n=47$) |
| **GBSA** | Gradient boosting | 906.026 | +0.009% | 1.019 | 4.70% | 0.361 ($n=41$) | 0.208 ($n=47$) |
| **DeepSurv** | Deep network | 906.048 | +0.011% | 1.030 | 4.21% | 0.572 ($n=40$) | 0.290 ($n=47$) |
| **MTLR** | Deep network | 905.969 | +0.014% | 1.033 | 4.42% | 0.888 ($n=36$) | 0.498 ($n=47$) |
| **SSVM** | Kernel / ranking | 906.248 | +0.018% | 1.084 | 6.51% | 0.464 ($n=40$) | 0.236 ($n=47$) |
| **Cox PH** | Linear | 906.415 | +0.049% | 1.041 | 8.24% | 0.567 ($n=39$) | 0.857 ($n=47$) |

*Degradation averages matched run-level cost ratios $100(f_{mi} / f_{0i} - 1)$. Wilcoxon tests use per-instance mean differences (53 instances for quality; 47 instances with rejections for speed). All $p \ge 0.05$.*

---

## Repository Structure

```
MA-CETSP/
├── Article.pdf                   # The complete manuscript PDF (19 pages, cas-dc format)
├── src/                          # C++ Solver implementation
│   ├── Algo.cpp                  # Main memetic search loop & stopping logic
│   ├── Genetic/                  # Population management, crossover, mutation
│   ├── LocalSearch/              # VND, greedy adjustment, LKH interface
│   ├── ML/                       # SurvivalModel C++ bridge & native Cox
│   └── Features/                 # Geometry & cost feature extractors
├── include/                      # C++ header files & global parameters (Defs.hpp)
├── ml/
│   ├── scripts/                  # Model training, prediction, threshold calibration
│   └── models/                   # Serialised models and training logs
├── datasets/                     # 68 CETSP benchmark instances (Mennell & Car-door)
├── solutions/                    # Best-known tours and solution logs
├── external/
│   └── LKH-2.0.11/               # Bundled Lin-Kernighan-Helsgaun solver
├── analysis_test/                # Parsed end-to-end experiment records (2,650 runs)
│   ├── parsed/                   # Run summaries, rejections, thresholds, convergence
│   └── results/                  # Model comparisons, Wilcoxon tests, threshold stats
├── analysis_la_cetsp/            # Diagnostic analyses & retained measurement CSVs
│   ├── make_figures.py           # Generates all manuscript figures
│   ├── oracle_ceiling.py         # Empirical oracle reference
│   ├── lkh_character.py          # LKH rank-gain contraction analysis
│   └── calibrate_recorded_cohorts.py # Full-VND copula sensitivity analysis
├── tools/                        # Reproduction & verification tools
│   ├── export_paper_results.py   # Rebuilds paper tables and LaTeX input fragments
│   ├── test_diagnostic_metrics.py# Unit tests for rank concordance metrics
│   ├── texcheck.py               # LaTeX document integrity checker
│   └── prepare_anonymous_submission.py # Double-anonymized submission generator
├── paper/                        # Complete LaTeX manuscript sources
│   ├── main.tex                  # Local source of truth (cas-dc template)
│   ├── references.bib            # Canonical bibliography (54 entries)
│   ├── figures/                  # PDF and PNG figures (Figures 1-6)
│   ├── generated/                # Exported LaTeX input fragments
│   └── submission/               # Cover letter, title page, editor info
├── run_experiment.ps1            # Automated PowerShell experiment runner
├── CMakeLists.txt                # CMake build configuration
└── README.md                     # This file
```

---

## Installation & Build Instructions

### Prerequisites
- **C++17 Compiler:** MSVC 2022 (Windows) or GCC 10+ / Clang 12+ (Linux)
- **Build System:** CMake 3.17+
- **Exact MIP Solver:** [Gurobi Optimizer](https://www.gurobi.com/) (tested with 12.0.3). Ensure `GUROBI_HOME` points to your installation.
- **Python 3.10+:** Required for survival model training and figure reproduction.

### Python Dependencies
```bash
# Minimal dependencies for result reproduction and plotting:
pip install -r paper/requirements-results.txt

# Full dependencies for online ML model training:
pip install scikit-survival lifelines scikit-learn torch pycox torchtuples numpy pandas scipy matplotlib
```

### Building the Solver
From the repository root:

```powershell
# Windows PowerShell
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
```

```bash
# Linux / macOS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable `MA-CETSP.exe` (or `MA-CETSP`) is generated in `build/Release/`.

---

## Running the Solver & Experiments

### Command-Line Usage
Run from `build/Release/` (default paths resolve relative to this directory):

```powershell
Set-Location build/Release

# 1. Single-island unfiltered control on instance index 1 (bubbles1):
.\MA-CETSP.exe -i 1 -s 1 -r 5000 --islands 1 --ml_enable 0

# 2. Single ML model island (e.g., Cox Proportional Hazards):
.\MA-CETSP.exe -i 1 -s 1 -r 5000 --islands 1 --ml_model COX --ml_enable 1

# 3. Full multi-island experiment (9 ML models + 1 in-run control):
.\MA-CETSP.exe -i 1 -s 1 -r 5000 --islands 10
```

### Batch Experiment Execution
To replicate the 10-island seeded benchmark:

```powershell
# Run on instance 0 (bonus1000) across all 5 seeds with 10 parallel islands:
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instances 0 -Islands 10 -Iterations 5000

# Run across multiple instances:
powershell -ExecutionPolicy Bypass -File run_experiment.ps1 -Instances 0,1,2,3 -Islands 10 -Iterations 5000
```
Console logs and summary files are written to `solutions/experiment_results/`.

### CLI Options Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `-i` | `int` | `0` | Instance index into `FILENAMES` in `Defs.hpp` |
| `-s` | `int` | `0` | Base seed (island $k$ receives $s + k$) |
| `-r` | `int` | `200` | Maximum iteration budget $T$ |
| `-p` | `int` | `20` | Population size $\mu$ |
| `-b` | `float` | `0.96` | Diversity balance weight $\beta$ in fitness function |
| `-d` | `float` | `5.0` | Minimum edit distance threshold for admission |
| `-n` | `int` | `50` | Candidate neighbourhood size for local search |
| `--islands` | `int` | `1` | Number of concurrent search threads (1–10) |
| `--ml_model` | `string` | `MTLR` | Model: `COX`, `RSF`, `GBSA`, `DEEPSURV`, `SSVM`, `WEIBULLAFT`, `KNN`, `ELASTICNET`, `MTLR` |
| `--ml_enable` | `int` | `1` | Enable ML surrogate filtering (`1` = yes, `0` = no) |
| `--python_exe` | `string` | `python` | Path to Python interpreter for resident worker |
| `--scripts_dir`| `string` | `../../ml/scripts/` | Path to ML Python scripts directory |
| `--lkh_exe` | `string` | `../../external/LKH-2.0.11/LKH.exe` | Path to compiled LKH executable |
| `--lkh_tmp` | `string` | `../../external/LKH-2.0.11/tmp/` | Path to temporary scratch directory for LKH |

---

## Reproducing Article Results

All tables, Wilcoxon signed-rank tests, diagnostic metrics, and publication figures can be reconstructed directly from the retained measurement CSVs without rerunning the C++ solver:

```bash
# 1. Run unit tests for concordance metrics:
python tools/test_diagnostic_metrics.py

# 2. Export paper summary tables and LaTeX macros:
python tools/export_paper_results.py

# 3. Regenerate all manuscript figures (Figures 1-6):
python analysis_la_cetsp/make_figures.py

# 4. Verify LaTeX document structure, environments, and citations:
python tools/texcheck.py
```

### Compiling the Manuscript
```bash
cd paper
latexmk -pdf main.tex
```

Or compile with Tectonic:
```bash
tectonic -X compile paper/main.tex --outdir output/pdf --untrusted
```

---

## Citation

```bibtex
@article{balemir2026assessing,
  title   = {Assessing the Viability of Pre-Local-Search Surrogate Filtering: 
             A Diagnostic Study of Survival-Based Offspring Pruning in Memetic {CETSP}},
  author  = {Balemir, Demir and Cant{\"u}rk, Deniz},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}

@article{lei2024effective,
  title   = {An effective memetic algorithm for the close-enough traveling salesman problem},
  author  = {Lei, Zhenyu and Hao, Jin-Kao},
  journal = {Applied Soft Computing},
  volume  = {153},
  pages   = {111266},
  year    = {2024},
  doi     = {10.1016/j.asoc.2024.111266}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Bundled external software (LKH 2.0.11) is subject to its original academic license terms.

