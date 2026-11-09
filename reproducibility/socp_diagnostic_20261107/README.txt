SOCP diagnostic bundle - Supplementary Note S1

Scope: one saved 37-point model, not an end-to-end filtering rerun.
Original LP SHA256:
f16cd27735893959309fb6096f3529da5229233f44c60e2d6fa7c64162bf79f1

Requirements: Windows, 64-bit Python 3.10+ (standard library only), local
Gurobi 12.0.3 installation, and your own valid license. No license is included.
GUROBI_HOME defaults to C:\gurobi1203\win64; override it if needed.
Gurobi's normal license discovery or GRB_LICENSE_FILE is used.

Run: python reproduce.py
This runs all nine cases with a 60-second cap per solve. Presolve=0 can use
the full minute and does not establish optimality. For a small reproduction:
python reproduce.py baseline coordinate_bounds

New outputs go into rerun/. Archived evidence in this directory is preserved.
summary.csv consolidates all nine published cases; results.json contains the
first eight and results_extra.json contains the tight-tolerance case.
Objective and geometry are checked from variables retrieved by name.
Timing is descriptive and not a solver-performance benchmark.
LP serialization is not a reconstruction of the historical in-memory model.
See Supplementary Note S1 for geometric residuals and interpretation limits.
