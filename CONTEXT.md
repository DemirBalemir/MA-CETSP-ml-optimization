# MA-CETSP session context (2026-09-02)

Read this file, `paper/README.md`, then `paper/main.tex`. The current sources and
CSV definitions supersede older session memories, including `feature-signal-ceiling`
and `paper-reframed-as-diagnostic`. Those historical notes contain stronger claims
and an obsolete kappa definition; do not reinstate them.

## Current research framing

The manuscript is **Assessing the Viability of Pre-Local-Search Surrogate Filtering:
A Diagnostic Study of Survival-Based Offspring Pruning in Memetic CETSP**.
The end-to-end experiment found no statistically detectable quality or runtime
effect under its protocol. This does not prove equivalence or impossibility.
The proposed one-sided gate is an exploratory diagnostic: adverse measurements
can support reconsideration of a tested design; favourable measurements permit
further evaluation without guaranteeing useful filtering. High kappa alone is
not a sufficient rejection rule.

## Read the evidence with its scope

- End-to-end test: 53 held-out instances, five runs, nine models plus control.
  Speed tests use the 47 instances with at least one recorded rejection.
  Ratios average matched run-level ratios; tests use paired per-instance means.
  Island seeds differ by island ID. Runtime is to the stagnation stopping rule,
  under concurrent-island contention; it is not time to a common quality target.
- Offline diagnostics are conditional on recorded admitted positive-lifetime
  solutions. Unadmitted offspring and terminal control survivors were not logged;
  zero-lifetime records are excluded. Reconstructed cohorts are incomplete.
- Five-instance model/oracle diagnostics include development instance bonus1000.
  The 25-instance full-VND calibration includes eight development instances.
  `analysis_la_cetsp/results/instance_manifest.csv` records memberships.
- Model, oracle and lineage diagnostics use GroupKFold by run. Univariate and
  partial-stage scores are direct concordances. The historical population-feature
  multivariate probe uses solution-level KFold and is exploratory.
- The oracle is a fitted empirical reference (0.687), not a formal ceiling.
  Best-of-three feature scores select the model within each instance and are
  optimistic. Only oracle and lineage comparisons have matched clock controls.
- Diversity is a directed Chamfer proxy, not the solver's edit distance.
  The combined lineage representation has 23 features, not 24.
- Full-VND kappa is Spearman(pre-VND recorded-cohort percentile, relative VND gain).
  The corrected 25-instance mean is approximately 0.889, observed direct cost-rank
  concordance 0.560. The historical 0.953 used pooled cost ranks and is obsolete.
  LKH uses the same definition at its own input/output stage: 0.921/0.944/0.820.
- Synthetic sensitivity holds recorded cohort membership and gain marginals
  fixed. Five reassignments per run are used at eleven nominal settings; both
  nominal and realised kappa are reported. Direct rank concordance excludes
  output ties and gives half credit for input ties. It is not rescaled Kendall
  tau-b, a survival C-index, or a universal performance guarantee.
- Negative penalised threshold objective need not mean negative survival gap.
  The gap uses observed (possibly censored) durations as a calibration heuristic.

## Reproduction

Use the scientific Python environment (locally Python 3.10, NumPy, pandas,
SciPy, matplotlib; survival analyses additionally require scikit-survival,
lifelines, torch, pycox). `paper/README.md` describes released inputs and commands.

```text
python tools/test_diagnostic_metrics.py
python tools/export_paper_results.py
python analysis_la_cetsp/make_figures.py
```

Raw-log recalculation, when the corresponding historical logs are available:

```text
python analysis_la_cetsp/pre_predicts_post.py
python analysis_la_cetsp/lkh_character.py
python analysis_la_cetsp/calibrate_recorded_cohorts.py
python tools/export_paper_results.py
python analysis_la_cetsp/make_figures.py
```

`kappa_calibration.py` is retained as a historical implementation and defaults to
`legacy_results/`. Do not use it to regenerate the manuscript. Current calibration
run IDs, counts and hashes are in `kappa_run_manifest.csv`. Changing the raw-log
collection can change selected runs and results; inspect the manifest.

Tables and repeated numeric prose values come from `paper/generated/`; edit their
generator, not the generated numbers. Figure 1 is TikZ inside main.tex. Other
figures are matplotlib PDFs regenerated from CSVs. The `accept` arrow now has
12 mm between the filter and VND; learning arrows route below the search loop.

## Publication and source files

- `paper/main.tex` is the local source of truth; Overleaf is a separate copy.
- `paper/references.bib` is the canonical bibliography; the additions/new files
  are historical leftovers. No new references were introduced in this review.
- The historical 60% recent-reference target was advisor-provided guidance, not
  independently verified as a universal ESWA rule. Check current journal guidance
  before submission rather than asserting it as policy.
- The source author block contains both ORCIDs; no identifier should be invented.
- No journal submission, Overleaf synchronization or new full solver experiment
  is implied by a repository release. State precisely which delivery happened.

## LaTeX

Elsevier cas-dc loads stfloats. Never add dblfloatfix. Full-width floats use
`[!tbp]`; keep declarations near their first citations. Shorten running titles.
Use the packaged bibliography and generated inputs when uploading to Overleaf.
For a local build use `latexmk -pdf main.tex` in paper/, or Tectonic as described
in paper/README.md. Static checks cannot substitute for PDF inspection.

## Code map and future work

`src/Algo.cpp`: training trigger, stopping rule, reject-neutral patience.
`src/Genetic/Population.cpp`: filtering, admission and eviction, lineage snapshot.
`src/LocalSearch/LocalSearch.cpp`: VND and intermediate-stage probes.
`src/Utils/Data.cpp`: log schema. `include/Genetic/List.hpp`: probe fields.
`src/Features/GeometryFeatures.cpp`: deployed geometry (mean edge excludes closing edge).
`src/ML/SurvivalModel.cpp`: native Cox and resident Python inference.

Complete logging of all admission outcomes, population membership and terminal
censoring needs new experiments. Matched-rate random pruning, within-time and
time-plus-feature controls, and time-to-target evaluation remain prospective
work. They are stated limitations, not experiments completed by editorial fixes.
