# November 2026 revision experiments and saved-result reproduction

This directory preserves the verified reporting inputs and frozen reference
code for the revision. It does not contain manuscript drafts, author photos,
cover letters, full process environments, model-weight payloads or raw run ZIPs.

## Distinct studies

| Study | Design | Interpretation |
| --- | --- | --- |
| Historical experiment | Nine survival-model islands plus an unfiltered control; 53 instances and five seeds; 2,650 island outcomes | Historical implementation results; neural training and Cox executable-identity limitations apply. Quality tests use 53 instances, runtime tests 47 active instances. |
| Corrected Cox control | 159 disjoint calibration runs, then Cox/random/unfiltered on 53 instances and five common seeds: 795 evaluation runs | No detectable adjusted cost advantage; Cox took 7.95% longer than random and 6.11% longer than unfiltered search under concurrent execution. |
| Corrected neural re-evaluation | Existing DeepSurv and MTLR implementations versus a fresh unfiltered control; 53 instances and five common seeds: 795 runs | No detectable cost advantage. Runtime increased 5.28% and 4.02%; six-test Holm p=0.004252 and 0.045651. |
| Common offline diagnostic | Nine families, 18,115 positive-lifetime admitted eviction records, five instances, three run-held-out folds; 30 reused neural fits plus 105 other fits | Equal-instance mean C-index 0.526–0.569; not an additional solver experiment or a nine-model within-time analysis. |
| Within-time diagnostic | One fixed ridge-Cox model, 29,908 selected records, 80 historical control runs, 16 test instances | Separate descriptive geometry/time analysis; selected-record and limited-model scope remain. |

Training repair and scoring are separate changes. The invalid validation-loss
early-stopping path was removed; fixed epochs are 100 for offline/temporary
fits and 200 for final online fits. Each neural arm has 175 trained runs and
90 runs that stopped before training, with zero training failures. The 350
saved final networks were audited for changed weights and BatchNorm updates.
Separately, MTLR uses `negative_rmst_step_v1`, the negative step integral to
the last learned cut. Rescoring the same 15 saved networks changed mean
concordance from 0.493444 to 0.533342, without retraining those networks.

The old 17,568-record model table and the updated common sample differ. The
older oracle value 0.687 is an empirical reference on a different selected
sample, not a universal upper bound. Historical neural equivalence results do
not establish equivalence of corrected models. Runtime comparisons concern
the recorded execution/stopping regime, including the conic-refinement fallback.

## Reproduce saved results without running an experiment

From the repository root, with Python 3.10 and the packages in
`requirements-saved-results.txt`:

```console
python reproducibility/revision_20261109/reproduce_revision_results.py
```

This reads existing CSV/JSON/prediction files and checks:

- all six corrected-neural cost/runtime contrasts, bootstrap intervals and Holm p-values;
- all 135 saved-prediction fold concordances and the common nine-family table;
- generated/rejected/admitted counts and SOCP-state totals;
- the six separate final Cox contrasts.

It writes verification outputs under `rerun/`. It does not fit a model or
launch MA-CETSP/Gurobi. Those generated verification outputs are ignored by Git.
The supplied requirements record the environment used for this publication check.

## Locate the evidence and code

- `solutions/neural53_v1/`: frozen protocol, 795-row manifest/results, comparisons,
  compact per-run validated metrics, training-audit summaries, exact deployed
  Python scripts and driver/maintenance snapshots.
- `analysis_revision/neural_diagnostic_rerun_20261108/`: selected offline inputs,
  input selection and run-held-out fold assignments, predictions and fit audits.
- `analysis_revision/common_model_diagnostics_20261108/`: seven other families'
  saved fold predictions/results and the updated common table.
- `analysis_revision/mtlr_score_correction_20261108/`: paired saved-network scores.
- `legacy_20261107/`: final Cox records and historical precision, within-time and
  traceability materials. Earlier superseded data are labelled separately.
- `revision_audit/`: compact verification and operational totals.
- `analysis_la_cetsp/`: older diagnostic tables/manifests and source snapshots;
  their sample/protocol limits remain as described above.
- `../../tools/`: current campaign and diagnostic drivers. These are reference
  code for conducting studies, not commands needed to verify saved results.
- `../../ml/scripts/`: corrected training and shared MTLR scoring code.
- `../socp_diagnostic_20261107/`: separate saved numerical subproblem.

Frozen manifests retain the original execution paths and hashes as provenance.
Some campaign drivers have Windows-specific paths; recreating full studies
requires configuring an independent environment, licensed Gurobi and external
LKH. Do not edit the frozen records to make a new run appear identical.
Full raw archives and network weights remain with the authors and are available
on request. The older historical console-log release linked in the repository
README does not contain the new neural campaign's raw archives.

`.gitattributes` preserves byte identity in `reproducibility/`; this matters for
the stored source hashes and original CSV/JSON inventories. The archived
`README.txt` and `SHA256.csv` are retained unchanged. `PUBLIC_SOURCE_MANIFEST.json`
records the source archive members and byte hashes of the extracted public files.

## Publication check of the editable controller

The current top-level controller adds bounded retries for transient Windows
report-file sharing failures. This is an I/O-only maintenance change discovered
with synthetic unit tests, not a new solver run or training experiment. All
frozen deployed snapshots in this directory remain unchanged. See
`../../tests/README.md` for the code checks.
