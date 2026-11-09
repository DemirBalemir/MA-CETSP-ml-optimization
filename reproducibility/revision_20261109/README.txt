MA-CETSP revision reporting reference - 9 November 2026

Experiments are distinct: historical 2650 island outcomes; legacy Cox/random/
unfiltered 795 runs; corrected neural DeepSurv/MTLR/fresh-unfiltered 795 runs.
The common offline sample has 18115 rows, 30 reused neural fits and 105 additional
other-family fits. Training repair and MTLR operational scoring are separate.
The older neural protocol retains its last-cut score description as history;
common_model_diagnostics_20261108/protocol.json and mtlr_score_correction_20261108/
identify the revised score used in the common table. Read Supplementary Note S2.

Result reproduction (no solver, no model fitting): install numpy, pandas, scipy,
scikit-survival, then run python reproduce_revision_results.py from any folder.
This recomputes six neural contrasts, all 135 common-sample fold C-indices and
the six old Cox contrasts. New audit outputs go to rerun/. The inherited legacy
reproducer writes its own rerun subfolder. Python 3.10 was used for research;
version metadata is in the frozen diagnostic and online protocols.

Other source files are reference snapshots, often expecting the original repository
layout, raw logs, trained weights or Gurobi. Do not run campaign/training/reference
scripts merely to reproduce published tables. No solver, Gurobi license or full
raw archive payload is bundled. Full raw per-run archives and network weights are
retained locally and available from the authors on request. Runtime binary hashes
identify the executed versions; the binary itself is not included. Paths in frozen
manifests identify the original local execution and are preserved as provenance.
Compact audit records omit full process environments. This revision did not rehash
every byte of every raw archive; see revision_audit/evidence_verification.json.

legacy_20261107/ preserves the previously supplied reporting bundle, including
historical auxiliary-field evidence flags and unresolved-field limits. The older
model/oracle/kappa result CSVs, manifests and source references are now included
under analysis_la_cetsp/; their exact older selections are not all reconstructed.
The updated model figure uses the new common table; the other panel retains older,
different samples. No remote default-branch identity/publication is assumed.
