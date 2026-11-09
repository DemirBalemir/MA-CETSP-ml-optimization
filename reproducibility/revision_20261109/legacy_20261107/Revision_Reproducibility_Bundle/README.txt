Revision reproduction reference - 7 November 2026

Read Supplementary Note S2 before interpreting these materials.
control/final_* identify the final 795 validated evaluation runs and their results.
Use these files consistently for reproduction of the reported comparisons.

To recalculate the six contrasts without a solver, install numpy, pandas and scipy,
then run python reproduce_results.py. Outputs go into rerun/; bundled inputs are
unchanged. reference_scripts/ are the original repository-layout analysis and
execution references, not a portable one-click experiment launcher. Do not run
the experiment-control scripts merely to reproduce the tables.

historical/ contains the original parsed input, evidence-labelled reference copy and field provenance.
All 2650 best costs, runtimes and rejection counts match the post-join summaries.
Literal summaries support 2572 best-iteration and 2563 effective-iteration values.
A further 5 best and 9 effective values have separately labelled stopping-counter
corroboration. 73 best and 78 effective values remain unresolved. The reference
copy records 115 supported changes (36 best and 79 effective), with evidence labels.
Cost/runtime are unchanged exactly. Some unrounded rejection means change slightly;
all displayed two-decimal rejection rates and reported activation summaries remain
unchanged by the verified corrections. Unresolved fields remain a limitation. The original parsed CSV
is preserved. This does not reconstruct every historical event or identify the
historical executable. Full original logs are available from the authors on request.
The trace scripts expect to run from the repository root with those logs present.

The public repository can contain older revisions. This bundle is the versioned
submission reference. No license file, solver executable or full raw log is bundled.
