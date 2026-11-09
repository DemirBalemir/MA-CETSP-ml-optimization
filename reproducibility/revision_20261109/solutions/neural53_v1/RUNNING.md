# Overnight neural/control campaign

Started 8 November 2026 at 01:54 Europe/Istanbul. After 11 validated runs, execution was handed off at 02:03 to the Windows Task Scheduler task MA-CETSP-Neural53-20261108. Current driver PID is recorded in launch.json and progress.json. The initial launch was found still inside a Windows job by a typed external API query; the handoff preserves all completed archives and avoids relying on Codex process ancestry. See scheduler_handoff.json and detachment_verification.json. System-idle sleep prevention is active. This is local Python/C++ work with no LLM or API requests.

Scope: 53 test instances, five common search seeds, corrected DeepSurv, corrected MTLR and unfiltered search: 795 full solver runs. Four concurrent workers. Two short preflight runs passed before scientific execution. The existing verified C++ solver binary and its explicitly counted SOCP numerical fallback are retained.

MTLR uses negative restricted mean lifetime from its entire fitted discrete survival curve. Its calibration and resident inference use the same function. Neural fits use 100 validation-fit and 200 final-fit epochs, and persist training-state audit metadata. Parameters are frozen before new outcomes; no automatic performance-based tuning is performed.

The separate common-sample offline diagnostic completed 105 additional fits and reused the 30 saved neural fits. All nine models now have a comparison on the same 18,115 recorded positive-lifetime observations and outer folds. See analysis_revision/common_model_diagnostics_20261108/common_sample_model_comparison.csv. The retained sampling and temporal limitations still apply.

Live state: progress.json. It contains completed count, active runs, errors and free space. results.csv contains accepted outcomes. terminal.log contains execution messages. first_30_verification.json is written after the initial 30 scientific runs have passed their independent archive/model audit.

Completed outputs are losslessly compressed under archives/. Every archive member is checked by SHA-256 before the expanded solutions, LKH temporary files and full console are removed. Full raw logs remain inside ZIPs; model files, statuses, validated metrics and a console tail remain accessible outside ZIPs. Previous research data and submission packages are untouched.

New work waits if C: free space is below 5 GiB. Unexpected technical/validation failures stop new work and let active peers drain. The run then records halted rather than accepting or silently excluding bad data. This local driver does not invent code fixes or change scientific parameters unattended. Codex is authorized to diagnose and repair technical defects while checking the initial runs.

Creating an empty STOP_AFTER_CURRENT file in this campaign directory pauses new launches after active runs finish. Completed archives can be resumed with the same frozen driver; a partial failed attempt requires diagnosis and preservation before retry. Do not launch a second copy while runner.lock belongs to a live process.

At 795 accepted outcomes the driver automatically writes comparisons.csv (six contrasts with Holm adjustment and descriptive instance-bootstrap intervals). Manuscript/source submission packages are not automatically published or submitted.

Initial supervision completed: first 20 scientific runs passed independent archive/model checks; see first_20_verification.json. The four-worker local driver continues all remaining runs without Codex/model calls.

9 November restart maintenance: the first restart encountered WinError 5 while replacing progress.json during archive revalidation. The initial progress call was outside the worker's exception handler, so one worker exited and its existing archived run was absent from the transient counter. The other three workers completed runs 655–657 before being drained. All 657 completed run folders, statuses and archives were retained; the affected old archive and the three new archives were independently hash-checked. The temporary counter of 656 did not represent a lost scientific result. Original logs, progress, results and launcher are preserved under maintenance/20261109_restart_failure/.

The scheduler launcher now invokes maintenance/io_guard_v1.py, a hash-recorded adapter around the unchanged frozen driver. It retries transient PermissionError in JSON/progress-file operations and stops new work if reporting fails persistently. Six checks passed, including real Windows reader-lock collisions, timeout/error propagation and unchanged hashes for all 26 frozen runtime files; see maintenance/20261109_io_probe/guard_verification.json. The adapter, frozen driver and protocol hashes are recorded per launch. Solver, models, seeds, datasets, protocol and manifest are unchanged. The three runs completed during the three-worker interval are identified in maintenance/io_guard_install.json for timing provenance.

The guarded restart restored all 657 completed results and began runs 658–661 with four workers. Its progress-file sharing collision was retried successfully. The original error remains in terminal.log as history; current errors are reported in progress.json. See maintenance/guarded_resume_verified.json for process/service ancestry and active-run verification.
