"""
threshold_utils.py
------------------
Adaptive rejection-threshold search for survival-analysis ML filters.

Problem
-------
A fixed percentile (e.g. 80th) is independent of instance difficulty and
model quality.  On easy instances the model may discriminate well at 60 %,
while on a hard instance 90 % may be needed to get any useful signal.

Approach
--------
We use a small held-out validation split (20 % of logged solutions) on which
the model has NEVER been fitted.  For each candidate rejection percentile we
compute an objective that rewards two things simultaneously:

    objective = rejection_rate * survival_gap

    rejection_rate  = fraction of val solutions that would be filtered out
                      (we want this HIGH — more saved VND calls)
    survival_gap    = (mean_survival(accepted) - mean_survival(rejected))
                      / mean_survival(all)
                      (we want this HIGH — accepted solutions genuinely live
                       longer, so we are filtering the right ones)

The percentile that maximises this objective on the validation set is then
applied to the FULL-data risk scores (from the final model trained on 100 %
of data) to obtain the saved threshold.

Fallback
--------
If the dataset is too small to split reliably (< MIN_SAMPLES), or if the
validation set yields degenerate results, the function falls back to
DEFAULT_PERCENTILE (80).
"""

from __future__ import annotations

import numpy as np

# ── tuneable constants ────────────────────────────────────────────────────────
CANDIDATE_PERCENTILES = list(range(50, 96, 5))   # 50, 55, 60, …, 95
VAL_FRACTION          = 0.20                      # 20 % held-out
MIN_SAMPLES           = 30                        # fallback below this
DEFAULT_PERCENTILE    = 80

# Soft-cap objective (replaces the old `rejection_rate * survival_gap`).
# The old objective multiplied by rejection_rate, but on the validation set
# rejection_rate is mechanically (100-pct)/100 — a built-in pull toward LOW
# percentiles (over-rejection) that ignores real model quality.  Instead we
# reward discrimination (survival_gap) and only allow rejection up to a target,
# penalising anything beyond it.  Percentile RANGE is NOT restricted (so RSF/
# GBSA, which need lower percentiles, keep all candidates) — only the rejection
# *rate* is softly capped.
TARGET_REJECT_RATE = 0.20   # aim to reject ~20 % (safe margin below the 30 % cap)
OVER_PENALTY       = 2.0    # weight on rejection beyond the target
# ─────────────────────────────────────────────────────────────────────────────


def _soft_cap_objective(rejection_rate: float, survival_gap: float) -> float:
    """Discrimination reward, ramped in up to TARGET_REJECT_RATE, then penalised.

    - `ramp` (0→1 as rejection goes 0→target) avoids collapsing to ~zero
      rejection: some filtering is needed to earn the full survival_gap credit.
    - `over` penalises rejecting beyond the target, which is what previously
      drove COX/SSVM/GBSA to ~30-46 % rejection and the quality degradation.
    """
    ramp = min(1.0, rejection_rate / TARGET_REJECT_RATE) if TARGET_REJECT_RATE > 0 else 1.0
    over = max(0.0, rejection_rate - TARGET_REJECT_RATE)
    return survival_gap * ramp - OVER_PENALTY * over
# ─────────────────────────────────────────────────────────────────────────────


def search_optimal_percentile(
    val_risks:      np.ndarray,
    val_survival:   np.ndarray,
    candidates:     list[int] | None = None,
) -> int:
    """
    Given unbiased (validation-set) risk scores and their survival times,
    return the percentile that maximises  rejection_rate * survival_gap.

    Parameters
    ----------
    val_risks     : 1-D array, higher value = model predicts shorter survival
    val_survival  : 1-D array, actual survival durations (iterations alive)
    candidates    : list of integer percentiles to try; defaults to
                    CANDIDATE_PERCENTILES (50, 55, …, 95)

    Returns
    -------
    best_percentile : int
    """
    if candidates is None:
        candidates = CANDIDATE_PERCENTILES

    mean_all = float(val_survival.mean())
    if mean_all < 1e-12:
        return DEFAULT_PERCENTILE

    best_obj = -np.inf
    best_pct = DEFAULT_PERCENTILE

    for pct in candidates:
        threshold = np.percentile(val_risks, pct)

        accepted = val_risks <= threshold
        rejected = val_risks > threshold

        n_acc = int(accepted.sum())
        n_rej = int(rejected.sum())
        if n_acc == 0 or n_rej == 0:
            continue

        rejection_rate = n_rej / len(val_risks)

        mean_acc = float(val_survival[accepted].mean())
        mean_rej = float(val_survival[rejected].mean())

        # normalised gap: > 0 means accepted survive longer (good)
        survival_gap = (mean_acc - mean_rej) / mean_all

        objective = _soft_cap_objective(rejection_rate, survival_gap)

        if objective > best_obj:
            best_obj  = objective
            best_pct  = pct

    return best_pct


def compute_threshold(
    full_risks:     np.ndarray,
    val_risks:      np.ndarray,
    val_survival:   np.ndarray,
    candidates:     list[int] | None = None,
) -> tuple[float, int, float]:
    """
    Full pipeline: find the optimal percentile on the validation set, then
    apply it to the full-data risk scores to get the saved threshold.

    Parameters
    ----------
    full_risks    : 1-D array of risk scores from the FINAL model on ALL data
    val_risks     : 1-D array of risk scores on the held-out validation set
    val_survival  : 1-D array of survival times on the held-out validation set
    candidates    : candidate percentiles (default: 50–95 step 5)

    Returns
    -------
    (threshold, best_percentile, objective_score)
    """
    if len(val_risks) < MIN_SAMPLES // 5:
        # Validation set too small — skip search, use default
        pct = DEFAULT_PERCENTILE
        obj = float("nan")
    else:
        pct = search_optimal_percentile(val_risks, val_survival, candidates)
        # Re-compute objective for logging
        threshold_val = np.percentile(val_risks, pct)
        accepted  = val_risks <= threshold_val
        rejected  = val_risks >  threshold_val
        mean_all  = float(val_survival.mean()) or 1.0
        mean_acc  = float(val_survival[accepted].mean()) if accepted.any() else 0.0
        mean_rej  = float(val_survival[rejected].mean()) if rejected.any() else 0.0
        rej_rate  = float(rejected.mean())
        obj       = _soft_cap_objective(rej_rate, (mean_acc - mean_rej) / mean_all)

    threshold = float(np.percentile(full_risks, pct))
    return threshold, pct, obj
