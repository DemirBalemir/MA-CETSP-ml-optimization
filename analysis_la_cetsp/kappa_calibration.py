"""
LEGACY ANALYSIS, superseded by calibrate_recorded_cohorts.py.
This historical implementation uses pooled cost ranks for kappa and rescaled
Kendall tau-b for agreement. These are NOT the manuscript's current metrics.
The original interpretation below is retained for provenance, not endorsed.
Its output defaults to legacy_results and must not replace results/*.csv.

CALIBRATION CURVE: how much surrogate performance is reachable at a given
local-search contraction coefficient kappa?

Why this exists
---------------
Section 6.5 shows that the local search contracts: the worse an offspring enters,
the more it gains, with Spearman(input cohort rank, relative gain) = kappa ~ 0.9.
That explains why the filter fails here. It does not, on its own, license a
threshold — we have exactly one case, and "kappa < 0.3 means go" would be
extrapolation from n = 1.

Rather than assert a cut-point we derive the mapping. Holding the real cost and
gain distributions fixed and varying ONLY the coupling between input rank and
gain, we measure how well the pre-local-search state can rank the
post-local-search outcome as a function of kappa. The reader then reads their own
cut-point off the curve given their own cost/benefit trade-off.

Design
------
For each run we take the real logged offspring: pre-VND cost p_i, post-VND cost
q_i, birth and death iterations. The real relative gain is g_i = (p_i - q_i)/p_i.

For a target kappa we REASSIGN the observed gains among offspring so that
Spearman(rank(p), g) = kappa, using a Gaussian copula. The multiset of gains is
unchanged — only who receives which gain changes — so gain magnitudes stay
exactly as observed and kappa is the single quantity being varied.

Synthetic outcome:      q_i' = p_i * (1 - g_i')
Target to predict:      the offspring's post-cost rank within its LIVE cohort
                        (the value-rank half of the eviction fitness, which is
                        what actually decides survival)
Predictor:              pre-VND cost, the strongest legitimate single feature
Score:                  C = (tau + 1) / 2, the concordance probability implied
                        by Kendall's tau between predicted and realised ordering

Validation
----------
At each instance's OWN observed kappa the simulation should reproduce that
instance's observed pre->post in-cohort agreement. The script reports both so the
curve can be checked rather than trusted.

Usage
-----
python analysis_la_cetsp/kappa_calibration.py --max_instances 25
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, norm, rankdata

KAPPAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_run(run_dir: Path, max_files: int):
    """Real per-offspring records: pre cost, post cost, birth, death."""
    P, Q, B, D = [], [], [], []
    files = sorted(os.scandir(run_dir), key=lambda f: f.name)
    files = [f for f in files if f.name.startswith("sol-")][-max_files:]
    for f in files:
        try:
            d = json.loads(Path(f.path).read_text())
        except Exception:
            continue
        p, q = d.get("pre_vnd_cost"), d.get("post_vnd_cost")
        s = d.get("survival_iters")
        if not p or not q or p <= 0 or q <= 0 or s is None or s <= 0:
            continue
        P.append(p); Q.append(q); B.append(d["birth_iter"]); D.append(d["death_iter"])
    if len(P) < 120:
        return None
    return (np.array(P, float), np.array(Q, float),
            np.array(B, float), np.array(D, float))


def couple_to_kappa(base_rank: np.ndarray, values: np.ndarray,
                    kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Reassign `values` so their Spearman correlation with `base_rank` is ~kappa.

    Gaussian copula: build a latent normal correlated with the base ranks at the
    requested strength, then hand out the sorted values in the latent order. The
    multiset of values is preserved exactly, so only the coupling changes.
    """
    n = len(values)
    u = (rankdata(base_rank) - 0.5) / n
    z_base = norm.ppf(u)
    # Spearman ~ (6/pi) * arcsin(rho/2); invert for the latent correlation.
    rho = 2.0 * np.sin(np.pi * np.clip(kappa, -1, 1) / 6.0)
    z = rho * z_base + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal(n)
    out = np.empty(n)
    out[np.argsort(z)] = np.sort(values)
    return out


def cohort_rank(vals: np.ndarray, birth: np.ndarray, death: np.ndarray) -> np.ndarray:
    """Percentile of each offspring's value among those alive at its birth."""
    n = len(vals)
    pct = np.full(n, np.nan)
    for i in range(n):
        alive = (birth <= birth[i]) & (death > birth[i])
        alive[i] = False
        if alive.sum() >= 3:
            pct[i] = (vals[alive] < vals[i]).mean()
    return pct


def concordance_from_tau(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 30:
        return np.nan
    tau = kendalltau(x[m], y[m]).correlation
    if tau is None or np.isnan(tau):
        return np.nan
    return (tau + 1.0) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--island", type=int, default=9)
    ap.add_argument("--max_instances", type=int, default=25)
    ap.add_argument("--max_runs", type=int, default=3)
    ap.add_argument("--max_per_run", type=int, default=800)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/legacy_results")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    instances = sorted(d.name for d in os.scandir(logs) if d.is_dir())
    curve = {k: [] for k in KAPPAS}
    observed = []          # (instance, observed kappa, observed C)
    used = 0

    for inst in instances:
        if used >= args.max_instances:
            break
        idir = logs / inst / f"island_{args.island}"
        if not idir.is_dir():
            continue
        runs = [d for d in sorted(os.scandir(idir), key=lambda f: f.name) if d.is_dir()]
        runs = runs[-args.max_runs:]
        per_inst = {k: [] for k in KAPPAS}
        obs_k, obs_c = [], []

        for rd in runs:
            got = load_run(Path(rd.path), args.max_per_run)
            if got is None:
                continue
            P, Q, B, D = got
            gains = (P - Q) / P
            pre_pct = cohort_rank(P, B, D)

            # --- what the real pipeline did ---
            k_real = spearmanr(rankdata(P), gains).correlation
            c_real = concordance_from_tau(pre_pct, cohort_rank(Q, B, D))
            if not np.isnan(c_real):
                obs_k.append(k_real); obs_c.append(c_real)

            # --- sweep kappa, holding cost and gain distributions fixed ---
            for k in KAPPAS:
                g = couple_to_kappa(rankdata(P), gains, k, rng)
                q_syn = P * (1.0 - g)
                c = concordance_from_tau(pre_pct, cohort_rank(q_syn, B, D))
                if not np.isnan(c):
                    per_inst[k].append(c)

        if not per_inst[KAPPAS[0]]:
            continue
        used += 1
        for k in KAPPAS:
            if per_inst[k]:
                curve[k].append(float(np.mean(per_inst[k])))
        if obs_k:
            observed.append((inst, float(np.mean(obs_k)), float(np.mean(obs_c))))
        print(f"[{used:>2}] {inst:<22} kappa_obs={np.mean(obs_k):+.2f} "
              f"C_obs={np.mean(obs_c):.3f}", flush=True)

    if used == 0:
        print("no usable instances")
        return

    rows = [{"kappa": k,
             "C_mean": round(float(np.mean(curve[k])), 4),
             "C_std": round(float(np.std(curve[k])), 4),
             "n_instances": len(curve[k])} for k in KAPPAS if curve[k]]
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out_dir / "kappa_calibration.csv", index=False, encoding="utf-8")

    odf = pd.DataFrame(observed, columns=["instance", "kappa_observed", "C_observed"])
    odf.to_csv(out_dir / "kappa_observed.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 54)
    print(f"CALIBRATION CURVE  ({used} instances, island {args.island})")
    print("=" * 54)
    print(f"{'kappa':>6}{'C reachable':>14}{'sd':>8}")
    for r in rows:
        print(f"{r['kappa']:>6.2f}{r['C_mean']:>14.3f}{r['C_std']:>8.3f}")
    print("=" * 54)
    print(f"observed kappa : {odf.kappa_observed.mean():+.3f} "
          f"(sd {odf.kappa_observed.std():.3f})")
    print(f"observed C     : {odf.C_observed.mean():.3f} "
          f"(sd {odf.C_observed.std():.3f})")
    print("\nValidation: the observed C should sit near the curve at the observed")
    print("kappa. If it does, the curve is calibrated and a reader can pick a")
    print("cut-point from it instead of being handed one.")


if __name__ == "__main__":
    main()
