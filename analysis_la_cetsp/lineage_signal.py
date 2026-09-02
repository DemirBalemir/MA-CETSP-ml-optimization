"""Lineage-feature survival discrimination with a matched clock control.
Compare geometry/cohort, lineage, combined 23-feature and birth-iteration sets
using run-aware GroupKFold on recorded positive-lifetime island-9 observations.
Neither the sample nor reconstructed cohorts represent all generated offspring.
Best-of-three model scores select within each instance and are optimistic.
A favourable score alone does not establish a useful online pruning policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_ceiling import extract_features
from oracle_ceiling import GEOM, PRE_POP, _cohort_stats, _sol_num

LINEAGE = ["parent_best_cost", "parent_mean_cost", "parent_cost_gap",
           "parent_best_fitness", "parent_mean_fitness",
           "dist_min", "dist_mean", "mutated",
           "offspring_vs_best_parent"]

FEATURE_SETS = {
    "PRE_all":     GEOM + PRE_POP,
    "LINEAGE":     LINEAGE,
    "PRE+LINEAGE": GEOM + PRE_POP + LINEAGE,
    "DRIFT":       ["birth_iter"],
}


def load_run(run_dir: Path, max_files: int):
    recs = []
    for fp in sorted(run_dir.glob("sol-*.json"), key=_sol_num)[-max_files:]:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        # skip logs written before the lineage instrumentation existed
        if "parent1_cost" not in d:
            continue
        s = d.get("survival_iters")
        pre = d.get("pre_vnd_cost")
        coords = d.get("pre_vnd_coords")
        p1, p2 = d.get("parent1_cost"), d.get("parent2_cost")
        f1, f2 = d.get("parent1_fitness"), d.get("parent2_fitness")
        d1, d2 = d.get("parent1_dist"), d.get("parent2_dist")
        if s is None or s <= 0 or not coords or not pre or pre <= 0:
            continue
        if p1 is None or p2 is None or p1 <= 0 or p2 <= 0:
            continue
        if d1 is None or d2 is None or d1 < 0 or d2 < 0:
            continue

        f = extract_features(coords, pre)
        if f is None:
            continue

        best_parent = min(p1, p2)
        f["parent_best_cost"] = best_parent
        f["parent_mean_cost"] = 0.5 * (p1 + p2)
        f["parent_cost_gap"] = abs(p1 - p2)
        # NOTE on fitness sign: populationManagement sorts ASCENDING by fitness
        # (= value_rank + beta*distance_rank), so LOWER fitness is BETTER.
        f["parent_best_fitness"] = min(f1, f2) if (f1 is not None and f2 is not None) else np.nan
        f["parent_mean_fitness"] = 0.5 * (f1 + f2) if (f1 is not None and f2 is not None) else np.nan
        f["dist_min"] = min(d1, d2)
        f["dist_mean"] = 0.5 * (d1 + d2)
        f["mutated"] = float(d.get("mutated", 0))
        # did the offspring start out better than the better of its parents?
        f["offspring_vs_best_parent"] = pre / best_parent if best_parent > 0 else np.nan

        f["birth_iter"] = float(d["birth_iter"])
        f["survival_time"] = float(s)
        f["event"] = (not d.get("censored", False))
        f["_birth"] = float(d["birth_iter"])
        f["_death"] = float(d["death_iter"])
        recs.append(f)

    if len(recs) < 40:
        return None
    df = pd.DataFrame(recs)
    pct, rmed, rbest, z = _cohort_stats(df["pre_vnd_cost"].values,
                                        df["_birth"].values, df["_death"].values)
    df["cost_pct_live"] = pct
    df["cost_ratio_median"] = rmed
    df["cost_ratio_best"] = rbest
    df["cost_z"] = z
    return df.drop(columns=["_birth", "_death"])


def cv_cindex(X, y, groups, model_fn, n_splits=3):
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return np.nan
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in kf.split(X, groups=groups):
        sc = StandardScaler().fit(X[tr])
        try:
            m = model_fn()
            m.fit(sc.transform(X[tr]), y[tr])
            risk = m.predict(sc.transform(X[te]))
            scores.append(concordance_index_censored(
                y["event"][te], y["time"][te], risk)[0])
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--island", type=int, default=9)
    ap.add_argument("--max_per_run", type=int, default=1500)
    ap.add_argument("--rsf_trees", type=int, default=30)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "COX": lambda: CoxPHSurvivalAnalysis(alpha=1.0),
        "RSF": lambda: RandomSurvivalForest(n_estimators=args.rsf_trees,
                                            min_samples_leaf=30,
                                            max_features="sqrt", n_jobs=-1,
                                            random_state=0),
        "GBSA": lambda: GradientBoostingSurvivalAnalysis(n_estimators=100,
                                                         learning_rate=0.1,
                                                         max_depth=3,
                                                         random_state=0),
    }

    rows = []
    for inst in args.instances:
        idir = logs / inst / f"island_{args.island}"
        if not idir.exists():
            print(f"[skip] {inst}: no island_{args.island} logs", flush=True)
            continue
        parts = []
        for run_id, rd in enumerate(sorted(idir.glob("run-*"))):
            df = load_run(rd, args.max_per_run)
            if df is not None:
                df["_run"] = run_id
                parts.append(df)
        if not parts:
            print(f"[skip] {inst}: no INSTRUMENTED runs (rerun the solver first)",
                  flush=True)
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 200:
            print(f"[skip] {inst}: only {len(df)} usable rows", flush=True)
            continue

        groups = df["_run"].values
        y = Surv.from_arrays(event=df["event"].values,
                             time=df["survival_time"].values)
        print(f"\n{inst}  (n={len(df)}, runs={len(np.unique(groups))})", flush=True)
        for set_name, feats in FEATURE_SETS.items():
            X = df[feats].values.astype(float)
            cells = {m: cv_cindex(X, y, groups, fn) for m, fn in models.items()}
            rows.append({"instance": inst, "n": len(df), "feature_set": set_name,
                         **{k: round(v, 4) for k, v in cells.items()},
                         "best": round(max(cells.values()), 4)})
            print(f"   {set_name:<13}" +
                  "".join(f"{m}={cells[m]:.3f}  " for m in models) +
                  f"| best={max(cells.values()):.3f}", flush=True)

        # univariate signal of each lineage feature, for the paper's table
        for f in LINEAGE:
            X = df[[f]].values.astype(float)
            c = cv_cindex(X, y, groups, models["COX"])
            rows.append({"instance": inst, "n": len(df),
                         "feature_set": f"uni:{f}", "COX": round(c, 4),
                         "best": round(c, 4)})

    if not rows:
        print("\nNo instrumented data found. Run the rebuilt solver first so the "
              "logs contain parent1_cost / parent1_dist / mutated.")
        return

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out_dir / "lineage_signal_per_instance.csv", index=False,
               encoding="utf-8")
    summary = (rdf[~rdf.feature_set.str.startswith("uni:")]
               .groupby("feature_set")[["COX", "RSF", "GBSA", "best"]]
               .mean().round(4).reindex(list(FEATURE_SETS.keys())))
    summary.to_csv(out_dir / "lineage_signal.csv", encoding="utf-8")

    print("\n" + "=" * 58)
    print("MEAN ACROSS INSTANCES")
    print("=" * 58)
    print(summary.to_string())
    print("=" * 58)
    print("\nRead:  PRE+LINEAGE ~ PRE_all (~0.54) => lineage adds nothing, the")
    print("       feature search is exhausted across every nameable family.")
    print("       PRE+LINEAGE >= 0.60 => real signal; rebuild the filter on it.")
    print("       (oracle ceiling from oracle_ceiling.py is 0.687)")


if __name__ == "__main__":
    main()
