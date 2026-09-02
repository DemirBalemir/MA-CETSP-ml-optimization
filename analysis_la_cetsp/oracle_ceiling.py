"""Empirical post-VND reference for survival prediction.
Compare pre-VND geometry/cohort descriptors, recorded post-VND descriptors and
a birth-iteration baseline on island 9 using GroupKFold by run. The retained
positive-lifetime eviction sample and reconstructed cohorts are incomplete.
Best-of-three model scores select within each instance and are optimistic.
The fitted oracle score is not a mathematical ceiling for other predictors.
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
from feature_ceiling import extract_features   # exact C++ geometry port

GEOM = ["avg_edge_length", "var_edge_length", "bbox_width", "bbox_height",
        "bbox_area", "centroid_x", "centroid_y", "centroid_dist_sum",
        "angle_variance", "pre_vnd_cost"]
PRE_POP = ["cost_pct_live", "cost_ratio_median", "cost_ratio_best", "cost_z"]
ORACLE = ["post_vnd_cost", "post_pct_live", "post_ratio_median", "post_ratio_best",
          "post_z", "vnd_gain"]
DRIFT = ["birth_iter"]

FEATURE_SETS = {
    "PRE_geom":   GEOM,
    "PRE_pop":    PRE_POP,
    "PRE_all":    GEOM + PRE_POP,
    "ORACLE":     ORACLE,
    "ORACLE_all": GEOM + PRE_POP + ORACLE,
    "DRIFT":      DRIFT,
}

DEFAULT_INSTANCES = ["concentricCircles4", "car_door_50", "bonus1000",
                     "kroD100_or2", "rat195_or30"]


def _sol_num(fp: Path) -> int:
    try:
        return int(fp.stem.split("-")[1])
    except Exception:
        return 1 << 30


def _cohort_stats(vals, birth, death):
    """For each solution, its standing among the members ALIVE at its birth."""
    n = len(vals)
    pct = np.full(n, np.nan)
    ratio_med = np.full(n, np.nan)
    ratio_best = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for i in range(n):
        t = birth[i]
        alive = (birth <= t) & (death > t)
        alive[i] = False
        if alive.sum() < 3:
            continue
        c = vals[alive]
        pct[i] = (c < vals[i]).mean()
        med = np.median(c)
        ratio_med[i] = vals[i] / med if med > 0 else np.nan
        best = c.min()
        ratio_best[i] = vals[i] / best if best > 0 else np.nan
        sd = c.std()
        z[i] = (vals[i] - c.mean()) / sd if sd > 1e-12 else 0.0
    return pct, ratio_med, ratio_best, z


def load_run(run_dir: Path, max_files: int):
    """Load the most recent `max_files` solutions of one run and build every
    feature set. Cohort statistics are computed WITHIN the run (each run has its
    own iteration numbering and its own population)."""
    recs = []
    files = sorted(run_dir.glob("sol-*.json"), key=_sol_num)[-max_files:]
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        s = d.get("survival_iters")
        pre = d.get("pre_vnd_cost")
        post = d.get("post_vnd_cost")
        coords = d.get("pre_vnd_coords")
        if s is None or s <= 0 or not coords:
            continue
        if pre is None or post is None or pre <= 0 or post <= 0:
            continue
        f = extract_features(coords, pre)
        if f is None:
            continue
        f["post_vnd_cost"] = float(post)
        f["vnd_gain"] = (pre - post) / pre
        f["birth_iter"] = float(d["birth_iter"])
        f["survival_time"] = float(s)
        f["event"] = (not d.get("censored", False))
        f["_birth"] = float(d["birth_iter"])
        f["_death"] = float(d["death_iter"])
        recs.append(f)
    if len(recs) < 40:
        return None

    df = pd.DataFrame(recs)
    b = df["_birth"].values
    de = df["_death"].values
    for src, prefix in ((df["pre_vnd_cost"].values, "cost"),
                        (df["post_vnd_cost"].values, "post")):
        pct, rmed, rbest, z = _cohort_stats(src, b, de)
        df[f"{prefix}_pct_live"] = pct
        df[f"{prefix}_ratio_median"] = rmed
        df[f"{prefix}_ratio_best"] = rbest
        df[f"{prefix}_z"] = z
    return df.drop(columns=["_birth", "_death"])


def cv_cindex(X, y, groups, model_fn, n_splits=3):
    """Cross-validated concordance with RUN-AWARE folds.

    GroupKFold (groups = run id) keeps every solution of a run entirely inside
    one fold. Plain KFold lets the same run appear in train and test, which lets
    a flexible model identify the run from run-level offsets in the features and
    exploit the run's survival distribution — inflating the score without any
    real per-offspring signal. This was observed on bonus1000, where pooling
    ~51k rows across many runs pushed RSF on the PRE_geom set from ~0.53 to
    0.66 under plain KFold."""
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return np.nan
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in kf.split(X, groups=groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        try:
            m = model_fn()
            m.fit(Xtr, y[tr])
            risk = m.predict(Xte)
            scores.append(concordance_index_censored(
                y["event"][te], y["time"][te], risk)[0])
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--island", type=int, default=9, help="baseline island (unbiased)")
    ap.add_argument("--max_per_run", type=int, default=1200)
    ap.add_argument("--max_runs", type=int, default=6,
                    help="use at most this many (most recent) runs per instance")
    ap.add_argument("--rsf_trees", type=int, default=30,
                    help="RSF size; the ceiling question does not need 100 trees")
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
                                                         max_depth=3, random_state=0),
    }

    rows = []
    for inst in args.instances:
        idir = logs / inst / f"island_{args.island}"
        if not idir.exists():
            print(f"[skip] {inst}: no island_{args.island} logs", flush=True)
            continue
        parts = []
        for run_id, rd in enumerate(sorted(idir.glob("run-*"))[-args.max_runs:]):
            df = load_run(rd, args.max_per_run)
            if df is not None:
                df["_run"] = run_id          # fold grouping key
                parts.append(df)
        if not parts:
            print(f"[skip] {inst}: no usable runs", flush=True)
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 200:
            print(f"[skip] {inst}: only {len(df)} rows after cleaning", flush=True)
            continue

        groups = df["_run"].values
        y = Surv.from_arrays(event=df["event"].values,
                             time=df["survival_time"].values)
        print(f"\n{inst}  (n={len(df)}, runs={len(np.unique(groups))}, "
              f"events={df['event'].mean()*100:.0f}%)", flush=True)
        for set_name, feats in FEATURE_SETS.items():
            X = df[feats].values.astype(float)
            cells = {}
            for mname, fn in models.items():
                cells[mname] = cv_cindex(X, y, groups, fn)
            best = max(cells.values())
            rows.append({"instance": inst, "n": len(df), "feature_set": set_name,
                         **{k: round(v, 4) for k, v in cells.items()},
                         "best": round(best, 4)})
            print(f"   {set_name:<12}" +
                  "".join(f"{m}={cells[m]:.3f}  " for m in models) +
                  f"| best={best:.3f}", flush=True)

    if not rows:
        print("no results")
        return

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out_dir / "oracle_ceiling_per_instance.csv", index=False,
               encoding="utf-8")

    summary = (rdf.groupby("feature_set")[["COX", "RSF", "GBSA", "best"]]
                  .mean().round(4)
                  .reindex(list(FEATURE_SETS.keys())))
    summary.to_csv(out_dir / "oracle_ceiling.csv", encoding="utf-8")

    print("\n" + "=" * 62)
    print("MEAN ACROSS INSTANCES")
    print("=" * 62)
    print(summary.to_string())
    print("=" * 62)
    print(f"saved -> {(out_dir / 'oracle_ceiling.csv').resolve()}")
    print("\nRead:")
    print("  ORACLE_all ~0.65-0.70  => label is intrinsically noisy; no pre-VND")
    print("                            feature can work. Feature search CLOSED.")
    print("  ORACLE_all >= 0.85     => information exists post-VND, destroyed by")
    print("                            VND. Keep searching for what survives LKH.")
    print("  Compare DRIFT: whatever birth_iter alone scores is NOT a real")
    print("  mechanism, just search-phase drift, and does not generalise.")


if __name__ == "__main__":
    main()
