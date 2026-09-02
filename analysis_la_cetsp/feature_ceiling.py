"""Three-model geometry-feature survival diagnostic.
Cox, RSF and GBSA use GroupKFold by run on recorded positive-lifetime island-9
solutions. Neither absence of filtering nor low recorded censoring makes this
sample unbiased: unadmitted candidates and terminal survivors are not logged.
Observed discrimination describes tested models and features, not a ceiling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

FEATURES = ["avg_edge_length", "var_edge_length", "bbox_width", "bbox_height",
            "bbox_area", "centroid_x", "centroid_y", "centroid_dist_sum",
            "angle_variance", "pre_vnd_cost"]


def extract_features(coords, pre_vnd_cost):
    """Exact port of C++ GeometryFeatures::extract."""
    n = len(coords)
    if n < 2:
        return None
    xs = np.array([c[0] for c in coords], dtype=float)
    ys = np.array([c[1] for c in coords], dtype=float)

    dx = np.diff(xs); dy = np.diff(ys)
    edges = np.sqrt(dx * dx + dy * dy)
    avg_e = edges.mean()
    var_e = ((edges - avg_e) ** 2).mean()

    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    cx = xs.mean(); cy = ys.mean()
    dist_sum = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).sum()

    # angle variance at interior vertices
    angs = []
    for i in range(1, n - 1):
        ax, ay = xs[i - 1] - xs[i], ys[i - 1] - ys[i]
        bx, by = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        nA = np.hypot(ax, ay); nB = np.hypot(bx, by)
        if nA < 1e-9 or nB < 1e-9:
            continue
        cosang = np.clip((ax * bx + ay * by) / (nA * nB), -1, 1)
        angs.append(np.arccos(cosang))
    var_ang = float(np.var(angs)) if angs else 0.0

    return {
        "avg_edge_length": avg_e, "var_edge_length": var_e,
        "bbox_width": width, "bbox_height": height, "bbox_area": width * height,
        "centroid_x": cx, "centroid_y": cy, "centroid_dist_sum": dist_sum,
        "angle_variance": var_ang, "pre_vnd_cost": pre_vnd_cost,
    }


def load_instance(inst_dir: Path, island: int, max_n: int, rng: random.Random):
    files = list((inst_dir / f"island_{island}").glob("run-*/sol-*.json"))
    if len(files) > max_n:
        files = rng.sample(files, max_n)
    rows = []
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        surv = d.get("survival_iters")
        coords = d.get("pre_vnd_coords")
        if surv is None or surv <= 0 or not coords:
            continue
        feats = extract_features(coords, d.get("pre_vnd_cost", 0.0))
        if feats is None:
            continue
        feats["survival_time"] = float(surv)
        feats["event"] = (not d.get("censored", False))
        feats["run"] = fp.parent.name          # fold grouping key
        rows.append(feats)
    return pd.DataFrame(rows)


def cv_cindex(X, y, groups, model_fn, n_splits=3):
    """Run-aware CV. Plain KFold puts solutions from the SAME run in both train
    and test, letting a flexible model recognise the run from run-level offsets
    and exploit its survival distribution. On bonus1000 that inflated RSF from
    0.512 to 0.660 with no real per-offspring signal (see oracle_ceiling.py)."""
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
            c = concordance_index_censored(y["event"][te], y["time"][te], risk)[0]
            scores.append(c)
        except Exception as e:
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+",
                    default=["concentricCircles4", "car_door_50", "bonus1000",
                             "kroD100_or2", "rat195_or30"])
    ap.add_argument("--island", type=int, default=9, help="baseline island (unbiased)")
    ap.add_argument("--max_per_instance", type=int, default=5000)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    rng = random.Random(0)
    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "COX(linear)":   lambda: CoxPHSurvivalAnalysis(alpha=1.0),  # ridge: collinearity-safe
        "RSF(nonlin)":   lambda: RandomSurvivalForest(n_estimators=100, min_samples_leaf=15,
                                                      max_features="sqrt", n_jobs=-1, random_state=0),
        "GBSA(nonlin)":  lambda: GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1,
                                                                 max_depth=3, random_state=0),
    }

    print(f"{'instance':<20}{'n':>7}{'events%':>9}   " +
          "".join(f"{k:>15}" for k in models))
    print("-" * (20 + 7 + 9 + 3 + 15 * len(models)))

    all_rows = []
    for inst in args.instances:
        inst_dir = logs / inst
        if not inst_dir.exists():
            print(f"{inst:<20}  [no logs]")
            continue
        df = load_instance(inst_dir, args.island, args.max_per_instance, rng)
        if len(df) < 100:
            print(f"{inst:<20}{len(df):>7}   [too few]")
            continue
        X = df[FEATURES].values.astype(float)
        groups = df["run"].values
        y = Surv.from_arrays(event=df["event"].values, time=df["survival_time"].values)
        ev = df["event"].mean() * 100
        res = {"instance": inst, "n": len(df), "events%": round(ev, 1)}
        cells = []
        for name, fn in models.items():
            c = cv_cindex(X, y, groups, fn)
            res[name] = round(c, 4)
            cells.append(f"{c:>15.3f}")
        all_rows.append(res)
        print(f"{inst:<20}{len(df):>7}{ev:>8.1f}%   " + "".join(cells))

    if all_rows:
        rdf = pd.DataFrame(all_rows)
        # mean row
        mean_row = {"instance": "MEAN", "n": rdf["n"].sum(), "events%": round(rdf["events%"].mean(), 1)}
        for k in models:
            mean_row[k] = round(rdf[k].mean(), 4)
        rdf = pd.concat([rdf, pd.DataFrame([mean_row])], ignore_index=True)
        rdf.to_csv(out_dir / "feature_ceiling.csv", index=False, encoding="utf-8")
        print("-" * (20 + 7 + 9 + 3 + 15 * len(models)))
        mr = rdf.iloc[-1]
        print(f"{'MEAN':<20}{'':>7}{'':>9}   " + "".join(f"{mr[k]:>15.3f}" for k in models))
        print(f"\nsaved -> {(out_dir / 'feature_ceiling.csv').resolve()}")
        print("\nRead: nonlinear ~0.5-0.55 => features are the ceiling (redesign);"
              " >=0.65 => signal exists, model choice matters.")


if __name__ == "__main__":
    main()
