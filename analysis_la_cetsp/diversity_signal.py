"""Exploratory directed-Chamfer diversity proxy for survival discrimination.
Compare pre-VND point sets with recorded-cohort peers (at most 80 points and
15 peers). This is not the solver's edit-distance criterion. Reported paper
scores use raw univariate concordance; below 0.5 can mean a protective direction.
The optional multivariate probe uses solution-level KFold, not run-aware folds.
Incomplete logged cohorts limit the interpretation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

DEFAULT_INSTANCES = ["concentricCircles4", "car_door_50", "bonus1000",
                     "kroD100_or2", "rat195_or30"]
MAX_PTS = 80          # subsample points per tour to bound Chamfer cost
MAX_COHORT = 15       # subsample live cohort members per offspring


def _sol_num(fp):
    try: return int(fp.stem.split("-")[1])
    except Exception: return 1 << 30


def load_run(run_dir: Path, max_files: int, rng: random.Random):
    """Load recent sols with coords + survival + cost."""
    recs = []
    files = sorted(run_dir.glob("sol-*.json"), key=_sol_num)[-max_files:]
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        s = d.get("survival_iters"); coords = d.get("pre_vnd_coords")
        pv = d.get("pre_vnd_cost")
        if s is None or s <= 0 or not coords or pv is None or pv <= 0:
            continue
        pts = np.asarray(coords, dtype=float)
        if pts.ndim != 2 or len(pts) < 4:
            continue
        if len(pts) > MAX_PTS:
            pts = pts[rng.sample(range(len(pts)), MAX_PTS)]
        recs.append({
            "birth": d["birth_iter"], "death": d["death_iter"],
            "surv": float(s), "event": (not d.get("censored", False)),
            "pre": float(pv), "pts": pts,
        })
    return recs if len(recs) >= 40 else None


def chamfer_dir(a: np.ndarray, tree_b: cKDTree) -> float:
    """Directed Chamfer: mean over a-points of nearest distance into b."""
    dd, _ = tree_b.query(a, k=1)
    return float(dd.mean())


def features_for_run(recs, max_score, rng):
    n = len(recs)
    birth = np.array([r["birth"] for r in recs], float)
    death = np.array([r["death"] for r in recs], float)
    pre = np.array([r["pre"] for r in recs], float)
    trees = [cKDTree(r["pts"]) for r in recs]   # one tree per solution, reused

    idxs = list(range(n))
    if n > max_score:
        idxs = rng.sample(idxs, max_score)

    rows = []
    for i in idxs:
        t = birth[i]
        alive = np.where((birth <= t) & (death > t))[0]
        alive = alive[alive != i]
        if len(alive) < 3:
            continue
        if len(alive) > MAX_COHORT:
            alive = np.array(rng.sample(list(alive), MAX_COHORT))
        pts_i = recs[i]["pts"]
        dists = [chamfer_dir(pts_i, trees[j]) for j in alive]
        div_min = float(np.min(dists))
        div_mean = float(np.mean(dists))
        cp = pre[alive]
        rows.append({
            "div_min": div_min, "div_mean": div_mean,
            "cost_pct_live": float((cp < pre[i]).mean()),
            "survival_time": recs[i]["surv"], "event": recs[i]["event"],
        })
    return rows


def cv_cindex(X, y, n_splits=3, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        try:
            m = GradientBoostingSurvivalAnalysis(n_estimators=80, learning_rate=0.1,
                                                 max_depth=3, random_state=0)
            m.fit(sc.transform(X[tr]), y[tr])
            risk = m.predict(sc.transform(X[te]))
            scores.append(concordance_index_censored(y["event"][te], y["time"][te], risk)[0])
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--island", type=int, default=9)
    ap.add_argument("--max_files_per_run", type=int, default=1200)
    ap.add_argument("--runs_per_instance", type=int, default=2)
    ap.add_argument("--max_score_per_run", type=int, default=600)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    rng = random.Random(0)
    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    uni_feats = ["div_min", "div_mean", "cost_pct_live"]
    per_inst = {}
    multi = []
    for inst in args.instances:
        idir = logs / inst / f"island_{args.island}"
        if not idir.exists():
            print(f"{inst:<20} [no logs]", flush=True); continue
        print(f"  loading {inst} ...", flush=True)
        rows = []
        for run_dir in sorted(idir.glob("run-*"))[:args.runs_per_instance]:
            recs = load_run(run_dir, args.max_files_per_run, rng)
            if recs is None:
                continue
            rows.extend(features_for_run(recs, args.max_score_per_run, rng))
        if len(rows) < 200:
            print(f"{inst:<20} n={len(rows)} [too few]", flush=True); continue
        df = pd.DataFrame(rows)
        ev = df["event"].values.astype(bool); tt = df["survival_time"].values.astype(float)
        cs = {}
        for f in uni_feats:
            try: cs[f] = concordance_index_censored(ev, tt, df[f].values.astype(float))[0]
            except Exception: cs[f] = np.nan
        per_inst[inst] = cs
        y = Surv.from_arrays(event=ev, time=tt)
        c_div = cv_cindex(df[["div_min", "div_mean"]].values.astype(float), y)
        c_all = cv_cindex(df[["div_min", "div_mean", "cost_pct_live"]].values.astype(float), y)
        multi.append({"instance": inst, "n": len(df), "GBSA_div": round(c_div, 4),
                      "GBSA_div+cost": round(c_all, 4)})
        print(f"loaded {inst:<20} n={len(df):>5}  div_min={cs['div_min']:.3f} "
              f"div_mean={cs['div_mean']:.3f} cost_pct={cs['cost_pct_live']:.3f} | "
              f"GBSA_div={c_div:.3f} div+cost={c_all:.3f}", flush=True)

    if not per_inst:
        print("no data"); return
    insts = list(per_inst.keys())
    rows = []
    for f in uni_feats:
        cvals = np.array([per_inst[i][f] for i in insts], float)
        rows.append({"feature": f, "signal(|C-.5|)": round(np.nanmean(np.abs(cvals - .5)), 4),
                     "C_mean": round(np.nanmean(cvals), 4),
                     **{i: round(per_inst[i][f], 3) for i in insts}})
    res = pd.DataFrame(rows).sort_values("signal(|C-.5|)", ascending=False)
    res.to_csv(out_dir / "diversity_signal.csv", index=False, encoding="utf-8")
    md = pd.DataFrame(multi)
    if len(md):
        md.loc[len(md)] = {"instance": "MEAN", "n": md["n"].sum(),
                           "GBSA_div": round(md["GBSA_div"].mean(), 4),
                           "GBSA_div+cost": round(md["GBSA_div+cost"].mean(), 4)}

    print("\n" + "=" * 80)
    print(" DIVERSITY SURVIVAL SIGNAL  (|C-.5|: 0=noise; div high should PROLONG => C<0.5)")
    print("=" * 80)
    hdr = f"{'feature':<16}{'signal':>9}{'C_mean':>9}   " + "".join(f"{i[:10]:>12}" for i in insts)
    print(hdr); print("-" * len(hdr))
    for _, r in res.iterrows():
        print(f"{r['feature']:<16}{r['signal(|C-.5|)']:>9.3f}{r['C_mean']:>9.3f}   " +
              "".join(f"{r[i]:>12.3f}" for i in insts))
    print("\n MULTIVARIATE (GBSA):")
    print(md.to_string(index=False))
    print("\nRead: div signal >= ~0.60 (i.e. C<=0.40 or C>=0.60) => diversity is a real "
          "pre-VND lever; ~0.50 => the last stone is also empty, null is airtight.")
    print(f"\nsaved -> {out_dir/'diversity_signal.csv'}")


if __name__ == "__main__":
    main()
