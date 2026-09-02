"""Exploratory population-relative feature probes on recorded island-9 data.
Pre-VND features describe relative standing in incomplete recorded cohorts;
post-VND features are empirical diagnostic references, not mathematical bounds.
The multivariate analysis uses shuffled solution-level KFold and can retain
run-specific information. See oracle_ceiling.py for run-aware comparisons.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

LEGIT = ["pre_vnd_cost", "cost_pct_live", "cost_ratio_median", "cost_ratio_best",
         "cost_z", "cost_ratio_runbest"]
LEAK = ["post_pct_live", "vnd_gain"]
ALL_FEATS = LEGIT + LEAK

DEFAULT_INSTANCES = ["concentricCircles4", "car_door_50", "bonus1000",
                     "kroD100_or2", "rat195_or30"]


def _sol_num(fp: Path) -> int:
    try:
        return int(fp.stem.split("-")[1])
    except Exception:
        return 1 << 30


def load_run(run_dir: Path, max_files: int):
    """Load the MOST RECENT `max_files` sols of one run (sol-N.json are numbered in
    death/log order, so the LAST N files == the late/mature phase of the search —
    where the population has settled and the real filter actually operates. This
    also matches the real on-the-fly training size, ~1000 rows)."""
    births, deaths, surv, censored, pre, post = [], [], [], [], [], []
    files = sorted(run_dir.glob("sol-*.json"), key=_sol_num)[-max_files:]
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        s = d.get("survival_iters")
        if s is None or s <= 0:
            continue
        b = d.get("birth_iter"); de = d.get("death_iter")
        pv = d.get("pre_vnd_cost"); po = d.get("post_vnd_cost")
        if b is None or de is None or pv is None or po is None or pv <= 0:
            continue
        births.append(b); deaths.append(de); surv.append(s)
        censored.append(bool(d.get("censored", False)))
        pre.append(pv); post.append(po)
    if len(births) < 50:
        return None
    return {
        "birth": np.array(births, dtype=float),
        "death": np.array(deaths, dtype=float),
        "surv": np.array(surv, dtype=float),
        "cens": np.array(censored, dtype=bool),
        "pre": np.array(pre, dtype=float),
        "post": np.array(post, dtype=float),
    }


def features_for_run(r, max_score: int, rng: random.Random):
    """Compute population-relative features for (a sample of) offspring in a run."""
    birth, death = r["birth"], r["death"]
    pre, post = r["pre"], r["post"]
    n = len(birth)

    # running-best pre cost up to (and including) each birth iter.
    order = np.argsort(birth, kind="stable")
    run_best = np.empty(n)
    cur = np.inf
    # walk in birth order, running min of pre among those born earlier-or-equal
    for idx in order:
        cur = min(cur, pre[idx])
        run_best[idx] = cur

    # choose which offspring to score
    idxs = list(range(n))
    if n > max_score:
        idxs = rng.sample(idxs, max_score)

    rows = []
    for i in idxs:
        t = birth[i]
        alive = (birth <= t) & (death > t)
        alive[i] = False  # exclude self
        if alive.sum() < 3:
            continue
        cp = pre[alive]; cpo = post[alive]
        pv, po = pre[i], post[i]
        med = np.median(cp); mn = cp.min()
        mu = cp.mean(); sd = cp.std()
        rows.append({
            "pre_vnd_cost": pv,
            "cost_pct_live": float((cp < pv).mean()),
            "cost_ratio_median": pv / med if med > 0 else 1.0,
            "cost_ratio_best": pv / mn if mn > 0 else 1.0,
            "cost_z": (pv - mu) / sd if sd > 1e-9 else 0.0,
            "cost_ratio_runbest": pv / run_best[i] if run_best[i] > 0 else 1.0,
            "post_pct_live": float((cpo < po).mean()),
            "vnd_gain": (pv - po) / pv if pv > 0 else 0.0,
            "survival_time": r["surv"][i],
            "event": (not r["cens"][i]),
        })
    return rows


def load_instance(inst_dir: Path, island: int, max_files: int, max_score: int,
                  runs_per_instance: int, rng: random.Random):
    rows = []
    run_dirs = sorted((inst_dir / f"island_{island}").glob("run-*"))[:runs_per_instance]
    for run_dir in run_dirs:
        r = load_run(run_dir, max_files)
        if r is None:
            continue
        rows.extend(features_for_run(r, max_score, rng))
    return pd.DataFrame(rows)


def cv_cindex(X, y, model_fn, n_splits=3, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        try:
            m = model_fn(); m.fit(Xtr, y[tr])
            risk = m.predict(Xte)
            c = concordance_index_censored(y["event"][te], y["time"][te], risk)[0]
            scores.append(c)
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else np.nan


def uni_cindex(x, ev, t):
    try:
        return concordance_index_censored(ev, t, x)[0]
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--island", type=int, default=9)
    ap.add_argument("--max_files_per_run", type=int, default=1200,
                    help="earliest N sol files per run == the real training window (~1k rows)")
    ap.add_argument("--runs_per_instance", type=int, default=2)
    ap.add_argument("--max_score_per_run", type=int, default=2000)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    rng = random.Random(0)
    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    per_inst_uni = {}          # instance -> {feature -> signed C}
    multi_rows = []            # per-instance multivariate C for feature sets
    models = {
        "COX": lambda: CoxPHSurvivalAnalysis(alpha=1.0),
        "RSF": lambda: RandomSurvivalForest(n_estimators=50, min_samples_leaf=20,
                                            max_features="sqrt", n_jobs=-1, random_state=0),
        "GBSA": lambda: GradientBoostingSurvivalAnalysis(n_estimators=80, learning_rate=0.1,
                                                         max_depth=3, random_state=0),
    }

    for inst in args.instances:
        inst_dir = logs / inst
        if not inst_dir.exists():
            print(f"{inst:<20} [no logs]", flush=True); continue
        print(f"  loading {inst} ...", flush=True)
        df = load_instance(inst_dir, args.island, args.max_files_per_run,
                           args.max_score_per_run, args.runs_per_instance, rng)
        if len(df) < 200:
            print(f"{inst:<20} n={len(df)} [too few]", flush=True); continue

        ev = df["event"].values.astype(bool)
        t = df["survival_time"].values.astype(float)

        # univariate signal per feature
        cs = {f: uni_cindex(df[f].values.astype(float), ev, t) for f in ALL_FEATS}
        per_inst_uni[inst] = cs

        # multivariate: legit-only vs legit+leak, across models
        y = Surv.from_arrays(event=ev, time=t)
        Xleg = df[LEGIT].values.astype(float)
        Xall = df[ALL_FEATS].values.astype(float)
        row = {"instance": inst, "n": len(df), "events%": round(ev.mean() * 100, 1)}
        for mname, fn in models.items():
            row[f"{mname}_legit"] = round(cv_cindex(Xleg, y, fn), 4)
        # only need one model to gauge the leakage ceiling
        row["GBSA_all(leak)"] = round(cv_cindex(Xall, y, models["GBSA"]), 4)
        multi_rows.append(row)
        print(f"loaded {inst:<20} n={len(df):>6}  events={ev.mean()*100:4.1f}%  "
              f"COXleg={row['COX_legit']:.3f} RSFleg={row['RSF_legit']:.3f} "
              f"GBSAleg={row['GBSA_legit']:.3f} | GBSA+leak={row['GBSA_all(leak)']:.3f}", flush=True)

    if not per_inst_uni:
        print("no data"); return

    # ---- univariate table ----
    insts = list(per_inst_uni.keys())
    uni_rows = []
    for f in ALL_FEATS:
        cvals = np.array([per_inst_uni[i][f] for i in insts], dtype=float)
        uni_rows.append({
            "feature": f,
            "kind": "LEAK" if f in LEAK else "legit",
            "signal(|C-.5|)": round(np.nanmean(np.abs(cvals - 0.5)), 4),
            "C_mean": round(np.nanmean(cvals), 4),
            **{i: round(per_inst_uni[i][f], 3) for i in insts},
        })
    uni = pd.DataFrame(uni_rows).sort_values("signal(|C-.5|)", ascending=False)
    uni.to_csv(out_dir / "feature_prototype_univariate.csv", index=False, encoding="utf-8")

    multi = pd.DataFrame(multi_rows)
    if len(multi):
        mean_row = {"instance": "MEAN", "n": multi["n"].sum(), "events%": ""}
        for c in multi.columns:
            if c not in ("instance", "n", "events%"):
                mean_row[c] = round(multi[c].mean(), 4)
        multi = pd.concat([multi, pd.DataFrame([mean_row])], ignore_index=True)
    multi.to_csv(out_dir / "feature_prototype_multivariate.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 90)
    print(" UNIVARIATE SIGNAL  (|C-.5|: 0=noise; legit features are the real candidates)")
    print("=" * 90)
    hdr = f"{'feature':<20}{'kind':>6}{'signal':>9}{'C_mean':>9}   " + "".join(f"{i[:9]:>10}" for i in insts)
    print(hdr); print("-" * len(hdr))
    for _, r in uni.iterrows():
        print(f"{r['feature']:<20}{r['kind']:>6}{r['signal(|C-.5|)']:>9.3f}{r['C_mean']:>9.3f}   " +
              "".join(f"{r[i]:>10.3f}" for i in insts))

    print("\n" + "=" * 90)
    print(" MULTIVARIATE CROSS-VAL C-INDEX")
    print("=" * 90)
    print(multi.to_string(index=False))
    print(f"\nsaved -> {out_dir/'feature_prototype_univariate.csv'}")
    print(f"saved -> {out_dir/'feature_prototype_multivariate.csv'}")
    print("\nRead: legit multivariate >= ~0.60 => real signal from population-relative"
          " cost; GBSA+leak shows the ceiling if post-VND rank were knowable pre-VND.")


if __name__ == "__main__":
    main()
