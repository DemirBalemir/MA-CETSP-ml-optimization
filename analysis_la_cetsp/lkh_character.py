"""LKH rank--gain association in recorded positive-lifetime solutions.
Spearman(post-greedy recorded-cohort percentile, relative LKH gain) is computed
within each run then averaged. Positive association alone does not establish
compression or lost ordering. The points CSV and kappa CSV share the same runs.
Default instrumented island: 0.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from feature_ceiling import extract_features   # exact C++ geometry port

DEF = ["concentricCircles4", "kroD100_or2", "rat195_or30"]
FEATS = ["pre_vnd_cost", "post_greed_cost", "n_points",
         "var_edge_length", "avg_edge_length", "angle_variance", "bbox_area"]


def load_run(run_dir):
    rows = []
    b, d = [], []
    for fp in run_dir.glob("sol-*.json"):
        try: x = json.loads(fp.read_text())
        except Exception: continue
        if "post_greed_cost" not in x: continue
        if x.get("survival_iters", 0) <= 0: continue
        pg, pl = x.get("post_greed_cost"), x.get("post_lkh_cost")
        coords = x.get("pre_vnd_coords")
        if not pg or not pl or pg <= 0 or not coords: continue
        f = extract_features(coords, x.get("pre_vnd_cost", 0.0))
        if f is None: continue
        f["post_greed_cost"] = pg
        f["n_points"] = len(coords)
        f["lkh_gain"] = (pg - pl) / pg
        rows.append(f); b.append(x["birth_iter"]); d.append(x["death_iter"])
    if len(rows) < 40: return None
    return rows, np.array(b, float), np.array(d, float)


def cohort_pct(vals, birth, death):
    n = len(vals); pct = np.full(n, np.nan)
    for i in range(n):
        t = birth[i]; al = (birth <= t) & (death > t); al[i] = False
        if al.sum() >= 3: pct[i] = (vals[al] < vals[i]).mean()
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEF)
    ap.add_argument("--island", type=int, default=0)
    a = ap.parse_args()
    logs = Path(a.logs_dir)

    print("Spearman( lkh_gain , feature )   [+ => that trait gets a BIGGER LKH gain]")
    print(f"{'instance':<20}" + "".join(f"{f[:11]:>13}" for f in FEATS))
    print("-" * (20 + 13 * len(FEATS)))
    compress = {}
    plot_rows = []
    for inst in a.instances:
        idir = logs / inst / f"island_{a.island}"
        if not idir.exists(): print(f"{inst:<20}[no logs]"); continue
        allrows = []; spreads_g = []; spreads_l = []; rank_gain = []
        for rd in sorted(idir.glob("run-*")):
            got = load_run(rd)
            if got is None: continue
            rows, b, d = got
            allrows.extend(rows)
            pg = np.array([r["post_greed_cost"] for r in rows])
            gain = np.array([r["lkh_gain"] for r in rows])
            pgpct = cohort_pct(pg, b, d)
            m = ~np.isnan(pgpct)
            plot_rows.extend(dict(instance=inst, run=rd.name, cohort_rank=float(x),
                                  relative_gain=float(y)) for x,y in zip(pgpct[m], gain[m]))
            # does a worse (higher-cost) input rank gain more?
            if m.sum() > 10:
                rank_gain.append(spearmanr(pgpct[m], gain[m]).correlation)
        if not allrows: print(f"{inst:<20}[no instrumented]"); continue
        gain = np.array([r["lkh_gain"] for r in allrows])
        cells = []
        for f in FEATS:
            x = np.array([r.get(f, np.nan) for r in allrows], float)
            mm = ~np.isnan(x)
            cells.append(spearmanr(x[mm], gain[mm]).correlation)
        print(f"{inst:<20}" + "".join(f"{c:>+13.2f}" for c in cells))
        compress[inst] = np.nanmean(rank_gain) if rank_gain else np.nan

    print("\nRegression-to-the-mean check  Spearman( post_greed cohort-rank , lkh_gain ):")
    for inst, v in compress.items():
        print(f"  {inst:<20}{v:>+7.2f}   "
              + ("(lower-ranked recorded inputs tend to gain more)" if v > 0.15 else ""))

    # Persist kappa so the paper has ONE source for it. make_figures.py reads this
    # file for the panel titles of fig4_contraction. Previously these numbers only
    # ever reached the paper by being typed out of this console output, while the
    # figure recomputed them with a different estimator (pooled across runs, last
    # three runs only) -- the two had drifted apart on two of three instances.
    out = Path(__file__).parent / "results" / "contraction_kappa.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("instance,kappa,estimator\n")
        for inst, v in compress.items():
            fh.write(f"{inst},{v:.6f},per_run_spearman_mean\n")
    print(f"\nwrote {out}")
    pd.DataFrame(plot_rows).to_csv(out.with_name('contraction_points.csv'), index=False)
    print("Rank--gain association should be read alongside measured input/output rank transmission.")


if __name__ == "__main__":
    main()
