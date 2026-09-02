"""Recorded positive-lifetime stage-cost rank transmission.
Pooled pre/post Spearman uses all selected runs. Recorded-cohort percentiles are
computed per run then pooled; this is not a comparison only of contemporaneous
pairs and cannot guarantee removal of temporal drift. Adjacent-stage Spearman
correlations are averaged across runs. Default instrumented island: 0.
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

STAGES = ["pre_vnd_cost", "post_greed_cost", "post_lkh_cost", "post_vnd_cost"]
DEF = ["concentricCircles4", "kroD100_or2", "rat195_or30"]


def load_run(run_dir):
    B, D = [], []
    st = {s: [] for s in STAGES}
    for fp in run_dir.glob("sol-*.json"):
        try: d = json.loads(fp.read_text())
        except Exception: continue
        if "post_greed_cost" not in d: continue
        if d.get("survival_iters", 0) <= 0: continue
        vals = [d.get(k) for k in STAGES]
        if any(v is None or v <= 0 for v in vals): continue
        B.append(d["birth_iter"]); D.append(d["death_iter"])
        for k, v in zip(STAGES, vals): st[k].append(v)
    if len(B) < 40: return None
    out = {"birth": np.array(B, float), "death": np.array(D, float)}
    for k in STAGES: out[k] = np.array(st[k], float)
    return out


def incohort_pct(r, stage):
    b, d = r["birth"], r["death"]; x = r[stage]; n = len(b)
    pct = np.full(n, np.nan)
    for i in range(n):
        t = b[i]; al = (b <= t) & (d > t); al[i] = False
        if al.sum() >= 3: pct[i] = (x[al] < x[i]).mean()
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEF)
    ap.add_argument("--island", type=int, default=0)
    a = ap.parse_args()
    logs = Path(a.logs_dir)

    print(f"{'instance':<20}{'GLOBAL pre>post':>16}{'IN-COHORT pre>post':>20}   "
          "stage-by-stage GLOBAL Spearman")
    print("-" * 100)
    results = []
    for inst in a.instances:
        idir = logs / inst / f"island_{a.island}"
        if not idir.exists(): print(f"{inst:<20} [no logs]"); continue
        pre_g, post_g = [], []
        preP, postP = [], []
        chain = {("pre_vnd_cost","post_greed_cost"):[], ("post_greed_cost","post_lkh_cost"):[],
                 ("post_lkh_cost","post_vnd_cost"):[]}
        for rd in sorted(idir.glob("run-*")):
            r = load_run(rd)
            if r is None: continue
            pre_g.append(r["pre_vnd_cost"]); post_g.append(r["post_vnd_cost"])
            pp = incohort_pct(r, "pre_vnd_cost"); qq = incohort_pct(r, "post_vnd_cost")
            m = ~(np.isnan(pp) | np.isnan(qq))
            preP.append(pp[m]); postP.append(qq[m])
            for (u, v) in chain:
                chain[(u, v)].append(spearmanr(r[u], r[v]).correlation)
        if not pre_g: print(f"{inst:<20} [no instrumented runs]"); continue
        g = spearmanr(np.concatenate(pre_g), np.concatenate(post_g)).correlation
        c = spearmanr(np.concatenate(preP), np.concatenate(postP)).correlation
        chain_str = "  ".join(f"{u.split('_')[0][:4]}>{v.replace('_cost','').split('_')[-1][:4]}="
                              f"{np.nanmean(vals):+.2f}" for (u, v), vals in chain.items())
        print(f"{inst:<20}{g:>16.3f}{c:>20.3f}   {chain_str}")
        results.append(dict(instance=inst, global_rho=g, cohort_rho=c,
                            greedy_rho=np.nanmean(chain[(STAGES[0],STAGES[1])]),
                            lkh_rho=np.nanmean(chain[(STAGES[1],STAGES[2])]),
                            refine_rho=np.nanmean(chain[(STAGES[2],STAGES[3])])))
    pd.DataFrame(results).to_csv(Path(__file__).parent / 'results' / 'rank_transmission.csv', index=False)
    print("Interpret correlations jointly; incomplete cohorts and temporal drift limit inference.")


if __name__ == "__main__":
    main()
