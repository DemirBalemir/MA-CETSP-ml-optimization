"""Direct survival concordance of recorded costs at stages of VND.
Uses instrumented island-0 positive-lifetime records and reconstructed recorded
cohort percentiles. No early-exit filtering policy is deployed by this script.
High or low concordance alone does not establish filtering utility. The sample
omits unadmitted solutions and terminal survivors, and cohorts are incomplete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

STAGES = ["pre_vnd_cost", "post_greed_cost", "post_lkh_cost", "post_vnd_cost"]
DEFAULT_INSTANCES = ["concentricCircles4", "kroD100_or2", "rat195_or30"]


def load_run(run_dir: Path):
    birth, death, surv, cens = [], [], [], []
    stage = {s: [] for s in STAGES}
    for fp in run_dir.glob("sol-*.json"):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if "post_greed_cost" not in d:          # skip old, un-instrumented logs
            continue
        s = d.get("survival_iters")
        if s is None or s <= 0:
            continue
        vals = [d.get(k) for k in STAGES]
        if any(v is None or v <= 0 for v in vals):
            continue
        birth.append(d["birth_iter"]); death.append(d["death_iter"])
        surv.append(s); cens.append(bool(d.get("censored", False)))
        for k, v in zip(STAGES, vals):
            stage[k].append(v)
    if len(birth) < 40:
        return None
    out = {"birth": np.array(birth, float), "death": np.array(death, float),
           "surv": np.array(surv, float), "cens": np.array(cens, bool)}
    for k in STAGES:
        out[k] = np.array(stage[k], float)
    return out


def cohort_percentiles(r):
    """For every offspring, percentile of each stage cost within its live cohort."""
    birth, death = r["birth"], r["death"]
    n = len(birth)
    pct = {s: np.full(n, np.nan) for s in STAGES}
    for i in range(n):
        t = birth[i]
        alive = (birth <= t) & (death > t)
        alive[i] = False
        if alive.sum() < 3:
            continue
        for s in STAGES:
            pct[s][i] = (r[s][alive] < r[s][i]).mean()
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--island", type=int, default=0)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    per_inst = {}   # instance -> {stage -> C of cohort-percentile vs survival}
    for inst in args.instances:
        idir = logs / inst / f"island_{args.island}"
        if not idir.exists():
            print(f"{inst:<20} [no logs]", flush=True); continue
        pct_all = {s: [] for s in STAGES}
        surv_all, ev_all = [], []
        nrun = 0
        for run_dir in sorted(idir.glob("run-*")):
            r = load_run(run_dir)
            if r is None:
                continue
            nrun += 1
            pct = cohort_percentiles(r)
            for s in STAGES:
                pct_all[s].append(pct[s])
            surv_all.append(r["surv"]); ev_all.append(~r["cens"])
        if nrun == 0:
            print(f"{inst:<20} [no instrumented runs]", flush=True); continue
        surv = np.concatenate(surv_all)
        ev = np.concatenate(ev_all)
        cs = {}
        for s in STAGES:
            x = np.concatenate(pct_all[s])
            m = ~np.isnan(x)
            try:
                cs[s] = concordance_index_censored(ev[m], surv[m], x[m])[0]
            except Exception:
                cs[s] = np.nan
        per_inst[inst] = cs
        n = int(m.sum()) if 'm' in dir() else len(surv)
        print(f"loaded {inst:<20} runs={nrun} n={len(surv):>5}  " +
              "  ".join(f"{s.replace('_cost','').replace('_vnd',''):>10}={cs[s]:.3f}" for s in STAGES),
              flush=True)

    if not per_inst:
        print("no data"); return

    insts = list(per_inst.keys())
    rows = []
    for s in STAGES:
        cvals = np.array([per_inst[i][s] for i in insts], float)
        rows.append({"stage": s, "C_mean": round(np.nanmean(cvals), 4),
                     **{i: round(per_inst[i][s], 3) for i in insts}})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "partial_vnd_signal.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print(" COHORT-PERCENTILE SURVIVAL C-INDEX BY VND STAGE  (0.5=noise; higher=more signal)")
    print("=" * 78)
    hdr = f"{'stage':<18}{'C_mean':>9}   " + "".join(f"{i[:11]:>13}" for i in insts)
    print(hdr); print("-" * len(hdr))
    for _, r in res.iterrows():
        print(f"{r['stage']:<18}{r['C_mean']:>9.3f}   " + "".join(f"{r[i]:>13.3f}" for i in insts))
    print("\nRead the ladder pre -> post_greed -> post_lkh -> post_vnd:")
    print("  jump AT post_greed  => cheap pre-LKH filter is viable (WIN)")
    print("  jump only AT post_lkh => LKH creates the signal; no cheap filter (null)")
    print(f"\nsaved -> {out_dir/'partial_vnd_signal.csv'}")


if __name__ == "__main__":
    main()
