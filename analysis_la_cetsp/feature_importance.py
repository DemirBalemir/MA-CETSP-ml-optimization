"""
Per-feature signal strength: univariate concordance index.

For each feature we use its raw value as the risk score and compute the
concordance index against survival. |C - 0.5| is the discrimination strength of
that single feature (0 = pure noise, 0.5 = perfect). Sign tells direction
(C>0.5: higher value => shorter survival; C<0.5: protective). Averaged over the
same baseline-island instances used by feature_ceiling.py.

Usage:
    python feature_importance.py --instances concentricCircles4 car_door_50 \
        bonus1000 kroD100_or2 rat195_or30 --max_per_instance 4000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_ceiling import FEATURES, load_instance
from sksurv.metrics import concordance_index_censored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+",
                    default=["concentricCircles4", "car_door_50", "bonus1000",
                             "kroD100_or2", "rat195_or30"])
    ap.add_argument("--island", type=int, default=9)
    ap.add_argument("--max_per_instance", type=int, default=4000)
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    rng = random.Random(0)
    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    per_inst = {}          # instance -> {feature -> signed C}
    for inst in args.instances:
        inst_dir = logs / inst
        if not inst_dir.exists():
            continue
        df = load_instance(inst_dir, args.island, args.max_per_instance, rng)
        if len(df) < 100:
            continue
        ev = df["event"].values.astype(bool)
        t = df["survival_time"].values.astype(float)
        cs = {}
        for f in FEATURES:
            x = df[f].values.astype(float)
            try:
                c = concordance_index_censored(ev, t, x)[0]
            except Exception:
                c = np.nan
            cs[f] = c
        per_inst[inst] = cs
        print(f"  loaded {inst}: n={len(df)}")

    if not per_inst:
        print("no data"); return

    # signal strength = |C - 0.5|, averaged over instances
    rows = []
    for f in FEATURES:
        cvals = np.array([per_inst[i][f] for i in per_inst], dtype=float)
        strength = np.nanmean(np.abs(cvals - 0.5))
        c_mean = np.nanmean(cvals)
        rows.append({
            "feature": f,
            "signal(|C-0.5|)": round(strength, 4),
            "C_mean": round(c_mean, 4),
            **{i: round(per_inst[i][f], 3) for i in per_inst},
        })
    res = pd.DataFrame(rows).sort_values("signal(|C-0.5|)", ascending=False)
    res.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8")

    inst_cols = list(per_inst.keys())
    print("\n" + "=" * (28 + 16 + 9 + 9 * len(inst_cols)))
    print("  PER-FEATURE UNIVARIATE SIGNAL  (|C-0.5|: 0=noise; sorted strongest first)")
    print("=" * (28 + 16 + 9 + 9 * len(inst_cols)))
    hdr = f"{'feature':<20}{'signal':>9}{'C_mean':>9}   " + "".join(f"{i[:8]:>9}" for i in inst_cols)
    print(hdr); print("-" * len(hdr))
    for _, r in res.iterrows():
        print(f"{r['feature']:<20}{r['signal(|C-0.5|)']:>9.3f}{r['C_mean']:>9.3f}   " +
              "".join(f"{r[i]:>9.3f}" for i in inst_cols))
    print(f"\nsaved -> {(out_dir / 'feature_importance.csv').resolve()}")
    print("Reference: multivariate ceiling (all features together) was ~0.53.")


if __name__ == "__main__":
    main()
