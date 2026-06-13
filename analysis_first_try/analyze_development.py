"""
Development set analysis for MA-CETSP.

Reads parsed CSVs from analysis/parsed/ and produces:
  - Model comparison table (degradation, speedup, rejection rate, wins)
  - Instance group breakdown
  - best_iter convergence analysis
  - Threshold stability analysis
  - Wilcoxon signed-rank test (each model vs BASELINE)
  - Decision table with recommendations

Usage:
    python analysis/analyze_development.py
    python analysis/analyze_development.py --parsed_dir analysis/parsed --out_dir analysis/results
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ── instance → group mapping ──────────────────────────────────────────────────
INSTANCE_GROUP = {
    "bonus1000":          "varied",
    "bubbles5":           "varied",
    "concentricCircles1": "varied",
    "rotatingDiamonds1":  "varied",
    "team4_400":          "varied",
    "d493_or2":           "or2",
    "kroD100_or10":       "or10",
    "rat195_or10":        "or10",
    "lin318_or30":        "or30",
    "dsj1000rdmRad":      "rdmRad",
    "kroD100rdmRad":      "rdmRad",
    "team6_500rdmRad":    "rdmRad",
    "car_door_25":        "real_world",
    "car_door_35":        "real_world",
    "car_door_45":        "real_world",
}

MODEL_ORDER = [
    "BASELINE", "RSF", "DEEPSURV", "WEIBULLAFT", "GBSA",
    "MTLR", "KNN", "ELASTICNET", "COX", "SSVM",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def separator(char="=", width=72):
    print(char * width)

def section(title):
    print()
    separator()
    print(f"  {title}")
    separator()

def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  saved {path.name}")

def flag(degradation, reject_rate):
    """Simple recommendation flag based on degradation and rejection rate."""
    if degradation > 5.0 or reject_rate > 35.0:
        return "[!!] AGGRESSIVE"
    if degradation > 2.0:
        return "[~]  MODERATE"
    return "[OK] good"

# ── main analysis ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_dir", default="analysis/parsed")
    parser.add_argument("--out_dir",    default="analysis/results")
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading CSVs...")
    summaries  = pd.read_csv(parsed_dir / "island_summaries.csv")
    thresholds = pd.read_csv(parsed_dir / "thresholds.csv")
    training   = pd.read_csv(parsed_dir / "training_events.csv")

    # attach group
    summaries["group"] = summaries["instance"].map(INSTANCE_GROUP).fillna("unknown")

    # rejection rate per island run
    summaries["reject_rate"] = (
        summaries["ml_rejects"] / summaries["effective_iter"] * 100
    ).round(2)

    instances = sorted(summaries["instance"].unique())
    models    = [m for m in MODEL_ORDER if m in summaries["model"].unique()]

    # ── per-run baseline lookup: instance × run × seed → baseline best & time ─
    baseline = summaries[summaries["model"] == "BASELINE"][
        ["instance", "run", "seed", "best", "total_t"]
    ].rename(columns={"best": "bl_best", "total_t": "bl_time"})

    merged = summaries.merge(baseline, on=["instance", "run", "seed"], how="left")
    merged["degradation"] = (
        (merged["best"] - merged["bl_best"]) / merged["bl_best"] * 100
    ).round(4)
    merged["speedup"] = (merged["bl_time"] / merged["total_t"]).round(4)

    # ── 1. MODEL COMPARISON TABLE ─────────────────────────────────────────────
    section("1. MODEL COMPARISON  (aggregated over all instances × runs)")

    # compute wins: per (instance,run,seed) which model has global min best
    merged["run_min_best"] = merged.groupby(
        ["instance", "run", "seed"])["best"].transform("min")
    merged["is_winner"] = merged["best"] == merged["run_min_best"]

    rows = []
    for model in models:
        sub = merged[merged["model"] == model]
        if sub.empty:
            continue
        wins = int(sub["is_winner"].sum())
        total_runs = len(sub)
        rows.append({
            "model":        model,
            "mean_best":    round(sub["best"].mean(), 4),
            "degradation%": round(sub["degradation"].mean(), 3),
            "speedup":      round(sub["speedup"].mean(), 3),
            "reject_rate%": round(sub["reject_rate"].mean(), 2),
            "wins":         wins,
            "win_rate%":    round(wins / total_runs * 100, 1),
        })

    comp = pd.DataFrame(rows)
    # sort: baseline first, rest by degradation
    baseline_row = comp[comp["model"] == "BASELINE"]
    rest = comp[comp["model"] != "BASELINE"].sort_values("degradation%")
    comp = pd.concat([baseline_row, rest], ignore_index=True)

    print(f"\n{'Model':<14} {'Deg%':>7} {'Speedup':>8} {'Rej%':>7} {'Wins':>6} {'Win%':>6}")
    print("-" * 52)
    for _, r in comp.iterrows():
        print(
            f"{r['model']:<14} {r['degradation%']:>7.2f} {r['speedup']:>8.3f}"
            f" {r['reject_rate%']:>7.1f} {r['wins']:>6} {r['win_rate%']:>5.1f}%"
        )

    write_csv(comp, out_dir / "model_comparison.csv")

    # ── 2. INSTANCE GROUP BREAKDOWN ───────────────────────────────────────────
    section("2. INSTANCE GROUP BREAKDOWN")

    group_rows = []
    for group in sorted(merged["group"].unique()):
        g = merged[merged["group"] == group]
        for model in models:
            sub = g[g["model"] == model]
            if sub.empty:
                continue
            group_rows.append({
                "group":        group,
                "model":        model,
                "degradation%": round(sub["degradation"].mean(), 3),
                "speedup":      round(sub["speedup"].mean(), 3),
                "reject_rate%": round(sub["reject_rate"].mean(), 2),
                "n_runs":       len(sub),
            })

    group_df = pd.DataFrame(group_rows)

    for group in sorted(group_df["group"].unique()):
        print(f"\n  [{group}]")
        sub = group_df[group_df["group"] == group].sort_values("degradation%")
        print(f"  {'Model':<14} {'Deg%':>7} {'Speedup':>8} {'Rej%':>7}")
        print("  " + "-" * 40)
        for _, r in sub.iterrows():
            if r["model"] == "BASELINE":
                print(f"  {r['model']:<14} {' -- ':>7} {'1.000':>8} {'0.0':>7}")
            else:
                print(
                    f"  {r['model']:<14} {r['degradation%']:>7.2f}"
                    f" {r['speedup']:>8.3f} {r['reject_rate%']:>7.1f}"
                )

    write_csv(group_df, out_dir / "group_breakdown.csv")

    # ── 3. BEST_ITER ANALYSIS (convergence speed) ─────────────────────────────
    section("3. CONVERGENCE SPEED  (best_iter = iteration where best solution found)")

    iter_rows = []
    for model in models:
        sub = merged[merged["model"] == model]
        if sub.empty:
            continue
        iter_rows.append({
            "model":          model,
            "best_iter_mean": round(sub["best_iter"].mean(), 1),
            "best_iter_std":  round(sub["best_iter"].std(), 1),
            "best_iter_min":  int(sub["best_iter"].min()),
            "best_iter_max":  int(sub["best_iter"].max()),
            "eff_iter_mean":  round(sub["effective_iter"].mean(), 1),
        })

    iter_df = pd.DataFrame(iter_rows)
    print(f"\n{'Model':<14} {'BestIter_mean':>14} {'std':>7} {'min':>6} {'max':>6} {'EffIter_mean':>13}")
    print("-" * 62)
    for _, r in iter_df.iterrows():
        print(
            f"{r['model']:<14} {r['best_iter_mean']:>14.1f} {r['best_iter_std']:>7.1f}"
            f" {r['best_iter_min']:>6} {r['best_iter_max']:>6} {r['eff_iter_mean']:>13.1f}"
        )

    write_csv(iter_df, out_dir / "best_iter_analysis.csv")

    # ── 4. THRESHOLD STABILITY ────────────────────────────────────────────────
    section("4. THRESHOLD STABILITY  (std across 10 seeds per instance)")

    thr_rows = []
    for model in thresholds["model"].unique():
        if model in ("BASELINE", ""):
            continue
        sub = thresholds[thresholds["model"] == model]
        # per instance: std of threshold across seeds
        per_inst = sub.groupby("instance")["threshold"].std().reset_index()
        per_inst.columns = ["instance", "thr_std"]
        thr_rows.append({
            "model":          model,
            "thr_mean":       round(sub["threshold"].mean(), 4),
            "thr_std_mean":   round(per_inst["thr_std"].mean(), 4),  # avg std across instances
            "pct_mean":       round(sub["percentile"].mean(), 1),
            "obj_mean":       round(sub["objective"].mean(), 4),
            "obj_negative%":  round((sub["objective"] < 0).mean() * 100, 1),
        })

    thr_df = pd.DataFrame(thr_rows).sort_values("thr_std_mean")
    print(f"\n{'Model':<14} {'Thr_mean':>10} {'AvgStd':>8} {'Pct%':>6} {'Obj_mean':>9} {'NegObj%':>8}")
    print("-" * 60)
    for _, r in thr_df.iterrows():
        print(
            f"{r['model']:<14} {r['thr_mean']:>10.4f} {r['thr_std_mean']:>8.4f}"
            f" {r['pct_mean']:>6.1f} {r['obj_mean']:>9.4f} {r['obj_negative%']:>7.1f}%"
        )
    print("\n  NegObj% = fraction of runs where threshold objective < 0 (model not learning)")

    write_csv(thr_df, out_dir / "threshold_stability.csv")

    # ── 5. WILCOXON TEST (per model vs BASELINE) ──────────────────────────────
    section("5. WILCOXON SIGNED-RANK TEST  (model vs BASELINE, per instance mean)")

    # For each model: 15 data points (one per instance = mean best over 10 seeds)
    bl_per_inst = (
        merged[merged["model"] == "BASELINE"]
        .groupby("instance")["best"]
        .mean()
    )

    wilcoxon_rows = []
    print(f"\n{'Model':<14} {'p-value':>9} {'Stat':>8} {'Sig?':>6}  note")
    print("-" * 55)
    for model in models:
        if model == "BASELINE":
            continue
        sub = merged[merged["model"] == model].groupby("instance")["best"].mean()
        # align on same instances
        common = sub.index.intersection(bl_per_inst.index)
        x = sub[common].values
        y = bl_per_inst[common].values
        diff = x - y
        if (diff == 0).all():
            print(f"{model:<14} {'---':>9} {'---':>8} {'---':>6}  all ties (easy instances)")
            wilcoxon_rows.append({"model": model, "p_value": None, "statistic": None, "significant": False, "note": "all ties"})
            continue
        try:
            stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            sig = "YES" if p < 0.05 else "no"
            note = "[!!]" if p < 0.05 else ""
            print(f"{model:<14} {p:>9.4f} {stat:>8.1f} {sig:>6}  {note}")
            wilcoxon_rows.append({"model": model, "p_value": round(p, 4), "statistic": round(stat, 1), "significant": p < 0.05, "note": note})
        except Exception as e:
            print(f"{model:<14}  error: {e}")
            wilcoxon_rows.append({"model": model, "p_value": None, "statistic": None, "significant": False, "note": str(e)})

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    write_csv(wilcoxon_df, out_dir / "wilcoxon_test.csv")

    # ── 6. TRAINING EVENT SUMMARY ─────────────────────────────────────────────
    section("6. TRAINING EVENTS  (when & how model gets trained)")

    early = training[training["stagnation"] == 1]
    hard  = training[training["soft_ceil"] == 1]
    print(f"\n  Total training events : {len(training)}")
    print(f"  Early (stagnation)    : {len(early)}  ({len(early)/len(training)*100:.1f}%)")
    print(f"  Hard ceil (iter=250)  : {len(hard)}   ({len(hard)/len(training)*100:.1f}%)")
    print(f"  Avg events at training: {training['events'].mean():.1f}")
    print(f"  Avg patience at train : {training['patience'].mean():.1f}")

    # per instance: how often early trigger
    inst_early = (
        training.groupby("instance")
        .apply(lambda df: (df["stagnation"] == 1).mean() * 100)
        .reset_index()
    )
    inst_early.columns = ["instance", "early_trigger%"]
    inst_early["group"] = inst_early["instance"].map(INSTANCE_GROUP)
    inst_early = inst_early.sort_values("early_trigger%", ascending=False)

    print(f"\n  {'Instance':<22} {'Group':<12} {'EarlyTrig%':>11}")
    print("  " + "-" * 48)
    for _, r in inst_early.iterrows():
        print(f"  {r['instance']:<22} {r['group']:<12} {r['early_trigger%']:>10.1f}%")

    write_csv(inst_early, out_dir / "training_events_summary.csv")

    # ── 7. DECISION TABLE ─────────────────────────────────────────────────────
    section("7. DECISION TABLE  (recommendations before test set)")

    print(f"\n{'Model':<14} {'Deg%':>7} {'Speedup':>8} {'Rej%':>7}  {'Recommendation'}")
    print("-" * 68)
    for _, r in comp.iterrows():
        if r["model"] == "BASELINE":
            print(f"{r['model']:<14} {' -- ':>7} {'1.000':>8} {'0.0':>7}  reference")
            continue
        rec = flag(r["degradation%"], r["reject_rate%"])
        print(
            f"{r['model']:<14} {r['degradation%']:>7.2f} {r['speedup']:>8.3f}"
            f" {r['reject_rate%']:>7.1f}  {rec}"
        )

    print("\n  Thresholds:")
    print("    degradation > 5% OR reject_rate > 35%  ->  [!!] AGGRESSIVE")
    print("    degradation > 2%                        ->  [~]  MODERATE")
    print("    otherwise                               ->  [OK] good")

    separator()
    print(f"\nAll results saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
