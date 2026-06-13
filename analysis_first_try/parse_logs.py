"""
Log parser for MA-CETSP experiment results.

Extracts from run*.log files:
  - island_summaries.csv    : final per-island results
  - training_events.csv     : when ML training triggered + details
  - thresholds.csv          : threshold calibration per island per run
  - convergence.csv         : best cost over iterations (sampled every SAMPLE_STEP iters)
  - rejection_timeline.csv  : rejection counts per WINDOW_SIZE iter window
  - rejections.csv          : individual rejection events with score

Usage:
    python analysis/parse_logs.py
    python analysis/parse_logs.py --results_dir solutions/experiment_results --out_dir analysis/parsed
"""

import re
import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

SAMPLE_STEP = 10   # convergence sampled every N iterations
WINDOW_SIZE = 50   # rejection timeline window size

# ── regex patterns ────────────────────────────────────────────────────────────

RE_LOG = re.compile(
    r"\[Island (\d+)\] \[LOG\] iter=(\d+) best=([\d.]+) patience=(\d+) "
    r"ml=(\S+) iter_t=([\d.]+) total_t=([\d.]+)"
)
RE_TRAIN = re.compile(
    r"\[Island (\d+)\] \[ML\] Training triggered at iter (\d+) "
    r"\(events=(\d+), patience=(\d+), stagnation=(\d+), soft_ceil=(\d+), hard_ceil=(\d+)\)"
)
RE_STARTING = re.compile(
    r"\[Island (\d+)\].*?(?:starting|Starting training)\s*[—\-]*\s*(?:model=)?(\w+)"
)
RE_VAL_SET = re.compile(
    r"\[Island (\d+)\]\s+\[INFO\] Validation set: (\d+) samples"
)
RE_THRESHOLD = re.compile(
    r"\[Island (\d+)\]\s+\[INFO\] Optimal rejection percentile: (\d+)%\s+"
    r"\(objective=([-\d.]+)\)\s+threshold(?:\(pct\))?=([-\d.e+]+)"
)
RE_SCORE = re.compile(
    r"\[ML-(\w+)\] score=([-\d.e+]+) threshold=([-\d.e+]+)"
)
RE_REJECT = re.compile(
    r"\[ML\] Offspring rejected before VND \((\w+)\)\. (?:Score|score)=([-\d.e+]+)"
)
RE_SUMMARY = re.compile(
    r"\[Island (\d+)\] \[SUMMARY\] instance=(\S+) model=(\S+) best=([\d.]+) "
    r"best_iter=(\d+) effective_iter=(\d+) total_t=([\d.]+) ml_rejects=(\d+)"
)
RE_PARALLEL_ROW = re.compile(
    r"Island (\d+)\s+model=(\S+)\s+best=([\d.]+)\s+time=([\d.]+)s\s+rejected=(\d+)"
)
RE_GLOBAL_BEST = re.compile(
    r"GLOBAL BEST.*?island=(\d+)\s+model=(\S+)\s+value=([\d.]+)\s+time=([\d.]+)s"
)


def parse_log(log_path: Path, instance_name: str, run_id: int, seed: int):
    """Parse a single run log file, return dict of extracted tables."""

    summaries = []
    training_events = []
    thresholds = []
    convergence = []
    rejection_timeline = []  # will be built from rejections
    rejections = []

    # per-island tracking state
    island_model = {}          # island_id -> model name (from first LOG line)
    island_current_iter = {}   # island_id -> current iter
    island_best = {}           # island_id -> best value seen (from LOG lines)
    island_best_iter = {}      # island_id -> iter where best was last improved
    # rejection counts per (island, window)
    island_rej_window = defaultdict(int)  # (island, window_idx) -> count
    # pending threshold info (val_set size comes before threshold line)
    pending_val_samples = {}   # island_id -> val_samples
    # parallel row fallback: island_id -> row dict (from end-of-run table)
    parallel_rows = {}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # ── model name from "starting" or "Starting training" lines ──────
            if "starting" in line.lower() and "[Island" in line:
                m = re.search(r"\[Island (\d+)\].*?[Ss]tarting training \((\w+)\)", line)
                if m:
                    island_model[int(m.group(1))] = m.group(2)
                else:
                    m = re.search(r"\[Island (\d+)\].*?model=(\w+)", line)
                    if m:
                        island_model[int(m.group(1))] = m.group(2)

            # ── convergence (sampled) ────────────────────────────────────────
            m = RE_LOG.search(line)
            if m:
                isl, it, best, pat, ml, iter_t, total_t = m.groups()
                try:
                    isl, it = int(isl), int(it)
                    best_f, iter_t_f, total_t_f = float(best), float(iter_t), float(total_t)
                except ValueError:
                    continue  # malformed line from parallel interleaving
                if ml not in ("pending",):
                    island_model[isl] = ml
                island_current_iter[isl] = it
                # track best and the iter it was achieved for fallback recovery
                if isl not in island_best or best_f < island_best[isl]:
                    island_best[isl] = best_f
                    island_best_iter[isl] = it
                if it % SAMPLE_STEP == 0 or it == 1:
                    convergence.append({
                        "instance": instance_name,
                        "run": run_id,
                        "seed": seed,
                        "island": isl,
                        "model": ml,
                        "iter": it,
                        "best": best_f,
                        "patience": int(pat),
                        "iter_t": iter_t_f,
                        "total_t": total_t_f,
                    })
                continue

            # ── training events ──────────────────────────────────────────────
            m = RE_TRAIN.search(line)
            if m:
                isl, it, evts, pat, stag, soft, hard = m.groups()
                training_events.append({
                    "instance": instance_name,
                    "run": run_id,
                    "seed": seed,
                    "island": int(isl),
                    "train_iter": int(it),
                    "events": int(evts),
                    "patience": int(pat),
                    "stagnation": int(stag),
                    "soft_ceil": int(soft),
                    "hard_ceil": int(hard),
                })
                continue

            # ── validation set size (preceding threshold line) ───────────────
            m = RE_VAL_SET.search(line)
            if m:
                isl, n = m.groups()
                pending_val_samples[int(isl)] = int(n)
                continue

            # ── threshold calibration ────────────────────────────────────────
            m = RE_THRESHOLD.search(line)
            if m:
                isl, pct, obj, thr = m.groups()
                isl = int(isl)
                thresholds.append({
                    "instance": instance_name,
                    "run": run_id,
                    "seed": seed,
                    "island": isl,
                    "model": island_model.get(isl, ""),
                    "percentile": int(pct),
                    "threshold": float(thr),
                    "objective": float(obj),
                    "val_samples": pending_val_samples.pop(isl, None),
                })
                continue

            # ── rejection events ─────────────────────────────────────────────
            # Score line comes first, then optionally the "rejected" line
            m = RE_REJECT.search(line)
            if m:
                model, score = m.groups()
                try:
                    score_f = float(score)
                except ValueError:
                    continue  # malformed line from parallel interleaving, skip
                isl_guess = next(
                    (k for k, v in island_model.items() if v == model), None
                )
                cur_iter = island_current_iter.get(isl_guess, 0)
                window = cur_iter // WINDOW_SIZE
                island_rej_window[(isl_guess, window)] += 1
                rejections.append({
                    "instance": instance_name,
                    "run": run_id,
                    "seed": seed,
                    "island": isl_guess,
                    "model": model,
                    "score": score_f,
                    "approx_iter": cur_iter,
                    "window": window,
                })
                continue

            # ── island summary ───────────────────────────────────────────────
            m = RE_SUMMARY.search(line)
            if m:
                isl, inst, mdl, best, best_it, eff_it, t, rej = m.groups()
                summaries.append({
                    "instance": instance_name,
                    "run": run_id,
                    "seed": seed,
                    "island": int(isl),
                    "model": mdl,
                    "best": float(best),
                    "best_iter": int(best_it),
                    "effective_iter": int(eff_it),
                    "total_t": float(t),
                    "ml_rejects": int(rej),
                })
                continue

            # ── parallel row fallback (end-of-run summary table) ────────────
            # Catches islands whose [SUMMARY] line was corrupted by interleaved
            # thread output. best_iter / effective_iter are recovered from the
            # LOG-line tracking above.
            m = RE_PARALLEL_ROW.search(line)
            if m:
                isl, mdl, best, t, rej = m.groups()
                isl = int(isl)
                parallel_rows[isl] = {
                    "model": mdl,
                    "best": float(best),
                    "total_t": float(t),
                    "ml_rejects": int(rej),
                }
                continue

    # ── fill in missing summaries from parallel rows ─────────────────────────
    seen_islands = {s["island"] for s in summaries}
    for isl, row in parallel_rows.items():
        if isl in seen_islands:
            continue
        summaries.append({
            "instance":       instance_name,
            "run":            run_id,
            "seed":           seed,
            "island":         isl,
            "model":          row["model"],
            "best":           row["best"],
            "best_iter":      island_best_iter.get(isl, -1),
            "effective_iter": island_current_iter.get(isl, -1),
            "total_t":        row["total_t"],
            "ml_rejects":     row["ml_rejects"],
        })

    # ── build rejection timeline from accumulated window counts ─────────────
    for (isl, window), count in island_rej_window.items():
        rejection_timeline.append({
            "instance": instance_name,
            "run": run_id,
            "seed": seed,
            "island": isl,
            "model": island_model.get(isl, ""),
            "window": window,
            "window_start_iter": window * WINDOW_SIZE,
            "window_end_iter": (window + 1) * WINDOW_SIZE - 1,
            "rejections": count,
        })

    return {
        "summaries": summaries,
        "training_events": training_events,
        "thresholds": thresholds,
        "convergence": convergence,
        "rejection_timeline": rejection_timeline,
        "rejections": rejections,
    }


def write_csv(rows: list, path: Path, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>6} rows -> {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="solutions/experiment_results",
        help="Path to experiment_results folder",
    )
    parser.add_argument(
        "--out_dir",
        default="analysis/parsed",
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    all_summaries = []
    all_training = []
    all_thresholds = []
    all_convergence = []
    all_timeline = []
    all_rejections = []

    # discover all run log files
    log_files = sorted(results_root.rglob("run*.log"))
    if not log_files:
        print(f"No run*.log files found under {results_root}")
        return

    print(f"Found {len(log_files)} log files\n")

    for log_path in log_files:
        # path structure: .../instance<N>_<timestamp>/run<R>_seed<S>.log
        instance_dir = log_path.parent.name          # e.g. instance0_20260525_110438
        run_file = log_path.stem                     # e.g. run1_seed1

        # parse instance index + run/seed from filenames
        m_inst = re.match(r"instance(\d+)", instance_dir)
        m_run  = re.match(r"run(\d+)_seed(\d+)", run_file)
        if not m_inst or not m_run:
            print(f"  [SKIP] unrecognised path: {log_path}")
            continue

        instance_idx = int(m_inst.group(1))
        run_id       = int(m_run.group(1))
        seed         = int(m_run.group(2))

        # derive instance name from log header (first [LOG] line's instance field)
        # Get instance name from [SUMMARY] lines — structured and interleave-safe
        instance_name = f"instance{instance_idx}"  # fallback
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = re.search(r'\[SUMMARY\] instance=(\S+) model=', line)
                    if m:
                        instance_name = m.group(1)
                        break
        except Exception:
            pass

        print(f"  parsing {instance_dir}/{run_file}.log  ({instance_name}, run={run_id}, seed={seed})")

        data = parse_log(log_path, instance_name, run_id, seed)
        all_summaries.extend(data["summaries"])
        all_training.extend(data["training_events"])
        all_thresholds.extend(data["thresholds"])
        all_convergence.extend(data["convergence"])
        all_timeline.extend(data["rejection_timeline"])
        all_rejections.extend(data["rejections"])

    print(f"\nWriting CSVs to {out_dir}/\n")

    write_csv(all_summaries, out_dir / "island_summaries.csv", [
        "instance", "run", "seed", "island", "model",
        "best", "best_iter", "effective_iter", "total_t", "ml_rejects",
    ])
    write_csv(all_training, out_dir / "training_events.csv", [
        "instance", "run", "seed", "island",
        "train_iter", "events", "patience", "stagnation", "soft_ceil", "hard_ceil",
    ])
    write_csv(all_thresholds, out_dir / "thresholds.csv", [
        "instance", "run", "seed", "island", "model",
        "percentile", "threshold", "objective", "val_samples",
    ])
    write_csv(all_convergence, out_dir / "convergence.csv", [
        "instance", "run", "seed", "island", "model",
        "iter", "best", "patience", "iter_t", "total_t",
    ])
    write_csv(all_timeline, out_dir / "rejection_timeline.csv", [
        "instance", "run", "seed", "island", "model",
        "window", "window_start_iter", "window_end_iter", "rejections",
    ])
    write_csv(all_rejections, out_dir / "rejections.csv", [
        "instance", "run", "seed", "island", "model",
        "score", "approx_iter", "window",
    ])

    print("\nDone.")


if __name__ == "__main__":
    main()
