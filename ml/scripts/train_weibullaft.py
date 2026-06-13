from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import WeibullAFTFitter

from loader import load_all_logs, load_run_dir, get_default_log_root
from features import build_feature_dataset
from threshold_utils import compute_threshold


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None,
                    help="Override output model directory (for parallel islands)")
    ap.add_argument("--log_dir", default=None,
                    help="Island-specific log folder (train only on this island's solutions)")
    ap.add_argument("--logfile", default=None,
                    help="Redirect stdout+stderr to this file (avoids shell-redirect quoting issues)")
    args, _ = ap.parse_known_args()

    _logfile_handle = None
    if args.logfile:
        _logfile_handle = open(args.logfile, "w", buffering=1)
        sys.stdout = _logfile_handle
        sys.stderr = _logfile_handle

    project_root = get_project_root()

    # ---- 1) Load logs ----
    if args.log_dir:
        log_dir_path = Path(args.log_dir)
        print(f"[INFO] Loading island logs from {log_dir_path}")
        if log_dir_path.exists() and log_dir_path.is_dir():
            rows = load_run_dir(log_dir_path)
            if rows:
                df_logs = pd.DataFrame(rows)
            else:
                print("[WARN] Island log dir is empty, falling back to all logs")
                df_logs = load_all_logs(get_default_log_root())
        else:
            print(f"[WARN] Island log dir not found: {log_dir_path}, falling back to all logs")
            df_logs = load_all_logs(get_default_log_root())
    else:
        log_root = get_default_log_root()
        print(f"[INFO] Loading logs from {log_root}")
        df_logs = load_all_logs(log_root)

    # ---- 2) Feature extraction ----
    df = build_feature_dataset(df_logs)

    # ---- 3) Prepare data ----
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    if "instance_index" in df.columns:
        df = df.drop(columns=["instance_index"])

    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.dropna()
    df = df[df["survival_time"] > 0]   # WeibullAFT requires strictly positive durations

    X = df.drop(columns=["event_observed", "survival_time"])
    survival_times = df["survival_time"].values
    events         = df["event_observed"].values
    feature_cols   = list(X.columns)

    n = len(X)
    print(f"[INFO] Dataset size: {n} samples")

    # ---- 4) Hold-out split for threshold search ----
    if n >= 30:
        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=0.20,
            stratify=events,
            random_state=42,
        )
        df_train_val = X.iloc[train_idx].copy()
        df_train_val["duration"] = survival_times[train_idx]
        df_train_val["event"]    = events[train_idx]

        aft_val = WeibullAFTFitter()
        aft_val.fit(df_train_val, duration_col="duration", event_col="event")
        val_risks    = -aft_val.predict_median(X.iloc[val_idx]).values
        val_survival = survival_times[val_idx]
        print(f"[INFO] Validation set: {len(val_idx)} samples for threshold search")
    else:
        val_risks    = np.array([])
        val_survival = np.array([])
        print("[WARN] Too few samples for hold-out split; using default percentile")

    # ---- 5) Train final model on ALL data ----
    df_all = X.copy()
    df_all["duration"] = survival_times
    df_all["event"]    = events

    aft = WeibullAFTFitter()
    aft.fit(df_all, duration_col="duration", event_col="event")
    print("[INFO] WeibullAFT training completed.")

    # Risk score: negate median survival (longer survival = lower risk)
    full_risks = -aft.predict_median(X).values

    # ---- 6) Adaptive threshold search ----
    threshold, best_pct, obj = compute_threshold(full_risks, val_risks, val_survival)

    print(f"[INFO] Optimal rejection percentile: {best_pct}%  "
          f"(objective={obj:.4f})  threshold={threshold:.6f}")

    # ---- 7) Save model and meta ----
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(model_dir / "weibullaft_model.pkl", "wb") as f:
        pickle.dump((aft, feature_cols), f)
    print(f"[INFO] Saved WeibullAFT model -> {model_dir / 'weibullaft_model.pkl'}")

    meta = {
        "threshold":            threshold,
        "rejection_percentile": best_pct,
        "objective":            obj,
        "n_samples":            n,
    }
    with open(model_dir / "weibullaft_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved WeibullAFT meta  -> {model_dir / 'weibullaft_meta.json'}")


if __name__ == "__main__":
    main()
