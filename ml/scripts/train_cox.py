from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter

from loader import load_all_logs, load_run_dir, get_default_log_root
from features import build_feature_dataset
from threshold_utils import compute_threshold


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None,
                    help="Override output model directory (for parallel islands)")
    ap.add_argument("--log_dir", default=None,
                    help="Island-specific log folder (train only on this island's solutions)")
    args, _ = ap.parse_known_args()

    project_root = get_project_root()

    # ------------------------------------------
    # 1) Load raw logs
    # ------------------------------------------
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
        print(f"[INFO] Reading logs from: {log_root}")
        df_logs = load_all_logs(log_root)
    print(f"[INFO] Loaded {len(df_logs)} raw solutions")

    # ------------------------------------------
    # 2) Build feature dataset
    # ------------------------------------------
    df = build_feature_dataset(df_logs)

    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    # ------------------------------------------
    # 3) Data cleaning
    # ------------------------------------------
    if "instance_index" in df.columns:
        df = df.drop(columns=["instance_index"])

    df = df.replace([float("inf"), float("-inf")], 0.0)

    before = len(df)
    df = df.dropna()
    print(f"[INFO] Dropped {before - len(df)} rows with NaN")

    low_var_cols = [c for c in df.columns if df[c].var() < 1e-9]
    if low_var_cols:
        print(f"[INFO] Dropping low-variance columns: {low_var_cols}")
        df = df.drop(columns=low_var_cols)

    # ------------------------------------------
    # 4) Save cleaned dataset
    # ------------------------------------------
    dataset_dir = project_root / "ml" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "cox_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"[INFO] Saved clean dataset → {dataset_path}")

    # ------------------------------------------
    # 5) Hold-out split for threshold search
    # ------------------------------------------
    n = len(df)
    print(f"[INFO] Dataset size: {n} samples")
    events = df["event_observed"].values

    if n >= 30:
        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=0.20,
            stratify=events,
            random_state=42,
        )
        df_train_val = df.iloc[train_idx]
        df_val       = df.iloc[val_idx]

        cph_val = CoxPHFitter()
        cph_val.fit(df_train_val, duration_col="survival_time", event_col="event_observed")

        val_scores   = cph_val.predict_partial_hazard(df_val).values
        val_survival = df_val["survival_time"].values
        print(f"[INFO] Validation set: {len(val_idx)} samples for threshold search")
    else:
        val_scores   = np.array([])
        val_survival = np.array([])
        print("[WARN] Too few samples for hold-out split; using default percentile")

    # ------------------------------------------
    # 6) Train final Cox model on ALL data
    # ------------------------------------------
    cph = CoxPHFitter()
    cph.fit(df, duration_col="survival_time", event_col="event_observed")
    print("[INFO] Cox model fitted.")
    print(f"[INFO] C-index: {cph.concordance_index_:.4f}")

    if args.model_dir:
        models_dir = Path(args.model_dir)
    else:
        models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # 7) Save normalization stats and coefficients
    # ------------------------------------------
    X = df.drop(columns=["survival_time", "event_observed"])

    norm = {
        col: {"mean": float(cph._norm_mean[col]), "std": float(cph._norm_std[col])}
        for col in X.columns
    }
    with open(models_dir / "cox_norm.json", "w") as f:
        json.dump(norm, f, indent=2)
    print("[INFO] Saved Cox normalization stats → cox_norm.json")

    coeffs = dict(zip(cph.params_.index.tolist(), cph.params_.values.tolist()))
    with open(models_dir / "cox_coeffs.json", "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"[INFO] Saved Cox coefficients → cox_coeffs.json")

    # ------------------------------------------
    # 8) Adaptive threshold search
    # ------------------------------------------
    full_scores = cph.predict_partial_hazard(df).values

    threshold, best_pct, obj = compute_threshold(full_scores, val_scores, val_survival)

    print(f"[INFO] Optimal rejection percentile: {best_pct}%  "
          f"(objective={obj:.4f})  threshold={threshold:.6f}")

    cox_meta = {
        "threshold":            threshold,
        "rejection_percentile": best_pct,
        "objective":            obj,
        "concordance_index":    float(cph.concordance_index_),
        "n_samples":            n,
    }
    with open(models_dir / "cox_meta.json", "w") as f:
        json.dump(cox_meta, f, indent=2)
    print(f"[INFO] Saved Cox meta → cox_meta.json")


if __name__ == "__main__":
    main()
