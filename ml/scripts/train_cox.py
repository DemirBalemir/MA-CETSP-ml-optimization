# ml/scripts/train_cox.py

from pathlib import Path
import pickle
import pandas as pd
from lifelines import CoxPHFitter

from loader import load_all_logs, get_default_log_root
from features import build_feature_dataset


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()

    # ------------------------------------------
    # 1) Load raw logs
    # ------------------------------------------
    log_root = get_default_log_root()
    print(f"[INFO] Reading logs from: {log_root}")

    df_logs = load_all_logs(log_root)
    print(f"[INFO] Loaded {len(df_logs)} raw solutions")

    # ------------------------------------------
    # 2) Build initial feature dataset
    # ------------------------------------------
    df = build_feature_dataset(df_logs)

    # Add event_observed column
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    # ======================================================
    # 3) DATA CLEANING BEFORE FITTING  (CRITICAL)
    # ======================================================

    # Drop instance_index (causes convergence issues)
    if "instance_index" in df.columns:
        print("[INFO] Dropping column: instance_index")
        df = df.drop(columns=["instance_index"])

    # Replace infinities with 0
    df = df.replace([float("inf"), float("-inf")], 0.0)

    # Drop rows with NaN
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with NaN")

    # Drop low-variance (constant) columns
    low_var_cols = [c for c in df.columns if df[c].var() < 1e-9]
    if low_var_cols:
        print("[INFO] Dropping low-variance columns:", low_var_cols)
        df = df.drop(columns=low_var_cols)

    # ------------------------------------------
    # 4) Save cleaned dataset for debugging
    # ------------------------------------------
    dataset_dir = project_root / "ml" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "cox_dataset.csv"
    df.to_csv(dataset_path, index=False)

    print(f"[INFO] Saved CLEAN dataset to: {dataset_path}")

    # ------------------------------------------
    # 5) Train Cox model
    # ------------------------------------------
    cph = CoxPHFitter()
    cph.fit(df, duration_col="survival_time", event_col="event_observed")

    print("[INFO] Cox model fitted.")
    print("[INFO] Training concordance index:", cph.concordance_index_)

    # ------------------------------------------
    # 6) Save model
    # ------------------------------------------
    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "cox_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(cph, f)

    print(f"[INFO] Saved Cox model to: {model_path}")


if __name__ == "__main__":
    main()
