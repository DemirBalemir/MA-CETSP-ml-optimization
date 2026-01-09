from pathlib import Path
import json
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
        df = df.drop(columns=low_var_cols)

    # ------------------------------------------
    # 4) Save cleaned dataset
    # ------------------------------------------
    dataset_dir = project_root / "ml" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "cox_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"[INFO] Saved CLEAN dataset → {dataset_path}")
    
    # ------------------------------------------
    # 5) Train Cox model
    # ------------------------------------------
    cph = CoxPHFitter()
    cph.fit(df, duration_col="survival_time", event_col="event_observed")

    print("[INFO] Cox model fitted.")
    print("[INFO] C-index:", cph.concordance_index_)

    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["survival_time", "event_observed"])

    norm = {}
    for col in X.columns:
        norm[col] = {
            "mean": float(cph._norm_mean[col]),
            "std": float(cph._norm_std[col]),
        }

    with open(models_dir / "cox_norm.json", "w") as f:
        json.dump(norm, f, indent=2)

    print("[INFO] Saved Cox normalization stats → cox_norm.json")

    # ------------------------------------------
    # 6) Save coefficients
    # ------------------------------------------


    coeffs = dict(zip(
        cph.params_.index.tolist(),
        cph.params_.values.tolist()
    ))

    coef_path = models_dir / "cox_coeffs.json"
    with open(coef_path, "w") as f:
        json.dump(coeffs, f, indent=2)

    print(f"[INFO] Saved Cox coefficients → {coef_path}")
    # ------------------------------------------
    # 7) Compute Cox threshold (percentile-based)
    # ------------------------------------------
    scores = cph.predict_partial_hazard(df)
    
    # Reject worst 30% (tune this)
    q = 0.7
    threshold = float(scores.quantile(q))

    cox_meta = {
        "threshold": threshold,
        "quantile": q,
        "n_samples": len(df)
    }

    meta_path = models_dir / "cox_meta.json"
    with open(meta_path, "w") as f:
        json.dump(cox_meta, f, indent=2)

    print(f"[INFO] Saved Cox threshold → {meta_path}")
    print(f"[INFO] Cox threshold (q={q}): {threshold}")


if __name__ == "__main__":
    main()
