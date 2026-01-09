# ml/scripts/train_gbsa.py

from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from loader import load_all_logs, get_default_log_root
from features import build_feature_dataset


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()

    # ---- 1) Load logs ----
    log_root = get_default_log_root()
    print(f"[INFO] Loading logs from {log_root}")
    df_logs = load_all_logs(log_root)

    # ---- 2) Feature extraction ----
    df = build_feature_dataset(df_logs)

    # ---- 3) Convert to survival format ----
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    # Remove instance_index (noise)
    if "instance_index" in df.columns:
        df = df.drop(columns=["instance_index"])

    # Clean dataset
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.dropna()

    # Prepare survival target
    y = np.array(
        [(bool(ev), float(t)) for ev, t in zip(df["event_observed"], df["survival_time"])],
        dtype=[("event", "bool"), ("time", "float64")]
    )
    X = df.drop(columns=["event_observed", "survival_time"])

    # ---- 4) Train Gradient Boosting Survival model ----
    gbsa = GradientBoostingSurvivalAnalysis(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        random_state=0
    )

    print("[INFO] Training GBSA model...")
    gbsa.fit(X, y)
    print("[INFO] GBSA training completed!")

    # ---- 5) Save GBSA model ----
    model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(model_dir / "gbsa_model.pkl", "wb") as f:
        pickle.dump((gbsa, list(X.columns)), f)

    print(f"[INFO] Saved GBSA model → {model_dir / 'gbsa_model.pkl'}")

    # ---- 6) Compute and save threshold (median risk score) ----
    risks = gbsa.predict(X)
    median_threshold = float(np.percentile(risks, 80))   # top 20% risk

    meta_path = model_dir / "gbsa_meta.json"
    with open(meta_path, "w") as f:
        f.write(f"{{\"threshold\": {median_threshold}}}")

    print(f"[INFO] Saved GBSA threshold → {meta_path}")


if __name__ == "__main__":
    main()