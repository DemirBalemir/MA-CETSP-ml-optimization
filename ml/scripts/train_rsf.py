from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import json

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

    # ---- 3) Convert to scikit-survival format ----
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    # Remove instance_index
    if "instance_index" in df.columns:
        print("[INFO] Dropping instance_index")
        df = df.drop(columns=["instance_index"])

    # Drop NA + inf
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.dropna()

    # Prepare y label for scikit-survival
    y = np.array(
        [(bool(ev), float(t)) for ev, t in zip(df["event_observed"], df["survival_time"])],
        dtype=[("event", "bool"), ("time", "float64")]
    )

    X = df.drop(columns=["event_observed", "survival_time"])

    # ---- 4) Train RSF ----
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0
    )
    rsf.fit(X, y)

    print("[INFO] RSF training completed.")

    # =====================================================
    # 5) CALCULATE THRESHOLD FROM TRAINING SET (80 percentile)
    # =====================================================
    hazard_scores = rsf.predict(X)   # RSF hazard predictions
    threshold = float(np.percentile(hazard_scores, 80))

    print(f"[INFO] Computed RSF Threshold (80% percentile): {threshold}")

    # ---- save threshold ----
    model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(model_dir / "rsf_meta.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    print(f"[INFO] Saved meta file to: {model_dir / 'rsf_meta.json'}")

    # ---- 6) Save RSF model ----
    with open(model_dir / "rsf_model.pkl", "wb") as f:
        pickle.dump((rsf, list(X.columns)), f)

    print(f"[INFO] Saved RSF model to: {model_dir / 'rsf_model.pkl'}")


if __name__ == "__main__":
    main()
