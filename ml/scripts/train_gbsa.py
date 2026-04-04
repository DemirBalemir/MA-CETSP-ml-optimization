from pathlib import Path
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from loader import load_all_logs, get_default_log_root
from features import build_feature_dataset
from threshold_utils import compute_threshold


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

    if "instance_index" in df.columns:
        df = df.drop(columns=["instance_index"])

    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.dropna()

    y = np.array(
        [(bool(ev), float(t)) for ev, t in zip(df["event_observed"], df["survival_time"])],
        dtype=[("event", "bool"), ("time", "float64")]
    )
    X = df.drop(columns=["event_observed", "survival_time"])
    survival_times = y["time"]
    events         = y["event"].astype(int)

    # ---- 4) Hold-out split for threshold search ----
    # A lightweight validation model (fewer estimators) gives unbiased
    # val-set predictions used only to find the optimal rejection percentile.
    n = len(X)
    print(f"[INFO] Dataset size: {n} samples")

    if n >= 30:
        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=0.20,
            stratify=events,
            random_state=42,
        )
        gbsa_val = GradientBoostingSurvivalAnalysis(
            learning_rate=0.1,
            n_estimators=100,          # fast — only used for threshold search
            max_depth=3,
            random_state=0,
        )
        gbsa_val.fit(X.iloc[train_idx], y[train_idx])
        val_risks    = gbsa_val.predict(X.iloc[val_idx])
        val_survival = survival_times[val_idx]
        print(f"[INFO] Validation set: {len(val_idx)} samples for threshold search")
    else:
        val_risks    = np.array([])
        val_survival = np.array([])
        print("[WARN] Too few samples for hold-out split; using default percentile")

    # ---- 5) Train final GBSA on ALL data ----
    gbsa = GradientBoostingSurvivalAnalysis(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        random_state=0,
    )
    print("[INFO] Training GBSA model...")
    gbsa.fit(X, y)
    print("[INFO] GBSA training completed!")

    # ---- 6) Adaptive threshold search ----
    full_risks = gbsa.predict(X)

    threshold, best_pct, obj = compute_threshold(full_risks, val_risks, val_survival)

    print(f"[INFO] Optimal rejection percentile: {best_pct}%  "
          f"(objective={obj:.4f})  threshold={threshold:.6f}")

    # ---- 7) Save model and meta ----
    model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(model_dir / "gbsa_model.pkl", "wb") as f:
        pickle.dump((gbsa, list(X.columns)), f)
    print(f"[INFO] Saved GBSA model → {model_dir / 'gbsa_model.pkl'}")

    meta = {
        "threshold":            threshold,
        "rejection_percentile": best_pct,
        "objective":            obj,
        "n_samples":            n,
    }
    with open(model_dir / "gbsa_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved GBSA meta  → {model_dir / 'gbsa_meta.json'}")


if __name__ == "__main__":
    main()
