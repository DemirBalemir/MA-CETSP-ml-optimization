from pathlib import Path
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from loader import load_all_logs, get_default_log_root
from features import build_feature_dataset
from threshold_utils import compute_threshold


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None,
                    help="Override output model directory (for parallel islands)")
    args, _ = ap.parse_known_args()

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
    # A lightweight validation model (fewer trees) gives unbiased val-set
    # predictions used only to find the optimal rejection percentile.
    n = len(X)
    print(f"[INFO] Dataset size: {n} samples")

    if n >= 30:
        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=0.20,
            stratify=events,
            random_state=42,
        )
        rsf_val = RandomSurvivalForest(
            n_estimators=100,          # fast — only used for threshold search
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=0,
        )
        rsf_val.fit(X.iloc[train_idx], y[train_idx])
        val_risks    = rsf_val.predict(X.iloc[val_idx])
        val_survival = survival_times[val_idx]
        print(f"[INFO] Validation set: {len(val_idx)} samples for threshold search")
    else:
        # Too few samples — val_risks == full_risks (degenerate fallback)
        val_risks    = np.array([])
        val_survival = np.array([])
        print("[WARN] Too few samples for hold-out split; using default percentile")

    # ---- 5) Train final RSF on ALL data ----
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )
    rsf.fit(X, y)
    print("[INFO] RSF training completed.")

    # ---- 6) Adaptive threshold search ----
    full_risks = rsf.predict(X)

    threshold, best_pct, obj = compute_threshold(full_risks, val_risks, val_survival)

    print(f"[INFO] Optimal rejection percentile: {best_pct}%  "
          f"(objective={obj:.4f})  threshold={threshold:.6f}")

    # ---- 7) Save model and meta ----
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(model_dir / "rsf_model.pkl", "wb") as f:
        pickle.dump((rsf, list(X.columns)), f)
    print(f"[INFO] Saved RSF model → {model_dir / 'rsf_model.pkl'}")

    meta = {
        "threshold":          threshold,
        "rejection_percentile": best_pct,
        "objective":          obj,
        "n_samples":          n,
    }
    with open(model_dir / "rsf_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved RSF meta  → {model_dir / 'rsf_meta.json'}")


if __name__ == "__main__":
    main()
