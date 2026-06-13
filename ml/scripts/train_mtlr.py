from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
from pycox.models import MTLR
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from loader import load_all_logs, load_run_dir, get_default_log_root
from features import build_feature_dataset
from threshold_utils import compute_threshold


NUM_DURATIONS = 20


def _build_net(in_features: int) -> tt.practical.MLPVanilla:
    return tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=[64, 64],
        out_features=NUM_DURATIONS,
        batch_norm=True,
        dropout=0.1,
    )


def _train_model(X_norm, durations, events, epochs=200):
    labtrans = LabTransDiscreteTime(NUM_DURATIONS)
    y_discrete = labtrans.fit_transform(
        durations.astype("float32"),
        events.astype("float32"),
    )
    net = _build_net(X_norm.shape[1])
    model = MTLR(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    model.fit(
        X_norm, y_discrete,
        batch_size=min(256, len(X_norm)),
        epochs=epochs,
        verbose=False,
        callbacks=[tt.callbacks.EarlyStopping(patience=20)],
    )
    model.duration_index = labtrans.cuts
    return model, labtrans


def _risks(model, X_norm):
    surv = model.predict_surv_df(X_norm)   # (num_durations, n_samples)
    return 1.0 - surv.iloc[-1].values       # higher = shorter survival = higher risk


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
        scaler_val = StandardScaler()
        X_tr_norm  = scaler_val.fit_transform(X.iloc[train_idx].values).astype("float32")
        X_val_norm = scaler_val.transform(X.iloc[val_idx].values).astype("float32")

        model_val, _ = _train_model(
            X_tr_norm, survival_times[train_idx], events[train_idx], epochs=100
        )
        val_risks    = _risks(model_val, X_val_norm)
        val_survival = survival_times[val_idx]
        print(f"[INFO] Validation set: {len(val_idx)} samples for threshold search")
    else:
        val_risks    = np.array([])
        val_survival = np.array([])
        print("[WARN] Too few samples for hold-out split; using default percentile")

    # ---- 5) Train final MTLR on ALL data ----
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X.values).astype("float32")

    print("[INFO] Training MTLR model...")
    model, labtrans = _train_model(X_norm, survival_times, events, epochs=200)
    print("[INFO] MTLR training completed.")

    # ---- 6) Adaptive threshold search ----
    full_risks = _risks(model, X_norm)

    threshold, best_pct, obj = compute_threshold(full_risks, val_risks, val_survival)

    print(f"[INFO] Optimal rejection percentile: {best_pct}%  "
          f"(objective={obj:.4f})  threshold={threshold:.6f}")

    # ---- 7) Save all artifacts ----
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    torch.save(model.net.state_dict(), model_dir / "mtlr_net.pt")
    print(f"[INFO] Saved MTLR weights   -> {model_dir / 'mtlr_net.pt'}")

    with open(model_dir / "mtlr_labtrans.pkl", "wb") as f:
        pickle.dump(labtrans, f)
    print(f"[INFO] Saved MTLR labtrans  -> {model_dir / 'mtlr_labtrans.pkl'}")

    with open(model_dir / "mtlr_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Saved MTLR scaler    -> {model_dir / 'mtlr_scaler.pkl'}")

    config = {
        "in_features":   X_norm.shape[1],
        "num_nodes":     [64, 64],
        "num_durations": NUM_DURATIONS,
        "batch_norm":    True,
        "dropout":       0.1,
        "feature_cols":  feature_cols,
    }
    with open(model_dir / "mtlr_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved MTLR config    -> {model_dir / 'mtlr_config.json'}")

    meta = {
        "threshold":            threshold,
        "rejection_percentile": best_pct,
        "objective":            obj,
        "n_samples":            n,
    }
    with open(model_dir / "mtlr_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved MTLR meta      -> {model_dir / 'mtlr_meta.json'}")


if __name__ == "__main__":
    main()
