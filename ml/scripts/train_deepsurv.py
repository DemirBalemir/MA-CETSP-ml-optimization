from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
from pycox.models import CoxPH

from loader import load_all_logs, load_run_dir, get_default_log_root
from features import build_feature_dataset
from threshold_utils import compute_threshold


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_net(in_features: int) -> tt.practical.MLPVanilla:
    return tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=[64, 64],
        out_features=1,
        batch_norm=True,
        dropout=0.1,
        output_bias=False,
    )


def _train_model(X_norm: np.ndarray, durations: np.ndarray, events: np.ndarray,
                 epochs: int = 200, batch_size: int = 256) -> tt.practical.MLPVanilla:
    net = _build_net(X_norm.shape[1])
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    y = (durations.astype("float32"), events.astype("float32"))
    bs = min(batch_size, len(X_norm))
    model.fit(X_norm, y, batch_size=bs, epochs=epochs, verbose=False,
              callbacks=[tt.callbacks.EarlyStopping(patience=20)])
    return net


def _predict(net: tt.practical.MLPVanilla, X_norm: np.ndarray) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return net(torch.FloatTensor(X_norm)).numpy().flatten()


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

    # ---- 3) Convert to survival format ----
    df["event_observed"] = (~df["censored"]).astype(int)
    df = df.drop(columns=["censored"])

    if "instance_index" in df.columns:
        df = df.drop(columns=["instance_index"])

    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.dropna()

    X = df.drop(columns=["event_observed", "survival_time"])
    survival_times = df["survival_time"].values
    events         = df["event_observed"].values

    n = len(X)
    print(f"[INFO] Dataset size: {n} samples")

    feature_cols = list(X.columns)

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

        net_val = _train_model(
            X_tr_norm,
            survival_times[train_idx],
            events[train_idx],
            epochs=100,
        )
        val_risks    = _predict(net_val, X_val_norm)
        val_survival = survival_times[val_idx]
        print(f"[INFO] Validation set: {len(val_idx)} samples for threshold search")
    else:
        val_risks    = np.array([])
        val_survival = np.array([])
        print("[WARN] Too few samples for hold-out split; using default percentile")

    # ---- 5) Train final DeepSurv on ALL data ----
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X.values).astype("float32")

    print("[INFO] Training DeepSurv model...")
    net = _train_model(X_norm, survival_times, events, epochs=200)
    print("[INFO] DeepSurv training completed.")

    # ---- 6) Adaptive threshold search ----
    full_risks = _predict(net, X_norm)

    _, best_pct, obj = compute_threshold(full_risks, val_risks, val_survival)

    # Use percentile rank as threshold so scale changes between training runs
    # don't cause all inference scores to fall below the threshold.
    # The predict server returns a percentile rank (0-100) relative to the
    # training distribution, and we store best_pct as the threshold.
    full_risks_sorted = np.sort(full_risks)
    threshold = float(best_pct)

    print(f"[INFO] Optimal rejection percentile: {best_pct}%  "
          f"(objective={obj:.4f})  threshold(pct)={threshold:.1f}")

    # ---- 7) Save model and meta ----
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = project_root / "ml" / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    torch.save(net.state_dict(), model_dir / "deepsurv_net.pt")
    print(f"[INFO] Saved DeepSurv weights -> {model_dir / 'deepsurv_net.pt'}")

    np.save(model_dir / "deepsurv_risks_sorted.npy", full_risks_sorted)
    print(f"[INFO] Saved sorted risks     -> {model_dir / 'deepsurv_risks_sorted.npy'}")

    config = {
        "in_features": X_norm.shape[1],
        "num_nodes":   [64, 64],
        "out_features": 1,
        "batch_norm":  True,
        "dropout":     0.1,
        "feature_cols": feature_cols,
    }
    with open(model_dir / "deepsurv_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved DeepSurv config  -> {model_dir / 'deepsurv_config.json'}")

    with open(model_dir / "deepsurv_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Saved DeepSurv scaler  -> {model_dir / 'deepsurv_scaler.pkl'}")

    meta = {
        "threshold":            threshold,
        "rejection_percentile": best_pct,
        "objective":            obj,
        "n_samples":            n,
    }
    with open(model_dir / "deepsurv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved DeepSurv meta    -> {model_dir / 'deepsurv_meta.json'}")


if __name__ == "__main__":
    main()
