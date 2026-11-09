import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pathlib import Path
from bisect import bisect_left

# ---- resolve model directory ----
ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", default=None,
                help="Path to the island's model directory")
args, remaining = ap.parse_known_args()

if args.model_dir:
    model_dir = Path(args.model_dir)
else:
    this_file = Path(__file__).resolve()
    model_dir = this_file.parents[2] / "ml" / "models"

# Load model once at startup
with open(model_dir / "deepsurv_config.json", "r") as f:
    config = json.load(f)

with open(model_dir / "deepsurv_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_cols = config["feature_cols"]

risks_sorted_path = model_dir / "deepsurv_risks_sorted.npy"
risks_sorted = np.load(risks_sorted_path) if risks_sorted_path.exists() else None

net = tt.practical.MLPVanilla(
    in_features=config["in_features"],
    num_nodes=config["num_nodes"],
    out_features=config["out_features"],
    batch_norm=config["batch_norm"],
    dropout=config["dropout"],
    output_bias=False,
)
net.load_state_dict(torch.load(model_dir / "deepsurv_net.pt", map_location="cpu", weights_only=True))
net.eval()


def _predict(feats: dict) -> float:
    df = pd.DataFrame([feats])
    df = df.reindex(columns=feature_cols, fill_value=0.0)
    X_norm = scaler.transform(df.values).astype("float32")
    with torch.no_grad():
        raw = float(net(torch.FloatTensor(X_norm)).numpy().flatten()[0])
    if risks_sorted is not None:
        # Return percentile rank (0-100) so threshold is scale-invariant
        rank = bisect_left(risks_sorted, raw)
        return rank / len(risks_sorted) * 100.0
    return raw


if remaining and remaining[0] != "--model_dir":
    # One-shot mode (legacy): python predict_deepsurv.py <json_path>
    with open(remaining[0], "r") as f:
        feats = json.load(f)
    print(_predict(feats))
else:
    # Server mode: read one JSON line from stdin, write one score to stdout, repeat.
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "EXIT":
            break
        try:
            feats = json.loads(line)
            print(_predict(feats), flush=True)
        except Exception as e:
            print(f"[DEEPSURV ERROR] {e}", file=sys.stderr)
            print(0.0, flush=True)
