import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

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
model_path = model_dir / "elasticnet_model.pkl"
with open(model_path, "rb") as f:
    model, scaler, feature_cols = pickle.load(f)


def _predict(feats: dict) -> float:
    df = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
    X_norm = scaler.transform(df.values).astype("float64")
    risks = model.predict(X_norm)
    if risks.ndim == 2:
        risks = risks[:, risks.shape[1] // 2]
    return float(risks[0])


if remaining and remaining[0] != "--model_dir":
    # One-shot mode (legacy): python predict_elasticnet.py <json_path>
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
            print(f"[ELASTICNET ERROR] {e}", file=sys.stderr)
            print(0.0, flush=True)
