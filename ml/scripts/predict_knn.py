import sys
import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from knn_survival import KNNSurvival  # noqa: F401 — needed for pickle to resolve the class

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
model_path = model_dir / "knn_model.pkl"
with open(model_path, "rb") as f:
    knn, scaler, feature_cols = pickle.load(f)


def _predict(feats: dict) -> float:
    df = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
    X_norm = scaler.transform(df.values)
    return float(knn.predict(X_norm)[0])


if remaining and remaining[0] != "--model_dir":
    # One-shot mode (legacy): python predict_knn.py <json_path>
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
            print(f"[KNN ERROR] {e}", file=sys.stderr)
            print(0.0, flush=True)
