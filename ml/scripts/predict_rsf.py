import sys
import json
import pickle
import pandas as pd
from pathlib import Path

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

# Load model once at startup
model_path = project_root / "ml" / "models" / "rsf_model.pkl"
with open(model_path, "rb") as f:
    rsf, feature_cols = pickle.load(f)


def _predict(feats: dict) -> float:
    df = pd.DataFrame([feats])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]
    return float(rsf.predict(df)[0])


if len(sys.argv) > 1:
    # One-shot mode (legacy): python predict_rsf.py <json_path>
    with open(sys.argv[1], "r") as f:
        feats = json.load(f)
    print(_predict(feats))
else:
    # Server mode: read one JSON line from stdin, write one score to stdout, repeat.
    # The C++ persistent process uses this mode.
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "EXIT":
            break
        try:
            feats = json.loads(line)
            print(_predict(feats), flush=True)
        except Exception as e:
            print(sys.stderr, f"[RSF ERROR] {e}", file=sys.stderr)
            print(0.0, flush=True)
