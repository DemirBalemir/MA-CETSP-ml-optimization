import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchtuples as tt
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

# Load all artifacts once at startup
with open(model_dir / "mtlr_config.json", "r") as f:
    config = json.load(f)

with open(model_dir / "mtlr_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_cols  = config["feature_cols"]
num_durations = config["num_durations"]

net = tt.practical.MLPVanilla(
    in_features=config["in_features"],
    num_nodes=config["num_nodes"],
    out_features=num_durations,
    batch_norm=config["batch_norm"],
    dropout=config["dropout"],
)
net.load_state_dict(torch.load(model_dir / "mtlr_net.pt", map_location="cpu",
                               weights_only=True))
net.eval()

# Replicate pycox MTLR.predict_pmf directly (no optimizer / wrapper needed):
#   preds → cumsum_reverse → pad_col(0) → softmax → drop last col → PMF
# risk = sum(PMF) = 1 - survival_at_last_bin
def _predict(feats: dict) -> float:
    df = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
    X_norm = scaler.transform(df.values).astype("float32")
    with torch.no_grad():
        phi      = net(torch.FloatTensor(X_norm))           # (1, num_durations)
        phi_crev = phi.flip(1).cumsum(1).flip(1)            # cumsum_reverse
        phi_pad  = F.pad(phi_crev, [0, 1], value=0.0)      # pad zero at end
        pmf      = F.softmax(phi_pad, dim=1)[:, :-1]       # (1, num_durations)
        risk     = float(pmf.sum(1)[0])                    # = 1 - surv_at_last_bin
    return risk


if remaining and remaining[0] != "--model_dir":
    # One-shot mode (legacy): python predict_mtlr.py <json_path>
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
            print(f"[MTLR ERROR] {e}", file=sys.stderr)
            print(0.0, flush=True)
