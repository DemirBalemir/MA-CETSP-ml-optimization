# ml/scripts/predict_gbsa.py

import sys
import json
import pickle
import pandas as pd
from pathlib import Path


# Resolve project root
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

# Load model
model_path = project_root / "ml" / "models" / "gbsa_model.pkl"
with open(model_path, "rb") as f:
    gbsa, feature_names = pickle.load(f)

# Read JSON path argument
json_file = sys.argv[1]
with open(json_file, "r") as f:
    feats = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([feats])

# Reorder columns to match training
df = df.reindex(columns=feature_names, fill_value=0.0)

# Predict (returns risk score)
score = gbsa.predict(df)[0]

# Output ONLY a float
print(float(score))
