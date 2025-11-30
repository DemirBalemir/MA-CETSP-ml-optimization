import sys
import json
import pickle
import pandas as pd
from pathlib import Path

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

# Load RSF model
model_path = project_root / "ml" / "models" / "rsf_model.pkl"
with open(model_path, "rb") as f:
    rsf, feature_cols = pickle.load(f)

# Load JSON file path
json_path = sys.argv[1]

with open(json_path, "r") as f:
    feats = json.load(f)

# Convert JSON into DataFrame
df = pd.DataFrame([feats])

# Ensure consistent feature columns
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

df = df[feature_cols]

# Predict risk score
score = float(rsf.predict(df)[0])

print(score)     # C++ expects float only
