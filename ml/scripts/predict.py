import sys
import json
import pickle
import pandas as pd
from pathlib import Path

# Resolve project root (same logic as before)
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

# Load Cox model
model_path = project_root / "ml" / "models" / "cox_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# === CHANGE ONLY THIS PART ===
# Instead of raw JSON string → read JSON FILE PATH
json_file_path = sys.argv[1]

# Load JSON features from file
with open(json_file_path, "r") as f:
    features = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([features])
""""""""""""""""""""
# Predict hazard score
hazard_score = model.predict_partial_hazard(df)[0]


# Output ONLY the score (C++ expects plain float)
print(float(hazard_score))
