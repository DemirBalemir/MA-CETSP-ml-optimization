# ml/scripts/features.py

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_geometry_features(coords: list[list[float]]) -> dict:
 
    coords_arr = np.asarray(coords, dtype=float)

    if coords_arr.shape[0] < 2:
        return {
            "avg_edge_length": 0.0,
            "var_edge_length": 0.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_area": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "centroid_dist_sum": 0.0,
            "angle_variance": 0.0,
        }

    xs = coords_arr[:, 0]
    ys = coords_arr[:, 1]

    # --- Edge lengths ---
    deltas = np.diff(coords_arr, axis=0)
    edges = np.sqrt(np.sum(deltas**2, axis=1))
    avg_edge = float(edges.mean())
    var_edge = float(edges.var())

    # --- Bounding box ---
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    bbox_area = float(width * height)

    # --- Centroid ---
    cx = float(xs.mean())
    cy = float(ys.mean())

    # --- Distance to centroid (spread) ---
    centroid_distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    centroid_dist_sum = float(centroid_distances.sum())

    # --- Turning angles (smoothness) ---
    angles = []
    for i in range(1, len(coords_arr) - 1):
        a = coords_arr[i - 1]
        b = coords_arr[i]
        c = coords_arr[i + 1]

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-9 or norm_bc < 1e-9:
            continue

        cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.arccos(cosang)
        angles.append(ang)

    angle_variance = float(np.var(angles)) if len(angles) > 0 else 0.0

    return {
        "avg_edge_length": avg_edge,
        "var_edge_length": var_edge,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": bbox_area,
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_dist_sum": centroid_dist_sum,
        "angle_variance": angle_variance,
    }


def build_feature_dataset(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw dataframe output from loader.load_all_logs(),
    extracts geometry features from 'coords', and
    returns a purely numerical dataframe ready for ML.
    """
    feature_rows = []

    for _, row in df_logs.iterrows():
        feats = extract_geometry_features(row["coords"])

        # Also adding the label and other numeric columns:
        feats["pre_vnd_cost"] = float(row["pre_vnd_cost"])
        feats["survival_time"] = int(row["survival_time"])
        feats["survival_time"] = int(row["survival_time"])
        feats["censored"] = bool(row["censored"])
        feats["instance_index"] = int(row["instance_index"])

        feature_rows.append(feats)

    df_feat = pd.DataFrame(feature_rows)
    return df_feat


if __name__ == "__main__":
    # for a small smoke test (e.g., use with loader)
    from loader import load_all_logs

    df_logs = load_all_logs()
    df_feat = build_feature_dataset(df_logs)
    print(df_feat.head())
    print("Feature dataset shape:", df_feat.shape)