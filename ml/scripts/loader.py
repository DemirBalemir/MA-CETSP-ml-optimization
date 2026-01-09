# ml/scripts/loader.py

import json
from pathlib import Path
import pandas as pd


def get_default_log_root() -> Path:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    return project_root / "solutions" / "ml_logs"


def load_run_dir(run_dir: Path) -> list[dict]:

    rows = []

    for json_file in sorted(run_dir.glob("sol-*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        rows.append({
            "run_folder": run_dir.name,
            "solution_file": json_file.name,
            "instance_index": data["instance_index"],
            "birth_iter": data["birth_iter"],
            "death_iter": data["death_iter"],
            "survival_time": data["survival_iters"],
            "censored": data["censored"],
            "pre_vnd_cost": data["pre_vnd_cost"],
            "post_vnd_cost": data["post_vnd_cost"],
            "coords": data["pre_vnd_coords"],
            "post_vnd_fitness": data.get("post_vnd_fitness", None),
            "final_fitness": data.get("final_fitness", None),
        })

    return rows


def load_all_logs(log_root: Path | None = None) -> pd.DataFrame:
    """
    Iterates through ALL run folders under solutions/ml_logs,
    collects all sol-*.json files into a single DataFrame.
    """
    if log_root is None:
        log_root = get_default_log_root()

    all_rows = []

    for run_dir in sorted(log_root.iterdir()):
        if not run_dir.is_dir():
            continue
        rows = load_run_dir(run_dir)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No logs found. Folder: {log_root}")

    df = pd.DataFrame(all_rows)
    return df


if __name__ == "__main__":
    df_logs = load_all_logs()
    print(df_logs.head())
    print("Total records:", len(df_logs))