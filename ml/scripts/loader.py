# ml/scripts/loader.py

import json
from pathlib import Path
import pandas as pd


def get_default_log_root() -> Path:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    return project_root / "solutions" / "ml_logs"


def _load_single_run_dir(run_dir: Path) -> list[dict]:
    """Load all sol-*.json from one flat run directory."""
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


def load_run_dir(path: Path) -> list[dict]:
    """
    Load solutions from `path`.

    Accepts two layouts:
    - Island dir:  path/run-<timestamp>/sol-*.json   (new nested structure)
    - Flat dir:    path/sol-*.json                   (legacy / single run)
    """
    run_subdirs = sorted(path.glob("run-*/"))
    if run_subdirs:
        # New layout: island dir containing run-* subdirectories
        rows = []
        for run_dir in run_subdirs:
            if run_dir.is_dir():
                rows.extend(_load_single_run_dir(run_dir))
        return rows
    else:
        # Legacy / single-run flat layout
        return _load_single_run_dir(path)


def load_all_logs(log_root: Path | None = None) -> pd.DataFrame:
    """
    Traverse the full log tree and return all solutions as a DataFrame.

    New structure:  log_root/<instance>/island_<N>/run-<timestamp>/sol-*.json
    Legacy fallback: log_root/<run_folder>/sol-*.json
    """
    if log_root is None:
        log_root = get_default_log_root()

    all_rows = []

    for entry in sorted(log_root.iterdir()):
        if not entry.is_dir():
            continue

        # Check if this is an instance dir (contains island_* subdirs)
        island_subdirs = sorted(entry.glob("island_*/"))
        if island_subdirs:
            for island_dir in island_subdirs:
                if island_dir.is_dir():
                    all_rows.extend(load_run_dir(island_dir))
        else:
            # Legacy flat dir directly under log_root
            all_rows.extend(_load_single_run_dir(entry))

    if not all_rows:
        raise RuntimeError(f"No logs found. Folder: {log_root}")

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df_logs = load_all_logs()
    print(df_logs.head())
    print("Total records:", len(df_logs))
