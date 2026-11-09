"""Run only the 30 corrected offline neural CV fits; never launch a solver.

The previous unsorted-glob sampling rule is retained, then every selected path,
file hash and fold is frozen. Existing paper results and submission files are
not overwritten. No model tuning or outer-test-dependent stopping is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_la_cetsp"))
import model_ceiling as diagnostic

OUT = ROOT / "analysis_revision/neural_diagnostic_rerun_20261108"
MODELS = {"DEEPSURV": diagnostic._fit_deepsurv, "MTLR": diagnostic._fit_mtlr}
SOURCES = ["tools/rerun_neural_diagnostics.py", "analysis_la_cetsp/model_ceiling.py",
           "analysis_la_cetsp/feature_ceiling.py", "ml/scripts/neural_training.py",
           "ml/scripts/train_deepsurv.py", "ml/scripts/train_mtlr.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    (OUT / "inputs").mkdir(exist_ok=True)
    previous = pd.read_csv(OUT / "before/analysis_la_cetsp/results/model_ceiling.csv")
    rng = random.Random(0)
    selection, assignments, fits, counts = [], [], [], []
    for number, instance in enumerate(diagnostic.DEFAULT_INSTANCES):
        folder = ROOT / "solutions/ml_logs" / instance / "island_9"
        pool = list(folder.glob("run-*/sol-*.json"))
        files = rng.sample(pool, 4000) if len(pool) > 4000 else pool
        rows = []
        for selected_index, path in enumerate(files):
            raw = path.read_bytes()
            item = dict(instance=instance, selected_index=selected_index,
                        path=path.relative_to(ROOT).as_posix(),
                        sha256=hashlib.sha256(raw).hexdigest(), record_index=None,
                        exclusion_reason="")
            try:
                obj = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                item["exclusion_reason"] = "invalid_json"
                selection.append(item)
                continue
            if obj.get("survival_iters", 0) <= 0 or not obj.get("pre_vnd_coords"):
                item["exclusion_reason"] = "nonpositive_lifetime_or_missing_coordinates"
            else:
                features = diagnostic.extract_features(obj["pre_vnd_coords"], obj.get("pre_vnd_cost", 0.0))
                if features is None:
                    item["exclusion_reason"] = "insufficient_coordinates"
                else:
                    item["record_index"] = len(rows)
                    features.update(record_index=len(rows), survival_time=float(obj["survival_iters"]),
                                    event=not obj.get("censored", False), run=path.parent.name)
                    rows.append(features)
            selection.append(item)
        frame = pd.DataFrame(rows)
        if len(frame) < 200 or frame.run.nunique() < 3:
            raise ValueError(f"Insufficient recorded rows/runs for three folds: {instance}")
        if not np.isfinite(frame[diagnostic.FEATURES + ["survival_time"]].to_numpy()).all():
            raise ValueError(f"Nonfinite diagnostic input: {instance}")
        path = OUT / "inputs" / (instance + ".csv")
        frame.to_csv(path, index=False, float_format="%.17g")
        folds = GroupKFold(n_splits=3).split(frame, groups=frame.run.to_numpy())
        for fold, (train, test) in enumerate(folds, 1):
            if set(frame.iloc[train].run) & set(frame.iloc[test].run):
                raise ValueError("Run leakage across CV fold")
            for index in test:
                assignments.append(dict(instance=instance, record_index=int(index),
                                        run=frame.iloc[index].run, test_fold=fold))
            for model_number, model in enumerate(MODELS):
                fits.append(dict(fit_id=f"{instance}-{model.lower()}-fold{fold}", instance=instance,
                                 model=model, fold=fold, n_train=len(train), n_test=len(test),
                                 train_runs=frame.iloc[train].run.nunique(),
                                 test_runs=frame.iloc[test].run.nunique(),
                                 seed=20261108 + 100 * number + 10 * model_number + fold))
        old_n = int(previous.loc[previous.instance == instance, "n"].iloc[0])
        counts.append(dict(instance=instance, available_files=len(pool), selected_files=len(files),
                           usable_records=len(frame), runs=frame.run.nunique(), historical_n=old_n,
                           historical_count_matches=(len(frame) == old_n),
                           feature_file_sha256=digest(path)))
        print(f"PREPARED {instance}: {len(frame)} records, {frame.run.nunique()} runs; old n={old_n}", flush=True)
    if len(fits) != 30:
        raise ValueError("Exactly 30 offline fits are authorized")
    pd.DataFrame(selection).to_csv(OUT / "input_selection.csv", index=False)
    pd.DataFrame(assignments).to_csv(OUT / "fold_assignments.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT / "fit_manifest.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "torch", "pycox", "torchtuples"]}
    source_hashes = {name: digest(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        path = OUT / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), planned_fits=30,
                    instances=diagnostic.DEFAULT_INSTANCES, models=list(MODELS), folds=3, island=9,
                    feature_names=diagnostic.FEATURES, sampling_seed=0, max_files_per_instance=4000,
                    sampling_rule="Original unsorted glob, sample before positive-lifetime filtering; frozen paths below",
                    historical_selection_limit="Original fit-specific path hashes were not retained; matching n is not proof of identical samples",
                    validation="GroupKFold(3) by run; scaler fits outer training rows only; outer test never used for tuning/stopping",
                    epochs=100, training_policy="Fixed 100 epochs; remove invalid validation EarlyStopping callback",
                    architecture=dict(hidden_units=[64, 64], batch_norm=True, dropout=.1,
                                      optimizer="Adam", learning_rate=.01, mtlr_durations=20),
                    score_convention="Higher risk means shorter survival; preserve DeepSurv log-risk and MTLR last-bin risk",
                    metric="sksurv concordance_index_censored per outer fold, mean folds then equal-weight instances; same legacy scoring scope",
                    neural_seed_rule="20261108 + 100*instance_position + 10*model_position + fold (1..3)",
                    cpu_threads=2, interop_threads=1, torch_deterministic=True, versions=versions,
                    python=sys.version, source_sha256=source_hashes, counts=counts,
                    input_selection_sha256=digest(OUT / "input_selection.csv"),
                    fold_assignments_sha256=digest(OUT / "fold_assignments.csv"),
                    fit_manifest_sha256=digest(OUT / "fit_manifest.csv"), solver_runs=0)
    save(OUT / "protocol.json", protocol)
    return protocol


def validate_protocol(protocol):
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed since protocol freeze: {name}")
    for name in ["input_selection", "fold_assignments", "fit_manifest"]:
        if digest(OUT / (name + ".csv")) != protocol[name + "_sha256"]:
            raise ValueError(f"Frozen manifest changed: {name}")
    for item in protocol["counts"]:
        if digest(OUT / "inputs" / (item["instance"] + ".csv")) != item["feature_file_sha256"]:
            raise ValueError("Frozen feature data changed")


def execute(protocol):
    torch.set_num_threads(protocol["cpu_threads"])
    torch.set_num_interop_threads(protocol["interop_threads"])
    torch.use_deterministic_algorithms(True)
    manifest = pd.read_csv(OUT / "fit_manifest.csv")
    assignments = pd.read_csv(OUT / "fold_assignments.csv")
    results = []
    for record in manifest.to_dict("records"):
        validate_protocol(protocol)
        folder = OUT / "fits" / record["fit_id"]
        result_file = folder / "result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result["state"] != "completed":
                raise ValueError(f"Failed fit requires inspection: {folder}")
            results.append(result)
            continue
        if folder.exists():
            raise ValueError(f"Partial fit requires inspection before any retry: {folder}")
        folder.mkdir(parents=True)
        save(folder / "started.json", dict(**record, started_utc=datetime.now(timezone.utc).isoformat()))
        frame = pd.read_csv(OUT / "inputs" / (record["instance"] + ".csv"), float_precision="round_trip")
        a = assignments[assignments.instance == record["instance"]].set_index("record_index")
        test = np.flatnonzero(a.loc[frame.record_index, "test_fold"].to_numpy() == record["fold"])
        train = np.setdiff1d(np.arange(len(frame)), test)
        if len(train) != record["n_train"] or len(test) != record["n_test"]:
            raise ValueError("Fold sizes changed")
        scaler = StandardScaler().fit(frame.iloc[train][diagnostic.FEATURES])
        xtrain = scaler.transform(frame.iloc[train][diagnostic.FEATURES])
        xtest = scaler.transform(frame.iloc[test][diagnostic.FEATURES])
        save(folder / "scaler.json", dict(feature_names=diagnostic.FEATURES,
                                         mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist()))
        random.seed(record["seed"]); np.random.seed(record["seed"]); torch.manual_seed(record["seed"])
        started = time.monotonic()
        print(f"FIT {len(results)+1}/30 {record['fit_id']} seed={record['seed']}", flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                risk = MODELS[record["model"]](xtrain,
                    frame.survival_time.to_numpy()[train], frame.event.to_numpy(dtype=bool)[train],
                    xtest, audit_dir=folder)
            if not np.isfinite(risk).all() or len(risk) != len(test):
                raise ValueError("Invalid held-out predictions")
            c, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
                frame.event.to_numpy(dtype=bool)[test], frame.survival_time.to_numpy()[test], risk)
            predictions = frame.iloc[test][["record_index", "run", "survival_time", "event"]].copy()
            predictions["risk"] = risk
            predictions.to_csv(folder / "predictions.csv", index=False, float_format="%.17g")
            audit = json.loads((folder / "training_audit.json").read_text())
            result = dict(**record, state="completed", cindex=float(c), seconds=time.monotonic()-started,
                          concordant=int(concordant), discordant=int(discordant), tied_risk=int(tied_risk),
                          tied_time=int(tied_time), unique_risks=int(np.unique(risk).size),
                          risk_min=float(np.min(risk)), risk_max=float(np.max(risk)),
                          warning_messages=[str(item.message) for item in caught],
                          training_audit=audit, prediction_sha256=digest(folder / "predictions.csv"))
            save(result_file, result)
            results.append(result)
            save(OUT / "status.json", dict(state="running", completed=len(results), planned=30, solver_runs=0))
            print(f"DONE {len(results)}/30 C={c:.6f}, {result['seconds']:.2f}s, epochs={audit['epochs_completed']}", flush=True)
        except Exception as error:
            save(folder / "failure.json", dict(error=repr(error)))
            save(OUT / "status.json", dict(state="failed", completed=len(results), fit_id=record["fit_id"], error=repr(error)))
            raise
    if len(results) != 30:
        raise ValueError("Incomplete offline diagnostic")
    flat = pd.DataFrame([{key: value for key, value in row.items()
                          if key not in ["warning_messages", "training_audit"]} for row in results])
    flat.to_csv(OUT / "fold_results.csv", index=False)
    corrected = flat.groupby(["instance", "model"]).cindex.mean().unstack("model")
    corrected["n"] = [next(r["usable_records"] for r in protocol["counts"] if r["instance"] == name)
                      for name in corrected.index]
    corrected = corrected.reindex(diagnostic.DEFAULT_INSTANCES)
    corrected.loc["MEAN"] = dict(DEEPSURV=corrected.DEEPSURV.mean(), MTLR=corrected.MTLR.mean(), n=corrected.n.sum())
    corrected.to_csv(OUT / "corrected_neural_cindex.csv", index_label="instance")
    previous = pd.read_csv(OUT / "before/analysis_la_cetsp/results/model_ceiling.csv").set_index("instance")
    comparison = []
    for instance, row in corrected.iterrows():
        for model in MODELS:
            old = float(previous.loc[instance, model]); new = float(row[model])
            comparison.append(dict(instance=instance, model=model, historical_reported_cindex=old,
                                   corrected_cindex=new, difference=new-old, n=int(row["n"])))
    pd.DataFrame(comparison).to_csv(OUT / "comparison.csv", index=False)
    hashes = {path.relative_to(OUT).as_posix(): digest(path) for path in sorted(OUT.rglob("*"))
              if path.is_file() and path.name not in ["SHA256.json", "status.json", "console.log"]}
    save(OUT / "SHA256.json", hashes)
    save(OUT / "status.json", dict(state="complete", completed=30, planned=30, solver_runs=0,
                                   result="corrected_neural_cindex.csv"))
    print(corrected.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = prepare()
    validate_protocol(protocol)
    if not args.prepare_only:
        execute(protocol)
