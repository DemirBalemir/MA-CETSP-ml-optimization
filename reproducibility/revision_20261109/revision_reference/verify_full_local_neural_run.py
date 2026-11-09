"""Read existing results/checkpoints and reproduce statistics; never run a solver/fit.

Writes only verification.json beside this script. Archive payloads are not rehashed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "solutions/neural53_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    data = pd.read_csv(OUT / "results.csv")
    manifest = pd.read_csv(OUT / "evaluation_manifest.csv")
    protocol = json.loads((OUT / "PROTOCOL_FROZEN.json").read_text())
    assert len(data) == len(manifest) == 795
    assert set(data.run_id) == set(manifest.run_id)
    assert not data.duplicated(["problem", "search_seed", "ml_model"]).any()
    assert data.problem.nunique() == 53
    assert set(data.ml_model) == {"DEEPSURV", "MTLR", "NONE"}
    assert set(data.search_seed) == {1, 11, 21, 31, 41}
    coverage = data.groupby(["problem", "ml_model"]).search_seed.nunique()
    assert len(coverage) == 159 and (coverage == 5).all()
    frozen_mismatches = [n for n, h in protocol["files"].items() if sha(OUT / n) != h]
    assert not frozen_mismatches

    helper_path = OUT / "scripts/neural_training.py"
    spec = importlib.util.spec_from_file_location("frozen_training_audit", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    indexed = data.set_index("run_id")
    training = {m: {"trained": 0, "not_attempted": 0, "failed": 0,
                    "max_generated_without_training": 0, "weights_verified": 0}
                for m in ["DEEPSURV", "MTLR", "NONE"]}
    for row in manifest.itertuples():
        folder = Path(row.output_dir)
        status = json.loads((folder / "runner_status.json").read_text())
        metrics = json.loads((folder / "validated_metrics.json").read_text())
        archive = json.loads((folder / "archive.json").read_text())
        assert Path(archive["path"]).is_file()
        assert status["state"] == "validated_success" and status["exit_code"] == 0
        actual = indexed.loc[row.run_id]
        assert metrics["run_id"] == row.run_id
        assert math.isclose(metrics["best"], actual.best, abs_tol=1e-12)
        assert math.isclose(metrics["solver_seconds"], actual.solver_seconds, abs_tol=1e-9)
        model = actual.ml_model
        info = training[model]
        info["trained"] += int(status["training_success"])
        info["failed"] += int(status["training_failed"])
        info["not_attempted"] += int(not status["training_attempted"])
        if not status["training_success"]:
            info["max_generated_without_training"] = max(
                info["max_generated_without_training"], int(status["generated"]))
            continue
        assert model in {"DEEPSURV", "MTLR"}
        name = model.lower()
        meta = json.loads((folder / "model" / f"{name}_meta.json").read_text())
        audit = meta["training_audit"]
        assert audit["epochs_completed"] == 200
        assert audit["changed_parameter_tensors"] > 0
        assert audit["initial_state_sha256"] != audit["final_state_sha256"]
        assert audit["torch_seed"] == int(row.neural_seed)
        assert audit["torch_threads"] == 2
        assert math.isfinite(meta["threshold"])
        weights = torch.load(folder / "model" / f"{name}_net.pt",
                             map_location="cpu", weights_only=True)
        assert all(torch.isfinite(t).all() for t in weights.values())
        assert helper.state_digest(weights) == audit["final_state_sha256"]
        counts = {k: int(v.item()) for k, v in weights.items()
                  if k.endswith("num_batches_tracked")}
        assert counts and counts == audit["batchnorm_batches"]
        assert all(v > 0 for v in counts.values())
        if model == "MTLR":
            config = json.loads((folder / "model" / "mtlr_config.json").read_text())
            assert meta["score_version"] == config["score_version"] == "negative_rmst_step_v1"
        info["weights_verified"] += 1

    means = data.groupby(["problem", "ml_model"])[["best", "solver_seconds"]].mean()
    rng = np.random.default_rng(20261108)
    comparisons = []
    for a, b in [("DEEPSURV", "NONE"), ("MTLR", "NONE"), ("DEEPSURV", "MTLR")]:
        for metric in ["best", "solver_seconds"]:
            x = 100 * (means.xs(a, level=1)[metric] / means.xs(b, level=1)[metric] - 1)
            boot = x.to_numpy()[rng.integers(0, 53, size=(10000, 53))].mean(1)
            comparisons.append(dict(
                treatment=a, reference=b, metric=metric,
                mean_pct=float(x.mean()), median_pct=float(x.median()),
                lo95=float(np.quantile(boot, .025)), hi95=float(np.quantile(boot, .975)),
                p=1.0 if np.all(x.to_numpy() == 0) else float(wilcoxon(x).pvalue)))
    previous = 0.
    for rank, index in enumerate(np.argsort([r["p"] for r in comparisons])):
        previous = max(previous, min(1., (6-rank)*comparisons[index]["p"]))
        comparisons[index]["holm_p"] = previous
    numeric = ["mean_pct", "median_pct", "lo95", "hi95", "p", "holm_p"]
    saved = pd.read_csv(OUT / "comparisons.csv")
    rebuilt = pd.DataFrame(comparisons)
    assert saved[["treatment", "reference", "metric"]].equals(
        rebuilt[["treatment", "reference", "metric"]])
    error = float(np.max(np.abs(saved[numeric].to_numpy()-rebuilt[numeric].to_numpy())))
    assert error < 1e-10
    package = ROOT / "analysis_revision/IEEE_Access_48653_Resubmission_FINAL_20261107"
    with zipfile.ZipFile(package / "Main_Manuscript_Source.zip") as archive:
        main = archive.read("main.tex")
    common = pd.read_csv(ROOT / "analysis_revision/common_model_diagnostics_20261108/"
                        "common_sample_model_comparison.csv")
    common_means = common.loc[common.instance == "MEAN"].iloc[0].drop("instance").to_dict()
    result = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),
        rows=795, instances=53, seeds=[1, 11, 21, 31, 41],
        coverage_cells=159, all_cells_have_five_seeds=True,
        all_795_statuses_validated_success=True,
        all_795_archive_references_exist=True,
        archive_payloads_rehashed_this_audit=False,
        frozen_files_checked=len(protocol["files"]), frozen_mismatches=frozen_mismatches,
        training=training, saved_neural_weight_states_verified=sum(
            x["weights_verified"] for x in training.values()),
        statistics_recompute_max_absolute_error=error, comparisons=comparisons,
        common_sample_cindex_means=common_means,
        packaged_main_tex_sha256=hashlib.sha256(main).hexdigest(),
        cold_source_matches_package=(ROOT / "analysis_revision/final_packaging_20261107/"
                                     "cold_source/main.tex").read_bytes() == main,
        root_paper_main_matches_package=(ROOT / "paper/main.tex").read_bytes() == main,
        hashes={n: sha(OUT / n) for n in ["results.csv", "comparisons.csv",
                                         "evaluation_manifest.csv", "PROTOCOL_FROZEN.json"]})
    return result


if __name__ == "__main__":
    result = verify()
    target = Path(__file__).with_name("verification.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
