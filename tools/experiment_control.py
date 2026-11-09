"""Draft manifest, guarded sequential runner, and run-output validator.

Dry-run is the default. The pilot is technical only; its p=.08 is not calibrated.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "build-task2-validation" / "Release" / "MA-CETSP.exe"
WORK_DIR = ROOT / "build" / "Release"  # solver's legacy input paths resolve here
SEEDS = (1, 11, 21, 31, 41)
# Chosen from the test split by family/size, before seeing new-run outcomes.
DRAFT_INSTANCES = (
    "bubbles1", "bubbles4", "bubbles8", "chaoSingleDep",
    "concentricCircles2", "concentricCircles4", "rotatingDiamonds2", "rotatingDiamonds5",
    "team1_100", "team3_300", "team6_500", "kroD100_or2",
    "rat195_or30", "pcb442_or30", "dsj1000_or30", "bonus1000rdmRad",
    "team2_200rdmRad", "car_door_50",
)
PILOT_INSTANCES = ("bubbles1", "team1_100")
METHODS = ("none", "learned", "random")
FIELDS = ("order", "run_id", "problem", "index", "method", "search_seed", "prune_seed",
          "iterations", "population", "fit_beta", "dist_th", "neighbor_size",
          "train_frac_min", "train_frac_hard", "min_events", "solver_time_limit",
          "outer_timeout", "random_reject_rate", "random_start_iter", "schedule_path",
          "schedule_sha256", "binary_sha256", "source_sha256", "output_dir")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_record(root=None):
    root = Path(root or ROOT)
    paths = {root / "CMakeLists.txt", root / "tools/experiment_control.py",
             root / "tools/control_study.py", root / "tools/analyze_control_study.py",
             root / "tools/control_study_parallel.py", root / "tools/control_study_fixed.py",
             root / "tools/check_cox_export_parity.py"}
    for folder, suffixes in (("src", {".cpp", ".h", ".hpp"}),
                             ("include", {".h", ".hpp"}), ("ml/scripts", {".py"})):
        paths.update(p for p in (root / folder).rglob("*") if p.suffix in suffixes)
    paths.update(root.glob("*.cmake"))
    paths.update(root.glob("run_experiment.*"))
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths) if p.is_file()}
    return {"schema": 2, "files": files, "sha256": canonical_hash(files)}


def source_identity():
    return source_record()["sha256"]


def python_identity(executable):
    executable = Path(executable).resolve()
    probe = "import sys,platform,json,importlib.metadata as m; print(json.dumps(dict(version=sys.version,implementation=platform.python_implementation(),packages=sorted((d.metadata['Name'],d.version,__import__('hashlib').sha256((d.read_text('RECORD') or '').encode()).hexdigest()) for d in m.distributions()))))"
    result = subprocess.run([str(executable), "-c", probe], capture_output=True, text=True, check=True, timeout=30)
    return {"path": str(executable), "sha256": digest(executable), **json.loads(result.stdout)}


def configuration_identity(row, binary, python_exe, command):
    if instance_map().get(row["problem"]) != int(row["index"]):
        raise ValueError("problem/index mismatch")
    source = source_record()
    if digest(binary) != row["binary_sha256"] or source["sha256"] != row["source_sha256"]:
        raise ValueError("current binary/source differs from manifest; create a new versioned manifest")
    schedule = None
    if row["schedule_path"]:
        path = Path(row["schedule_path"]).resolve()
        if digest(path) != row["schedule_sha256"]:
            raise ValueError("schedule identity mismatch")
        schedule = {"path": str(path), "sha256": digest(path)}
    data = ROOT / "datasets" / (row["problem"] + ".cetsp")
    lkh = ROOT / "external/LKH-2.0.11/LKH.exe"
    record = dict(schema=2, row={k: str(v) for k,v in row.items()}, command=command,
                  binary={"path": str(Path(binary).resolve()), "sha256": digest(binary)},
                  source=source, schedule=schedule, python=python_identity(python_exe),
                  data={"path":str(data), "sha256":digest(data)},
                  lkh={"path":str(lkh), "sha256":digest(lkh)},
                  dependency_scope="Python executable bytes, version, installed distribution versions and RECORD digests; not a hash of installed package payloads or the OS DLL closure",
                  native_dependencies={str(p.resolve()):digest(p) for p in sorted(set(
                      list(Path(binary).parent.glob("*.dll")) + list(Path("C:/gurobi1203/win64/bin").glob("gurobi120.dll"))))},
                  native_scope="Adjacent DLLs and configured Gurobi runtime DLL if present; transitive/system DLL resolution, CPU load and license state are not certified",

                  environment=dict(platform=__import__('platform').platform(), machine=__import__('platform').machine(),
                      cwd=str(WORK_DIR.resolve()), variables=dict(sorted(os.environ.items()))))
    # Do not persist environment values (may contain credentials): bind their digest.
    if row.get("neural_seed"):
        record["environment"]["variables"]["CETSP_NEURAL_SEED"] = str(row["neural_seed"])
    record["environment"]["variables"] = canonical_hash(record["environment"]["variables"])
    return {"sha256":canonical_hash(record), "record":record}


def verify_resume(status, identity, command):
    if status.get("state") != "validated_success":
        raise ValueError("existing partial run; preserved for inspection")
    if "configuration_identity" not in status:
        raise ValueError("legacy status without configuration identity; historical inspection only")
    saved = status["configuration_identity"]
    if saved != identity or saved.get("sha256") != canonical_hash(saved.get("record")) or status.get("command") != command:
        raise ValueError("configuration identity mismatch; existing run preserved")


def instance_map():
    text = (ROOT / "include/Defs.hpp").read_text(encoding="utf-8")
    body = text.split("const std::vector<std::string> FILENAMES = {", 1)[1].split("};", 1)[0]
    return {name: int(index) for name, index in
            re.findall(r'"([^"]+)"\s*,?\s*//\s*(\d+)', body)}


def test_split():
    with (ROOT / "analysis_la_cetsp/results/instance_manifest.csv").open(newline="") as f:
        return {row["instance"] for row in csv.DictReader(f) if row["split"] == "test"}


def make_manifest(kind, output_root, binary, manifest):
    names = PILOT_INSTANCES if kind == "pilot" else DRAFT_INSTANCES
    mapping, tests = instance_map(), test_split()
    assert set(names) <= tests and all(name in mapping for name in names)
    if not binary.is_file():
        raise ValueError(f"binary missing: {binary}")
    binary_hash, source_hash = digest(binary), source_identity()
    rng = random.Random(20261030)
    rows = []
    for problem in names:
        for seed in ((11,) if kind == "pilot" else SEEDS):
            methods = list(METHODS)
            rng.shuffle(methods)  # fixed and recorded before execution
            for method in methods:
                run_id = f"{problem}-s{seed}-{method}"
                rows.append(dict(order=len(rows)+1, run_id=run_id, problem=problem,
                    index=mapping[problem], method=method, search_seed=seed,
                    prune_seed=(917000 + mapping[problem]*100 + seed if method == "random" else ""),
                    iterations=200 if kind == "pilot" else 5000,
                    population=20, fit_beta=.96, dist_th=5, neighbor_size=50,
                    train_frac_min=.20, train_frac_hard=.50, min_events=100,
                    solver_time_limit=30 if kind == "pilot" else 36000,
                    outer_timeout=90 if kind == "pilot" else 39600,
                    random_reject_rate=.08 if kind == "pilot" and method == "random" else "",
                    random_start_iter=40 if kind == "pilot" and method == "random" else "",
                    schedule_path="", schedule_sha256="", binary_sha256=binary_hash,
                    source_sha256=source_hash, output_dir=str(output_root / run_id)))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        with manifest.open(newline="") as f:
            old = list(csv.DictReader(f))
        if old != [{k: str(row[k]) for k in FIELDS} for row in rows]:
            raise ValueError(f"manifest already exists with different contents: {manifest}")
    else:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
    return rows


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid/nonfinite Cox numeric field")
    return value


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate model JSON key")
        obj[key] = value
    return obj


def validate_cox_model(folder, detail):
    try:
        coeff, norm, meta = [json.loads((folder / name).read_text(), object_pairs_hook=unique_object)
                            for name in ("cox_coeffs.json", "cox_norm.json", "cox_meta.json")]
        features = {"avg_edge_length", "var_edge_length", "bbox_width", "bbox_height", "bbox_area",
                    "centroid_x", "centroid_y", "centroid_dist_sum", "angle_variance", "pre_vnd_cost"}
        if not isinstance(coeff, dict) or not coeff or not set(coeff) <= features or not isinstance(norm, dict) or set(coeff) != set(norm):
            raise ValueError("Cox coefficient/normalization feature mismatch")
        for key, value in coeff.items():
            finite_number(value)
            if not isinstance(norm[key], dict) or set(norm[key]) != {"mean", "std"}:
                raise ValueError("invalid Cox normalization structure")
            finite_number(norm[key]["mean"])
            if finite_number(norm[key]["std"]) <= 1e-12:
                raise ValueError("unusable Cox normalization std")
        if not isinstance(meta, dict):
            raise ValueError("invalid Cox meta structure")
        if finite_number(meta["threshold"]) < 0 or meta["rejection_percentile"] not in range(50,96,5):
            raise ValueError("invalid Cox threshold/percentile")
        if not 0 <= finite_number(meta["concordance_index"]) <= 1:
            raise ValueError("invalid Cox concordance")
        n = meta["n_samples"]
        if type(n) is not int or n < 2:
            raise ValueError("invalid Cox sample count")
        objective = meta["objective"]
        fallback = (2 <= n < 30 and meta["rejection_percentile"] == 80 and
                    "Too few samples for hold-out split; using default percentile" in detail)
        if not (fallback and type(objective) is float and math.isnan(objective)):
            finite_number(objective)
        if ("Too few samples for hold-out split" in detail) != (n < 30):
            raise ValueError("Cox fallback/sample count mismatch")
        return meta
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid/missing Cox model structure: {exc}") from exc


def validate_startup(row, text, strict=True, python_exe=None):
    lines = [line for line in text.splitlines() if line.startswith("[Island 0] instance=")]
    if len(lines) != 1:
        raise ValueError("missing/duplicate startup record")
    line = lines[0]
    expected = {"instance":row["index"], "seed":row["search_seed"], "pop":row["population"],
                "iter":row["iterations"], "mode":row["method"],
                "ml":row.get("ml_model", "COX") if row["method"] == "learned" else "OFF",
                "train_min":row["train_frac_min"], "train_hard":row["train_frac_hard"], "min_events":row["min_events"]}
    if row["method"] == "random":
        expected["prune_seed"] = row["prune_seed"]
        if not row["schedule_path"]:
            expected.update(random_rate=row["random_reject_rate"], random_start_iter=row["random_start_iter"])
    if strict:
        expected.update(fit_beta=row["fit_beta"], dist_th=row["dist_th"], neighbor_size=row["neighbor_size"], max_time=row["solver_time_limit"],
                        init="KMEANS", select="RANDOM", crossover="KSX", improvement="BEST", greed="SQUEEZE", distance="EDIT", random="0", islands="1")
    for key, value in expected.items():
        match = re.search(r"(?:^| )" + key + r"=(\S+)", line)
        if not match:
            raise ValueError(f"startup missing {key}")
        actual = match.group(1)
        try:
            equal = math.isfinite(float(actual)) and float(actual) == float(value)
        except ValueError:
            equal = actual == str(value)
        if not equal:
            raise ValueError(f"startup mismatch: {key}")
    if strict:
        runtime = [l for l in text.splitlines() if l.startswith("[RUNTIME] ")]
        if len(runtime) != 1:
            raise ValueError("missing/duplicate runtime startup record")
        for key, expected in (("python", python_exe), ("scripts", row.get("scripts_dir", ROOT/"ml/scripts")), ("lkh", ROOT/"external/LKH-2.0.11/LKH.exe")):
            match = re.search(key + r'=("(?:[^"\\]|\\.)*")', runtime[0])
            if not match or expected is None or Path(json.loads(match.group(1))).resolve() != Path(expected).resolve():
                raise ValueError(f"runtime startup mismatch: {key}")
        for key, path in (("run_dir", Path(row["output_dir"])), ("model_dir", Path(row["output_dir"])/"model"),
                          ("results_dir", Path(row["output_dir"])/"solutions"), ("lkh_tmp", Path(row["output_dir"])/"lkh")):
            # Delimit by the following field rather than spaces within paths.
            following = {"run_dir":"results_dir", "model_dir":"run_dir", "results_dir":"lkh_tmp", "lkh_tmp":"timestamp"}[key]
            match = re.search(r" " + key + r"=(.*?) " + following + "=", line)
            if not match or Path(match.group(1)).resolve() != path.resolve():
                raise ValueError(f"startup path mismatch: {key}")
        schedule = row["schedule_path"]
        expected_schedule = str(Path(schedule).resolve()) if schedule else "NA"
        match = re.search(r" schedule=(.*?) schedule_fnv1a=(\S+)", line)
        if not match or match.group(1) != expected_schedule:
            raise ValueError("startup schedule mismatch")
        if schedule:
            fnv = 14695981039346656037
            for byte in Path(schedule).read_bytes():
                fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64)-1)
            if match.group(2).lower() != format(fnv, 'x'):
                raise ValueError("startup schedule hash mismatch")
    if f"[SUMMARY] instance={row['problem']} " not in text:
        raise ValueError("summary problem mismatch")


def validate_run(row, run_dir, log_path):
    if not log_path.is_file():
        raise ValueError("missing console log")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if row.get('require_socp') == '1':
        if re.search(r'Error code\s*=|License expired|\[SOCP ERROR\]|Exception during optimization|\[ERROR\]',text,re.I):
            raise ValueError('solver/subsolver error in console; not a valid completed run')
        states=re.findall(r'\[SOCP\] status=(\d+) solutions=(\d+)',text)
        if not states:raise ValueError('missing explicit SOCP status evidence')
        fallback=sum(int(s)==12 and int(n)==0 for s,n in states)
        success=sum(int(s) in (2,13) and int(n)>0 for s,n in states)
        if fallback+success!=len(states) or text.count('[SOCP FALLBACK]')!=fallback or text.count('[SOCP SUCCESS]')!=success:
            raise ValueError('inconsistent SOCP success/fallback evidence')
    summaries = re.findall(r"\[SUMMARY\].*?model=(\S+).*?best=([\d.eE+-]+).*?effective_iter=(\d+).*?total_t=([\d.eE+-]+).*?ml_rejects=(\d+)", text)
    counters = re.findall(r"\[CANDIDATES\] generated=(\d+) filter_rejected=(\d+) vnd_evaluated=(\d+) admitted=(\d+) not_admitted=(\d+)", text)
    if len(summaries) != 1 or len(counters) != 1:
        raise ValueError("missing or duplicate summary/counter line")
    model, best, iterations, seconds, rejects = summaries[0]
    expected={"none":("BASELINE",),"random":("RANDOM",),"learned":(row.get("ml_model", "COX"),"LEARNED_INACTIVE")}[row["method"]]
    if model not in expected:
        raise ValueError(f"wrong model in summary: {model} for {row['method']}")
    generated, filtered, vnd, admitted, not_admitted = map(int, counters[0])
    iterations, rejects = int(iterations), int(rejects)
    if not (math.isfinite(float(best)) and math.isfinite(float(seconds)) and float(seconds)>=0):
        raise ValueError("nonfinite summary")
    if iterations != generated or generated != filtered + vnd or vnd != admitted + not_admitted or rejects != filtered:
        raise ValueError("counter invariant failed")
    if generated > int(row["iterations"]):
        raise ValueError("processed more candidates than budget")
    research = run_dir / "solutions" / "research"
    with (research / "candidates.csv").open(newline="") as f:
        candidates=list(csv.DictReader(f))
    with (research / "lifetimes.csv").open(newline="") as f:
        lives=list(csv.DictReader(f))
    outcomes={"filter_rejected":filtered,"admitted":admitted,"not_admitted":not_admitted}
    if len(candidates)!=generated or len({int(c["candidate_id"]) for c in candidates})!=generated:
        raise ValueError("candidate rows/IDs invalid")
    if {int(c["candidate_id"]) for c in candidates} != set(range(1, generated+1)) or {int(c["iteration"]) for c in candidates} != set(range(1, generated+1)):
        raise ValueError("candidate IDs/iterations must be contiguous positive 1..generated")
    if any(int(c["candidate_id"])!=int(c["iteration"]) or c["outcome"] not in outcomes for c in candidates):
        raise ValueError("candidate iteration/outcome invalid")
    if any(sum(c["outcome"]==key for c in candidates)!=count for key,count in outcomes.items()):
        raise ValueError("candidate outcomes disagree with summary")
    admitted_ids={int(c["candidate_id"]) for c in candidates if c["outcome"]=="admitted"}
    life_ids=[int(l["candidate_id"]) for l in lives]
    if len(life_ids)!=len(set(life_ids)) or set(life_ids)!=admitted_ids:
        raise ValueError("missing/duplicate final lifetimes")
    for life in lives:
        birth, death, duration=map(int,(life["birth_iter"],life["death_iter"],life["survival_iters"]))
        if birth!=int(life["candidate_id"]) or duration!=death-birth or duration<0 or death>iterations or life["censored"] not in ("0","1"):
            raise ValueError("invalid lifetime")
        if life["censored"]=="1" and death!=iterations:
            raise ValueError("survivor censored before final iteration")
    training_attempted="[ML] Training triggered" in text
    training_success="[ML] Training completed successfully." in text
    training_failed=training_attempted and not training_success
    training_log=run_dir/"model"/"training_log.txt"
    detail=training_log.read_text(encoding="utf-8",errors="replace") if training_log.is_file() else ""
    convergence_warning="ConvergenceWarning" in detail
    threshold_fallback="Too few samples for hold-out split" in detail
    predictions = []
    for line in text.splitlines():
        if "[ML-COX]" in line:
            match = re.fullmatch(r"\[ML-COX\] score=(\S+) threshold=(\S+)", line.strip())
            if not match:
                raise ValueError("malformed Cox score/threshold record")
            values = tuple(float(v) for v in match.groups())
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("nonfinite/negative Cox score/threshold")
            predictions.append(values)
    activated = training_success and bool(predictions)
    if predictions and (row["method"] != "learned" or not training_success):
        raise ValueError("Cox prediction without successful learned training")
    if row["method"]=="learned" and training_success and row.get("ml_model", "COX") == "COX":
        meta = validate_cox_model(run_dir / "model", detail)
        if row.get('require_socp')=='1' and meta.get('coefficient_scale')!='lifelines_raw_centered_v1':
            raise ValueError('Cox export scale version mismatch')
        for score, threshold in predictions:
            if not math.isclose(threshold, meta["threshold"], rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("logged Cox threshold disagrees with model")
    if row.get("problem"):
        validate_startup(row, text, strict=False)
    return dict(best=float(best), solver_seconds=float(seconds), processed_iterations=iterations,
        generated=generated,filter_rejected=filtered,vnd_evaluated=vnd,admitted=admitted,
        not_admitted=not_admitted, reject_per_generated=(filtered/generated if generated else None),
        admitted_per_generated=(admitted/generated if generated else None),
        admitted_per_vnd=(admitted/vnd if vnd else None), training_attempted=training_attempted,
        training_success=training_success,training_failed=training_failed, filter_activated=activated,
        convergence_warning=convergence_warning, threshold_fallback=threshold_fallback,
        censored=sum(l["censored"]=="1" for l in lives),
        socp_successes=text.count('[SOCP SUCCESS]'),
        socp_numerical_fallbacks=text.count('[SOCP FALLBACK]'))


def save_json(path, data):
    tmp=path.with_suffix(path.suffix+".tmp")
    payload=json.dumps(data,indent=2,allow_nan=False)
    deadline=time.monotonic()+30.0
    while True:
        try:
            tmp.write_text(payload,encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError:
            # Windows readers can briefly hold either report file open.
            # Keep the atomic replacement and propagate persistent errors.
            if time.monotonic()>=deadline:
                raise
            time.sleep(.05)


def refresh_results(root, rows):
    columns=("order","run_id","problem","method","search_seed","prune_seed","state",
             "exit_code","wall_seconds","best","solver_seconds","processed_iterations",
             "generated","filter_rejected","vnd_evaluated","admitted","not_admitted",
             "reject_per_generated","admitted_per_generated","admitted_per_vnd","training_attempted","training_success",
             "training_failed","filter_activated","convergence_warning","threshold_fallback",
             "censored","socp_successes","socp_numerical_fallbacks","error")
    path=root/"results.csv"
    tmp=path.with_suffix(".csv.tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            status_path=Path(row["output_dir"])/"runner_status.json"
            if not status_path.is_file(): continue
            status=json.loads(status_path.read_text(encoding="utf-8"))
            merged={**row,**status}
            writer.writerow({key:("" if merged.get(key) is None else merged.get(key,"")) for key in columns})
    tmp.replace(path)


def run_rows(rows, binary, python_exe, execute, kind, max_runs, authorized_protocol=None):
    if not execute:
        print(f"DRY RUN: {len(rows)} sequential {kind} runs; no solver started")
        for row in rows[:min(9,len(rows))]: print(row["order"],row["run_id"],row["output_dir"])
        return
    if kind in ("calibration", "evaluation"):
        if authorized_protocol is None:
            raise ValueError("scientific execution requires frozen authorized protocol")
        protocol_path = Path(authorized_protocol)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("execution_authorized") is not True or protocol.get("study_id") not in ("control53-v1", "control53-v2", "control53-parallel4-v1", "control53-fixed-v1", "neural53-v1"):
            raise ValueError("protocol is not authorized for this study")
        if any(row.get("protocol_sha256") != digest(protocol_path) or row.get("stage") != kind for row in rows):
            raise ValueError("manifest/protocol stage identity mismatch")
    elif kind != "pilot":
        raise ValueError("draft main manifest is not executable until calibration and protocol approval")
    if not WORK_DIR.is_dir(): raise ValueError(f"solver working directory missing: {WORK_DIR}")
    for row in rows[:max_runs or len(rows)]:
        run_dir=Path(row["output_dir"])
        status_path=run_dir/"runner_status.json"
        log_path=run_dir/"console.log"
        if run_dir.exists() and not status_path.is_file():
            raise ValueError("existing directory preserved")
        if run_dir.exists():
            saved = json.loads(status_path.read_text())
            if "configuration_identity" not in saved:
                raise ValueError("legacy status without configuration identity; historical inspection only")
        command=[str(binary),"-i",str(row["index"]),"-s",str(row["search_seed"]),
            "-r",str(row["iterations"]),"-p",str(row["population"]),
            "-b",str(row["fit_beta"]),"-d",str(row["dist_th"]),"-n",str(row["neighbor_size"]),
            "-t",str(row["solver_time_limit"]),"--islands","1","--filter_mode",str(row["method"]),
            "--train_frac_min",str(row["train_frac_min"]),"--train_frac_hard",str(row["train_frac_hard"]),
            "--min_events",str(row["min_events"]),"--output_root",str(run_dir.parent),"--run_id",row["run_id"]]
        if row["method"]=="learned": command += ["--ml_model",row.get("ml_model", "COX")]
        if row["method"]=="random":
            command += ["--prune_seed",str(row["prune_seed"])]
            if row["schedule_path"]:
                path=Path(row["schedule_path"]).resolve()
                if not path.is_file() or digest(path)!=row["schedule_sha256"]:
                    if run_dir.exists(): raise ValueError("changed schedule; existing run preserved")
                    run_dir.mkdir(parents=True)
                    save_json(status_path,dict(state="invalid_schedule",error=f"missing/changed schedule: {path}",
                                                  binary_sha256=row["binary_sha256"]))
                    refresh_results(run_dir.parent,rows)
                    print("invalid_schedule",path,flush=True)
                    break
                command += ["--prune_schedule",str(path)]
            elif row["random_reject_rate"] != "":
                command += ["--random_reject_rate",str(row["random_reject_rate"]),
                            "--random_start_iter",str(row["random_start_iter"])]
            else:
                if run_dir.exists(): raise ValueError("invalid schedule; existing run preserved")
                run_dir.mkdir(parents=True)
                save_json(status_path,dict(state="invalid_schedule",error="random schedule is not calibrated",
                                              binary_sha256=row["binary_sha256"]))
                refresh_results(run_dir.parent,rows)
                print("invalid_schedule: random schedule is not calibrated",flush=True)
                break
        command += ["--python_exe", str(python_exe), "--scripts_dir", str(row.get("scripts_dir", ROOT / "ml/scripts")),
                    "--lkh_exe", str(ROOT / "external/LKH-2.0.11/LKH.exe"),
                    "--init", "KMEANS", "--select", "RANDOM", "--crossover", "KSX",
                    "--improvement", "BEST", "--greed", "SQUEEZE", "--distance", "EDIT", "-g", "0"]
        identity = configuration_identity(row, binary, python_exe, command)
        if run_dir.exists():
            if not status_path.is_file():
                raise ValueError("existing directory preserved; missing status")
            status = json.loads(status_path.read_text())
            verify_resume(status, identity, command)
            validate_run(row, run_dir, log_path)
            validate_startup(row, log_path.read_text(), strict=True, python_exe=python_exe)
            print("SKIP verified", row["run_id"])
            continue
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        print("RUN",row["order"],row["run_id"],flush=True)
        started=time.monotonic()
        flags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=="nt" else 0
        run_env = dict(os.environ)
        if row.get("neural_seed"): run_env["CETSP_NEURAL_SEED"] = str(row["neural_seed"])
        process=subprocess.Popen(command,cwd=WORK_DIR,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 creationflags=flags,env=run_env)
        try:
            output,_=process.communicate(timeout=float(row["outer_timeout"]))
            state="finished" if process.returncode==0 else "solver_nonzero"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if os.name=="nt":
                subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
            else:
                process.kill()
            output,_=process.communicate()
            state="user_interrupted" if isinstance(exc, KeyboardInterrupt) else "outer_timeout"
        elapsed=time.monotonic()-started
        if not run_dir.is_dir(): run_dir.mkdir(parents=True)  # preserve failure record
        log_path.write_bytes(output)
        status=dict(state=state,exit_code=process.returncode,wall_seconds=elapsed,
                    binary_sha256=row["binary_sha256"],command=command,configuration_identity=identity)
        console=output.decode("utf-8",errors="replace")
        training_log=run_dir/"model"/"training_log.txt"
        training_detail=training_log.read_text(encoding="utf-8",errors="replace") if training_log.is_file() else ""
        status.update(training_attempted="[ML] Training triggered" in console,
                      training_success="[ML] Training completed successfully." in console,
                      convergence_warning="ConvergenceWarning" in training_detail,
                      threshold_fallback="Too few samples for hold-out split" in training_detail)
        status["training_failed"]=status["training_attempted"] and not status["training_success"]
        if state=="solver_nonzero":
            diagnostic=console
            errors=re.findall(r'^.*(?:\[ERROR\]|\[SOCP ERROR\]|\[ML ERROR\]).*$',diagnostic,re.M)
            status["error"]=errors[-1] if errors else 'solver exited nonzero; inspect console.log'
        if state=="finished":
            try:
                status.update(validate_run(row,run_dir,log_path))
                validate_startup(row, console, strict=True, python_exe=python_exe)
                if configuration_identity(row, binary, python_exe, command) != identity:
                    raise ValueError("runtime inputs changed during execution")
                status["state"]="validated_success"
            except Exception as exc:
                status["state"]="invalid_output"; status["error"]=str(exc)
        save_json(status_path,status)
        refresh_results(run_dir.parent,rows)
        print(status["state"],f"{elapsed:.2f}s",status.get("error",""),flush=True)
        if status["state"]!="validated_success": break


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind",choices=("draft","pilot"),default="draft")
    ap.add_argument("--binary",type=Path,default=DEFAULT_EXE)
    ap.add_argument("--output-root",type=Path,default=None)
    ap.add_argument("--manifest",type=Path,default=None)
    ap.add_argument("--python-exe",type=Path,default=Path(sys.executable))
    ap.add_argument("--execute",action="store_true",help="run the bounded technical pilot")
    ap.add_argument("--max-runs",type=int,default=0)
    args=ap.parse_args()
    output=(args.output_root or ROOT/"solutions"/("technical_pilot_task3" if args.kind=="pilot" else "additional_control_draft")).resolve()
    manifest=(args.manifest or ROOT/("pilot_manifest_task3.csv" if args.kind=="pilot" else "experiment_manifest_draft.csv")).resolve()
    rows=make_manifest(args.kind,output,args.binary.resolve(),manifest)
    run_rows(rows,args.binary.resolve(),args.python_exe.resolve(),args.execute,args.kind,args.max_runs)


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise SystemExit(2)
