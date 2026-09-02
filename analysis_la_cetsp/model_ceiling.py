"""Nine-model survival discrimination on recorded island-9 solutions.
GroupKFold by run uses the deployed feature family and diagnostic model settings.
Adjustments: Cox/Weibull penalties 0.01, neural fits 100 rather than 200 epochs,
and AFT negative expected rather than median survival. These are family-level
diagnostics, not exact re-evaluation of saved deployed models. Positive-lifetime
eviction records form a selected sample; the results are not universal ceilings.
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))
from feature_ceiling import extract_features, FEATURES

warnings.filterwarnings("ignore")

DEFAULT_INSTANCES = ["concentricCircles4", "car_door_50", "bonus1000",
                     "kroD100_or2", "rat195_or30"]


# ---------------------------------------------------------------------------
# Risk scorers. Each returns risk where HIGHER = shorter expected survival,
# matching the convention the C++ filter uses (score > threshold => reject).
# ---------------------------------------------------------------------------

def _fit_cox(Xtr, ttr, etr, Xte):
    from lifelines import CoxPHFitter
    df = pd.DataFrame(Xtr, columns=FEATURES)
    df["survival_time"] = ttr
    df["event_observed"] = etr.astype(int)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="survival_time", event_col="event_observed")
    return cph.predict_partial_hazard(pd.DataFrame(Xte, columns=FEATURES)).values


def _fit_elasticnet(Xtr, ttr, etr, Xte):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    m = CoxnetSurvivalAnalysis(l1_ratio=0.5, n_alphas=10, max_iter=5000)
    m.fit(Xtr, Surv.from_arrays(event=etr, time=ttr))
    p = m.predict(Xte)
    # may return (n_samples, n_alphas): take the middle alpha, as train_elasticnet.py does
    return p[:, p.shape[1] // 2] if p.ndim == 2 else p


def _fit_rsf(Xtr, ttr, etr, Xte, trees):
    from sksurv.ensemble import RandomSurvivalForest
    m = RandomSurvivalForest(n_estimators=trees, min_samples_split=10,
                             min_samples_leaf=5, max_features="sqrt",
                             n_jobs=-1, random_state=0)
    m.fit(Xtr, Surv.from_arrays(event=etr, time=ttr))
    return m.predict(Xte)


def _fit_gbsa(Xtr, ttr, etr, Xte):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    m = GradientBoostingSurvivalAnalysis(learning_rate=0.1, n_estimators=300,
                                         max_depth=3, random_state=0)
    m.fit(Xtr, Surv.from_arrays(event=etr, time=ttr))
    return m.predict(Xte)


def _fit_ssvm(Xtr, ttr, etr, Xte):
    from sksurv.svm import FastSurvivalSVM
    m = FastSurvivalSVM(rank_ratio=1.0, max_iter=100, tol=1e-3, random_state=0)
    m.fit(Xtr, Surv.from_arrays(event=etr, time=ttr))
    return m.predict(Xte)


def _fit_weibull(Xtr, ttr, etr, Xte):
    from lifelines import WeibullAFTFitter
    df = pd.DataFrame(Xtr, columns=FEATURES)
    df["survival_time"] = ttr
    df["event_observed"] = etr.astype(int)
    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(df, duration_col="survival_time", event_col="event_observed")
    # AFT predicts survival TIME; risk is its negation
    return -aft.predict_expectation(pd.DataFrame(Xte, columns=FEATURES)).values


def _fit_knn(Xtr, ttr, etr, Xte):
    from knn_survival import KNNSurvival
    m = KNNSurvival(n_neighbors=min(15, len(Xtr) - 1))
    m.fit(Xtr, ttr)
    return m.predict(Xte)


def _fit_deepsurv(Xtr, ttr, etr, Xte):
    import torch, torchtuples as tt
    from pycox.models import CoxPH
    net = tt.practical.MLPVanilla(in_features=Xtr.shape[1], num_nodes=[64, 64],
                                  out_features=1, batch_norm=True, dropout=0.1,
                                  output_bias=False)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    model.fit(Xtr.astype("float32"),
              (ttr.astype("float32"), etr.astype("float32")),
              batch_size=min(256, len(Xtr)), epochs=100, verbose=False,
              callbacks=[tt.callbacks.EarlyStopping(patience=20)])
    net.eval()
    with torch.no_grad():
        return net(torch.FloatTensor(Xte.astype("float32"))).numpy().flatten()


def _fit_mtlr(Xtr, ttr, etr, Xte):
    import torchtuples as tt
    from pycox.models import MTLR
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime
    ND = 20
    labtrans = LabTransDiscreteTime(ND)
    y = labtrans.fit_transform(ttr.astype("float32"), etr.astype("float32"))
    net = tt.practical.MLPVanilla(in_features=Xtr.shape[1], num_nodes=[64, 64],
                                  out_features=ND, batch_norm=True, dropout=0.1)
    model = MTLR(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    model.fit(Xtr.astype("float32"), y, batch_size=min(256, len(Xtr)),
              epochs=100, verbose=False,
              callbacks=[tt.callbacks.EarlyStopping(patience=20)])
    model.duration_index = labtrans.cuts
    surv = model.predict_surv_df(Xte.astype("float32"))
    return 1.0 - surv.iloc[-1].values      # higher = shorter survival


def build_models(args):
    m = {
        "COX":        _fit_cox,
        "ELASTICNET": _fit_elasticnet,
        "RSF":        lambda a, b, c, d: _fit_rsf(a, b, c, d, args.rsf_trees),
        "GBSA":       _fit_gbsa,
        "SSVM":       _fit_ssvm,
        "WEIBULLAFT": _fit_weibull,
        "KNN":        _fit_knn,
    }
    if not args.skip_neural:
        m["DEEPSURV"] = _fit_deepsurv
        m["MTLR"] = _fit_mtlr
    return m


def load_instance(inst_dir: Path, island: int, max_n: int, rng: random.Random):
    files = list((inst_dir / f"island_{island}").glob("run-*/sol-*.json"))
    if len(files) > max_n:
        files = rng.sample(files, max_n)
    rows = []
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        surv = d.get("survival_iters")
        coords = d.get("pre_vnd_coords")
        if surv is None or surv <= 0 or not coords:
            continue
        f = extract_features(coords, d.get("pre_vnd_cost", 0.0))
        if f is None:
            continue
        f["survival_time"] = float(surv)
        f["event"] = (not d.get("censored", False))
        f["run"] = fp.parent.name
        rows.append(f)
    return pd.DataFrame(rows)


def cv_cindex(X, t, e, groups, fit_fn, n_splits=3):
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return np.nan
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        sc = StandardScaler().fit(X[tr])
        try:
            risk = fit_fn(sc.transform(X[tr]), t[tr], e[tr], sc.transform(X[te]))
            scores.append(concordance_index_censored(e[te], t[te], risk)[0])
        except Exception as ex:
            print(f"      [fold failed] {type(ex).__name__}: {ex}", flush=True)
            scores.append(np.nan)
    return float(np.nanmean(scores)) if scores else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="solutions/ml_logs")
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--island", type=int, default=9)
    ap.add_argument("--max_per_instance", type=int, default=4000)
    ap.add_argument("--rsf_trees", type=int, default=200)
    ap.add_argument("--skip_neural", action="store_true")
    ap.add_argument("--out_dir", default="analysis_la_cetsp/results")
    args = ap.parse_args()

    rng = random.Random(0)
    logs = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = build_models(args)

    rows = []
    for inst in args.instances:
        idir = logs / inst
        if not (idir / f"island_{args.island}").exists():
            print(f"[skip] {inst}", flush=True)
            continue
        df = load_instance(idir, args.island, args.max_per_instance, rng)
        if len(df) < 200:
            print(f"[skip] {inst}: {len(df)} rows", flush=True)
            continue
        X = df[FEATURES].values.astype(float)
        t = df["survival_time"].values.astype(float)
        e = df["event"].values.astype(bool)
        groups = df["run"].values

        print(f"\n{inst}  (n={len(df)}, runs={len(np.unique(groups))})", flush=True)
        rec = {"instance": inst, "n": len(df)}
        for name, fn in models.items():
            c = cv_cindex(X, t, e, groups, fn)
            rec[name] = round(c, 4)
            print(f"   {name:<12} C={c:.3f}", flush=True)
        rows.append(rec)

    if not rows:
        print("no results")
        return

    rdf = pd.DataFrame(rows)
    mean_row = {"instance": "MEAN", "n": int(rdf["n"].sum())}
    for name in models:
        mean_row[name] = round(float(rdf[name].mean()), 4)
    rdf = pd.concat([rdf, pd.DataFrame([mean_row])], ignore_index=True)
    rdf.to_csv(out_dir / "model_ceiling.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 58)
    print("MEAN ACROSS INSTANCES  (chance = 0.500)")
    print("=" * 58)
    for name in models:
        print(f"  {name:<12} {mean_row[name]:.3f}")
    print("=" * 58)
    print(f"saved -> {(out_dir / 'model_ceiling.csv').resolve()}")
    print("\nRead: if all nine cluster near 0.53, model class is not the")
    print("      bottleneck and the limitation lies upstream of the model.")


if __name__ == "__main__":
    main()
