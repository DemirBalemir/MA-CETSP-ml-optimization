"""Sensitivity of recorded-cohort rank agreement to full-VND gain reassignment.

kappa_VND = Spearman(pre-VND recorded-cohort percentile, relative VND gain).
kappa_LKH has the same definition at the LKH stage. Neither is a bound on
predictability. Samples omit unadmitted candidates and terminal survivors.
Membership is held fixed in the simulation; changed survival is not simulated.
Run this file directly as the command-line entry point.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from diagnostic_metrics import cohort_mask, cohort_rank, pairwise_concordance
from diagnostic_metrics import rank_gain_kappa, couple_to_kappa

KAPPAS = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
DEFAULT_INSTANCES = [
    'bonus1000', 'bubbles1', 'bubbles2', 'bubbles4', 'bubbles5', 'bubbles6',
    'bubbles7', 'bubbles8', 'car_door_25', 'car_door_35', 'car_door_40',
    'car_door_45', 'car_door_50', 'chaoSingleDep', 'concentricCircles1',
    'concentricCircles3', 'concentricCircles4', 'concentricCircles5',
    'd493_or10', 'd493_or2', 'd493_or30', 'dsj1000_or10', 'dsj1000_or2',
    'dsj1000_or30', 'dsj1000rdmRad',
]


def load_run(path, max_files):
    files = sorted((f for f in os.scandir(path)
                    if f.is_file() and f.name.startswith('sol-') and f.name.endswith('.json')),
                   key=lambda f: int(f.name[4:-5]))[-max_files:]
    rows, digest = [], hashlib.sha256()
    for f in files:
        raw = Path(f.path).read_bytes()
        digest.update(f.name.encode() + b'\0' + raw)
        try:
            d = json.loads(raw)
            p, q, s = d.get('pre_vnd_cost'), d.get('post_vnd_cost'), d.get('survival_iters')
            if not p or not q or p <= 0 or q <= 0 or s is None or s <= 0:
                continue
            rows.append((p, q, d['birth_iter'], d['death_iter']))
        except (ValueError, KeyError, TypeError):
            continue
    if len(rows) < 120:
        return None
    return np.asarray(rows, float).T, digest.hexdigest(), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_dir', default='solutions/ml_logs')
    ap.add_argument('--instances', nargs='+', default=DEFAULT_INSTANCES)
    ap.add_argument('--max_instances', type=int, default=25)
    ap.add_argument('--island', type=int, default=9)
    ap.add_argument('--max_runs', type=int, default=3)
    ap.add_argument('--max_per_run', type=int, default=800)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--out_dir', default='analysis_la_cetsp/results')
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    observed, simulated, manifest = [], [], []
    for inst in args.instances[:args.max_instances]:
        idir = Path(args.logs_dir) / inst / f'island_{args.island}'
        if not idir.is_dir():
            raise FileNotFoundError(idir)
        runs = sorted((Path(d.path) for d in os.scandir(idir)
                       if d.is_dir() and d.name.startswith('run-')))[-args.max_runs:]
        for run in runs:
            loaded = load_run(run, args.max_per_run)
            if loaded is None:
                continue
            (p, q, birth, death), digest, n_files = loaded
            mask = cohort_mask(birth, death)
            pre_rank = cohort_rank(p, mask)
            valid = np.isfinite(pre_rank)
            if valid.sum() < 30:
                continue
            gain = (p-q)/p
            observed.append(dict(instance=inst, run=run.name,
                                 kappa_observed=rank_gain_kappa(pre_rank, gain),
                                 C_observed=pairwise_concordance(pre_rank, cohort_rank(q, mask))))
            manifest.append(dict(instance=inst, island=args.island, run=run.name,
                                 selected_files=n_files, usable_records=len(p),
                                 ranked_records=int(valid.sum()), sha256=digest))
            for target in KAPPAS:
                for rep in range(args.repeats):
                    g = gain.copy()
                    g[valid] = couple_to_kappa(pre_rank[valid], gain[valid], target, rng)
                    simulated.append(dict(instance=inst, run=run.name, target_kappa=target,
                                          repeat=rep, achieved_kappa=rank_gain_kappa(pre_rank, g),
                                          C=pairwise_concordance(pre_rank, cohort_rank(p*(1-g), mask))))
        print(f'processed {inst}', flush=True)
    if not observed:
        raise RuntimeError('No usable runs')
    obs = pd.DataFrame(observed)
    obs.to_csv(out/'kappa_observed_runs.csv', index=False)
    obs.groupby('instance')[['kappa_observed', 'C_observed']].mean().to_csv(out/'kappa_observed.csv')
    sim = pd.DataFrame(simulated)
    sim.to_csv(out/'kappa_simulations.csv', index=False)
    # Equal instance weighting after equal run/repeat weighting within instances.
    per_instance = sim.groupby(['target_kappa', 'instance'])[['achieved_kappa', 'C']].mean()
    summary = per_instance.groupby('target_kappa').agg(
        kappa_mean=('achieved_kappa', 'mean'), kappa_std=('achieved_kappa', 'std'),
        C_mean=('C', 'mean'), C_std=('C', 'std'), n_instances=('C', 'count'))
    summary.index.name = 'kappa'
    summary.to_csv(out/'kappa_calibration.csv')
    pd.DataFrame(manifest).to_csv(out/'kappa_run_manifest.csv', index=False)
    print(summary.round(4).to_string())
    print('Observed instance means:', obs.groupby('instance')[['kappa_observed','C_observed']].mean().mean().to_dict())


if __name__ == '__main__':
    main()
