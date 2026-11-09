"""Frozen analysis of all 53 problem clusters; no solver launches."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(values):
    order=np.argsort(values);result=np.zeros(len(values));prev=0.
    for rank,index in enumerate(order):
        prev=max(prev,min(1.,(len(values)-rank)*values[index]));result[index]=prev
    return result.tolist()


def analyze(root,rows,inspect,rates):
    records=[]
    for row in rows:
        v=inspect(row)
        v.update(problem=row['problem'],seed=int(row['search_seed']),method=row['method'],
                 calibration_basis=rates[row['problem']]['basis'],
                 calibration_probability=rates[row['problem']]['probability'],
                 zero_fallback_mismatch=(rates[row['problem']]['basis']=='calibration_pre_activation_stop'
                                         and row['method']=='learned' and v['filter_rejected']>0))
        records.append(v)
    d=pd.DataFrame(records)
    if len(d)!=795 or d.duplicated(['problem','seed','method']).any() or d.problem.nunique()!=53:
        raise ValueError('complete unique 53x5x3 results required; no silent complete-case analysis')
    d.to_csv(root/'evaluation_diagnostics.csv',index=False)
    means=d.groupby(['problem','method'])[['best','solver_seconds']].mean()
    rng=np.random.default_rng(20261030);effects=[];per=[]
    for a,b in [('learned','random'),('learned','none'),('random','none')]:
        for metric in ['best','solver_seconds']:
            # Equal problem weights; each problem aggregates its five matched seeds.
            x=100*(means.xs(a,level='method')[metric]/means.xs(b,level='method')[metric]-1)
            if not np.isfinite(x).all():raise ValueError('nonfinite comparison')
            boot=x.to_numpy()[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
            p=1. if (x==0).all() else float(wilcoxon(x,alternative='two-sided',method='auto').pvalue)
            effects.append(dict(treatment=a,reference=b,metric=metric,n_problems=len(x),
                mean_effect_pct=float(x.mean()),median_effect_pct=float(x.median()),
                lo95=float(np.quantile(boot,.025)),hi95=float(np.quantile(boot,.975)),p=p))
            per.extend(dict(problem=name,treatment=a,reference=b,metric=metric,effect_pct=float(v)) for name,v in x.items())
    for row,p in zip(effects,holm([r['p'] for r in effects])):row['holm_p_six_tests']=p
    pd.DataFrame(effects).to_csv(root/'comparisons.csv',index=False)
    pd.DataFrame(per).to_csv(root/'problem_effects.csv',index=False)
    # Preserve matched-seed contrasts as descriptive diagnostics, not 265 independent problems.
    wide=d.pivot(index=['problem','seed'],columns='method',values=['best','solver_seconds'])
    wide.to_csv(root/'paired_seed_outcomes.csv')
    (root/'analysis_status.json').write_text(json.dumps(dict(state='complete',runs=len(d),
        primary='learned versus random: best and solver_seconds',
        uncertainty='10000 problem-cluster bootstrap draws, seed 20261030; percentile 95% descriptive intervals',
        tests='two-sided Wilcoxon on per-problem relative mean effects; Holm across all six contrasts',
        caveat='No equivalence claim. Active-only diagnostics are descriptive. Interval family is not simultaneous.'),indent=2))
