"""105 remaining fits on the frozen neural sample; reuse the 30 saved neural fits."""
from pathlib import Path
import hashlib
import json
import random
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis_revision/neural_diagnostic_rerun_20261108'
OUT = ROOT / 'analysis_revision/common_model_diagnostics_20261108'
sys.path.insert(0, str(ROOT / 'analysis_la_cetsp'))
import model_ceiling as mc


def main():
    from threadpoolctl import threadpool_limits
    OUT.mkdir(exist_ok=True)
    results = []
    manifest = pd.read_csv(BASE / 'fit_manifest.csv')
    assign = pd.read_csv(BASE / 'fold_assignments.csv')
    models = mc.build_models(type('Args', (), dict(skip_neural=True, rsf_trees=200))())
    protocol = dict(models=list(models), new_fits=105, reused_neural_fits=30,
                    source_sample=str(BASE), mtlr_score='negative_rmst_step_v1',
                    settings='Same 3 outer folds, features and model hyperparameters as model_ceiling.py; no tuning',
                    scoring='Mean fold C then equal mean of five instances; selected positive-lifetime eviction sample',
                    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                             ['analysis_la_cetsp/model_ceiling.py','tools/common_model_diagnostics.py','ml/scripts/mtlr_scores.py']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for instance in mc.DEFAULT_INSTANCES:
        d = pd.read_csv(BASE/'inputs'/f'{instance}.csv',float_precision='round_trip')
        a = assign[assign.instance==instance].set_index('record_index').loc[d.record_index]
        for fold in [1,2,3]:
            te = np.flatnonzero(a.test_fold.to_numpy()==fold);tr=np.setdiff1d(np.arange(len(d)),te)
            sc=StandardScaler().fit(d.iloc[tr][mc.FEATURES]);xtr=sc.transform(d.iloc[tr][mc.FEATURES]);xte=sc.transform(d.iloc[te][mc.FEATURES])
            for name,fit in models.items():
                f=OUT/f'{instance}-{name}-fold{fold}.json'
                if f.exists():results.append(json.loads(f.read_text()));continue
                print('OFFLINE',len(results)+1,'/105',instance,name,fold,flush=True)
                start=time.monotonic();random.seed(0);np.random.seed(0)
                with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    risk=fit(xtr,d.survival_time.to_numpy()[tr],d.event.to_numpy(dtype=bool)[tr],xte)
                if not np.isfinite(risk).all():raise RuntimeError('Nonfinite risk')
                c=concordance_index_censored(d.event.to_numpy(dtype=bool)[te],d.survival_time.to_numpy()[te],risk)[0]
                pd.DataFrame(dict(record_index=te,risk=risk)).to_csv(f.with_suffix('.csv'),index=False,float_format='%.17g')
                row=dict(instance=instance,model=name,fold=fold,cindex=c,seconds=time.monotonic()-start,
                         warnings=[str(w.message) for w in ws]);f.write_text(json.dumps(row,indent=2));results.append(row)
                (OUT/'status.json').write_text(json.dumps(dict(state='running',completed=len(results),planned=105)))
    rows=results.copy()
    for r in manifest.to_dict('records'):
        if r['model']=='DEEPSURV':
            v=json.loads((BASE/'fits'/r['fit_id']/'result.json').read_text());c=v['cindex']
        else:
            pred=pd.read_csv(ROOT/'analysis_revision/mtlr_score_correction_20261108'/f"{r['fit_id']}.csv",float_precision='round_trip')
            c=concordance_index_censored(pred.event.to_numpy(dtype=bool),pred.survival_time.to_numpy(),pred.negative_rmst_risk.to_numpy())[0]
        rows.append(dict(instance=r['instance'],model=r['model'],fold=r['fold'],cindex=c))
    df=pd.DataFrame(rows);df.to_csv(OUT/'all_fold_results.csv',index=False)
    table=df.groupby(['instance','model']).cindex.mean().unstack('model').reindex(mc.DEFAULT_INSTANCES)
    table.loc['MEAN']=table.mean();table.to_csv(OUT/'common_sample_model_comparison.csv',index_label='instance')
    (OUT/'status.json').write_text(json.dumps(dict(state='complete',completed=105,reused_neural_fits=30,solver_runs=0)))
    print(table.to_string(),flush=True)


if __name__=='__main__':main()
