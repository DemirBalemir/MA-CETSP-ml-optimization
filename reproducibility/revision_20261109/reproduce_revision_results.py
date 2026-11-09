"""Recalculate saved-result statistics and held-out concordance. Never fit/run a solver."""
from pathlib import Path
import json,runpy,subprocess,sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
R=Path(__file__).resolve().parent
O=R/'solutions/neural53_v1'; out=R/'rerun';out.mkdir(exist_ok=True)
d=pd.read_csv(O/'results.csv');m=pd.read_csv(O/'evaluation_manifest.csv')
assert len(d)==len(m)==795 and set(d.run_id)==set(m.run_id)
assert d.problem.nunique()==53 and not d.duplicated(['problem','search_seed','ml_model']).any()
assert (d.groupby(['problem','ml_model']).search_seed.nunique()==5).all()
means=d.groupby(['problem','ml_model'])[['best','solver_seconds']].mean()
rng=np.random.default_rng(20261108);rows=[]
for a,b in [('DEEPSURV','NONE'),('MTLR','NONE'),('DEEPSURV','MTLR')]:
 for metric in ['best','solver_seconds']:
  x=100*(means.xs(a,level=1)[metric]/means.xs(b,level=1)[metric]-1)
  boot=x.to_numpy()[rng.integers(0,53,size=(10000,53))].mean(axis=1)
  rows.append(dict(treatment=a,reference=b,metric=metric,mean_pct=float(x.mean()),median_pct=float(x.median()),lo95=float(np.quantile(boot,.025)),hi95=float(np.quantile(boot,.975)),p=1. if np.all(x.to_numpy()==0) else float(wilcoxon(x).pvalue)))
previous=0.
for rank,index in enumerate(np.argsort([r['p'] for r in rows])):
 previous=max(previous,min(1.,(6-rank)*rows[index]['p']));rows[index]['holm_p']=previous
actual=pd.DataFrame(rows);expected=pd.read_csv(O/'comparisons.csv')
assert actual[['treatment','reference','metric']].equals(expected[['treatment','reference','metric']])
cols=['mean_pct','median_pct','lo95','hi95','p','holm_p']
assert np.allclose(actual[cols],expected[cols],rtol=0,atol=1e-10)
actual.to_csv(out/'neural_comparisons.csv',index=False)
metrics=pd.DataFrame([json.loads((O/'evaluation'/r/'validated_metrics.json').read_text()) for r in d.run_id])
assert (metrics.generated==metrics.filter_rejected+metrics.vnd_evaluated).all()
assert (metrics.vnd_evaluated==metrics.admitted+metrics.not_admitted).all()
assert (metrics.vnd_evaluated==metrics.socp_successes+metrics.socp_numerical_fallbacks).all()
counts=metrics.groupby('ml_model')[['generated','filter_rejected','vnd_evaluated','admitted','socp_successes','socp_numerical_fallbacks']].sum()
assert counts.equals(pd.read_csv(R/'revision_audit/neural_operational_totals.csv').set_index('ml_model'))
counts.to_csv(out/'neural_operational_totals.csv')
runpy.run_path(str(R/'verify_saved_offline_predictions.py'),run_name='__main__')
legacy=R/'legacy_20261107/Revision_Reproducibility_Bundle/reproduce_results.py'
subprocess.run([sys.executable,str(legacy)],check=True)
print('Six neural contrasts, 135 saved-prediction C-indices and six legacy Cox contrasts reproduced. No fitting or solver execution.')
