from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
R=Path(__file__).resolve().parent
d=pd.read_csv(R/'control/final_evaluation_diagnostics.csv')
assert len(d)==795 and d.problem.nunique()==53 and not d.duplicated(['problem','seed','method']).any()
m=d.groupby(['problem','method'])[['best','solver_seconds']].mean();rng=np.random.default_rng(20261030);rows=[]
for a,b in [('learned','random'),('learned','none'),('random','none')]:
 for metric in ['best','solver_seconds']:
  x=100*(m.xs(a,level=1)[metric]/m.xs(b,level=1)[metric]-1);boot=x.to_numpy()[rng.integers(0,53,size=(10000,53))].mean(axis=1)
  rows.append(dict(treatment=a,reference=b,metric=metric,mean_effect_pct=x.mean(),median_effect_pct=x.median(),lo95=np.quantile(boot,.025),hi95=np.quantile(boot,.975),p=wilcoxon(x).pvalue))
order=np.argsort([r['p'] for r in rows]);previous=0.
for rank,index in enumerate(order):
 previous=max(previous,min(1.,(6-rank)*rows[index]['p']));rows[index]['holm_p_six_tests']=previous
actual=pd.DataFrame(rows);expected=pd.read_csv(R/'control/final_comparisons.csv')
for c in ['mean_effect_pct','median_effect_pct','lo95','hi95','p','holm_p_six_tests']:assert np.allclose(actual[c],expected[c],rtol=1e-9,atol=1e-9),c
out=R/'rerun';out.mkdir(exist_ok=True);actual.to_csv(out/'comparisons.csv',index=False)
print('All six published contrasts reproduced; no solver was launched.')
