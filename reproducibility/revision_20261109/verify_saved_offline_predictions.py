"""Read saved predictions only; no fitting, solver, or original-file writes."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
W=Path(__file__).resolve().parent; R=W
B=R/'analysis_revision/neural_diagnostic_rerun_20261108'
C=R/'analysis_revision/common_model_diagnostics_20261108'
A=pd.read_csv(B/'fold_assignments.csv'); rows=[]; total=0
for p in (B/'inputs').glob('*.csv'):
 d=pd.read_csv(p,float_precision='round_trip');total+=len(d)
 a=A[A.instance==p.stem].set_index('record_index').loc[d.record_index]
 assert (a.run.to_numpy()==d.run.to_numpy()).all()
 assert a.groupby('run').test_fold.nunique().max()==1
 for fold in [1,2,3]:
  ix=np.flatnonzero(a.test_fold.to_numpy()==fold);tr=np.setdiff1d(np.arange(len(d)),ix)
  assert not set(d.iloc[ix].run)&set(d.iloc[tr].run)
  target=d.iloc[ix]
  for model in ['COX','ELASTICNET','RSF','GBSA','SSVM','WEIBULLAFT','KNN','DEEPSURV','MTLR']:
   if model=='DEEPSURV':
    pred=pd.read_csv(B/'fits'/f'{p.stem}-deepsurv-fold{fold}'/'predictions.csv',float_precision='round_trip');risk=pred.risk
   elif model=='MTLR':
    pred=pd.read_csv(R/'analysis_revision/mtlr_score_correction_20261108'/f'{p.stem}-mtlr-fold{fold}.csv',float_precision='round_trip');risk=pred.negative_rmst_risk
   else:
    pred=pd.read_csv(C/f'{p.stem}-{model}-fold{fold}.csv',float_precision='round_trip');risk=pred.risk
   assert np.array_equal(pred.record_index,ix)
   assert np.isfinite(risk).all()
   score=concordance_index_censored(target.event.to_numpy(dtype=bool),target.survival_time.to_numpy(),risk.to_numpy())[0]
   rows.append(dict(instance=p.stem,model=model,fold=fold,cindex=score))
assert total==18115 and len(rows)==135
rebuilt=pd.DataFrame(rows)
saved=pd.read_csv(C/'all_fold_results.csv')
j=rebuilt.merge(saved,on=['instance','model','fold'],suffixes=('_new','_old'),validate='one_to_one')
assert len(j)==135
error=float(abs(j.cindex_new-j.cindex_old).max());assert error<1e-12
table=rebuilt.groupby(['instance','model']).cindex.mean().unstack();table.loc['MEAN']=table.mean()
expected=pd.read_csv(C/'common_sample_model_comparison.csv').set_index('instance')
assert np.allclose(table.loc[expected.index,expected.columns],expected,rtol=0,atol=1e-12)
for p in json.loads((C/'protocol.json').read_text())['sources']:
 assert hashlib.sha256((R/p).read_bytes()).hexdigest()==json.loads((C/'protocol.json').read_text())['sources'][p]
report=dict(records=total,fold_scores=135,training_runs_held_out=True,prediction_cindex_max_error=error,source_hashes_match=True,no_training_or_solver_launched=True)
(W/'rerun'/'offline_verification.json').write_text(json.dumps(report,indent=2));print(report)
