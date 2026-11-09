from pathlib import Path
import pandas as pd,json
R=Path.cwd();O=R/'analysis_revision/auxiliary_recheck_20261107';a=pd.read_csv(O/'safe_field_trace.csv');old=pd.read_csv(R/'analysis_test/parsed/island_summaries.csv');new=old.copy();changes=[]
for row in a.itertuples():
 ix=new.index[(new.instance==row.instance)&(new.run==row.run)&(new.model==row.model)];assert len(ix)==1
 for field in ['best_iter','effective_iter']:
  val=getattr(row,field);kind=getattr(row,field+'_evidence_kind');new.loc[ix,field+'_evidence_kind']=kind
  if pd.notna(val):
   before=int(new.loc[ix,field].iloc[0]);after=int(val)
   if before!=after:changes.append(dict(instance=row.instance,run=row.run,model=row.model,field=field,old_value=before,reference_value=after,evidence_kind=kind,console=row.console))
   new.loc[ix,field]=after
new.to_csv(O/'verified_historical_summaries.csv',index=False);pd.DataFrame(changes).to_csv(O/'iteration_corrections.csv',index=False);a[a.best_iter.isna()|a.effective_iter.isna()].to_csv(O/'unresolved_iteration_fields.csv',index=False)
core=pd.read_csv(R/'analysis_revision/remaining_issues_20261107/historical_traceability.csv');core=core[['instance','run','model','island','console','tail_start_byte','joined_summary','best','total_t','ml_rejects','core_fields_match']];assert core.core_fields_match.all();core.to_csv(O/'historical_traceability.csv',index=False)
impact=[]
for model,g in old.groupby('model'):
 h=new[new.model==model];x=100*(g.ml_rejects/g.effective_iter).mean();y=100*(h.ml_rejects/h.effective_iter).mean();impact.append(dict(model=model,original_rejection_pct=x,reference_rejection_pct=y,display_2dp_unchanged=f'{x:.2f}'==f'{y:.2f}'))
pd.DataFrame(impact).to_csv(O/'rejection_rate_impact.csv',index=False)
activity={}
for tag,data in [('original',old),('safe_reference',new)]:
 ml=data[data.model!='BASELINE'];tr=ml[ml.effective_iter>=1000];activity[tag]=dict(before_trigger=int((ml.effective_iter<1000).sum()),at_trigger=int((ml.effective_iter==1000).sum()),median_window=float((tr.effective_iter-1000).median()),median_window_fraction=float(((tr.effective_iter-1000)/tr.effective_iter).median()))
assert (new[['best','total_t','ml_rejects']].values==old[['best','total_t','ml_rejects']].values).all()
known=a[(a.instance=='concentricCircles2')&(a.run==5)&(a.model=='RSF')].iloc[0];assert pd.isna(known.best_iter) or known.best_iter!=7,'Foreign island ID must not be accepted as best_iter'
summary=json.loads((O/'summary.json').read_text());summary.update(corrected_fields=len(changes),cost_and_runtime_unchanged_exactly=True,displayed_rejection_rates_unchanged=all(v['display_2dp_unchanged'] for v in impact),activation_summaries_unchanged=activity['original']==activity['safe_reference'],activity=activity,supersedes='The prior permissive 139-field correction list is withdrawn. Only this evidence-labelled reference is used in the current submission.',core_rows_verified=2650)
(O/'correction_impact.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert summary['displayed_rejection_rates_unchanged'];assert summary['activation_summaries_unchanged']
