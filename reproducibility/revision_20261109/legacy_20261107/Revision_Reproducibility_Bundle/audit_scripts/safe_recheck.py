from pathlib import Path
import pandas as pd,re,json,hashlib
R=Path.cwd();O=R/'analysis_revision/auxiliary_recheck_20261107';d=pd.read_csv(R/'analysis_test/parsed/island_summaries.csv');mapping=pd.read_csv(R/'analysis_revision/matched_run_manifest.csv');out=[];num=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
for k,m in enumerate(mapping.itertuples()):
 p=R/m.console;b=p.read_bytes();s=b.decode('utf-8-sig',errors='replace');run=int(re.search(r'run(\d+)_seed',p.name)[1]);sha=hashlib.sha256(b).hexdigest();starts=[x.start() for x in re.finditer(r'\[SUMMARY\]',s)]
 for row in d[(d.instance==m.instance)&(d.run==run)].itertuples():
  candidates={'best_iter':[],'effective_iter':[]};proof={'best_iter':[],'effective_iter':[]}
  # Only literal scalar field grammar is accepted. No arbitrary integer mining.
  for x in re.finditer(rf'model=\s*{row.model}\s+best=\s*({num})\s+best_iter=\s*(\d+)\s+effective_iter=\s*(\d+)(?=\s)',s):
   if float(x[1])!=row.best:continue
   for field,val in [('best_iter',int(x[2])),('effective_iter',int(x[3]))]:candidates[field].append(val);proof[field].append(x[0])
  for x in re.finditer(rf'\[Island {row.island}\]\s+\[SUMMARY\]',s):
   ends=[v for v in starts if v>x.end()];part=s[x.start():min(x.start()+2200,ends[0] if ends else len(s))]
   if 'ml_rejects=' in part:part=part[:part.index('ml_rejects=')]
   names=re.findall(r'model=\s*(WEIBULLAFT|ELASTICNET|DEEPSURV|BASELINE|COX|RSF|GBSA|SSVM|KNN|MTLR)(?![A-Z])',part)
   if names and names!=[row.model]:continue
   for field,pat in [('best_iter',r'best_iter=\s*(\d+)\s+effective_iter='),('effective_iter',r'effective_iter=\s*(\d+)\s+total_t=')]:
    matches=list(re.finditer(pat,part))
    if len(matches)==1:candidates[field].append(int(matches[0][1]));proof[field].append(part)
  values={};kinds={};issues=[]
  for field,vals in candidates.items():
   vals=sorted(set(v for v in vals if 0<=v<=(5000 if field=='best_iter' else 5001)))
   values[field]=vals[0] if len(vals)==1 else None;kinds[field]='literal_summary_field' if len(vals)==1 else 'unresolved'
   if len(vals)>1:issues.append(field+' conflicting scalar values')
  logs=[]
  pat=rf'\[Island {row.island}\] \[LOG\] iter=(\d+) best=({num}) patience=(\d+) ml=(\S+) iter_t=({num}) total_t=({num})(?=\r?$)'
  for x in re.finditer(pat,s,re.M):
   i,c,pa,t=int(x[1]),float(x[2]),int(x[3]),float(x[6])
   if 1<=i<=5000 and 0<=pa<=500 and 0<=t<=row.total_t+0.1:logs.append((i,c,pa,x[0]))
  logs.sort();maxi=max([x[0] for x in logs],default=0)
  if values['effective_iter'] is not None and values['effective_iter']<maxi:
   issues.append('effective_iter below complete LOG');values['effective_iter']=None;kinds['effective_iter']='unresolved'
  stop=bool(re.search(rf'\[Island {row.island}\] \[STOP\] no improvement for 500 iterations',s))
  terminal=[x for x in logs if x[2]==500 and x[1]==row.best and x[0]==maxi]
  if values['effective_iter'] is None and len(terminal)==1 and not any(x[0]>terminal[0][0] for x in logs):
   values['effective_iter']=terminal[0][0];kinds['effective_iter']='terminal_log_corroboration';proof['effective_iter'].append(terminal[0][3])
  bi,eff=values['best_iter'],values['effective_iter']
  if bi is not None and eff is not None and (bi>eff or (stop and not 500<=eff-bi<=500+row.ml_rejects)):
   issues.append('summary pair conflicts with stopping counters');values={'best_iter':None,'effective_iter':None};kinds={x:'unresolved' for x in values}
  # Exact counter relation is used only as explicitly labelled corroboration,
  # with zero recorded rejections, natural stop, and compatible complete logs.
  if row.ml_rejects==0 and stop:
   bi,eff=values['best_iter'],values['effective_iter'];profile=max([i-pa for i,c,pa,_ in logs],default=0)
   if bi is not None and eff is None and bi>=profile and bi+500>=maxi:
    values['effective_iter']=bi+500;kinds['effective_iter']='zero_rejection_stop_rule_corroboration'
   elif eff is not None and bi is None and eff>=500 and eff-500>=profile:
    values['best_iter']=eff-500;kinds['best_iter']='zero_rejection_stop_rule_corroboration'
  rec=dict(instance=row.instance,run=run,model=row.model,island=row.island,console=m.console,console_sha256=sha,old_best_iter=row.best_iter,old_effective_iter=row.effective_iter,ml_rejects=row.ml_rejects,complete_log_count=len(logs),natural_stop_literal=stop,issues='; '.join(issues))
  for field in values:rec[field]=values[field];rec[field+'_evidence_kind']=kinds[field];rec[field+'_excerpt']='\n---\n'.join(proof[field][:2])
  out.append(rec)
 if (k+1)%50==0:print('rechecked logs',k+1,flush=True)
a=pd.DataFrame(out);a.to_csv(O/'safe_field_trace.csv',index=False)
summary=dict(rows=len(a),fields={field:dict(recovered=int(a[field].notna().sum()),unresolved=int(a[field].isna().sum()),evidence=a[field+'_evidence_kind'].value_counts().to_dict(),different_from_original=int(((a[field].notna())&(a[field]!=a['old_'+field])).sum())) for field in ['best_iter','effective_iter']})
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
