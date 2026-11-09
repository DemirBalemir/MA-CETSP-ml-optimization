"""Read-only reanalysis of the original 53-instance experiment; no solver calls.

Post-hoc TOST sensitivity uses instance-level percentage effects, not run pairs.
Within-time diagnostics use only control logs matched by console timestamps.
"""
from pathlib import Path
import sys, re, json, hashlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import t as student
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis_revision'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'analysis_la_cetsp'))
from feature_ceiling import extract_features, FEATURES

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def effects():
    p = ROOT/'analysis_test/parsed/island_summaries.csv'
    d = pd.read_csv(p)
    assert len(d) == 2650 and d.instance.nunique() == 53
    assert not d.duplicated(['instance','run','model']).any()
    assert (d.groupby(['instance','model']).size() == 5).all()
    means = d.groupby(['instance','model'])[['best','total_t']].mean()
    active = set(d.groupby('instance').ml_rejects.sum().loc[lambda x:x>0].index)
    rows=[]; raw=[]
    for model in sorted(set(d.model)-{'BASELINE'}):
        for metric,col,margins in [('quality','best',[.1,.5,1]),('runtime','total_t',[1,5,10])]:
            ratio=100*(means.xs(model,level='model')[col]/means.xs('BASELINE',level='model')[col]-1)
            for subset in ['full','active']:
                x=ratio if subset=='full' else ratio.loc[sorted(active)]
                for inst,v in x.items(): raw.append(dict(model=model,metric=metric,subset=subset,instance=inst,effect_pct=v))
                n=len(x); mean=x.mean(); se=x.std(ddof=1)/np.sqrt(n)
                lo90,hi90=mean+np.array([-1,1])*student.ppf(.95,n-1)*se
                lo95,hi95=mean+np.array([-1,1])*student.ppf(.975,n-1)*se
                for margin in margins:
                    pv=max(student.sf((mean+margin)/se,n-1),student.cdf((mean-margin)/se,n-1)) if se else float(not(-margin<mean<margin))
                    rows.append(dict(model=model,metric=metric,subset=subset,n=n,mean_pct=mean,lo95=lo95,hi95=hi95,lo90=lo90,hi90=hi90,margin_pct=margin,tost_p=pv))
    pd.DataFrame(rows).to_csv(OUT/'equivalence_sensitivity.csv',index=False)
    pd.DataFrame(raw).to_csv(OUT/'instance_effects.csv',index=False)
    return d,dict(path=str(p.relative_to(ROOT)),sha256=digest(p))

def matched_logs(d):
    manifest=[]
    defs=(ROOT/'include/Defs.hpp').read_text()
    names=re.findall(r'"([^"]+)"',defs.split('FILENAMES = {')[1].split('};')[0])
    for log in sorted((ROOT/'solutions/experiment_results').glob('*/run*_seed*.log')):
        # The unique timestamp is printed before concurrent output begins.
        with log.open(encoding='utf-8-sig',errors='replace') as f: head=f.read(16000)
        match=re.search(r'instance=(\d+) seed=(\d+).*?timestamp=(\d+)',head)
        if not match: raise ValueError(f'No header {log}')
        seed=int(match[2]); stamp=match[3]
        inst=names[int(match[1])]
        assert inst in set(d.instance)
        run=ROOT/'solutions/ml_logs'/inst/'island_9'/f'run-{stamp}_i9'
        files=sorted(run.glob('sol-*.json'),key=lambda p:int(p.stem.split('-')[1]))
        manifest.append(dict(instance=inst,seed=seed,timestamp=stamp,console=str(log.relative_to(ROOT)),raw_dir=str(run.relative_to(ROOT)),files=len(files)))
    m=pd.DataFrame(manifest)
    assert len(m)==265 and not m.duplicated(['instance','seed']).any()
    m.to_csv(OUT/'matched_run_manifest.csv',index=False)
    return m

def concordance(frame,risk,width):
    num=den=0
    groups=[np.arange(len(frame))] if width==0 else [np.flatnonzero((frame.birth.to_numpy()//width)==w) for w in np.unique(frame.birth.to_numpy()//width)]
    for ix in groups:
        if len(ix)<2: continue
        try:
            c,ok,bad,tied,_=concordance_index_censored(frame.event.to_numpy()[ix],frame.duration.to_numpy()[ix],risk[ix])
        except ValueError: continue
        num+=ok+.5*tied;den+=ok+bad+tied
    return num/den if den else np.nan,den

def diagnostics(manifest):
    cache=OUT/'diagnostic_features.csv'
    if cache.exists(): data=pd.read_csv(cache)
    else:
        records=[]; hashes=[]
        for r in manifest.itertuples():
            files=sorted((ROOT/r.raw_dir).glob('sol-*.json'),key=lambda p:int(p.stem.split('-')[1]))
            # Fixed cap chosen before seeing diagnostic outcomes; sample across entire run.
            if len(files)>400:
                rng=np.random.default_rng(int(r.timestamp));files=[files[i] for i in sorted(rng.choice(len(files),400,replace=False))]
            for p in files:
                b=p.read_bytes(); obj=json.loads(b)
                if obj.get('survival_iters',0)<=0 or not obj.get('pre_vnd_coords'): continue
                f=extract_features(obj['pre_vnd_coords'],obj['pre_vnd_cost'])
                if f is None: continue
                f.update(instance=r.instance,run=str(r.timestamp),birth=obj['birth_iter'],duration=obj['survival_iters'],event=not obj['censored'])
                records.append(f);hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest()))
            print('loaded',r.instance,r.seed,'sample',len(files),flush=True)
        data=pd.DataFrame(records);data.to_csv(cache,index=False)
        pd.DataFrame(hashes).to_csv(OUT/'sample_hashes.csv',index=False)
    rows=[]; failures=[]
    for inst,df in data.groupby('instance'):
        if df.run.nunique()<3: continue
        for run,te in df.groupby('run'):
            tr=df[df.run!=run]
            if len(tr)<40 or len(te)<10: continue
            for name,cols in [('geometry',FEATURES),('clock',['birth']),('geometry_clock',FEATURES+['birth'])]:
                try:
                    sc=StandardScaler().fit(tr[cols]);model=CoxPHSurvivalAnalysis(alpha=.1,n_iter=200)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');model.fit(sc.transform(tr[cols]),Surv.from_arrays(tr.event.astype(bool),tr.duration))
                    risk=model.predict(sc.transform(te[cols]))
                    assert np.isfinite(risk).all()
                    for width in [0,100,250,500]:
                        c,pairs=concordance(te,risk,width)
                        rows.append(dict(instance=inst,run=run,model=name,window=width,n_test=len(te),cindex=c,pairs=pairs))
                except Exception as exc: failures.append(dict(instance=inst,run=run,model=name,error=str(exc)))
        print('fitted',inst,flush=True)
    out=pd.DataFrame(rows);out.to_csv(OUT/'within_time_runs.csv',index=False)
    pd.DataFrame(failures,columns=['instance','run','model','error']).to_csv(OUT/'diagnostic_failures.csv',index=False)
    per=out.groupby(['instance','model','window']).cindex.mean().reset_index()
    per.to_csv(OUT/'within_time_instances.csv',index=False)
    per.groupby(['model','window']).agg(mean_c=('cindex','mean'),n_instances=('cindex','count')).to_csv(OUT/'within_time_summary.csv')
    print('FEATURE ROWS',len(data),'INSTANCES',data.instance.nunique(),'FAILURES',len(failures),flush=True)

if __name__=='__main__':
    d,h=effects();m=matched_logs(d)
    (OUT/'input_summary.json').write_text(json.dumps(h,indent=2))
    print('Matched',len(m),'runs; raw available',int((m.files>0).sum()),'instances',m[m.files>0].instance.nunique(),flush=True)
    diagnostics(m)
