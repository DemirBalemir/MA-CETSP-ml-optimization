"""Explicit current common-model input; older feature-panel inputs are separate."""
from pathlib import Path
import importlib.util, json, hashlib
import pandas as pd
W=Path(__file__).resolve().parent;R=W.parents[1]
spec=importlib.util.spec_from_file_location('legacy_figures',R/'analysis_la_cetsp/make_figures.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# The imported module is read-only except for mkdir(exist_ok=True); no old figure is saved.
m.OUT=W/'source/figures'
d=pd.read_csv(R/'analysis_revision/common_model_diagnostics_20261108/common_sample_model_comparison.csv')
v=d[d.instance=='MEAN'].iloc[0].drop('instance').sort_values()
fams,drift=m.feature_values()
fig,(a,b)=m.plt.subplots(1,2,figsize=(7,2.6))
y=m.np.arange(len(v));a.barh(y,v.to_numpy(dtype=float),color=m.BAR,height=.68,zorder=3)
a.set_yticks(y);a.set_yticklabels(v.index)
a.axvline(.5,color=m.CHANCE,lw=1.2,ls='--',label='chance')
a.axvline(.687,color=m.CEIL,lw=1.2,ls=':',label='older oracle (different sample)')
a.set_xlim(.45,.72);a.set_xlabel('cross-validated concordance');a.set_title('(a) common sample: 18,115 records',fontsize=8)
a.legend(loc='lower right',fontsize=6.3,framealpha=.95)
y=m.np.arange(len(fams));b.barh(y,[v for _,v in fams],color=[m.ACCENT if n in ('Lineage','All 23 features') else m.BAR for n,_ in fams],height=.68,zorder=3)
b.set_yticks(y);b.set_yticklabels([n for n,_ in fams]);b.axvline(.5,color=m.CHANCE,lw=1.2,ls='--',label='chance')
b.axvline(drift,color='k',lw=1.2,ls='-.',label=f'lineage clock ({drift:.3f})')
b.set_xlim(.45,.66);b.set_xlabel('concordance (protocols differ)');b.set_title('(b) older feature samples',fontsize=8)
b.legend(loc='lower right',framealpha=.95)
for ax in (a,b):ax.grid(axis='x',lw=.4,alpha=.4,zorder=0)
fig.tight_layout(w_pad=2);m.save(fig,'figA_landscape')
paths=['analysis_revision/common_model_diagnostics_20261108/common_sample_model_comparison.csv']+['analysis_la_cetsp/results/'+n+'.csv' for n in ['diversity_signal','partial_vnd_signal','oracle_ceiling','lineage_signal']]
(W/'figure_inputs.json').write_text(json.dumps({p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths},indent=2))
