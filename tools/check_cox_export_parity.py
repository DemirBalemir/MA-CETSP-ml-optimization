"""Fit/export Python Cox, compare actual native predictions on real714-row data."""
import json
from pathlib import Path
import subprocess
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'solutions/fixed_preflight_20261102/python_cpp_parity'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'solutions/control53_parallel4_v1/evaluation/worker-4/kroD100_or2-s11-learned-a1/model/cox_dataset.csv')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model=CoxPHFitter().fit(data,duration_col='survival_time',event_col='event_observed')
    x=data.drop(columns=['survival_time','event_observed'])
    (OUT/'cox_coeffs.json').write_text(json.dumps(model.params_.to_dict()))
    (OUT/'cox_norm.json').write_text(json.dumps({c:dict(mean=float(x[c].mean()),std=float(x[c].std())) for c in x}))
    # Python itself supplies reference values, not a reimplementation of C++.
    x['expected_hazard']=model.predict_partial_hazard(data).to_numpy()
    assert np.isfinite(x.expected_hazard).all()
    x.to_csv(OUT/'reference.csv',index=False,float_format='%.17g',lineterminator='\n')
    command=[str(ROOT/'build-task2-validation/Release/cox_model_test.exe'),'--parity',str(OUT),str(OUT/'reference.csv')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (OUT/'native_test.log').write_text(result.stdout+result.stderr)
    (OUT/'result.json').write_text(json.dumps(dict(exit_code=result.returncode,rows=len(x),
        reference='Actual installed lifelines.predict_partial_hazard',warnings=[str(w.message) for w in caught],
        output=result.stdout+result.stderr),indent=2))
    print(result.stdout+result.stderr)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
