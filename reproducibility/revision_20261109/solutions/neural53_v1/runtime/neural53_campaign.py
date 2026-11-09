"""User-authorized 795-run neural/control study, detached and independent of Codex.

No API calls, no automated scientific tuning. Four local workers; completed raw
outputs are verified and losslessly archived to fit the available disk space.
"""
import argparse
import concurrent.futures
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

ROOT = Path(os.environ.get('CETSP_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT/'solutions/neural53_v1'
sys.path.insert(0, str(OUT/'runtime') if (OUT/'runtime/experiment_control.py').exists() else str(ROOT/'tools'))
import experiment_control as ec
ec.ROOT=ROOT
ec.WORK_DIR=ROOT/'build/Release'
PY=Path(r'C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe')
PROTOCOL=OUT/'PROTOCOL_FROZEN.json'
LOCK=threading.Lock()
STOP=threading.Event()
ACTIVE={}
COMPLETED={}
ERRORS=[]
FIRST_PAIRS=[('car_door_30',21),('rotatingDiamonds3',11),('bubbles3',21),
             ('kroD100_or30',31),('team2_200rdmRad',41),('concentricCircles3',41),
             ('team2_200',1),('d493rdmRad',1),('concentricCircles2',41),('concentricCircles3',21)]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)


def prepare():
    if PROTOCOL.exists():return json.loads(PROTOCOL.read_text())
    original=ROOT/'solutions/control53_fixed_v1'
    old=json.loads((original/'source_identity.json').read_text())['files']
    changed=[name for name,h in old.items() if name.startswith(('src/','include/')) and digest(ROOT/name)!=h]
    if changed:raise RuntimeError(f'Existing binary C++ source differs: {changed}')
    for name in ['bin','scripts','runtime','archives','preflight','evaluation']:(OUT/name).mkdir(exist_ok=True)
    for f in (original/'bin').iterdir():
        if f.is_file():shutil.copy2(f,OUT/'bin'/f.name)
    for f in (ROOT/'ml/scripts').glob('*.py'):shutil.copy2(f,OUT/'scripts'/f.name)
    shutil.copy2(ROOT/'tools/experiment_control.py',OUT/'runtime/experiment_control.py')
    shutil.copy2(Path(__file__),OUT/'runtime/neural53_campaign.py')
    source=ec.source_record();save(OUT/'source_identity.json',source)
    save(OUT/'python_identity.json',ec.python_identity(PY))
    membership=sorted(ec.test_split());mapping=ec.instance_map()
    files={p.relative_to(OUT).as_posix():digest(p) for folder in ['bin','scripts','runtime'] for p in (OUT/folder).rglob('*') if p.is_file()}
    p=dict(study_id='neural53-v1',execution_authorized=True,
           authorization='User explicitly requested detached overnight long runs and autonomous technical fixes on 8 November 2026; no further prompt required',
           instances=membership,evaluation_seeds=[1,11,21,31,41],models=['DEEPSURV','MTLR','NONE'],runs=795,
           workers=4,iterations=5000,population=20,patience=500,train_frac_min=.2,train_frac_hard=.5,min_events=100,
           solver_time_limit=36000,outer_timeout=39600,order_seed=20261108,
           first_pairs=FIRST_PAIRS,first_pairs_basis='Technical coverage of naturally active short runs in prior Cox records; chosen before any new outcomes',
           training='Validation fit 100 fixed epochs, final fit 200 fixed epochs; two 64-unit layers; dropout 0.1; Adam lr .01; no tuning',
           neural_seeds='20261108 + 10000*model_index(Deep=0,MTLR=1) + 100*instance_index + search_seed; shared independent search seeds',
           mtlr_score='negative_rmst_step_v1; negative integral of full discrete survival curve to fitted final cut; fixed before campaign',
           deepsurv_score='Percentile of log-risk against final training scores; existing calibration and rolling cap retained',
           socp='Preserve verified C++ binary and numerical-no-solution fallback; count each status. No coordinate-bound change',
           invalid='No silent exclusions, no retaining unknown training failures. Technical errors drain peers and halt; no automatic hyperparameter changes',
           resume='Completed archives revalidated; partial attempts preserved and rerun with same settings only after documented technical diagnosis',
           disk='Lossless per-run ZIP, every member SHA-256 verified before removing expanded solutions/lkh/console; old research files untouched; pause new work below 5 GiB',
           analysis='53 equally weighted means of 5 seeds; DeepSurv/none, MTLR/none, DeepSurv/MTLR for cost and solver time; six Wilcoxon tests with Holm; descriptive bootstrap intervals; no equivalence claim',
           environment='Four concurrent single-island solver processes, OMP/MKL and Torch 2 threads; local archival may contend; timings conditional on this regime',
           files=files,source_sha256=source['sha256'],python_identity_sha256=digest(OUT/'python_identity.json'),
           data_sha256={n:digest(ROOT/'datasets'/f'{n}.cetsp') for n in membership},
           lkh_sha256=digest(ROOT/'external/LKH-2.0.11/LKH.exe'),
           license_sha256=digest(Path(r'C:/Users/Demir/gurobi.lic')),created_utc=datetime.now(timezone.utc).isoformat())
    save(PROTOCOL,p)
    rng=random.Random(p['order_seed']);pairs=[(n,s) for n in membership for s in p['evaluation_seeds'] if (n,s) not in FIRST_PAIRS];rng.shuffle(pairs);pairs=FIRST_PAIRS+pairs
    rows=[]
    for name,seed in pairs:
        methods=['DEEPSURV','MTLR','NONE'];rng.shuffle(methods)
        for model in methods:
            run_id=f'{name}-s{seed}-{model.lower()}-a1';method='none' if model=='NONE' else 'learned'
            row=dict(order=len(rows)+1,run_id=run_id,problem=name,index=mapping[name],method=method,ml_model=model if model!='NONE' else 'COX',
                     search_seed=seed,prune_seed='',iterations=5000,population=20,fit_beta=.96,dist_th=5,neighbor_size=50,
                     train_frac_min=.2,train_frac_hard=.5,min_events=100,solver_time_limit=36000,outer_timeout=39600,
                     random_reject_rate='',random_start_iter='',schedule_path='',schedule_sha256='',
                     binary_sha256=digest(OUT/'bin/MA-CETSP.exe'),source_sha256=source['sha256'],
                     output_dir=str(OUT/'evaluation'/run_id),stage='evaluation',protocol_sha256=digest(PROTOCOL),
                     require_socp='1',scripts_dir=str(OUT/'scripts'),
                     neural_seed=20261108+(10000 if model=='MTLR' else 0)+100*mapping[name]+seed)
            rows.append({k:str(v) for k,v in row.items()})
    if len(rows)!=795:raise RuntimeError('Invalid manifest size')
    with (OUT/'evaluation_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return p


def configure(p):
    for name,h in p['files'].items():
        if digest(OUT/name)!=h:raise RuntimeError('Frozen runtime changed: '+name)
    source=json.loads((OUT/'source_identity.json').read_text())
    python=json.loads((OUT/'python_identity.json').read_text())
    if digest(PY)!=python['sha256']:raise RuntimeError('Python executable changed')
    ec.source_record=lambda root=None:source
    ec.python_identity=lambda executable:python
    ec.refresh_results=lambda root,rows:None  # Driver writes one synchronized table.
    if digest(ROOT/'external/LKH-2.0.11/LKH.exe')!=p['lkh_sha256']:raise RuntimeError('LKH changed')
    return p


def inspect(row):
    folder=Path(row['output_dir']);v=ec.validate_run(row,folder,folder/'console.log')
    text=(folder/'console.log').read_text(errors='replace')
    if re.search(r'Traceback|\[ML ERROR\]|\[MTLR ERROR\]|\[DEEPSURV ERROR\]',text):raise RuntimeError('ML error in raw console')
    if v['training_failed']:raise RuntimeError('Training failure must not be silently retained')
    preds=re.findall(r'\[ML-(?:DEEPSURV|MTLR)\] score=(\S+) threshold=(\S+)',text)
    if preds and not v['training_success']:raise RuntimeError('Predictions before successful training')
    if v['training_success'] and row['method']=='learned':
        name=row['ml_model'].lower();meta=json.loads((folder/'model'/f'{name}_meta.json').read_text());audit=meta['training_audit']
        if audit['epochs_completed']!=200 or audit['changed_parameter_tensors']==0 or audit['initial_state_sha256']==audit['final_state_sha256']:raise RuntimeError('Invalid learned weights')
        if any(x<=0 for x in audit['batchnorm_batches'].values()):raise RuntimeError('Untrained BatchNorm')
        if audit['torch_seed']!=int(row['neural_seed']) or audit['torch_threads']!=2:raise RuntimeError('Neural seed/thread mismatch')
        if row['ml_model']=='MTLR' and meta.get('score_version')!='negative_rmst_step_v1':raise RuntimeError('Wrong MTLR scoring contract')
        for score,threshold in preds:
            if not all(__import__('math').isfinite(float(x)) for x in [score,threshold]):raise RuntimeError('Nonfinite prediction')
            if not __import__('math').isclose(float(threshold),meta['threshold'],rel_tol=1e-5,abs_tol=1e-6):raise RuntimeError('Threshold mismatch')
    if '[STOP] no improvement' in text:reason='stagnation'
    elif '[STOP] time limit' in text:reason='time_limit'
    elif v['generated']==int(row['iterations']):reason='iteration_budget'
    else:raise RuntimeError('Unexplained short run')
    v.update(filter_activated=bool(preds),stop_reason=reason,ml_model=row['ml_model'] if row['method']=='learned' else 'NONE',search_seed=int(row['search_seed']),problem=row['problem'])
    return v


def archive(row):
    folder=Path(row['output_dir']).resolve()
    if not folder.is_relative_to(OUT.resolve()) or folder==OUT.resolve() or folder.is_symlink():raise RuntimeError('Unsafe archive source')
    final=OUT/'archives'/f"{row['run_id']}.zip";part=final.with_suffix('.zip.part')
    hashes={};size=0
    with zipfile.ZipFile(part,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_symlink():raise RuntimeError('Unexpected symlink in new outputs')
            if f.is_file():
                key=f.relative_to(folder).as_posix();hashes[key]=digest(f);size+=f.stat().st_size;z.write(f,key)
        z.writestr('archive_manifest.json',json.dumps(hashes,sort_keys=True))
    with zipfile.ZipFile(part) as z:
        for name,h in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise RuntimeError('Archive verification failed')
    os.replace(part,final)
    record=dict(path=str(final),sha256=digest(final),raw_bytes=size,compressed_bytes=final.stat().st_size,member_hashes_verified=len(hashes))
    save(folder/'archive.json',record)
    # Only remove verified expanded copies produced by this campaign.
    for base in [folder/'solutions',folder/'lkh']:
        if base.exists():
            for d,dirs,files in os.walk(base,topdown=False):
                for name in files:
                    f=Path(d)/name
                    if f.is_symlink() or not f.resolve().is_relative_to(folder):raise RuntimeError('Unsafe cleanup path')
                    f.unlink()
                for name in dirs:(Path(d)/name).rmdir()
            base.rmdir()
    (folder/'console_tail.txt').write_bytes((folder/'console.log').read_bytes()[-18000:])
    (folder/'console.log').unlink()
    return record


def progress(state='running'):
    with LOCK:
        data=dict(state=state,pid=os.getpid(),completed=len(COMPLETED),planned=795,active=dict(ACTIVE),errors=list(ERRORS),
                  first_30_validated=sum(int(v['order'])<=30 for v in COMPLETED.values()),
                  free_GiB=shutil.disk_usage(OUT).free/1024**3,updated_utc=datetime.now(timezone.utc).isoformat())
        save(OUT/'progress.json',data)
        cols=['order','run_id','problem','search_seed','ml_model','best','solver_seconds','generated','filter_rejected','admitted','training_success','stop_reason']
        with (OUT/'results.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(sorted(COMPLETED.values(),key=lambda x:int(x['order'])))


def run_one(row,stage='evaluation'):
    p=json.loads(PROTOCOL.read_text())
    for name,h in p['files'].items():
        if digest(OUT/name)!=h:raise RuntimeError('Frozen runtime changed: '+name)
    if digest(ROOT/'datasets'/f"{row['problem']}.cetsp")!=p['data_sha256'][row['problem']]:raise RuntimeError('Benchmark data changed')
    folder=Path(row['output_dir'])
    if folder.exists() and (folder/'archive.json').exists():
        ar=json.loads((folder/'archive.json').read_text());status=json.loads((folder/'runner_status.json').read_text())
        if digest(ar['path'])!=ar['sha256'] or status['state']!='validated_success':raise RuntimeError('Archived result identity mismatch')
        return json.loads((folder/'validated_metrics.json').read_text())
    if folder.exists():
        status_path=folder/'runner_status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('state')=='validated_success':
            metrics=inspect(row);metrics.update(order=row['order'],run_id=row['run_id'])
            save(folder/'validated_metrics.json',metrics);archive(row);return metrics
        raise RuntimeError('Partial attempt preserved; technical diagnosis required: '+str(folder))
    ec.run_rows([row],OUT/'bin/MA-CETSP.exe',PY,True,stage,0,authorized_protocol=PROTOCOL)
    status=json.loads((folder/'runner_status.json').read_text())
    if status['state']!='validated_success':raise RuntimeError(status.get('error',status['state']))
    metrics=inspect(row);metrics.update(order=row['order'],run_id=row['run_id'])
    status.update(metrics);save(folder/'runner_status.json',status);save(folder/'validated_metrics.json',metrics)
    archive(row)
    return metrics


def analyze():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon
    d=pd.read_csv(OUT/'results.csv')
    if len(d)!=795 or d.duplicated(['problem','search_seed','ml_model']).any():raise RuntimeError('Incomplete final outcomes')
    m=d.groupby(['problem','ml_model'])[['best','solver_seconds']].mean();rng=np.random.default_rng(20261108);rows=[]
    for a,b in [('DEEPSURV','NONE'),('MTLR','NONE'),('DEEPSURV','MTLR')]:
        for metric in ['best','solver_seconds']:
            x=100*(m.xs(a,level=1)[metric]/m.xs(b,level=1)[metric]-1)
            boot=x.to_numpy()[rng.integers(0,53,size=(10000,53))].mean(1)
            pv=1.0 if np.all(x.to_numpy()==0) else float(wilcoxon(x).pvalue)
            rows.append(dict(treatment=a,reference=b,metric=metric,mean_pct=x.mean(),median_pct=x.median(),lo95=np.quantile(boot,.025),hi95=np.quantile(boot,.975),p=pv))
    previous=0.
    for rank,index in enumerate(np.argsort([r['p'] for r in rows])):
        previous=max(previous,min(1.,(6-rank)*rows[index]['p']));rows[index]['holm_p']=previous
    pd.DataFrame(rows).to_csv(OUT/'comparisons.csv',index=False)


def preflight(rows):
    if (OUT/'preflight_complete.json').exists():return
    cli=Path('C:/gurobi1203/win64/bin/gurobi_cl.exe');lp=OUT/'preflight/license.lp';lp.write_text('Minimize\n obj: x\nSubject To\n c: x >= 1\nBounds\n x >= 0\nEnd\n')
    check=subprocess.run([str(cli),str(lp)],capture_output=True,text=True,timeout=60)
    (OUT/'preflight/license.log').write_text(check.stdout+check.stderr)
    if check.returncode or 'Optimal objective' not in check.stdout:raise RuntimeError('Gurobi license preflight failed')
    for model in ['DEEPSURV','MTLR']:
        row=dict(next(x for x in rows if x['ml_model']==model));row.update(run_id='preflight-'+model.lower(),problem='bubbles1',index=str(ec.instance_map()['bubbles1']),
            iterations='200',min_events='10',output_dir=str(OUT/'preflight'/('preflight-'+model.lower())),stage='pilot',neural_seed=str(20261108+(10000 if model=='MTLR' else 0)+101))
        result=run_one(row,'pilot')
        if not result['training_success'] or not result['filter_activated']:raise RuntimeError('Preflight did not exercise training and inference')
    save(OUT/'preflight_complete.json',dict(state='passed',models=['DEEPSURV','MTLR']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    p=prepare();configure(p)
    if args.prepare_only:print('Prepared 795 runs and frozen runtime');return
    log=(OUT/'terminal.log').open('a',encoding='utf-8',buffering=1);sys.stdout=sys.stderr=log
    lock=OUT/'runner.lock'
    if lock.exists():raise RuntimeError('Runner lock exists; duplicate launch refused')
    lock.write_text(str(os.getpid()))
    power=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    in_job=ctypes.c_int();ctypes.windll.kernel32.IsProcessInJob(ctypes.windll.kernel32.GetCurrentProcess(),None,ctypes.byref(in_job))
    save(OUT/'launch.json',dict(pid=os.getpid(),in_windows_job=bool(in_job.value),sleep_prevention=bool(power),api_calls=False,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        rows=list(csv.DictReader((OUT/'evaluation_manifest.csv').open()))
        save(OUT/'progress.json',dict(state='preflight',pid=os.getpid(),completed=0,planned=795))
        preflight(rows)
        jobs=queue.Queue()
        for row in rows:jobs.put(row)
        def worker(number):
            while not STOP.is_set():
                if (OUT/'STOP_AFTER_CURRENT').exists():return
                if shutil.disk_usage(OUT).free<5*1024**3:
                    progress('waiting_for_disk_space');time.sleep(30);continue
                try:row=jobs.get_nowait()
                except queue.Empty:return
                with LOCK:ACTIVE[str(number)]=row['run_id']
                progress()
                try:
                    result=run_one(row)
                    with LOCK:COMPLETED[row['run_id']]=result
                except Exception as e:
                    traceback.print_exc()
                    with LOCK:ERRORS.append(dict(run_id=row['run_id'],error=str(e)))
                    STOP.set()
                finally:
                    with LOCK:ACTIVE.pop(str(number),None)
                    jobs.task_done();progress('draining_after_error' if STOP.is_set() else 'running')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fs=[pool.submit(worker,n) for n in range(1,5)]
            for f in fs:f.result()
        if not ERRORS and len(COMPLETED)==795:analyze()
        progress('halted' if ERRORS else 'complete' if len(COMPLETED)==795 else 'paused')
    except BaseException as error:
        traceback.print_exc();save(OUT/'progress.json',dict(state='halted',pid=os.getpid(),error=str(error),completed=len(COMPLETED),planned=795));raise
    finally:
        if power:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True);log.flush()


if __name__=='__main__':main()
