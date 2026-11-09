"""Four fixed worker streams; reuse frozen independent calibration, restart evaluation."""
import concurrent.futures
import csv
import ctypes
import json
import os
from pathlib import Path
import queue
import sys
import threading

import experiment_control as ec
import control_study as serial

BASE=ec.ROOT/'solutions/control53_v2'
OUT=ec.ROOT/'solutions/control53_parallel4_v1'
PROTOCOL=ec.ROOT/'CONTROL53_PARALLEL4_PROTOCOL_FROZEN.json'
PY=Path(sys.executable).resolve()
FIELDS=(*serial.FIELDS,'worker','execution_regime')


def verify_calibration():
    """Historical checks; never rewrite old statuses or claim current-source resume."""
    rows=list(csv.DictReader((BASE/'calibration_manifest.csv').open()))
    if len(rows)!=159:raise ValueError('159 calibration records required')
    current=ec.source_record()['files'];pidentity=ec.python_identity(PY)
    audit=[]
    for row in rows:
        d=Path(row['output_dir']);status=json.loads((d/'runner_status.json').read_text())
        identity=status['configuration_identity'];record=identity['record']
        if identity['sha256']!=ec.canonical_hash(record):raise ValueError('invalid historical identity')
        if status['state']!='validated_success':raise ValueError('invalid calibration run')
        for name,h in record['source']['files'].items():
            if name.startswith(('src/','include/','ml/scripts/')) or name=='CMakeLists.txt':
                if current.get(name)!=h:raise ValueError(f'solver/trainer changed: {name}')
        for key in ('data','lkh','binary'):
            if ec.digest(record[key]['path'])!=record[key]['sha256']:raise ValueError(f'changed calibration {key}')
        if record['python']!=pidentity:raise ValueError('Python changed since calibration')
        v=serial.inspect(row)
        ec.validate_startup(row,(d/'console.log').read_text(),strict=True,python_exe=PY)
        audit.append(dict(run_id=row['run_id'],status_sha256=ec.digest(d/'runner_status.json'),
                          console_sha256=ec.digest(d/'console.log'),generated=v['generated'],
                          rejected=v['filter_rejected']))
    details=json.loads((BASE/'calibration_details.json').read_text())
    for r in details:
        d=BASE/'calibration'/r['run_id']
        if ec.digest(d/'runner_status.json')!=r['status_sha256'] or ec.digest(d/'console.log')!=r['console_sha256']:
            raise ValueError('calibration changed after rates frozen')
    rates=json.loads((BASE/'calibration_rates.json').read_text())
    if set(rates)!=ec.test_split():raise ValueError('calibration membership mismatch')
    for name,rate in rates.items():
        runs=[r for r in details if r['problem']==name]
        exposure=sum(r['exposure'] for r in runs);rejected=sum(r['filter_rejected'] for r in runs)
        expected=rejected/exposure if exposure else 0.
        if len(runs)!=3 or expected!=rate['probability']:raise ValueError('frozen rate mismatch')
    return audit


def prepare():
    p=json.loads(PROTOCOL.read_text())
    if p.get('study_id')!='control53-parallel4-v1' or p.get('workers')!=4 or not p.get('execution_authorized'):
        raise ValueError('invalid four-worker authorization')
    if ec.digest(BASE/'calibration_rates.json')!=p['calibration_rates_sha256']:
        raise ValueError('calibration rate identity mismatch')
    if ec.digest(OUT/'bin/MA-CETSP.exe')!=ec.digest(BASE/'bin/MA-CETSP.exe'):
        raise ValueError('solver binary changed')
    audit=verify_calibration()
    serial.write_once(OUT/'calibration_reuse_audit.json',audit)
    serial.write_once(OUT/'source_identity.json',ec.source_record())
    old=list(csv.DictReader((BASE/'evaluation_manifest.csv').open()))
    if len(old)!=795 or len({(r['problem'],r['search_seed'],r['method']) for r in old})!=795:
        raise ValueError('full unique evaluation required')
    source=ec.source_identity();binary=ec.digest(OUT/'bin/MA-CETSP.exe')
    rows=[]
    for row in old:
        r=dict(row);worker=(int(r['order'])-1)%4+1
        r.update(output_dir=str(OUT/'evaluation'/f'worker-{worker}'/r['run_id']),
            worker=str(worker),execution_regime='four_fixed_worker_streams',source_sha256=source,
            binary_sha256=binary,protocol_sha256=ec.digest(PROTOCOL),
            calibration_sha256=ec.digest(OUT/'calibration_rates.json'))
        rows.append(r)
    path=OUT/'evaluation_manifest.csv'
    if path.exists():
        if list(csv.DictReader(path.open()))!=rows:raise ValueError('existing manifest differs')
    else:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    return rows


def run_parallel(rows,binary,python,protocol,root,runner=None,inspector=None,stage='evaluation'):
    runner=runner or ec.run_rows;inspector=inspector or serial.inspect
    events=queue.Queue();stop=threading.Event();active={};completed=set();failures=[]
    def worker(number):
        for row in (r for r in rows if int(r['worker'])==number):
            if stop.is_set() or (root/'STOP_AFTER_CURRENT').exists():break
            events.put(('start',number,row,None))
            try:
                runner([row],binary,python,True,stage,0,authorized_protocol=protocol)
                inspector(row)
                events.put(('done',number,row,None))
            except Exception as exc:
                stop.set();events.put(('error',number,row,str(exc)));break
    def save(event):
        kind,n,row,error=event
        if kind=='start':active[str(n)]=row['run_id']
        else:
            active.pop(str(n),None)
            if kind=='done':completed.add(row['run_id'])
            else:failures.append(dict(worker=n,run_id=row['run_id'],error=error))
            ec.refresh_results(root/stage,rows)
        ec.save_json(root/'progress.json',dict(stage=stage,workers=4,completed=len(completed),
            total=len(rows),active=active,failures=failures,state='draining_after_error' if failures else 'running'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,n) for n in range(1,5)]
        while not all(f.done() for f in futures) or not events.empty():
            try:save(events.get(timeout=.5))
            except queue.Empty:pass
        for f in futures:f.result()
    if failures:raise RuntimeError(f'failed rows preserved; peers finished current runs: {failures}')
    if len(completed)!=len(rows):
        ec.save_json(root/'pipeline_status.json',dict(state='paused_at_run_boundaries',completed=len(completed),total=len(rows)))
        return False
    return True


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'runner.lock').exists():raise ValueError('runner lock exists; do not launch duplicate')
    # Hold the exclusive campaign lock during preparation and execution.
    lock=(OUT/'runner.lock').open('x');lock.write(str(os.getpid()));lock.close()
    try:
        rows=prepare()
        if not args.execute:
            print('DRY RUN:795 evaluation runs, four fixed streams;159 existing calibration runs reused');return
        ec.save_json(OUT/'pipeline_status.json',dict(state='running',stage='evaluation',workers=4))
        if run_parallel(rows,OUT/'bin/MA-CETSP.exe',PY,PROTOCOL,OUT):
            from analyze_control_study import analyze
            rates=json.loads((OUT/'calibration_rates.json').read_text())
            analyze(OUT,rows,serial.inspect,rates)
            ec.save_json(OUT/'pipeline_status.json',dict(state='complete',evaluation=795,workers=4))
    finally:
        (OUT/'runner.lock').unlink()


if __name__=='__main__':
    power=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    print('Idle-system-sleep prevention:',bool(power),flush=True)
    try:main()
    except Exception as exc:
        print('PARALLEL STUDY HALTED:',exc,flush=True)
        # A duplicate invocation must not overwrite the live owner's status.
        if not (OUT/'runner.lock').exists():
            ec.save_json(OUT/'pipeline_status.json',dict(state='halted',error=str(exc)))
        raise
    finally:
        if power:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
