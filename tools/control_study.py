"""Authorized full-test study. Dry-run by default; historical runners stay guarded."""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import experiment_control as ec

ROOT = ec.ROOT
STUDY = ROOT / 'solutions/control53_v2'
PROTOCOL = ROOT / 'CONTROL53_PROTOCOL_V2_FROZEN.json'
FIELDS = (*ec.FIELDS, 'stage', 'protocol_sha256', 'calibration_sha256', 'rate_basis')


def write_once(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != data:
            raise ValueError(f'frozen file differs; preserved: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding='utf-8')


def protocol():
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if not p['execution_authorized'] or len(p['instances']) != 53:
        raise ValueError('invalid authorized protocol')
    if set(p['instances']) != ec.test_split():
        raise ValueError('test membership changed')
    return p


def build_rows(stage, p, binary, rates=None, study_root=None, protocol_path=None):
    study_root=Path(study_root or STUDY);protocol_path=Path(protocol_path or PROTOCOL)
    rows = []
    mapping = ec.instance_map()
    binary_hash, source_hash = ec.digest(binary), ec.source_identity()
    rng = random.Random(p['order_seed'] + (stage == 'evaluation'))
    pairs = [(name, seed) for name in p['instances'] for seed in
             p['calibration_seeds' if stage == 'calibration' else 'evaluation_seeds']]
    rng.shuffle(pairs)
    # Finish the first small instance early to expose startup/inference problems.
    if stage == 'calibration':
        pairs.sort(key=lambda x: x[0] != 'bubbles1')
    for name, seed in pairs:
        methods = ['learned'] if stage == 'calibration' else ['none', 'learned', 'random']
        rng.shuffle(methods)
        for method in methods:
            run_id = f'{name}-s{seed}-{method}-a1'
            rate = rates[name] if rates else None
            row = dict(order=len(rows)+1, run_id=run_id, problem=name, index=mapping[name],
                method=method, search_seed=seed,
                prune_seed=917000+mapping[name]*100+seed if method == 'random' else '',
                iterations=5000, population=20, fit_beta=.96, dist_th=5, neighbor_size=50,
                train_frac_min=.2, train_frac_hard=.5, min_events=100,
                solver_time_limit=p['solver_time_limit'], outer_timeout=p['outer_timeout'],
                random_reject_rate=rate['probability'] if method == 'random' else '',
                random_start_iter=1000 if method == 'random' else '',
                schedule_path='', schedule_sha256='', binary_sha256=binary_hash,
                source_sha256=source_hash, output_dir=str(study_root/stage/run_id),
                stage=stage, protocol_sha256=ec.digest(protocol_path),
                calibration_sha256=ec.digest(study_root/'calibration_rates.json') if rates else '',
                rate_basis=rate['basis'] if method == 'random' else '')
            rows.append({k: str(v) for k,v in row.items()})
    return rows


def manifest(stage, rows):
    path = STUDY / f'{stage}_manifest.csv'
    if path.exists():
        with path.open(newline='', encoding='utf-8') as f:
            if list(csv.DictReader(f)) != rows:
                raise ValueError(f'manifest differs; preserved: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    return path


def inspect(row):
    d=Path(row['output_dir'])
    status=json.loads((d/'runner_status.json').read_text(encoding='utf-8'))
    if status['state'] != 'validated_success':
        raise ValueError(f"technical failure; preserve and diagnose: {row['run_id']}: {status.get('error',status['state'])}")
    v=ec.validate_run(row,d,d/'console.log')
    text=(d/'console.log').read_text(encoding='utf-8', errors='replace')
    time_stop='[STOP] time limit reached' in text
    natural='[STOP] no improvement' in text
    reason='time_limit' if time_stop else 'stagnation' if natural else 'iteration_budget'
    if not time_stop and not natural and v['generated'] != int(row['iterations']):
        raise ValueError('unexplained short run')
    trigger=re.search(r'Training triggered at iter (\d+)',text)
    v.update(stop_reason=reason,training_iteration=int(trigger[1]) if trigger else None,
             wall_seconds=status['wall_seconds'])
    with (d/'solutions/research/candidates.csv').open(newline='') as f:
        if any(c['outcome']=='filter_rejected' and int(c['iteration'])<=1000 for c in csv.DictReader(f)):
            raise ValueError('rejection before allowed prefix')
    if v['training_failed']:
        detail=(d/'model/training_log.txt').read_text(encoding='utf-8',errors='replace')
        # Only a recognizable statistical Cox fitting failure may be retained.
        # Import, I/O, process and unknown exceptions halt for diagnosis.
        if not ('ConvergenceError' in detail and 'lifelines' in detail):
            raise ValueError(f"unclassified training failure: {row['run_id']}")
        v['fit_failure_retained']=True
    else:
        v['fit_failure_retained']=False
    return v


def calibrate(rows,output_root=None):
    output_root=Path(output_root or STUDY)
    details=[]; rates={}
    for row in rows:
        v=inspect(row)
        details.append(dict(problem=row['problem'],seed=row['search_seed'],run_id=row['run_id'],
            exposure=max(v['generated']-1000,0),**v,
            status_sha256=ec.digest(Path(row['output_dir'])/'runner_status.json'),
            console_sha256=ec.digest(Path(row['output_dir'])/'console.log')))
    for name in sorted({r['problem'] for r in rows}):
        runs=[r for r in details if r['problem']==name]
        if len(runs)!=3: raise ValueError('three calibration runs required')
        exposure=sum(r['exposure'] for r in runs)
        rejects=sum(r['filter_rejected'] for r in runs)
        if exposure==0:
            if not all(r['stop_reason']=='stagnation' and not r['training_failed'] for r in runs):
                raise ValueError(f'zero exposure not valid natural stops: {name}')
            probability=0.;basis='calibration_pre_activation_stop'
        else:
            probability=rejects/exposure
            basis='estimated_zero' if rejects==0 else 'pooled_post1000_rate'
        rates[name]=dict(probability=probability,basis=basis,exposure=exposure,rejected=rejects,
                         any_fit_failure=any(r['training_failed'] for r in runs))
    write_once(output_root/'calibration_details.json',details)
    write_once(output_root/'calibration_rates.json',rates)
    return rates


def execute_stage(stage,rows,binary,python,execute,max_runs=0):
    manifest(stage,rows)
    if not execute:
        ec.run_rows(rows,binary,python,False,stage,0)
        return False
    # One row at a time permits scientific failure classification before advancing.
    for row in rows[:max_runs or len(rows)]:
        ec.run_rows([row],binary,python,True,stage,0,authorized_protocol=PROTOCOL)
        inspect(row)
        ec.refresh_results(STUDY/stage,rows)
        ec.save_json(STUDY/'progress.json',dict(stage=stage,last_validated_run=row['run_id'],
            order=int(row['order']),stage_total=len(rows),state='running'))
    return not max_runs or max_runs>=len(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--stage',choices=['calibration','evaluation','all'],default='all')
    ap.add_argument('--max-runs',type=int,default=0)
    ap.add_argument('--python-exe',type=Path,default=Path(sys.executable))
    args=ap.parse_args()
    p=protocol();binary=STUDY/'bin/MA-CETSP.exe'
    rows=build_rows('calibration',p,binary)
    manifest('calibration',rows)
    write_once(STUDY/'source_identity.json',ec.source_record())
    if args.stage in ('calibration','all'):
        complete=execute_stage('calibration',rows,binary,args.python_exe.resolve(),args.execute,args.max_runs)
        if not complete:return
    # Revalidate/rebind every calibration identity even on evaluation-only restart.
    if args.execute and args.stage=='evaluation':
        execute_stage('calibration',rows,binary,args.python_exe.resolve(),True)
    if args.execute:
        rates=calibrate(rows)
        if args.stage=='calibration':return
        evaluation=build_rows('evaluation',p,binary,rates)
        if execute_stage('evaluation',evaluation,binary,args.python_exe.resolve(),True,args.max_runs):
            from analyze_control_study import analyze
            analyze(STUDY,evaluation,inspect,rates)
    else:
        print('Evaluation: 795 rows generated only after validated calibration rates freeze.')


if __name__=='__main__':
    power_request = False
    try:
        if '--execute' in sys.argv and sys.platform == 'win32':
            import ctypes
            power_request = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            print('Idle-system-sleep prevention:', 'active' if power_request else 'unavailable', flush=True)
        main()
        if '--execute' in sys.argv:
            ec.save_json(STUDY/'pipeline_status.json',dict(state='finished_requested_scope'))
    except Exception as exc:
        print(f'STUDY HALTED: {exc}',file=sys.stderr,flush=True)
        if '--execute' in sys.argv:
            STUDY.mkdir(parents=True,exist_ok=True)
            ec.save_json(STUDY/'pipeline_status.json',dict(state='halted',error=str(exc)))
        raise SystemExit(2)
    finally:
        if power_request:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
