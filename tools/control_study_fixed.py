"""Fresh licensed/correct-Cox campaign. No old calibration or outcomes reused."""
import argparse
import csv
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

import experiment_control as ec
import control_study as study
import control_study_parallel as parallel

OUT=ec.ROOT/'solutions/control53_fixed_v1'
PROTOCOL=ec.ROOT/'CONTROL53_FIXED_PROTOCOL_FROZEN.json'
BINARY=OUT/'bin/MA-CETSP.exe'
FIELDS=(*parallel.FIELDS,'require_socp')


def manifest(stage,p,rates=None):
    rows=study.build_rows(stage,p,BINARY,rates,study_root=OUT,protocol_path=PROTOCOL)
    for row in rows:
        worker=(int(row['order'])-1)%4+1
        row.update(worker=str(worker),execution_regime='four_fixed_worker_streams',require_socp='1',
            output_dir=str(OUT/stage/f'worker-{worker}'/row['run_id']))
    path=OUT/(stage+'_manifest.csv')
    if path.exists():
        if list(csv.DictReader(path.open()))!=rows:raise ValueError('existing frozen manifest changed')
    else:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    return rows


def license_check(p):
    license_path=Path(p['license_path'])
    if ec.digest(license_path)!=p['license_sha256']:raise ValueError('license identity changed')
    folder=OUT/'license_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    lp=folder/'check.lp';lp.write_text('Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd\n')
    command=[str(Path('C:/gurobi1203/win64/bin/gurobi_cl.exe')),'ResultFile='+str(folder/'check.sol'),str(lp)]
    r=subprocess.run(command,cwd=folder,capture_output=True,timeout=60)
    text=(r.stdout+r.stderr).decode(errors='replace');(folder/'console.log').write_text(text)
    if r.returncode or 'Optimal objective' not in text or not (folder/'check.sol').is_file():
        raise RuntimeError('Gurobi license/optimization preflight failed')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    p=json.loads(PROTOCOL.read_text())
    if p['study_id']!='control53-fixed-v1' or not p['execution_authorized'] or set(p['instances'])!=ec.test_split():
        raise ValueError('invalid fixed campaign protocol')
    lock=OUT/'runner.lock'
    if lock.exists():raise ValueError('runner lock exists; do not duplicate')
    with lock.open('x') as f:f.write(str(ec.os.getpid()))
    try:
        study.write_once(OUT/'source_identity.json',ec.source_record())
        cal=manifest('calibration',p)
        if not args.execute:
            print('DRY RUN:159 fresh calibration +795 fresh evaluation, four workers, no old data reused');return
        license_check(p)
        ec.save_json(OUT/'pipeline_status.json',dict(state='running',stage='calibration',workers=4))
        if not parallel.run_parallel(cal,BINARY,Path(sys.executable),PROTOCOL,OUT,stage='calibration'):return
        rates=study.calibrate(cal,output_root=OUT)
        evaluation=manifest('evaluation',p,rates)
        ec.save_json(OUT/'pipeline_status.json',dict(state='running',stage='evaluation',workers=4))
        if parallel.run_parallel(evaluation,BINARY,Path(sys.executable),PROTOCOL,OUT):
            from analyze_control_study import analyze
            analyze(OUT,evaluation,study.inspect,rates)
            ec.save_json(OUT/'pipeline_status.json',dict(state='complete',calibration=159,evaluation=795,workers=4))
    finally:lock.unlink()


if __name__=='__main__':
    power=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    print('Idle-system-sleep prevention:',bool(power),flush=True)
    try:main()
    except Exception as exc:
        print('FIXED CAMPAIGN HALTED:',exc,flush=True)
        if not (OUT/'runner.lock').exists():ec.save_json(OUT/'pipeline_status.json',dict(state='halted',error=str(exc)))
        raise
    finally:
        if power:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
