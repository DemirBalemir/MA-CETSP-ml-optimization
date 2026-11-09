"""Analyze the completed corrected campaign; never launches a solver."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import control_study_fixed as campaign
import experiment_control as ec
from analyze_control_study import analyze


def main():
    root = campaign.OUT
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit = root / ('analysis_execution_' + stamp)
    audit.mkdir()
    rows = list(csv.DictReader((root / 'evaluation_manifest.csv').open()))
    rates = json.loads((root / 'calibration_rates.json').read_text())
    count = 0
    started = time.monotonic()
    def inspect(row):
        nonlocal count
        result = campaign.study.inspect(row)
        count += 1
        if count % 25 == 0 or count == len(rows):
            message = f'ANALYSIS validated {count}/{len(rows)} ({time.monotonic()-started:.1f}s)'
            print(message, flush=True)
            (audit / 'progress.txt').write_text(message, encoding='utf-8')
        return result
    try:
        print('Analysis only; no solver launch', flush=True)
        analyze(root, rows, inspect, rates)
        # Preserve the prior terminal bookkeeping failure before marking complete.
        (audit / 'previous_pipeline_status.json').write_bytes((root / 'pipeline_status.json').read_bytes())
        ec.save_json(root / 'progress.json', dict(stage='evaluation', workers=4, completed=795,
            total=795, active={}, failures=[], state='complete'))
        ec.save_json(root / 'pipeline_status.json', dict(state='complete', calibration=159,
            evaluation=795, workers=4, analysis_audit=str(audit)))
        print('ANALYSIS COMPLETE', flush=True)
    except BaseException as exc:
        ec.save_json(audit / 'error.json', dict(error=str(exc), inspected=count))
        raise


if __name__ == '__main__':
    main()
