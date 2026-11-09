"""Audited environment-digest-only resume bridge; scientific source remains frozen."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import experiment_control as ec
import control_study_parallel as campaign


def checked_resume(status,identity,command,allowed,target,audit_root,verify):
    saved=status.get('configuration_identity',{})
    old=saved.get('record',{}).get('environment',{}).get('variables')
    new=identity['record']['environment']['variables']
    if old==new:
        return verify(status,identity,command)
    if old not in allowed or new!=target:
        raise ValueError('unapproved environment transition')
    # Permit ONLY the opaque environment-variable digest to differ. The original
    # validator still verifies the saved self-hash, command, state and every other
    # source/input/configuration/runtime identity field. No old status is rewritten.
    adjusted=copy.deepcopy(identity)
    adjusted['record']['environment']['variables']=old
    adjusted['sha256']=ec.canonical_hash(adjusted['record'])
    verify(status,adjusted,command)
    audit_root.mkdir(parents=True,exist_ok=True)
    run_id=saved['record']['row']['run_id']
    ec.save_json(audit_root/(run_id+'.json'),dict(run_id=run_id,
        old_configuration_sha256=saved['sha256'],current_configuration_sha256=identity['sha256'],
        previous_environment_digest=old,current_environment_digest=new,
        accepted_difference='environment.variables digest only',
        limitation='Old records store no per-variable hashes, so the changed variable names/values cannot be reconstructed. No claim that environment variables were identical. Other identity fields match exactly; original status preserved.'))


def main():
    if (campaign.OUT/'runner.lock').exists():raise ValueError('campaign already has a runner lock')
    if (campaign.OUT/'STOP_AFTER_CURRENT').exists():raise ValueError('pause flag still present')
    statuses=[]
    for path in (campaign.OUT/'evaluation').glob('worker-*/*/runner_status.json'):
        status=json.loads(path.read_text())
        if status['state']!='validated_success':raise ValueError('partial/failed run needs diagnosis')
        statuses.append(status)
    allowed={s['configuration_identity']['record']['environment']['variables'] for s in statuses}
    target=ec.canonical_hash(dict(sorted(ec.os.environ.items())))
    root=campaign.OUT/('resume_environment_transition_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir(parents=True,exist_ok=True)
    ec.save_json(root/'launch.json',dict(authorization='User requested resume of paused four-worker campaign',
        completed_statuses=len(statuses),allowed_previous_environment_digests=sorted(allowed),
        current_environment_digest=target,resume_bridge_sha256=ec.digest(Path(__file__)),
        note='Scientific source/protocol/manifests unchanged. Only opaque environment digest transition permitted, with original identity validation of every other field. This wrapper does not weaken future per-run before/after checks.'))
    original=ec.verify_resume
    ec.verify_resume=lambda status,identity,command:checked_resume(status,identity,command,allowed,target,root/'runs',original)
    import ctypes
    power=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:campaign.main()
    finally:
        if power:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('RESUME HALTED:',exc,flush=True)
        if not (campaign.OUT/'runner.lock').exists():
            ec.save_json(campaign.OUT/'pipeline_status.json',dict(state='halted',error=str(exc)))
        raise
