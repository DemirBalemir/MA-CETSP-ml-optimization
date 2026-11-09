import csv
import copy
import json
import shutil
from unittest.mock import patch
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import experiment_control as ec
from experiment_control import ROOT, validate_run, run_rows


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build-task2-validation")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p = self.root / "solutions" / "research"
        p.mkdir(parents=True)
        (p / "candidates.csv").write_text(
            "candidate_id,iteration,outcome\n1,1,admitted\n2,2,filter_rejected\n")
        (p / "lifetimes.csv").write_text(
            "candidate_id,birth_iter,death_iter,survival_iters,censored\n1,1,2,1,1\n")
        self.log = self.root / "console.log"
        self.log.write_text("[SUMMARY] instance=x model=RANDOM best=12 effective_iter=2 total_t=1 ml_rejects=1\n"
                            "[CANDIDATES] generated=2 filter_rejected=1 vnd_evaluated=1 admitted=1 not_admitted=0\n")
        self.row = {"iterations": "2", "method": "random"}

    def test_valid(self):
        result = validate_run(self.row, self.root, self.log)
        self.assertEqual(result["admitted_per_generated"], .5)
        self.assertEqual(result["admitted_per_vnd"], 1)

    def test_missing_lifetime(self):
        (self.root / "solutions/research/lifetimes.csv").write_text(
            "candidate_id,birth_iter,death_iter,survival_iters,censored\n")
        with self.assertRaisesRegex(ValueError, "lifetimes"):
            validate_run(self.row, self.root, self.log)

    def test_duplicate_candidate(self):
        with (self.root / "solutions/research/candidates.csv").open("a") as f:
            f.write("1,1,admitted\n")
        with self.assertRaisesRegex(ValueError, "candidate rows/IDs"):
            validate_run(self.row, self.root, self.log)

    def test_exit_zero_without_summary_is_not_success(self):
        self.log.write_text("solver exited normally\n")
        with self.assertRaisesRegex(ValueError, "summary"):
            validate_run(self.row, self.root, self.log)

    def test_wrong_model(self):
        self.log.write_text(self.log.read_text().replace("model=RANDOM", "model=COX"))
        with self.assertRaisesRegex(ValueError, "wrong model"):
            validate_run(self.row, self.root, self.log)

    def test_invalid_schedule_status_before_launch(self):
        output=self.root/"bad-schedule-run"
        row=dict(order="1",run_id="bad-schedule-run",output_dir=str(output),
            index="1",search_seed="11",iterations="2",population="4",fit_beta="0.96",
            dist_th="5",neighbor_size="50",solver_time_limit="2",train_frac_min=".2",
            train_frac_hard=".5",min_events="100",method="random",prune_seed="7",
            schedule_path=str(self.root/"missing.csv"),schedule_sha256="bad",
            random_reject_rate="",random_start_iter="",binary_sha256="fixture",problem="fixture")
        run_rows([row],self.root/"never-launched.exe",self.root/"python.exe",True,"pilot",0)
        self.assertIn('"state": "invalid_schedule"',(output/"runner_status.json").read_text())
        self.assertFalse((output/"console.log").exists())


class ReviewFixTest(unittest.TestCase):
    setUp = ValidatorTest.setUp
    def cox_fixture(self):
        self.row["method"] = "learned"
        self.log.write_text(self.log.read_text().replace("model=RANDOM", "model=COX") +
                            "[ML] Training triggered\n[ML] Training completed successfully.\n[ML-COX] score=1.2 threshold=1.5\n")
        self.model = self.root/"model"
        self.model.mkdir()
        self.coeff = {"pre_vnd_cost": 0.1}
        self.norm = {"pre_vnd_cost": {"mean":2., "std":1.}}
        self.meta = dict(threshold=1.5, rejection_percentile=80, objective=.1, concordance_index=.6, n_samples=100)
        self.write_models()

    def write_models(self):
        for name, obj in (("coeffs",self.coeff),("norm",self.norm),("meta",self.meta)):
            (self.model/f"cox_{name}.json").write_text(json.dumps(obj))

    def check(self):
        return validate_run(self.row,self.root,self.log)

    def test_cox_numeric_records(self):
        self.cox_fixture()
        original=self.log.read_text()
        for token in ("nan", "NaN", "inf", "-inf", "1.2garbage", ""):
            for field, old in (("score","1.2"),("threshold","1.5")):
                with self.subTest(field=field,token=token):
                    self.log.write_text(original.replace(f"{field}={old}",f"{field}={token}"))
                    with self.assertRaises(ValueError): self.check()
        self.log.write_text(original)
        result=self.check()
        self.assertTrue(result["filter_activated"])
        self.assertFalse(result["threshold_fallback"])
        self.log.write_text(original.replace("[ML-COX] score=1.2 threshold=1.5\n", ""))
        self.assertTrue(self.check()["training_success"])
        self.assertFalse(self.check()["filter_activated"])

    def test_validator_accepts_reordered_normalization(self):
        self.cox_fixture()
        self.norm={"pre_vnd_cost":{"std":1.,"mean":2.}}
        self.write_models()
        self.assertTrue(self.check()["filter_activated"])

    def test_models_and_normalization(self):
        self.cox_fixture()
        for obj,key,value in ((self.coeff,"pre_vnd_cost",float('nan')),
                              (self.norm,"pre_vnd_cost",{}),
                              (self.meta,"threshold",float('inf')),
                              (self.meta,"n_samples",True)):
            old=copy.deepcopy(obj[key]);obj[key]=value;self.write_models()
            with self.assertRaises(ValueError):self.check()
            obj[key]=old
        for value in (0,-1,1e-13,float('nan'),float('inf'),"1"):
            self.norm["pre_vnd_cost"]["std"]=value;self.write_models()
            with self.assertRaises(ValueError):self.check()
        self.norm["pre_vnd_cost"]["std"]=1
        self.norm["unknown_feature"]={"mean":0,"std":1};self.write_models()
        with self.assertRaises(ValueError):self.check()
        del self.norm["unknown_feature"];self.write_models()
        for content in ('{}','[]','{"pre_vnd_cost":1,"pre_vnd_cost":2}', 'broken'):
            (self.model/"cox_coeffs.json").write_text(content)
            with self.assertRaises(ValueError):self.check()
        (self.model/"cox_coeffs.json").unlink()
        with self.assertRaises(ValueError):self.check()

    def test_fallback_is_diagnostic_only(self):
        self.cox_fixture()
        self.meta.update(n_samples=20,objective=float('nan'))
        (self.model/"training_log.txt").write_text("ConvergenceWarning\n[WARN] Too few samples for hold-out split; using default percentile\n")
        self.write_models()
        result=self.check()
        self.assertTrue(result["threshold_fallback"])
        self.assertTrue(result["convergence_warning"])
        for key,value in (("n_samples",100),("rejection_percentile",85),("objective",float('inf')),("threshold",float('nan'))):
            old=self.meta[key]; self.meta[key]=value;self.write_models()
            with self.assertRaises(ValueError):self.check()
            self.meta[key]=old
        self.coeff["pre_vnd_cost"]=float('nan');self.write_models()
        with self.assertRaises(ValueError):self.check()
        self.coeff["pre_vnd_cost"]=.1;self.write_models()
        (self.model/"training_log.txt").write_text("")
        with self.assertRaises(ValueError):self.check()

    def test_positive_contiguous_candidates(self):
        path=self.root/"solutions/research/candidates.csv"
        for token in ("0","-1","3", ""):
            path.write_text(f"candidate_id,iteration,outcome\n{token},{token},admitted\n2,2,filter_rejected\n")
            with self.assertRaises(ValueError):self.check()

    def test_zero_lifetime_and_terminal_censoring(self):
        path=self.root/"solutions/research/lifetimes.csv"
        header="candidate_id,birth_iter,death_iter,survival_iters,censored\n"
        path.write_text(header+"1,1,1,0,0\n")
        self.assertEqual(self.check()["censored"],0)
        for record in ("1,1,1,0,1", "1,1,3,2,0", "1,1,2,0,0", "0,0,2,2,1"):
            path.write_text(header+record+"\n")
            with self.assertRaises(ValueError):self.check()
        path.write_text(header+"1,1,2,1,1\n")
        self.assertEqual(self.check()["censored"],1)

    def test_historical_status_not_upgraded(self):
        (self.root/"runner_status.json").write_text('{"state":"validated_success"}')
        before=(self.root/"runner_status.json").read_bytes()
        with self.assertRaisesRegex(ValueError,"legacy"):
            run_rows([dict(output_dir=str(self.root))],Path('no.exe'),Path('no.py'),True,'pilot',0)
        self.assertEqual(before,(self.root/"runner_status.json").read_bytes())

    def test_configuration_and_resume_changes(self):
        with (ROOT/"pilot_manifest_task3.csv").open() as manifest:
            row=next(csv.DictReader(manifest))
        binary=self.root/"solver.exe";binary.write_bytes(b"fixture")
        row.update(binary_sha256=ec.digest(binary),source_sha256=ec.source_identity())
        py={"path":"python","sha256":"p1","version":"3.10","packages":[["numpy","1"]]}
        with patch.object(ec,"python_identity",return_value=py):
            identity=ec.configuration_identity(row,binary,Path('python'),['command'])
            status=dict(state='validated_success',command=['command'],configuration_identity=identity)
            ec.verify_resume(status,identity,['command'])
            for key,value in (("search_seed","21"),("prune_seed","22"),("population","30"),("train_frac_min",".3"),("solver_time_limit","31"),("outer_timeout","91"),("problem","team1_100")):
                changed=dict(row);changed[key]=value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    new=ec.configuration_identity(changed,binary,Path('python'),['command'])
                    ec.verify_resume(status,new,['command'])
            for key,value in (("packages",[["numpy","2"]]),("path","other-python"),("sha256","changed-python"),("version","3.11")):
                changed_py=copy.deepcopy(py);changed_py[key]=value
                with patch.object(ec,"python_identity",return_value=changed_py):
                    with self.assertRaises(ValueError):
                        ec.verify_resume(status,ec.configuration_identity(row,binary,Path('python'),['command']),['command'])
            changed=dict(row,problem="team1_100",index="21")
            with self.assertRaises(ValueError):
                ec.verify_resume(status,ec.configuration_identity(changed,binary,Path('python'),['command']),['command'])
            with patch.object(ec,"source_record",return_value={"sha256":"changed-source"}):
                with self.assertRaisesRegex(ValueError,"binary/source"):
                    ec.configuration_identity(row,binary,Path('python'),['command'])
            schedule=self.root/'schedule.csv';schedule.write_text('first')
            scheduled=dict(row,schedule_path=str(schedule),schedule_sha256=ec.digest(schedule))
            first=ec.configuration_identity(scheduled,binary,Path('python'),['command'])
            schedule.write_text('second')
            with self.assertRaises(ValueError):ec.configuration_identity(scheduled,binary,Path('python'),['command'])
            scheduled['schedule_sha256']=ec.digest(schedule)
            self.assertNotEqual(first,ec.configuration_identity(scheduled,binary,Path('python'),['command']))
            binary.write_bytes(b'changed')
            with self.assertRaises(ValueError):ec.configuration_identity(row,binary,Path('python'),['command'])

    def test_source_coverage_deterministic(self):
        record=ec.source_record()
        self.assertEqual(record,ec.source_record())
        for name in ('ml/scripts/threshold_utils.py','ml/scripts/features.py','ml/scripts/loader.py','src/ML/SurvivalModel.cpp','src/LocalSearch/LKH.cpp'):
            self.assertIn(name,record['files'])
        fixture=self.root/'source';fixture.mkdir()
        for name in record['files']:
            target=fixture/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'fixture')
        old=ec.source_record(fixture)
        for name in ('ml/scripts/threshold_utils.py','src/ML/SurvivalModel.cpp','src/LocalSearch/LKH.cpp'):
            path=fixture/name;path.write_bytes(b'changed')
            self.assertNotEqual(old['sha256'],ec.source_record(fixture)['sha256'])
            path.write_bytes(b'fixture')
        (fixture/'build').mkdir();(fixture/'build/output.cpp').write_text('excluded')
        self.assertEqual(old,ec.source_record(fixture))

    def test_startup_comparison(self):
        with (ROOT/"pilot_manifest_task3.csv").open() as f:
            row=next(csv.DictReader(f))
        text=(Path(row['output_dir'])/'console.log').read_text()
        ec.validate_startup(row,text,strict=False)
        for key,value in (("search_seed","21"),("population","30"),("index","21"),("train_frac_min",".3")):
            changed=dict(row);changed[key]=value
            with self.assertRaises(ValueError):ec.validate_startup(changed,text,strict=False)
        with self.assertRaises(ValueError):ec.validate_startup(row,text,strict=True,python_exe=Path(sys.executable))
        runtime='[RUNTIME] python='+json.dumps(sys.executable)+' scripts='+json.dumps(str(ROOT/'ml/scripts')+'/')+' lkh='+json.dumps(str(ROOT/'external/LKH-2.0.11/LKH.exe'))+'\n'
        text=runtime+text.replace(' model_dir=', ' fit_beta=0.96 dist_th=5 neighbor_size=50 max_time=30 init=KMEANS select=RANDOM crossover=KSX improvement=BEST greed=SQUEEZE distance=EDIT random=0 islands=1 model_dir=')
        ec.validate_startup(row,text,strict=True,python_exe=Path(sys.executable))
        with self.assertRaises(ValueError):ec.validate_startup(row,text.replace('max_time=30','max_time=31'),strict=True,python_exe=Path(sys.executable))

    def test_python_probe_records_content_and_dependencies(self):
        identity=ec.python_identity(Path(sys.executable))
        self.assertEqual(identity['sha256'],ec.digest(sys.executable))
        self.assertTrue(identity['version'])
        self.assertTrue(identity['packages'])
        self.assertEqual(len(identity['packages'][0]),3)

    def test_main_guard(self):
        with self.assertRaisesRegex(ValueError,'not executable'):
            run_rows([],Path('no.exe'),Path('no.py'),True,'draft',0)


class RunnerFlowTest(unittest.TestCase):
    """Exercise real identity, logging, validators and resume; replace only solver process."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build-task2-validation',prefix='runner flow ')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.binary=self.root/'fixture-solver.exe'
        self.binary.write_bytes(b'not an executable; solver launch must be intercepted')
        with (ROOT/'pilot_manifest_task3.csv').open() as f:
            self.row=dict(next(csv.DictReader(f)))
        self.row.update(method='none',run_id='flow-none',output_dir=str(self.root/'runs'/'flow-none'),
                        iterations='2',prune_seed='',random_reject_rate='',random_start_iter='',
                        binary_sha256=ec.digest(self.binary),source_sha256=ec.source_identity())
        self.calls=[]
        self.bad_startup=False
        self.real_popen=ec.subprocess.Popen

    def process(self,command,**kwargs):
        if Path(command[0]) != self.binary:
            # Real scientific Python identity probes still run; nothing else may start.
            self.assertEqual(Path(command[0]).resolve(),Path(sys.executable).resolve())
            self.assertEqual(command[1],'-c')
            return self.real_popen(command,**kwargs)
        self.calls.append(command)
        self.assertEqual(kwargs['cwd'],ec.WORK_DIR)
        options=dict(zip(command[1::2],command[2::2]))
        run=Path(options['--output_root'])/options['--run_id']
        self.assertFalse(run.exists())
        research=run/'solutions/research';research.mkdir(parents=True)
        (run/'model').mkdir();(run/'lkh').mkdir()
        (research/'candidates.csv').write_text('candidate_id,iteration,outcome\n1,1,admitted\n2,2,not_admitted\n')
        (research/'lifetimes.csv').write_text('candidate_id,birth_iter,death_iter,survival_iters,censored\n1,1,2,1,1\n')
        fields={'instance':options['-i'],'seed':options['-s'],'pop':options['-p'],'iter':options['-r'],
                'mode':options['--filter_mode'],'ml':'OFF','prune_seed':'NA','random_start_iter':'NA',
                'random_rate':'NA','schedule':'NA','schedule_fnv1a':'NA',
                'train_min':options['--train_frac_min'],'train_hard':options['--train_frac_hard'],
                'min_events':options['--min_events'],'fit_beta':options['-b'],'dist_th':options['-d'],
                'neighbor_size':options['-n'],'max_time':'999' if self.bad_startup else options['-t']}
        for key in ('init','select','crossover','improvement','greed','distance','islands'):
            fields[key]=options['--'+key]
        fields.update(random=options['-g'],model_dir=str(run/'model')+'/',run_dir=str(run),
                      results_dir=str(run/'solutions')+'/',lkh_tmp=str(run/'lkh')+'/',timestamp='12345')
        runtime='[RUNTIME] python='+json.dumps(options['--python_exe'])+' scripts='+json.dumps(options['--scripts_dir']+'/')+' lkh='+json.dumps(options['--lkh_exe'])
        output=(runtime+'\n[Island 0] '+' '.join(k+'='+v for k,v in fields.items())+'\n'
                '[SUMMARY] instance=bubbles1 model=BASELINE best=12 effective_iter=2 total_t=1 ml_rejects=0\n'
                '[CANDIDATES] generated=2 filter_rejected=0 vnd_evaluated=2 admitted=1 not_admitted=1\n').encode()
        class FakeProcess:
            returncode=0
            def communicate(self,timeout): return output,None
        return FakeProcess()

    def invoke(self,row=None):
        with patch.object(ec.subprocess,'Popen',side_effect=self.process):
            run_rows([row or self.row],self.binary,Path(sys.executable),True,'pilot',0)

    def snapshot(self):
        return {str(p.relative_to(self.root/'runs')):p.read_bytes() for p in (self.root/'runs').rglob('*') if p.is_file()}

    def test_new_record_resume_and_rejections(self):
        self.invoke()
        run=Path(self.row['output_dir'])
        status=json.loads((run/'runner_status.json').read_text())
        self.assertEqual(status['state'],'validated_success')
        identity=status['configuration_identity']
        self.assertEqual(identity['sha256'],ec.canonical_hash(identity['record']))
        self.assertEqual(identity['record']['command'],self.calls[0])
        self.assertEqual(identity['record']['row'],self.row)
        self.assertEqual(identity['record']['source'],ec.source_record())
        self.assertIn('packages',identity['record']['python'])
        self.assertEqual(status['command'],self.calls[0])
        before=self.snapshot()
        self.invoke()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.snapshot(),before)  # status, results.csv, console and all outputs
        changed=dict(self.row,search_seed='21')
        with self.assertRaisesRegex(ValueError,'configuration identity mismatch'):
            self.invoke(changed)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.snapshot(),before)
        (run/'solutions/research/lifetimes.csv').write_text('candidate_id,birth_iter,death_iter,survival_iters,censored\n')
        corrupted=self.snapshot()
        with self.assertRaisesRegex(ValueError,'lifetimes'): self.invoke()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.snapshot(),corrupted)

    def test_actual_command_startup_mismatch_is_not_success(self):
        self.bad_startup=True
        self.invoke()
        status=json.loads((Path(self.row['output_dir'])/'runner_status.json').read_text())
        self.assertEqual(status['state'],'invalid_output')
        self.assertIn('startup mismatch: max_time',status['error'])
        before=self.snapshot()
        with self.assertRaisesRegex(ValueError,'partial run'): self.invoke()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.snapshot(),before)


if __name__ == "__main__":
    unittest.main()
