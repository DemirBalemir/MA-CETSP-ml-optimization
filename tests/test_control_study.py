import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import control_study as study
import experiment_control as ec
from analyze_control_study import holm


class StudyTest(unittest.TestCase):
    def test_full_membership_and_zero_random(self):
        p=study.protocol()
        with tempfile.TemporaryDirectory(dir=ec.ROOT/'build-task2-validation') as td:
            binary=Path(td)/'fake.exe';binary.write_bytes(b'fixture')
            rows=study.build_rows('calibration',p,binary)
            self.assertEqual(len(rows),159)
            self.assertEqual(set(r['problem'] for r in rows),ec.test_split())
            self.assertEqual(set(r['search_seed'] for r in rows),{'101','111','121'})
            rates={name:dict(probability=0.,basis='calibration_pre_activation_stop') for name in p['instances']}
            ratefile=Path(td)/'calibration_rates.json';ratefile.write_text('{}')
            with patch.object(study,'STUDY',Path(td)):
                rows=study.build_rows('evaluation',p,binary,rates)
            self.assertEqual(len(rows),795)
            self.assertEqual(sum(r['method']=='random' and r['random_reject_rate']=='0.0' for r in rows),265)
            self.assertTrue(all(r['random_start_iter']=='1000' for r in rows if r['method']=='random'))

    def test_unauthorized_study_stays_blocked(self):
        with self.assertRaisesRegex(ValueError,'frozen authorized'):
            ec.run_rows([],Path('none'),Path('none'),True,'calibration',0)
        with self.assertRaisesRegex(ValueError,'stage identity'):
            ec.run_rows([{'stage':'wrong'}],Path('none'),Path('none'),True,'evaluation',0,study.PROTOCOL)

    def test_pool_and_natural_zero_fallback(self):
        with tempfile.TemporaryDirectory(dir=ec.ROOT/'build-task2-validation') as td:
            root=Path(td);rows=[]
            for name in ['early','active']:
                for seed in [101,111,121]:
                    folder=root/f'{name}-{seed}';folder.mkdir()
                    (folder/'runner_status.json').write_text('{}');(folder/'console.log').write_text('fixture')
                    rows.append(dict(problem=name,search_seed=str(seed),run_id=folder.name,output_dir=str(folder)))
            def inspect(row):
                early=row['problem']=='early'
                return dict(generated=500 if early else 1100,filter_rejected=0 if early else 10,
                    stop_reason='stagnation',training_failed=False)
            with patch.object(study,'STUDY',root),patch.object(study,'inspect',side_effect=inspect):
                rates=study.calibrate(rows)
            self.assertEqual(rates['early']['probability'],0)
            self.assertEqual(rates['early']['basis'],'calibration_pre_activation_stop')
            self.assertAlmostEqual(rates['active']['probability'],.1)
            # A timed-out/limited zero exposure is not silently treated as a natural stop.
            bad=root/'other';bad.mkdir()
            with patch.object(study,'STUDY',bad),patch.object(study,'inspect',return_value=dict(
                generated=500,filter_rejected=0,stop_reason='time_limit',training_failed=False)):
                with self.assertRaisesRegex(ValueError,'not valid natural'):
                    study.calibrate(rows)

    def test_holm_known_values(self):
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])


if __name__=='__main__':unittest.main()
