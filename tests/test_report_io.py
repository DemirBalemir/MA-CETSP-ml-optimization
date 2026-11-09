"""Report-file sharing regression checks; no solver or model fitting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import experiment_control as ec

class ReportIOTest(unittest.TestCase):
    def test_transient_replace_lock_preserves_complete_json(self):
        with tempfile.TemporaryDirectory(dir=ec.ROOT/'build-task2-validation') as td:
            target=Path(td)/'progress.json';target.write_text('{"old": true}')
            replace=Path.replace;attempts=[]
            def locked_once(tmp,dest):
                attempts.append(tmp)
                if len(attempts)==1:raise PermissionError('sharing fixture')
                return replace(tmp,dest)
            with patch.object(Path,'replace',locked_once),patch.object(ec.time,'sleep'):
                ec.save_json(target,{'completed':795,'active':[]})
            self.assertEqual(json.loads(target.read_text()),{'completed':795,'active':[]})
            self.assertEqual(len(attempts),2)
            self.assertFalse(target.with_suffix('.json.tmp').exists())

    def test_persistent_lock_propagates_and_keeps_previous_report(self):
        with tempfile.TemporaryDirectory(dir=ec.ROOT/'build-task2-validation') as td:
            target=Path(td)/'progress.json';target.write_text('{"old": true}')
            with patch.object(Path,'replace',side_effect=PermissionError('persistent fixture')),patch.object(ec.time,'monotonic',side_effect=[0.,31.]),patch.object(ec.time,'sleep'):
                with self.assertRaises(PermissionError):ec.save_json(target,{'completed':795})
            self.assertEqual(json.loads(target.read_text()),{'old':True})

if __name__=='__main__':unittest.main()
