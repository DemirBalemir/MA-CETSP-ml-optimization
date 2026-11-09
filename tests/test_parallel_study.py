import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import control_study_parallel as p


class ParallelSchedulerTest(unittest.TestCase):
    def exercise(self,fail=False,pause=False):
        with tempfile.TemporaryDirectory(dir=p.ec.ROOT/'build-task2-validation') as td:
            root=Path(td);(root/'evaluation').mkdir()
            rows=[dict(run_id=f'run-{i}',worker=str(i%4+1),output_dir=str(root/'evaluation'/f'run-{i}')) for i in range(12)]
            lock=threading.Lock();active=set();seen=[];peak=[0];barrier=threading.Barrier(4)
            def runner(group,*args,**kwargs):
                row=group[0]
                with lock:
                    self.assertNotIn(row['worker'],active);active.add(row['worker'])
                    peak[0]=max(peak[0],len(active));seen.append(row['run_id'])
                if int(row['run_id'].split('-')[1])<4:barrier.wait(timeout=5)
                time.sleep(.02)
                if pause:(root/'STOP_AFTER_CURRENT').touch()
                with lock:active.remove(row['worker'])
                if fail and row['run_id']=='run-0':raise ValueError('fixture failure')
            if fail:
                with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                    p.run_parallel(rows,Path('unused'),Path('unused'),Path('unused'),root,runner,lambda r:None)
            else:
                complete=p.run_parallel(rows,Path('unused'),Path('unused'),Path('unused'),root,runner,lambda r:None)
                self.assertEqual(complete,not pause)
            self.assertEqual(peak[0],4);self.assertFalse(active)
            self.assertEqual(len(seen),len(set(seen)))
            if not fail and not pause:self.assertEqual(len(seen),12)
            if pause:self.assertEqual(len(seen),4)
            progress=json.loads((root/'progress.json').read_text())
            self.assertFalse(progress['active'])
    def test_four_streams_unique_and_complete(self):self.exercise()
    def test_failure_drains_running_peers(self):self.exercise(fail=True)
    def test_pause_after_current(self):self.exercise(pause=True)


if __name__=='__main__':unittest.main()
