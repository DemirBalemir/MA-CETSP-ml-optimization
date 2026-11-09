import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import experiment_control as ec
from resume_parallel_campaign import checked_resume


class ResumeBridgeTest(unittest.TestCase):
    def test_only_approved_environment_digest_is_accepted(self):
        old=dict(record=dict(environment=dict(variables='old'),row=dict(run_id='fixture'),binary='same'))
        old['sha256']=ec.canonical_hash(old['record'])
        status=dict(state='validated_success',configuration_identity=old,command=['same'])
        new=copy.deepcopy(old);new['record']['environment']['variables']='new'
        new['sha256']=ec.canonical_hash(new['record'])
        before=copy.deepcopy(status)
        with tempfile.TemporaryDirectory(dir=ec.ROOT/'build-task2-validation') as td:
            root=Path(td)
            checked_resume(status,new,['same'],{'old'},'new',root,ec.verify_resume)
            self.assertEqual(status,before);self.assertTrue((root/'fixture.json').exists())
            for field in ['binary','row']:
                changed=copy.deepcopy(new);changed['record'][field]='changed'
                changed['sha256']=ec.canonical_hash(changed['record'])
                with self.assertRaises(ValueError):checked_resume(status,changed,['same'],{'old'},'new',root,ec.verify_resume)
            with self.assertRaises(ValueError):checked_resume(status,new,['different'],{'old'},'new',root,ec.verify_resume)
            with self.assertRaises(ValueError):checked_resume(status,new,['same'],{'other'},'new',root,ec.verify_resume)
            with self.assertRaises(ValueError):checked_resume(status,new,['same'],{'old'},'other',root,ec.verify_resume)
            forged=copy.deepcopy(status);forged['configuration_identity']['sha256']='forged'
            with self.assertRaises(ValueError):checked_resume(forged,new,['same'],{'old'},'new',root,ec.verify_resume)


if __name__=='__main__':unittest.main()
