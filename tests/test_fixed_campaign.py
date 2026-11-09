import unittest
import test_experiment_control as fixtures


class FixedSocpEvidenceTest(unittest.TestCase):
    setUp=fixtures.ValidatorTest.setUp
    def test_explicit_success_and_numeric_fallback(self):
        self.row['require_socp']='1'
        original=self.log.read_text()
        with self.assertRaisesRegex(ValueError,'missing explicit SOCP'):
            fixtures.ValidatorTest.test_valid(self)
        evidence='[SOCP] status=2 solutions=1\n[SOCP SUCCESS] objective=12\n[SOCP] status=12 solutions=0\n[SOCP FALLBACK] numerical_no_solution; candidate_unchanged\n'
        self.log.write_text(original+evidence)
        fixtures.ValidatorTest.test_valid(self)
        for bad in ['Error code = 10009\nLicense expired\n','[SOCP ERROR] missing license\n','[ERROR] Nonfinite Cox score\n']:
            self.log.write_text(original+evidence+bad)
            with self.assertRaisesRegex(ValueError,'solver/subsolver error'):
                fixtures.ValidatorTest.test_valid(self)
        self.log.write_text(original+evidence.replace('status=12','status=3'))
        with self.assertRaisesRegex(ValueError,'inconsistent SOCP'):
            fixtures.ValidatorTest.test_valid(self)


if __name__=='__main__':unittest.main()
