"""Regression checks for the paper's numerical definitions."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_la_cetsp'))
from diagnostic_metrics import cohort_mask, cohort_rank, pairwise_concordance
from diagnostic_metrics import rank_gain_kappa, couple_to_kappa


class DiagnosticMetricsTests(unittest.TestCase):
    def test_concordance_ties(self):
        self.assertAlmostEqual(pairwise_concordance([0,0,1], [0,1,2]), 5/6)
        self.assertEqual(pairwise_concordance([0,1,2], [0,0,1]), 1)
        self.assertEqual(pairwise_concordance([2,1,0], [0,1,2]), 0)
        self.assertTrue(np.isnan(pairwise_concordance([0,1], [1,1])))

    def test_high_kappa_does_not_imply_rank_loss(self):
        p, q = np.array([10,20,30,40.]), np.array([9,16,21,24.])
        self.assertAlmostEqual(rank_gain_kappa([0,.3,.6,1], (p-q)/p), 1)
        self.assertEqual(pairwise_concordance(p,q), 1)

    def test_membership_excludes_self_and_deaths(self):
        mask = cohort_mask([0,0,0,0,1], [5,1,5,5,5])
        self.assertFalse(mask[4,1])
        self.assertFalse(np.diag(mask).any())
        self.assertEqual(cohort_rank([1,2,3,4,5], mask)[4], 1)

    def test_copula_preserves_gains(self):
        gains = np.linspace(.01,.9,100)
        actual = couple_to_kappa(np.arange(100), gains, .8, np.random.default_rng(0))
        np.testing.assert_array_equal(np.sort(actual), gains)


if __name__ == '__main__':
    unittest.main()
