import unittest

import numpy as np

from bev_tracking.lfrr_v1_phase2a import _quartiles, _rho, _classify_score_collapse


class LFRRV1Phase2ATest(unittest.TestCase):
    def test_quartiles_and_spearman_are_descriptive(self):
        self.assertEqual(_quartiles([1, 2, 3, 4])["P50"], 2.5)
        self.assertAlmostEqual(_rho([0, 1, 2], [0, 2, 4])["spearman_rho"], 1.0)

    def test_score_collapse_mapping_does_not_use_labels_as_features(self):
        def dist(p, n1, n0):
            return {
                "POSITIVE": {"probability": {"P50": p}},
                "N1": {"probability": {"P50": n1}},
                "N0": {"probability": {"P50": n0}},
            }
        result = _classify_score_collapse(
            dist(0.75, 0.30, 0.10), dist(0.35, 0.28, 0.08), 0.05
        )
        self.assertIn(result, {
            "POSITIVE_SPECIFIC_COLLAPSE", "CLASS_OVERLAP_EXPANSION", "MIXED"
        })


if __name__ == "__main__":
    unittest.main()
