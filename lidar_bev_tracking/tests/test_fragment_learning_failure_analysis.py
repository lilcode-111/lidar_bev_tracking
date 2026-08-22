import unittest

import numpy as np

from bev_tracking.fragment_learning_failure_analysis import (
    _decision_from_smd,
    _frame_error_concentration,
    _quartiles,
    _shift_summary,
)
from bev_tracking.fragment_learning_dataset import MODEL_FEATURE_FIELDS


class FragmentLearningFailureAnalysisTest(unittest.TestCase):
    def test_frame_concentration_uses_all_negative_false_positives(self):
        rows = [
            {"frame_id": "000001", "label": "POSITIVE", "score": 0.8},
            {"frame_id": "000001", "label": "N0", "score": 0.7},
            {"frame_id": "000001", "label": "N1", "score": 0.6},
            {"frame_id": "000002", "label": "POSITIVE", "score": 0.2},
        ]
        result = _frame_error_concentration(rows)
        self.assertEqual(result["total_FP"], 2)
        self.assertAlmostEqual(result["total_FP_score_mass"], 1.3)
        self.assertEqual(result["total_FN"], 1)
        self.assertEqual(result["recovered_positive_frame_count"], 1)
        self.assertEqual(result["zero_recovery_frames"], ["000002"])

    def test_shift_summary_and_decision_are_frozen(self):
        width = len(MODEL_FEATURE_FIELDS)
        matrix = np.zeros((8, width), dtype=np.float64)
        matrix[:4, :3] = 2.0
        matrix[:4, 3:] = np.arange(4)[:, None] * 0.01
        matrix[4:, :] = np.arange(4)[:, None] * 0.01
        left = np.asarray([True] * 4 + [False] * 4)
        summary = _shift_summary(matrix, left, ~left)
        self.assertGreaterEqual(summary["moderate_or_larger_feature_count"], 3)
        self.assertEqual(_decision_from_smd(summary), "YES")

    def test_quartiles_are_compact(self):
        self.assertEqual(_quartiles([1, 2, 3, 4]), {"N": 4, "P25": 1.75, "P50": 2.5, "P75": 3.25})


if __name__ == "__main__":
    unittest.main()
