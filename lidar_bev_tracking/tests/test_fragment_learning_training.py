import unittest

import numpy as np

from bev_tracking.fragment_learning_training import _binary_metrics, _interpret_gates


class FragmentLearningTrainingTest(unittest.TestCase):
    def test_binary_metrics_use_probability_ranking_and_fixed_threshold(self):
        result = _binary_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertEqual((result["TP"], result["FP"], result["FN"], result["TN"]), (2, 1, 0, 1))

    def test_interpretation_is_mechanical(self):
        keys = ("prerequisites", "M1_OOF_AP", "M1_AP_MINUS_PREVALENCE", "M1_AP_MINUS_L0_AP", "cross_fold_stability", "P_vs_N1_AP_lift")
        gates = {key: {"passed": True} for key in keys}
        self.assertEqual(_interpret_gates(gates), "SUPPORTED_FOR_RUNTIME_PROTOTYPE")
        gates["cross_fold_stability"]["passed"] = False
        self.assertEqual(_interpret_gates(gates), "INCONCLUSIVE")
        gates["M1_OOF_AP"]["passed"] = False
        self.assertEqual(_interpret_gates(gates), "NOT_SUPPORTED_BY_CURRENT_REPRESENTATION")


if __name__ == "__main__":
    unittest.main()
