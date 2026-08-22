import unittest
import json

import numpy as np

from bev_tracking.fragment_learning_v2_training import (
    _interpret,
    _margin_coverage,
    _to_builtin,
)
from bev_tracking.fragment_learning_multiseed_analysis import build_multiseed_record
from bev_tracking.fragment_phase0 import build_candidate_fragments
from tests.test_fragment_learning_multiseed_analysis import component


class FragmentLearningV2TrainingTest(unittest.TestCase):
    def test_margin_uses_distinct_seed_components_and_frozen_order(self):
        target = build_candidate_fragments(
            np.asarray([[3.0, 0.0, 0.0, 0.2], [3.1, 0.0, 0.0, 0.2]]),
            [10, 11],
        )[0]
        record = build_multiseed_record(
            target,
            [component(9, [0.0, 0.0]), component(3, [2.0, 0.0]), component(5, [1.0, 0.0])],
        )
        self.assertEqual(record["best_relation"]["seed_component_runtime_id"], 3)
        self.assertEqual(record["second_best_relation"]["seed_component_runtime_id"], 5)
        self.assertGreaterEqual(record["second_minus_best_endpoint_gap"], 0.0)

    def test_missing_margin_is_native_nan_not_zero(self):
        values = np.asarray([1.0, np.nan, 0.0])
        labels = np.asarray(["POSITIVE", "N1", "N0"], dtype=object)
        coverage = _margin_coverage(values, labels)
        self.assertEqual(coverage["ALL"]["valid"], 2)
        self.assertEqual(coverage["ALL"]["missing"], 1)
        self.assertEqual(coverage["N1"]["valid"], 0)
        self.assertEqual(coverage["N0"]["valid"], 1)

    def test_development_interpretation_is_mechanical(self):
        gates = {f"gate-{index}": {"passed": True} for index in range(7)}
        self.assertEqual(_interpret(gates, 0.02, 20, 29), "PASS")
        gates["gate-1"]["passed"] = False
        self.assertEqual(_interpret(gates, 0.005, 10, 29), "INCONCLUSIVE")
        self.assertEqual(_interpret(gates, -0.001, 2, 29), "NOT_SUPPORTED")
        self.assertEqual(_interpret(gates, 0.03, 30, 26), "NOT_SUPPORTED")

    def test_numpy_gate_scalars_are_json_serializable(self):
        value = {
            "passed": np.bool_(True),
            "count": np.int64(3),
            "metric": np.float64(0.25),
        }
        normalized = _to_builtin(value)
        self.assertEqual(json.loads(json.dumps(normalized)), {
            "passed": True, "count": 3, "metric": 0.25,
        })


if __name__ == "__main__":
    unittest.main()
