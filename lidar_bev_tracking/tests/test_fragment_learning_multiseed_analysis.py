import unittest
from types import SimpleNamespace

import numpy as np

from bev_tracking.fragment_learning_multiseed_analysis import (
    build_multiseed_record,
    enumerate_distinct_seed_relations,
    evaluate_context_signal,
)
from bev_tracking.fragment_phase0 import build_candidate_fragments


def component(runtime_id, center):
    geometry = SimpleNamespace(
        valid=True,
        center=np.asarray(center, dtype=np.float64),
        major=np.asarray([1.0, 0.0]),
        minor=np.asarray([0.0, 1.0]),
        u_min=-0.5, u_max=0.5, v_min=-0.2, v_max=0.2,
        lambda1=1.0, lambda2=0.1,
    )
    return SimpleNamespace(
        runtime_id=runtime_id,
        source_indices=np.asarray([runtime_id, runtime_id + 1]),
        geometry=geometry,
    )


class FragmentLearningMultiSeedAnalysisTest(unittest.TestCase):
    def test_second_best_is_a_distinct_seed_component(self):
        points = np.asarray([[3.0, 0.0, 0.0, 0.2], [3.1, 0.0, 0.0, 0.2]])
        fragment = build_candidate_fragments(points, [10, 11])[0]
        components = [component(100, [0.0, 0.0]), component(200, [1.0, 0.0])]
        relations = enumerate_distinct_seed_relations(fragment, components)
        self.assertEqual(len(relations), 2)
        self.assertNotEqual(relations[0]["seed_component_runtime_id"], relations[1]["seed_component_runtime_id"])
        record = build_multiseed_record(fragment, components)
        self.assertEqual(record["valid_seed_relation_count"], 2)
        self.assertEqual(record["frame_valid_seed_component_count"], 2)
        self.assertEqual(record["has_second_best_relation"], 1)
        self.assertGreaterEqual(record["second_minus_best_endpoint_gap"], 0.0)

    def test_singleton_orientation_contrast_remains_missing(self):
        fragment = build_candidate_fragments(
            np.asarray([[3.0, 0.0, 0.0, 0.2]]), [10]
        )[0]
        record = build_multiseed_record(
            fragment, [component(100, [0.0, 0.0]), component(200, [1.0, 0.0])]
        )
        self.assertIsNone(record["best_orientation_difference"])
        self.assertIsNone(record["second_minus_best_orientation_difference"])

    def test_signal_requires_overall_superfold_and_frame_consistency(self):
        records = []
        for fold, frame in ((1, "a"), (2, "b"), (5, "c")):
            for group, value in (("P_HIGH", 5.0), ("N0_FP", 1.0)):
                for offset in (0.0, 0.1):
                    row = {
                        "group": group, "fold": fold, "frame_id": frame,
                        "fragment_type": "STRUCTURED",
                    }
                    for field in (
                        "frame_valid_seed_component_count", "valid_seed_relation_count",
                        "valid_seed_relation_fraction", "has_second_best_relation",
                        "best_endpoint_gap", "best_lateral_offset", "best_orientation_difference",
                        "best_forward_projection", "second_endpoint_gap", "second_lateral_offset",
                        "second_orientation_difference", "second_forward_projection",
                        "second_minus_best_endpoint_gap", "second_minus_best_lateral_offset",
                        "second_minus_best_orientation_difference", "best_seed_point_count",
                        "best_seed_major_span", "best_seed_minor_span", "best_seed_linearity",
                        "second_seed_point_count", "second_seed_major_span", "second_seed_minor_span",
                        "second_seed_linearity", "second_minus_best_seed_point_count",
                        "second_minus_best_seed_major_span", "second_minus_best_seed_minor_span",
                        "second_minus_best_seed_linearity",
                    ):
                        row[field] = value + offset if field == "valid_seed_relation_count" else 1.0 + offset
                    records.append(row)
        result = evaluate_context_signal(records)
        self.assertTrue(result["families"]["seed_relation_multiplicity"]["supported"])
        self.assertEqual(result["MULTI_SEED_CONTEXT_SIGNAL"], "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
