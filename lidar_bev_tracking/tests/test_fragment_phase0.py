import tempfile
import unittest
from pathlib import Path

import numpy as np

from bev_tracking.fragment_phase0 import (
    best_endpoint_relation,
    build_candidate_fragments,
    build_summary,
    roc_auc_lower_is_positive,
    write_phase0_outputs,
)
from bev_tracking.gesr_v1 import build_seed_components_optimized


class FragmentPhase0Test(unittest.TestCase):
    def test_shared_candidate_graph_keeps_structured_and_singleton_fragments(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.20],
                [0.5, 0.0, 0.0, 0.20],
                [2.0, 0.0, 0.0, 0.20],
            ],
            dtype=np.float64,
        )
        fragments = build_candidate_fragments(points, [10, 11, 12])
        self.assertEqual([item["point_count"] for item in fragments], [2, 1])
        self.assertEqual(
            [item["fragment_type"] for item in fragments],
            ["STRUCTURED", "SINGLETON"],
        )
        self.assertTrue(fragments[0]["major_axis_valid"])
        self.assertFalse(fragments[1]["major_axis_valid"])

    def test_endpoint_relation_supports_structured_and_singleton_fragments(self):
        seed = np.asarray(
            [
                [0.0, -0.1, 0.0, 0.5],
                [0.0, 0.1, 0.0, 0.5],
                [0.3, -0.1, 0.0, 0.5],
                [0.3, 0.1, 0.0, 0.5],
            ],
            dtype=np.float64,
        )
        components = build_seed_components_optimized(seed, [1, 2, 3, 4])
        points = np.asarray(
            [[1.0, 0.0, 0.0, 0.2], [1.2, 0.0, 0.0, 0.2], [3.0, 0.0, 0.0, 0.2]],
            dtype=np.float64,
        )
        structured, singleton = build_candidate_fragments(points, [10, 11, 20])
        structured_relation = best_endpoint_relation(structured, components)
        singleton_relation = best_endpoint_relation(singleton, components)
        self.assertTrue(structured_relation["has_computable_seed_relation"])
        self.assertTrue(singleton_relation["has_computable_seed_relation"])
        self.assertIsNotNone(structured_relation["orientation_difference"])
        self.assertIsNone(singleton_relation["orientation_difference"])
        self.assertGreaterEqual(structured_relation["forward_projection"], 0.0)

    def test_auc_uses_lower_geometry_value_as_positive_score(self):
        self.assertEqual(roc_auc_lower_is_positive([1.0, 2.0], [3.0, 4.0]), 1.0)
        self.assertEqual(roc_auc_lower_is_positive([3.0], [1.0]), 0.0)
        self.assertIsNone(roc_auc_lower_is_positive([], [1.0]))

    def test_summary_uses_unique_fragments_for_distribution_and_pairs_for_gt_coverage(self):
        fragment_rows = [
            {
                "frame_id": "000001",
                "fragment_runtime_id": 1,
                "fragment_type": "STRUCTURED",
                "strict_background": False,
                "has_computable_seed_relation": True,
                "endpoint_gap": 1.0,
                "lateral_offset": 0.1,
                "orientation_difference": 0.2,
            },
            {
                "frame_id": "000001",
                "fragment_runtime_id": 2,
                "fragment_type": "SINGLETON",
                "strict_background": True,
                "has_computable_seed_relation": True,
                "endpoint_gap": 3.0,
                "lateral_offset": 1.0,
                "orientation_difference": None,
            },
        ]
        pairs = [
            {
                "frame_id": "000001",
                "gt_id": "gt_1",
                "fragment_runtime_id": 1,
                "fragment_type": "STRUCTURED",
                "material_positive": True,
                "oracle_fragment_purity": 0.75,
            }
        ]
        summary = build_summary(
            fragment_rows,
            pairs,
            [{"cooperative_recovery": False}],
            target_count=11,
        )
        self.assertEqual(summary["candidate_fragment_count"], 2)
        self.assertEqual(summary["material_positive_unique_fragment_count"], 1)
        self.assertEqual(summary["fragment_formation_GT_coverage"]["count"], 1)
        self.assertEqual(
            summary["positive_vs_background"]["endpoint_gap"][
                "ROC_AUC_lower_value_is_positive"
            ],
            1.0,
        )
        self.assertIsNone(
            summary["positive_vs_background"]["orientation_difference"][
                "ROC_AUC_lower_value_is_positive"
            ]
        )

    def test_output_writer_emits_summary_and_two_flat_tables(self):
        result = {
            "schema_version": "test",
            "summary": {},
            "fragment_rows": [{"frame_id": "1", "fragment_runtime_id": 2}],
            "fragment_GT_pairs": [{"frame_id": "1", "gt_id": "gt_1"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_phase0_outputs(result, tmp)
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["fragments"].exists())
            self.assertTrue(paths["pairs"].exists())
            self.assertNotIn("fragment_rows", paths["summary"].read_text())


if __name__ == "__main__":
    unittest.main()
