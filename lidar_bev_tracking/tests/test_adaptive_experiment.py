import unittest

import numpy as np

from bev_tracking.adaptive_experiment import (
    VariantSpec,
    build_gt_cluster_associations,
    compare_eligible_gt_sets,
    preregister_variant_specs,
    summarize_merging,
    validate_variant_specs,
)
from bev_tracking.clustering_policy import ClusteringPolicy


def policy(mode="fixed", eps=0.6, min_points=20):
    if mode == "fixed":
        return ClusteringPolicy(mode=mode, eps=eps, min_points=min_points)
    return ClusteringPolicy(
        mode="adaptive",
        distance_params={
            "near_0_15": {"eps": 0.5, "min_points": 20},
            "mid_15_30": {"eps": 0.65, "min_points": 15},
            "far_30_inf": {"eps": 0.85, "min_points": 8},
        },
    )


def variant_specs():
    return {
        "C0": VariantSpec("C0", policy()),
        "C1": VariantSpec("C1", policy("adaptive")),
        "C2": VariantSpec("C2", policy("adaptive")),
        "C3": VariantSpec("C3", policy("adaptive")),
    }


class AdaptiveExperimentTest(unittest.TestCase):
    def test_preregistered_variants_have_fixed_order_and_frozen_filters(self):
        specs = variant_specs()
        self.assertTrue(validate_variant_specs(specs))
        report = preregister_variant_specs(specs)
        self.assertEqual(report["variant_order"], ["C0", "C1", "C2", "C3"])
        self.assertEqual(report["variants"]["C0"]["eps"], 0.6)

    def test_variant_registration_rejects_filter_change(self):
        specs = variant_specs()
        specs["C2"] = VariantSpec("C2", policy("adaptive"), intensity_min=0.0)
        with self.assertRaises(ValueError):
            validate_variant_specs(specs)

    def test_gt_cluster_association_requires_count_and_fraction(self):
        points = np.asarray(
            [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0], [1.3, 0.0, 0.0]],
            dtype=np.float32,
        )
        gt = {
            "id": "gt_1", "class_name": "car", "x": 1.15, "y": 0.0, "z": 0.0,
            "length": 1.0, "width": 1.0, "height": 1.0, "yaw": 0.0,
        }
        records = build_gt_cluster_associations(points, [points[:3], points[3:]], [gt])
        self.assertTrue(records[0]["eligible"])
        self.assertEqual(records[0]["associated_cluster_indices"], [0])

    def test_merging_counts_unique_gt_once(self):
        records = [
            {"gt_id": "gt_1", "eligible": True, "associated_cluster_indices": [0]},
            {"gt_id": "gt_2", "eligible": True, "associated_cluster_indices": [0]},
            {"gt_id": "gt_3", "eligible": True, "associated_cluster_indices": [1]},
        ]
        summary = summarize_merging(records, [None, None])
        self.assertEqual(summary["merged_positive_gt_count"], 2)
        self.assertEqual(summary["eligible_positive_gt_count"], 3)
        self.assertAlmostEqual(summary["merging_rate"], 2.0 / 3.0)

    def test_eligible_gt_sets_are_compared_against_c0(self):
        reports = {
            "C0": [{"frame_id": "1", "gt_cluster_associations": [{"gt_id": "gt_1", "eligible": True}]}],
            "C1": [{"frame_id": "1", "gt_cluster_associations": [{"gt_id": "gt_1", "eligible": True}]}],
            "C2": [{"frame_id": "1", "gt_cluster_associations": [{"gt_id": "gt_2", "eligible": True}]}],
            "C3": [{"frame_id": "1", "gt_cluster_associations": [{"gt_id": "gt_1", "eligible": True}]}],
        }
        result = compare_eligible_gt_sets(reports)
        self.assertFalse(result["passed"])
        self.assertIn("C2", result["mismatches"])


if __name__ == "__main__":
    unittest.main()
