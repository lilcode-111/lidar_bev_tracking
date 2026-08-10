import copy
import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.cluster_separability import (
    build_cluster_feature_day2,
    build_cluster_group_day1,
    build_frame_cluster_groups,
    validate_cluster_feature_records,
)
from bev_tracking.clustering_policy import ClusteringPolicy


def car_box(x=10.0):
    return {
        "id": "gt_1", "class_name": "car", "x": x, "y": 0.0, "z": 0.0,
        "length": 2.0, "width": 2.0, "height": 2.0, "yaw": 0.0,
    }


class ClusterSeparabilityDay2Test(unittest.TestCase):
    def test_day2_aggregates_delta_p2_oracle_without_pipeline_mutation(self):
        points = np.column_stack([
            np.linspace(9.2, 10.8, 10), np.zeros(10), np.zeros(10), np.ones(10)
        ]).astype(np.float32)
        detection = {
            "id": "cluster_1", "class_name": "pedestrian", "x": 10.0, "y": 0.0,
            "length": 2.0, "width": 2.0, "yaw": 0.0,
        }
        c0_groups = build_frame_cluster_groups(
            frame_id="1", variant="C0", filtered_points=points, clusters=[points],
            raw_detections=[detection], gt_boxes=[],
        )
        c1_groups = build_frame_cluster_groups(
            frame_id="1", variant="C1", filtered_points=points, clusters=[points],
            raw_detections=[detection], gt_boxes=[car_box()],
        )
        reports = {
            "C0": [{
                "frame_id": "000001",
                "candidate_conversion": {"evidence": [{"frame_id": "000001", "gt_id": "gt_1", "cluster_ids": []}]},
                "cluster_separability": c0_groups,
            }],
            "C1": [{
                "frame_id": "000001",
                "candidate_conversion": {"evidence": [{"frame_id": "000001", "gt_id": "gt_1", "cluster_ids": ["cluster_1"]}]},
                "cluster_separability": c1_groups,
            }],
        }
        grouped = build_cluster_group_day1(reports)
        result = build_cluster_feature_day2(grouped)
        self.assertEqual(result["delta_22_p2_oracle"]["count"], 1)
        self.assertEqual(result["delta_22_p2_oracle"]["iou_ge_0_25_count"], 1)
        self.assertEqual(result["delta_22_p2_oracle"]["iou_ge_0_50_count"], 1)
        self.assertTrue(grouped["records_by_variant"]["C1"][0]["is_delta_22"])
        self.assertNotIn("records_by_variant", result)

    def test_day2_rejects_raw_reports_instead_of_recomputing_day1(self):
        with self.assertRaisesRegex(ValueError, "requires a valid Day 1"):
            build_cluster_feature_day2({"C0": [], "C1": []})

    def test_p2_features_use_frozen_coverage_and_purity_denominators(self):
        inside = np.column_stack([
            np.linspace(9.2, 9.8, 5), np.zeros(5), np.zeros(5), np.ones(5)
        ]).astype(np.float32)
        outside = np.column_stack([
            np.linspace(11.2, 11.8, 5), np.zeros(5), np.zeros(5), np.ones(5)
        ]).astype(np.float32)
        extra_gt = np.column_stack([
            np.linspace(10.1, 10.7, 5), np.zeros(5), np.zeros(5), np.ones(5)
        ]).astype(np.float32)
        candidate_cluster = np.vstack([inside, outside])
        detection = {
            "id": "cluster_1", "class_name": "pedestrian", "x": 10.5, "y": 0.0,
            "length": 1.5, "width": 0.5, "yaw": 0.0,
        }
        frozen_detection = copy.deepcopy(detection)
        payload = build_frame_cluster_groups(
            frame_id="1", variant="C1",
            filtered_points=np.vstack([candidate_cluster, extra_gt]),
            clusters=[candidate_cluster], raw_detections=[detection], gt_boxes=[car_box()],
        )
        record = payload["records"][0]
        features = record["features"]
        self.assertEqual(record["group"], "P2")
        self.assertAlmostEqual(features["gt_coverage_ratio"], 0.5)
        self.assertAlmostEqual(features["cluster_purity_ratio"], 0.5)
        self.assertEqual(features["num_points"], 10)
        self.assertIsNotNone(features["oracle_car_iou"])
        self.assertEqual(detection, frozen_detection)
        self.assertTrue(validate_cluster_feature_records(payload["records"]))

    def test_n_has_uniform_geometry_but_no_gt_or_oracle_values(self):
        cluster = np.asarray(
            [[30.0, 5.0, 0.0, 1.0], [30.5, 5.2, 0.5, 1.0]], dtype=np.float32
        )
        payload = build_frame_cluster_groups(
            frame_id="1", variant="C1", filtered_points=cluster, clusters=[cluster],
            raw_detections=[{
                "id": "cluster_1", "class_name": "cone", "x": 30.25, "y": 5.1,
                "length": 0.5, "width": 0.2, "yaw": 0.0,
            }],
            gt_boxes=[],
        )
        features = payload["records"][0]["features"]
        self.assertEqual(features["num_points"], 2)
        self.assertAlmostEqual(features["height_span"], 0.5)
        self.assertIsNone(features["gt_coverage_ratio"])
        self.assertIsNone(features["cluster_purity_ratio"])
        self.assertIsNone(features["oracle_car_iou"])

    def test_run_variant_frame_emits_valid_day2_features(self):
        points = np.column_stack([
            np.linspace(9.5, 10.5, 12), np.linspace(-0.3, 0.3, 12),
            np.zeros(12), np.ones(12),
        ]).astype(np.float32)
        report = run_variant_frame(
            points, [car_box()], frame_id="1",
            variant=VariantSpec("C1", ClusteringPolicy(mode="fixed", eps=0.6, min_points=3)),
        )
        records = report["cluster_separability"]["records"]
        self.assertTrue(validate_cluster_feature_records(records))
        self.assertIn("features", records[0])


if __name__ == "__main__":
    unittest.main()
