import json
import unittest

import numpy as np

from bev_tracking.failure_evidence import build_failure_evidence_report


def car_box(box_id="gt_1", x=10.0, y=0.0):
    return {
        "id": box_id,
        "class_name": "car",
        "x": x,
        "y": y,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": 0.0,
    }


def points_in_car(count, intensity=0.8, seed=7):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(9.0, 11.0, count),
            rng.uniform(-0.5, 0.5, count),
            rng.uniform(-0.4, 0.4, count),
            np.full(count, intensity),
        ]
    ).astype(np.float32)


class FailureEvidenceReplayTest(unittest.TestCase):
    def test_no_raw_points_has_one_primary_reason(self):
        report = build_failure_evidence_report(
            np.asarray([[30.0, 10.0, 0.0, 0.8]], dtype=np.float32),
            [car_box()],
            frame_id="317",
            min_points=3,
        )

        evidence = report["failure_evidence"][0]
        self.assertEqual(evidence["primary_reason"], "no_raw_points_in_gt")
        self.assertEqual(evidence["supporting_flags"], [])

    def test_intensity_filter_removal_is_traced(self):
        report = build_failure_evidence_report(
            points_in_car(8, intensity=0.2),
            [car_box()],
            frame_id="317",
            min_points=3,
            intensity_min=0.38,
        )

        evidence = report["failure_evidence"][0]
        self.assertEqual(evidence["stage_point_counts"]["raw"], 8)
        self.assertEqual(evidence["stage_point_counts"]["z_filter"], 8)
        self.assertEqual(evidence["stage_point_counts"]["intensity_filter"], 0)
        self.assertEqual(evidence["primary_reason"], "removed_by_intensity_filter")

    def test_intensity_filter_threshold_crossing_is_primary_cause(self):
        points = points_in_car(30, intensity=0.2)
        points[:10, 3] = 0.8

        report = build_failure_evidence_report(
            points,
            [car_box()],
            frame_id="317",
            min_points=20,
            intensity_min=0.38,
        )

        evidence = report["failure_evidence"][0]
        self.assertEqual(evidence["stage_point_counts"]["z_filter"], 30)
        self.assertEqual(evidence["stage_point_counts"]["intensity_filter"], 10)
        self.assertEqual(evidence["cluster_ids"], [])
        self.assertEqual(evidence["primary_reason"], "removed_by_intensity_filter")
        self.assertIn("insufficient_points_for_clustering", evidence["supporting_flags"])

    def test_points_without_cluster_are_insufficient_for_clustering(self):
        points = np.asarray(
            [
                [9.1, -0.8, 0.0, 0.8],
                [10.0, 0.0, 0.0, 0.8],
                [10.9, 0.8, 0.0, 0.8],
            ],
            dtype=np.float32,
        )
        report = build_failure_evidence_report(
            points,
            [car_box()],
            frame_id="317",
            eps=0.1,
            min_points=2,
        )

        evidence = report["failure_evidence"][0]
        self.assertEqual(evidence["primary_reason"], "insufficient_points_for_clustering")
        self.assertIn("final_iou_below_threshold", evidence["supporting_flags"])

    def test_report_is_json_serializable_and_counts_reasons(self):
        report = build_failure_evidence_report(
            points_in_car(8, intensity=0.2),
            [car_box()],
            frame_id="317",
            min_points=3,
        )

        self.assertEqual(report["frame_id"], "000317")
        self.assertEqual(report["summary"]["num_false_negatives"], 1)
        self.assertEqual(report["summary"]["primary_reason_counts"]["removed_by_intensity_filter"], 1)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
