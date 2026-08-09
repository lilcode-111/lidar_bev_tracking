import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.candidate_conversion import (
    build_candidate_conversion_report,
    effective_min_points_for_gt,
)
from bev_tracking.clustering_policy import ClusteringPolicy


def car_box(box_id="gt_1"):
    return {
        "id": box_id,
        "class_name": "car",
        "x": 10.0,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": 0.0,
    }


def points(count=30):
    return np.column_stack([
        np.linspace(9.0, 11.0, count),
        np.linspace(-0.5, 0.5, count),
        np.zeros(count),
        np.ones(count),
    ]).astype(np.float32)


class CandidateConversionDay2Test(unittest.TestCase):
    @staticmethod
    def adaptive_policy():
        return ClusteringPolicy(
            mode="adaptive",
            distance_params={
                "near_0_15": {"eps": 0.6, "min_points": 16},
                "mid_15_30": {"eps": 0.6, "min_points": 10},
                "far_30_inf": {"eps": 0.6, "min_points": 6},
            },
        )

    def test_effective_min_points_uses_gt_distance_bin_boundaries(self):
        policy = self.adaptive_policy()
        self.assertEqual(effective_min_points_for_gt(car_box(), 16, policy), 16)
        self.assertEqual(effective_min_points_for_gt({**car_box(), "x": 15.0}, 16, policy), 10)
        self.assertEqual(effective_min_points_for_gt({**car_box(), "x": 30.0}, 16, policy), 6)

    def test_mid_range_terminal_state_does_not_use_near_min_points(self):
        gt_box = {**car_box(), "x": 20.0}
        filtered = np.column_stack([
            np.linspace(19.0, 21.0, 12),
            np.linspace(-0.5, 0.5, 12),
            np.zeros(12),
            np.ones(12),
        ]).astype(np.float32)
        result = build_candidate_conversion_report(
            frame_id="1",
            gt_boxes=[gt_box],
            stages={name: filtered for name in ("raw", "roi", "z_filter", "intensity_filter")},
            clusters=[],
            raw_detections=[],
            detections_after_nms=[],
            evaluation={"matches": []},
            variant="C1",
            min_points=16,
            clustering_policy=self.adaptive_policy(),
        )
        self.assertEqual(result["evidence"][0]["terminal_state"], "no_associated_cluster")

    def test_run_variant_frame_uses_gt_distance_policy_at_boundaries(self):
        variant = VariantSpec("C1", self.adaptive_policy())
        cases = (
            (14.999, 12, "insufficient_filtered_points_for_association"),
            (15.0, 12, "no_associated_cluster"),
            (29.999, 7, "insufficient_filtered_points_for_association"),
            (30.0, 7, "no_associated_cluster"),
        )
        for index, (range_xy, count, expected) in enumerate(cases):
            with self.subTest(range_xy=range_xy):
                gt_box = {**car_box(), "x": range_xy}
                offsets = np.asarray(
                    [
                        (-1.5, -0.7), (-0.75, -0.7), (0.0, -0.7), (0.75, -0.7),
                        (1.5, -0.7), (-1.5, 0.0), (-0.75, 0.0), (0.0, 0.0),
                        (0.75, 0.0), (1.5, 0.0), (-1.5, 0.7), (-0.75, 0.7),
                    ][:count],
                    dtype=np.float32,
                )
                frame_points = np.column_stack(
                    [
                        offsets[:, 0] + range_xy,
                        offsets[:, 1],
                        np.zeros(count, dtype=np.float32),
                        np.ones(count, dtype=np.float32),
                    ]
                ).astype(np.float32)
                report = run_variant_frame(
                    frame_points,
                    [gt_box],
                    frame_id=str(index + 1),
                    variant=variant,
                )
                evidence = report["candidate_conversion"]["evidence"][0]
                self.assertEqual(evidence["terminal_state"], expected)

    def test_report_preserves_all_associated_branches(self):
        cluster_a = points(15)
        cluster_b = points(15)
        raw = [
            {"id": "cluster_1", "class_name": "pedestrian", "x": 10.0, "y": 0.0, "length": 1.0, "width": 1.0, "yaw": 0.0},
            {"id": "cluster_2", "class_name": "car", "x": 12.0, "y": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0},
        ]
        result = build_candidate_conversion_report(
            frame_id="1",
            gt_boxes=[car_box()],
            stages={"raw": np.vstack([cluster_a, cluster_b]), "roi": np.vstack([cluster_a, cluster_b]), "z_filter": np.vstack([cluster_a, cluster_b]), "intensity_filter": np.vstack([cluster_a, cluster_b])},
            clusters=[cluster_a, cluster_b],
            raw_detections=raw,
            detections_after_nms=raw,
            evaluation={"matches": []},
            variant="C1",
            min_points=20,
        )
        evidence = result["evidence"][0]
        self.assertEqual(evidence["cluster_ids"], ["cluster_1", "cluster_2"])
        self.assertEqual(len(evidence["candidate_branches"]), 2)
        self.assertEqual(evidence["terminal_state"], "box_iou_0_25_to_0_50")

    def test_unassociated_global_detection_does_not_match_gt(self):
        cluster = points(30)
        associated = {"id": "cluster_1", "class_name": "car", "x": 30.0, "y": 10.0, "length": 4.0, "width": 2.0, "yaw": 0.0}
        unrelated = {"id": "cluster_99", "class_name": "car", "x": 10.0, "y": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}
        result = build_candidate_conversion_report(
            frame_id="1",
            gt_boxes=[car_box()],
            stages={"raw": cluster, "roi": cluster, "z_filter": cluster, "intensity_filter": cluster},
            clusters=[cluster],
            raw_detections=[associated, unrelated],
            detections_after_nms=[associated, unrelated],
            evaluation={"matches": [{"gt_id": "gt_1", "det_id": "cluster_99"}]},
            variant="C0",
            min_points=20,
        )
        self.assertFalse(result["evidence"][0]["matched_at_primary_iou"])


if __name__ == "__main__":
    unittest.main()
