import json
from pathlib import Path
import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.point_retention import (
    build_point_retention_day1,
    pca_input_status,
    validate_stage_point_counts,
)


def car_box():
    return {
        "id": "gt_1", "class_name": "car", "x": 10.0, "y": 0.0, "z": 0.0,
        "length": 4.0, "width": 2.0, "height": 2.0, "yaw": 0.0,
    }


class PointRetentionDay1Test(unittest.TestCase):
    def test_closure_archive_is_valid_and_closed(self):
        with Path("docs/v15_3_1_closure.json").open("r", encoding="utf-8") as stream:
            closure = json.load(stream)
        self.assertEqual(closure["status"], "closed")
        self.assertEqual(closure["gates"]["delta_associated_gt_count"], 22)
        self.assertEqual(closure["oracle"]["iou_ge_0_25_count"], 0)

    def test_run_variant_frame_canonical_counts_feed_day1(self):
        points = np.asarray([
            [9.6, -0.1, 0.0, 0.8], [10.0, 0.1, 0.0, 0.8],
            [10.4, -0.1, 0.2, 0.8],
        ], dtype=np.float32)
        c0 = run_variant_frame(
            points,
            [car_box()],
            frame_id="1",
            variant=VariantSpec(
                "C0", ClusteringPolicy(mode="fixed", eps=0.6, min_points=20)
            ),
        )
        c1 = run_variant_frame(
            points,
            [car_box()],
            frame_id="1",
            variant=VariantSpec(
                "C1", ClusteringPolicy(mode="fixed", eps=0.6, min_points=3)
            ),
        )
        result = build_point_retention_day1({"C0": [c0], "C1": [c1]})
        self.assertEqual(result["delta_gt_count"], 1)
        self.assertEqual(result["delta_records"][0]["stage_point_counts"]["raw"], 3)
        self.assertEqual(
            result["delta_records"][0]["source_of_truth"],
            "candidate_conversion.evidence.stage_point_counts",
        )

    def test_pca_status_distinguishes_insufficient_degenerate_and_valid(self):
        self.assertEqual(pca_input_status(np.asarray([[0.0, 0.0], [1.0, 0.0]])), "insufficient_points")
        self.assertEqual(
            pca_input_status(np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
            "degenerate_geometry",
        )
        self.assertEqual(
            pca_input_status(np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
            "valid",
        )

    def test_stage_count_validation_rejects_non_monotonic_counts(self):
        with self.assertRaisesRegex(ValueError, "monotonically"):
            validate_stage_point_counts({
                "raw": 3, "roi": 4, "z_filter": 2, "intensity_filter": 1,
            })

    def test_batch_freezes_delta_and_p1_control_keys(self):
        def report(variant, cluster_ids, group):
            groups = [] if group is None else [{"group": group, "associated_gt_ids": ["gt_1"]}]
            return {
                "frame_id": "000001",
                "candidate_conversion": {"evidence": [{
                    "frame_id": "000001", "gt_id": "gt_1", "cluster_ids": cluster_ids,
                    "distance_bin": "near_0_15",
                    "stage_point_counts": {"raw": 5, "roi": 5, "z_filter": 4, "intensity_filter": 3},
                }]},
                "cluster_separability": {"records": groups},
            }

        result = build_point_retention_day1({
            "C0": [report("C0", [], None)],
            "C1": [report("C1", ["cluster_1"], "P1")],
        })
        self.assertEqual(result["delta_gt_count"], 1)
        self.assertEqual(result["p1_control_gt_count"], 1)
        self.assertIn("ROI_FILTER_LIMITED", result["attribution_contract"]["labels"])
        self.assertIn("O2_gt_clipped", result["oracle_ladder"])
        self.assertEqual(
            result["stage_count_source_of_truth"],
            "candidate_conversion.evidence.stage_point_counts",
        )


if __name__ == "__main__":
    unittest.main()
