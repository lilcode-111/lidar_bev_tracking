import unittest

import numpy as np

from bev_tracking.point_retention import build_pca_oracle
from bev_tracking.v15_5_seed_support import (
    V155SeedSupportError,
    build_delta22_representation_record,
    summarize_delta22_representations,
)


def source_record(indices, points, gt_box):
    oracle = build_pca_oracle(points[np.asarray(indices, dtype=np.int64)], gt_box)
    return {
        "point_identity": "raw_lidar_point_index",
        "stages": {
            "intensity_filter": {
                "count": len(indices),
                "source_point_indices": list(indices),
            }
        },
        "post_intensity_diagnostic_pca": oracle,
    }


class V155Phase0Day3Test(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray([
            [0.0, 0.0, 0.0, 0.50],
            [2.0, 0.0, 0.0, 0.45],
            [0.0, 1.0, 0.0, 0.35],
            [2.0, 1.0, 0.0, 0.25],
            [1.0, 0.5, 0.0, 0.20],
        ])
        self.gt = {"x": 1.0, "y": 0.5, "length": 2.0, "width": 1.0, "yaw": 0.0}

    def test_global_support_is_intersected_with_gt_hm_identity(self):
        result = build_delta22_representation_record(
            key=("000001", "gt_1"),
            raw_points=self.points,
            gt_box=self.gt,
            t0_record=source_record([0, 1], self.points, self.gt),
            t2_record=source_record([0, 1, 2, 3, 4], self.points, self.gt),
            globally_supported_hm=np.asarray([2, 4, 99]),
        )
        self.assertEqual(result["point_counts"]["H_M_available"], 3)
        self.assertEqual(result["point_counts"]["H_M_seed_supported"], 2)
        self.assertEqual(result["point_counts"]["Seed_Supported"], 4)
        self.assertEqual(result["point_counts"]["T2"], 5)

    def test_source_index_not_coordinate_value_controls_identity(self):
        points = self.points.copy()
        points[3, :3] = points[2, :3]
        result = build_delta22_representation_record(
            key=("000001", "gt_1"),
            raw_points=points,
            gt_box=self.gt,
            t0_record=source_record([0, 1], points, self.gt),
            t2_record=source_record([0, 1, 2, 3], points, self.gt),
            globally_supported_hm=np.asarray([2, 3]),
        )
        self.assertEqual(result["point_counts"]["H_M_seed_supported"], 2)
        self.assertEqual(result["point_counts"]["Seed_Supported"], 4)

    def test_t0_must_be_source_identity_subset_of_t2(self):
        with self.assertRaisesRegex(V155SeedSupportError, "not a T2 subset"):
            build_delta22_representation_record(
                key=("000001", "gt_1"),
                raw_points=self.points,
                gt_box=self.gt,
                t0_record=source_record([0, 4], self.points, self.gt),
                t2_record=source_record([0, 1, 2], self.points, self.gt),
                globally_supported_hm=np.asarray([2]),
            )

    def test_summary_reports_thresholds_material_recovery_and_retention(self):
        def item(gt_id, seed_iou, t2_iou):
            return {
                "frame_id": "000001", "gt_id": gt_id,
                "point_counts": {"T0": 2, "Seed_Supported": 4, "T2": 5},
                "representations": {
                    "T0": {"iou": 0.10},
                    "Seed_Supported": {"iou": seed_iou},
                    "T2": {"iou": t2_iou},
                },
                "material_recovery_vs_T0": {
                    "Seed_Supported": seed_iou - 0.10 >= 0.10,
                    "T2": t2_iou - 0.10 >= 0.10,
                },
            }
        summary = summarize_delta22_representations([
            item("gt_1", 0.30, 0.40),
            item("gt_2", 0.15, 0.35),
        ])
        self.assertEqual(summary["Seed_Supported"]["iou_ge_0_25_count"], 1)
        self.assertEqual(summary["T2"]["material_recovery_vs_T0_count"], 2)
        self.assertEqual(summary["recovery_retention_vs_T2"]["retained_T2_material_recovery_count"], 1)
        self.assertEqual(summary["recovery_retention_vs_T2"]["retention_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
