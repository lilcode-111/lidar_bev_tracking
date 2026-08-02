import unittest

import numpy as np

from bev_tracking.failure_evidence import (
    build_failure_evidence_report,
    summarize_gt_candidate_records,
)


def car_box(box_id, x):
    return {
        "id": box_id,
        "class_name": "car",
        "x": float(x),
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": 0.0,
    }


def dense_car_points(count=600, seed=11):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(9.0, 11.0, count),
            rng.uniform(-0.5, 0.5, count),
            rng.uniform(-0.4, 0.4, count),
            np.full(count, 0.8),
        ]
    ).astype(np.float32)


class GtCandidateCoverageTest(unittest.TestCase):
    def test_report_records_candidate_chain_for_every_positive_gt(self):
        report = build_failure_evidence_report(
            dense_car_points(),
            [car_box("gt_near", 10.0), car_box("gt_missing", 30.0)],
            frame_id="317",
            min_points=20,
            oriented=False,
        )

        records = {item["gt_id"]: item for item in report["gt_candidate_records"]}
        self.assertEqual(set(records), {"gt_near", "gt_missing"})
        self.assertTrue(records["gt_near"]["cluster_ids"])
        self.assertTrue(records["gt_near"]["raw_detection_ids"])
        self.assertTrue(records["gt_near"]["car_detection_ids_before_nms"])
        self.assertTrue(records["gt_near"]["car_detection_ids_after_nms"])
        self.assertEqual(records["gt_missing"]["candidate_outcome"], "no_cluster")
        self.assertEqual(records["gt_near"]["distance_bin"], "near_0_15")
        self.assertEqual(records["gt_missing"]["distance_bin"], "far_30_inf")

        coverage = report["summary"]["candidate_coverage"]
        self.assertEqual(coverage["counts"]["num_positive_gt"], 2)
        self.assertEqual(coverage["counts"]["gt_with_cluster"], 1)
        self.assertEqual(coverage["counts"]["gt_with_car_detection_after_nms"], 1)
        self.assertEqual(coverage["zero_detection_with_gt_eligible_count"], 1)
        self.assertEqual(coverage["zero_car_candidate_gt_count"], 1)

    def test_coverage_waterfall_uses_all_gt_as_denominator(self):
        records = [
            candidate_record("gt_1", iou=0.60, has_cluster=True, has_car=True),
            candidate_record("gt_2", iou=0.30, has_cluster=True, has_car=True),
            candidate_record("gt_3", iou=0.00, has_cluster=False, has_car=False),
        ]

        coverage = summarize_gt_candidate_records(records)

        self.assertEqual(coverage["counts"]["num_positive_gt"], 3)
        self.assertEqual(coverage["counts"]["gt_with_cluster"], 2)
        self.assertEqual(coverage["counts"]["gt_with_best_iou_ge_0_25"], 2)
        self.assertEqual(coverage["counts"]["gt_with_best_iou_ge_0_50"], 1)
        self.assertAlmostEqual(coverage["ratios"]["gt_with_cluster"], 2 / 3)
        self.assertEqual(coverage["zero_car_candidate_gt_count"], 1)

    def test_candidate_generation_separates_cluster_and_car_counts(self):
        report = build_failure_evidence_report(
            dense_car_points(),
            [car_box("gt_1", 10.0)],
            frame_id="317",
            min_points=20,
            oriented=False,
        )

        generation = report["summary"]["candidate_generation"]
        self.assertEqual(generation["cluster_count"], generation["raw_detection_count"])
        self.assertEqual(
            generation["raw_detection_count"],
            generation["car_candidate_count_before_nms"] + generation["non_car_candidate_count"],
        )
        self.assertEqual(
            generation["final_car_detection_count"],
            generation["effective_car_detection_count"],
        )


def candidate_record(gt_id, iou, has_cluster, has_car):
    cluster_ids = ["cluster_1"] if has_cluster else []
    car_ids = ["cluster_1"] if has_car else []
    return {
        "gt_id": gt_id,
        "cluster_ids": cluster_ids,
        "raw_detection_ids": cluster_ids,
        "car_detection_ids_before_nms": car_ids,
        "car_detection_ids_after_nms": car_ids,
        "best_iou_after_nms": iou,
        "candidate_outcome": "matched_at_primary_iou" if iou >= 0.5 else "no_cluster",
    }


if __name__ == "__main__":
    unittest.main()
