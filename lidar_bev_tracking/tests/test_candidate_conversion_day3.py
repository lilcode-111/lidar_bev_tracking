import unittest

from bev_tracking.candidate_conversion import summarize_candidate_conversion_records


def record(distance_bin, points, axis_length, car, iou, matched=False):
    return {
        "distance_bin": distance_bin,
        "stage_point_counts": {"intensity_filter": points},
        "cluster_ids": ["cluster_1"],
        "car_detection_ids_before_nms": ["cluster_1"] if car else [],
        "car_detection_ids_after_nms": ["cluster_1"] if car else [],
        "best_iou_after_nms": iou,
        "matched_at_primary_iou": matched,
        "terminal_state": "matched_at_0_50" if matched else "rejected_by_car_classifier",
        "candidate_branches": [
            {
                "is_car_candidate": car,
                "cluster_features": {
                    "num_points": points,
                    "axis_length": axis_length,
                    "axis_width": 1.0,
                    "height_span": 1.2,
                    "range_xy_m": 20.0,
                    "point_density_xy": 10.0,
                },
            }
        ],
    }


class CandidateConversionDay3Test(unittest.TestCase):
    def test_waterfall_is_reported_by_distance_and_total(self):
        result = summarize_candidate_conversion_records([
            record("near_0_15", 40, 2.2, True, 0.7, True),
            record("far_30_inf", 8, 1.4, False, 0.0),
        ])
        near = result["waterfall"]["near_0_15"]["counts"]
        far = result["waterfall"]["far_30_inf"]["counts"]
        total = result["waterfall"]["total"]["counts"]
        self.assertEqual(near["gt_matched_at_iou_0_50"], 1)
        self.assertEqual(far["gt_with_car_before_nms"], 0)
        self.assertEqual(total["num_positive_gt"], 2)
        self.assertEqual(total["gt_with_associated_iou_ge_0_50"], 1)

    def test_classifier_audit_does_not_change_thresholds(self):
        result = summarize_candidate_conversion_records([
            record("mid_15_30", 120, 1.8, False, 0.2),
            record("mid_15_30", 600, 1.5, True, 0.5, True),
        ])
        gates = result["classification_audit"]["mid_15_30"]["gate_counts"]
        self.assertEqual(gates["associated_branch_count"], 2)
        self.assertEqual(gates["axis_size_pass_count"], 0)
        self.assertEqual(gates["point_count_pass_count"], 1)
        self.assertEqual(gates["both_classifier_gates_pass_count"], 1)
        self.assertEqual(gates["neither_classifier_gate_pass_count"], 1)

    def test_invalid_distance_bin_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_candidate_conversion_records([record("unknown", 10, 1.0, False, 0.0)])


if __name__ == "__main__":
    unittest.main()
