import unittest

from bev_tracking.review_supplement import build_review_supplement_day2


def evidence(gt_id, distance_bin, terminal_state, after=None, downstream=None):
    return {
        "frame_id": "000001",
        "gt_id": gt_id,
        "distance_bin": distance_bin,
        "terminal_state": terminal_state,
        "car_detection_ids_after_nms": after or [],
        "downstream_attribution": downstream or {},
    }


class ReviewSupplementDay2Test(unittest.TestCase):
    def test_terminal_matrix_contains_all_states_by_distance(self):
        reports = {
            "C0": [{"candidate_conversion": {"evidence": [
                evidence("gt_1", "near_0_15", "no_associated_cluster"),
            ]}}],
            "C1": [{"candidate_conversion": {"evidence": [
                evidence("gt_1", "near_0_15", "rejected_by_car_classifier"),
            ]}}],
        }
        result = build_review_supplement_day2(reports)
        c1_near = result["terminal_state_matrix"]["C1"]["near_0_15"]
        self.assertEqual(c1_near["rejected_by_car_classifier"], 1)
        self.assertEqual(set(c1_near), {
            "no_filtered_points",
            "insufficient_filtered_points_for_association",
            "no_associated_cluster",
            "rejected_by_car_classifier",
            "removed_by_nms",
            "box_iou_below_0_25",
            "box_iou_0_25_to_0_50",
            "iou_ge_0_50_but_unmatched",
            "matched_at_0_50",
        })

    def test_downstream_summary_counts_competition_and_geometry(self):
        reports = {"C1": [{"candidate_conversion": {"evidence": [evidence(
            "gt_1", "mid_15_30", "box_iou_0_25_to_0_50", ["det_1"], {
                "geometry": {"candidates": [{
                    "center_error_m": 0.2,
                    "length_error_m": 0.3,
                    "width_error_m": 0.1,
                    "yaw_error_rad": 0.05,
                }]},
                "evaluation": {"competition_matches": [{"det_id": "det_1"}]},
            }
        )]}}]}
        summary = build_review_supplement_day2(reports)["downstream_summary"]["C1"]
        self.assertEqual(summary["associated_car_candidate_count_after_nms"], 1)
        self.assertEqual(summary["geometry_candidate_count"], 1)
        self.assertEqual(summary["evaluation_competition_gt_count"], 1)
        self.assertAlmostEqual(summary["geometry_error_mean"]["center_error_m"], 0.2)

    def test_unknown_terminal_state_is_rejected(self):
        reports = {"C0": [{"candidate_conversion": {"evidence": [
            evidence("gt_1", "near_0_15", "unknown"),
        ]}}]}
        with self.assertRaises(ValueError):
            build_review_supplement_day2(reports)


if __name__ == "__main__":
    unittest.main()
