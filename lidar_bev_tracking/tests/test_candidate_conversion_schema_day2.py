import unittest

from bev_tracking.adaptive_experiment import build_variant_diagnostics


def evidence(frame_id="000001", gt_id="gt_1", distance_bin="near_0_15"):
    return {
        "frame_id": frame_id,
        "gt_id": gt_id,
        "distance_bin": distance_bin,
        "stage_point_counts": {"intensity_filter": 10},
        "cluster_ids": ["cluster_1"],
        "car_detection_ids_before_nms": ["det_1"],
        "car_detection_ids_after_nms": ["det_1"],
        "best_iou_after_nms": 0.6,
        "matched_by_iou": {"0.50": True, "0.25": True},
        "matched_at_primary_iou": True,
        "terminal_state": "matched_at_0_50",
        "candidate_branches": [],
    }


class CandidateConversionSchemaDay2Test(unittest.TestCase):
    def test_diagnostics_use_canonical_evidence_without_gt_records(self):
        report = {
            "frame_id": "000001",
            "summary": {
                "metrics_by_iou": {
                    "0.50": {"tp": 1, "fp": 0, "fn": 0, "effective_car_detection_count": 1},
                    "0.25": {"tp": 1, "fp": 0, "fn": 0},
                },
                "candidate_generation": {},
                "primary_reason_counts": {},
            },
            "candidate_conversion": {"evidence": [evidence()]},
        }
        output = build_variant_diagnostics([report])
        self.assertNotIn("gt_records", output)
        self.assertEqual(output["candidate_conversion_evidence"], [evidence()])
        waterfall = output["candidate_conversion_analysis"]["waterfall"]
        self.assertEqual(waterfall["near_0_15"]["counts"]["num_positive_gt"], 1)
        self.assertNotIn("unknown", output["candidate_conversion_analysis"]["terminal_state_counts"]["near_0_15"])
        self.assertEqual(
            output["candidate_conversion_analysis"]["terminal_state_counts"]["near_0_15"],
            {"matched_at_0_50": 1},
        )
        self.assertEqual(waterfall["total"]["counts"]["gt_matched_at_iou_0_50"], 1)

    def test_missing_canonical_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            build_variant_diagnostics([{"frame_id": "000001", "summary": {}}])


if __name__ == "__main__":
    unittest.main()
