import unittest

from bev_tracking.adaptive_experiment import build_variant_diagnostics


class CandidateConversionCliIntegrationTest(unittest.TestCase):
    def test_variant_diagnostics_includes_candidate_conversion_analysis(self):
        reports = [
            {
                "frame_id": "1",
                "summary": {
                    "metrics_by_iou": {
                        "0.50": {"tp": 0, "fp": 0, "fn": 1, "effective_car_detection_count": 0},
                        "0.25": {"tp": 0, "fp": 0, "fn": 1},
                    },
                    "num_positive_gt": 1,
                    "candidate_generation": {},
                    "primary_reason_counts": {},
                },
                "gt_candidate_records": [],
                "candidate_conversion": {
                    "evidence": [
                        {
                            "distance_bin": "mid_15_30",
                            "stage_point_counts": {"intensity_filter": 20},
                            "cluster_ids": [],
                            "car_detection_ids_before_nms": [],
                            "car_detection_ids_after_nms": [],
                            "best_iou_after_nms": 0.0,
                            "matched_at_primary_iou": False,
                            "terminal_state": "no_associated_cluster",
                            "candidate_branches": [],
                        }
                    ]
                },
            }
        ]
        result = build_variant_diagnostics(reports)
        analysis = result["candidate_conversion_analysis"]
        self.assertEqual(analysis["waterfall"]["total"]["counts"]["num_positive_gt"], 1)
        self.assertEqual(analysis["terminal_state_counts"]["total"]["no_associated_cluster"], 1)


if __name__ == "__main__":
    unittest.main()
