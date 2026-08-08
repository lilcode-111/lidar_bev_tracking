import tempfile
import unittest
from pathlib import Path

from bev_tracking.adaptive_experiment import aggregate_distance_analysis, variant_specs_from_config


class ClusteringAblationConfigTest(unittest.TestCase):
    def test_config_parser_requires_all_four_variants(self):
        with self.assertRaises(ValueError):
            variant_specs_from_config({"variants": {"C0": {"mode": "fixed"}}})

    def test_config_parser_keeps_adaptive_distance_params(self):
        adaptive = {
            "mode": "adaptive",
            "distance_params": {
                "near_0_15": {"eps": 0.5, "min_points": 20},
                "mid_15_30": {"eps": 0.65, "min_points": 15},
                "far_30_inf": {"eps": 0.85, "min_points": 8},
            },
        }
        config = {
            "variants": {
                "C0": {"mode": "fixed", "eps": 0.6, "min_points": 20},
                "C1": adaptive,
                "C2": adaptive,
                "C3": adaptive,
            }
        }
        specs = variant_specs_from_config(config)
        self.assertEqual(specs["C2"].policy.distance_params["far_30_inf"]["eps"], 0.85)

    def test_frozen_preregistered_parameter_matrix(self):
        config = {
            "variants": {
                "C0": {"mode": "fixed", "eps": 0.60, "min_points": 20},
                "C1": {
                    "mode": "adaptive",
                    "distance_params": {
                        "near_0_15": {"eps": 0.60, "min_points": 16},
                        "mid_15_30": {"eps": 0.60, "min_points": 10},
                        "far_30_inf": {"eps": 0.60, "min_points": 6},
                    },
                },
                "C2": {
                    "mode": "adaptive",
                    "distance_params": {
                        "near_0_15": {"eps": 0.50, "min_points": 20},
                        "mid_15_30": {"eps": 0.70, "min_points": 20},
                        "far_30_inf": {"eps": 0.85, "min_points": 20},
                    },
                },
                "C3": {
                    "mode": "adaptive",
                    "distance_params": {
                        "near_0_15": {"eps": 0.50, "min_points": 16},
                        "mid_15_30": {"eps": 0.70, "min_points": 10},
                        "far_30_inf": {"eps": 0.85, "min_points": 6},
                    },
                },
            }
        }
        specs = variant_specs_from_config(config)
        self.assertEqual(specs["C1"].policy.distance_params["far_30_inf"]["min_points"], 6)
        self.assertEqual(specs["C2"].policy.distance_params["mid_15_30"]["eps"], 0.70)
        self.assertEqual(specs["C3"].policy.distance_params["near_0_15"]["min_points"], 16)

    def test_distance_analysis_keeps_gt_level_evidence(self):
        reports = [
            {
                "gt_candidate_records": [
                    {
                        "distance_bin": "far_30_inf",
                        "matched_by_iou": {"0.50": False, "0.25": True},
                        "cluster_ids": [],
                        "car_detection_ids_after_nms": [],
                        "candidate_outcome": "no_cluster",
                    }
                ]
            }
        ]
        result = aggregate_distance_analysis(reports)
        far = result["far_30_inf"]
        self.assertEqual(far["gt_count"], 1)
        self.assertEqual(far["fn_iou_0_50"], 1)
        self.assertEqual(far["tp_iou_0_25"], 1)
        self.assertEqual(far["zero_detection_with_gt"], 1)


if __name__ == "__main__":
    unittest.main()
