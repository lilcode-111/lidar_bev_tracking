import unittest

import numpy as np

from bev_tracking.v15_5_geometry_critical import (
    build_far_comparison_summary,
    build_far_frame_stability,
    feature_correlations,
    offline_pca_extent_features,
    range_segment_from_value,
    summarize_marginal_gain,
)


class V155GeometryCriticalTest(unittest.TestCase):
    def test_gt_range_boundaries_are_frozen(self):
        self.assertEqual(range_segment_from_value(14.999), "near")
        self.assertEqual(range_segment_from_value(15.0), "mid")
        self.assertEqual(range_segment_from_value(29.999), "mid")
        self.assertEqual(range_segment_from_value(30.0), "far")

    def test_far_frame_stability_excludes_zero_denominator_frames(self):
        frames = {
            "000001": self._frame(H=((10, 8), (20, 10)), M=((0, 0), (10, 5))),
            "000002": self._frame(H=((10, 2), (20, 10)), M=((10, 8), (0, 0))),
        }
        result = build_far_frame_stability(frames)
        h = result["H"]["frame_stability"]
        self.assertEqual(h["comparable_frame_count"], 2)
        self.assertEqual(h["positive_frame_count"], 1)
        self.assertEqual(h["negative_frame_count"], 1)
        m = result["M"]["frame_stability"]
        self.assertEqual(m["comparable_frame_count"], 0)
        self.assertEqual(m["non_comparable_frame_count"], 2)

    def test_concentration_is_reported_for_vehicle_and_background_separately(self):
        frames = {
            "000001": self._frame(H=((10, 8), (20, 5)), M=((1, 1), (1, 1))),
            "000002": self._frame(H=((10, 2), (20, 5)), M=((1, 1), (1, 1))),
        }
        result = build_far_frame_stability(frames)["H"]["frame_stability"]
        self.assertEqual(result["vehicle_rescue_concentration"]["top1_frame_share"], 0.8)
        self.assertEqual(result["background_rescue_concentration"]["top1_frame_share"], 0.5)

    def test_offline_extent_detects_point_beyond_current_axis(self):
        base = np.asarray([[-1.0, -0.5], [-1.0, 0.5], [1.0, -0.5], [1.0, 0.5]])
        result = offline_pca_extent_features(base, np.asarray([2.0, 0.0]))
        self.assertAlmostEqual(result["distance_to_current_axis_extent"], 1.0)
        self.assertEqual(result["role"], "offline_oracle_explanation_only")

    def test_far_comparison_uses_uniform_count_and_seed_supported_count(self):
        frames = {
            "000001": self._frame(H=((10, 8), (20, 5)), M=((4, 1), (8, 2))),
        }
        stability = build_far_frame_stability(frames)
        record = {
            "frame_id": "000001", "gt_id": "gt_1",
            "representations": {
                "T0": {"iou": 0.1},
                "R_far_seed": {"iou": 0.2},
                "R_far_uniform": {"iou": 0.3},
                "T2": {"iou": 0.4},
            },
            "material_recovery_vs_T0": {
                "R_far_seed": True, "R_far_uniform": True, "T2": True,
            },
        }
        modes = build_far_comparison_summary(stability, [record])["modes"]
        self.assertEqual(modes["R_far_uniform"]["vehicle_rescue_count"], 14)
        self.assertEqual(modes["R_far_seed"]["vehicle_rescue_count"], 9)
        self.assertEqual(modes["R_far_uniform"]["background_rescue_count"], 28)
        self.assertEqual(modes["R_far_seed"]["background_rescue_count"], 7)
        self.assertEqual(modes["R_far_uniform"]["material_recovery_count"], 1)

    def test_summary_does_not_define_high_gain_threshold(self):
        records = [self._marginal("gt_1", 0.2, 1.0), self._marginal("gt_2", -0.1, 2.0)]
        result = summarize_marginal_gain(records)
        distribution = result["G_p_distribution"]
        self.assertEqual(distribution["positive_count"], 1)
        self.assertEqual(distribution["negative_count"], 1)
        self.assertFalse(distribution["high_G_threshold_defined"])
        self.assertAlmostEqual(
            result["runtime_feature_spearman_correlation"]["range_xy"]["spearman_rho"], -1.0
        )

    def test_constant_feature_correlation_is_null(self):
        records = [self._marginal("gt_1", 0.1, 1.0), self._marginal("gt_2", 0.2, 1.0)]
        output = feature_correlations(records, "runtime_features", ("range_xy",))
        self.assertIsNone(output["range_xy"]["spearman_rho"])

    @staticmethod
    def _frame(H, M):
        def band(values):
            vehicle, background = values
            return {
                "vehicle": {"point_count": vehicle[0], "seed_supported_count": vehicle[1]},
                "background": {"point_count": background[0], "seed_supported_count": background[1]},
            }
        return {"H": band(H), "M": band(M)}

    @staticmethod
    def _marginal(gt_id, gain, feature):
        runtime = {
            "intensity": feature,
            "range_xy": feature,
            "d_seed": feature,
            "n_seed_0p6": feature,
            "local_neighbor_count_0p6": feature,
            "nearest_neighbor_distance": feature,
        }
        offline = {
            "projection_on_seed_pca_axis_1": feature,
            "projection_on_seed_pca_axis_2": feature,
            "distance_to_current_axis_extent": feature,
            "distance_to_axis_1_extent": feature,
            "distance_to_axis_2_extent": feature,
        }
        return {
            "frame_id": "000001",
            "gt_id": gt_id,
            "raw_lidar_point_index": 1,
            "band": "H",
            "G_p": gain,
            "runtime_features": runtime,
            "offline_oracle_geometry_features": offline,
        }


if __name__ == "__main__":
    unittest.main()
