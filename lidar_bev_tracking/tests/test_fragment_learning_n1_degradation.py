import unittest

from bev_tracking.fragment_learning_n1_degradation import (
    _conclusion,
    _score_movement,
)


class FragmentLearningN1DegradationTest(unittest.TestCase):
    def test_score_movement_uses_fixed_threshold_and_signed_delta(self):
        rows = [
            {"M1_score": 0.4, "M2_score": 0.6, "Delta_score": 0.2,
             "M1_prediction_at_0_50": 0, "M2_prediction_at_0_50": 1},
            {"M1_score": 0.7, "M2_score": 0.3, "Delta_score": -0.4,
             "M1_prediction_at_0_50": 1, "M2_prediction_at_0_50": 0},
            {"M1_score": 0.8, "M2_score": 0.9, "Delta_score": 0.1,
             "M1_prediction_at_0_50": 1, "M2_prediction_at_0_50": 1},
            {"M1_score": 0.2, "M2_score": 0.1, "Delta_score": -0.1,
             "M1_prediction_at_0_50": 0, "M2_prediction_at_0_50": 0},
        ]
        result = _score_movement(rows)
        self.assertEqual(result["score_increased_count"], 2)
        self.assertEqual(result["score_decreased_count"], 2)
        self.assertEqual(result["M1_below_to_M2_at_or_above_0_50"], 1)
        self.assertEqual(result["M1_at_or_above_to_M2_below_0_50"], 1)
        self.assertEqual(result["both_at_or_above_0_50"], 1)
        self.assertEqual(result["both_below_0_50"], 1)

    def test_boundary_diagnosis_requires_gain_link_and_no_scene_concentration(self):
        materiality = {
            "groups": {
                "ALL_N1": {"max_material_gain": {"P50": 0.03}},
                "SIGNIFICANT_SCORE_UP_N1": {"max_material_gain": {"P50": 0.07}},
            },
            "spearman": {"max_material_gain_vs_Delta_score": {"rho": 0.3}},
        }
        folds = [
            {"fold_id": 1, "Delta_AP": -0.1},
            {"fold_id": 2, "Delta_AP": -0.1},
            {"fold_id": 3, "Delta_AP": -0.1},
            {"fold_id": 4, "Delta_AP": 0.1},
            {"fold_id": 5, "Delta_AP": 0.1},
        ]
        scene = {"concentration": {
            "M2_N1_FP": {"top_3_frames": {"ratio": 0.3}},
            "significant_score_up_count": {"top_3_frames": {"ratio": 0.4}},
        }}
        result = _conclusion(materiality, folds, scene)
        self.assertEqual(
            result["N1_RANKING_DEGRADATION_DIAGNOSIS"],
            "MATERIALITY_BOUNDARY_CONFLICT",
        )

    def test_two_bad_folds_and_three_nonbad_folds_support_concentration(self):
        materiality = {
            "groups": {
                "ALL_N1": {"max_material_gain": {"P50": 0.03}},
                "SIGNIFICANT_SCORE_UP_N1": {"max_material_gain": {"P50": 0.04}},
            },
            "spearman": {"max_material_gain_vs_Delta_score": {"rho": 0.1}},
        }
        folds = [
            {"fold_id": 1, "Delta_AP": -0.11},
            {"fold_id": 2, "Delta_AP": 0.01},
            {"fold_id": 3, "Delta_AP": 0.12},
            {"fold_id": 4, "Delta_AP": 0.01},
            {"fold_id": 5, "Delta_AP": -0.12},
        ]
        scene = {"concentration": {
            "M2_N1_FP": {"top_3_frames": {"ratio": 0.26}},
            "significant_score_up_count": {"top_3_frames": {"ratio": 0.38}},
        }}
        result = _conclusion(materiality, folds, scene)
        self.assertEqual(
            result["N1_RANKING_DEGRADATION_DIAGNOSIS"],
            "FOLD / SCENE CONCENTRATED",
        )

    def test_overlapping_mechanisms_are_mixed(self):
        materiality = {
            "groups": {
                "ALL_N1": {"max_material_gain": {"P50": 0.03}},
                "SIGNIFICANT_SCORE_UP_N1": {"max_material_gain": {"P50": 0.07}},
            },
            "spearman": {"max_material_gain_vs_Delta_score": {"rho": 0.3}},
        }
        folds = []
        scene = {"concentration": {
            "M2_N1_FP": {"top_3_frames": {"ratio": 0.6}},
            "significant_score_up_count": {"top_3_frames": {"ratio": 0.2}},
        }}
        result = _conclusion(materiality, folds, scene)
        self.assertEqual(
            result["N1_RANKING_DEGRADATION_DIAGNOSIS"], "MIXED / INCONCLUSIVE"
        )


if __name__ == "__main__":
    unittest.main()
