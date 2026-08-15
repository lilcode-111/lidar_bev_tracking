import copy
import unittest

from bev_tracking.v15_4_materialization import load_json
from bev_tracking.v15_4_release import (
    ARTIFACT_SCHEMA_VERSION,
    absolute_relative_delta,
    build_intensity_filter_ablation_skeleton,
    evaluate_variant_release_gates,
    select_release_candidate,
    validate_release_gate_config,
)


def gate_config():
    return load_json("configs/experiments/v15_4/v15_4_release_gate.json")


def validity():
    required = gate_config()["gates"]["Gate 0"]["required_invariants"]
    return {
        "invariants": {name: True for name in required},
        "formal_100": {"requested": 100, "success": 100, "metric_valid": 100, "partial_success": 0, "skipped": 0, "failed": 0},
    }


def baseline():
    return {
        "validity": validity(),
        "delta_22": {"iou_ge_0_25_count": 1},
        "candidate": {"positive_gt_count": 100},
        "metrics_by_iou": {"0.50": {"tp": 20, "fp": 10, "f1": 0.50}, "0.25": {"tp": 40, "fp": 10, "f1": 0.60}},
        "counts": {"effective_car_detection": 100, "raw_car_candidate_before_nms": 200, "strict_background_candidate_after_nms": 10},
    }


def passing_variant():
    return {
        "validity": validity(),
        "delta_22": {"material_recovery_count": 6, "median_iou_gain_vs_T0": 0.05, "iou_ge_0_25_count": 5, "material_regression_count": 2, "material_recovery_unique_frames": 3},
        "candidate": {"improved_gt_count": 2, "regressed_gt_count": 0},
        "metrics_by_iou": {"0.50": {"tp": 21, "fp": 11, "f1": 0.51}, "0.25": {"tp": 43, "fp": 11, "f1": 0.61}},
        "counts": {"effective_car_detection": 110, "raw_car_candidate_before_nms": 220, "strict_background_candidate_after_nms": 11},
        "regression": {"tp_regressed_gt_count": {"0.50": 0, "0.25": 1}, "unexplained_count": 0},
        "gain_distribution": {"0.50": {"new_tp_gt_count": 1, "new_tp_frame_count": 1}, "0.25": {"new_tp_gt_count": 3, "new_tp_frame_count": 2}},
    }


def selection_metric(name, **updates):
    item = {
        "variant": name,
        "f1_0_50": 0.51,
        "tp_0_50": 21,
        "f1_0_25": 0.61,
        "tp_0_25": 43,
        "fp_0_50": 11,
        "effective_car_detection_growth": 10,
        "effective_car_detection_count": 110,
        "strict_background_candidate_after_nms": 11,
        "candidate_regressed_gt_count": 0,
        "background_car_candidate_growth": 1,
        "intensity_min": 0.30 if name == "T1" else 0.15,
    }
    item.update(updates)
    return item


class V154ReleaseGateTest(unittest.TestCase):
    def test_frozen_release_gate_is_valid(self):
        self.assertEqual(validate_release_gate_config(gate_config())["gate_count"], 8)

    def test_all_gate_checks_are_offline_recomputable(self):
        result = evaluate_variant_release_gates(baseline(), passing_variant(), "T1", gate_config())
        self.assertTrue(result["release_candidate_qualified"])
        self.assertEqual(list(result["gate_results"]), ["Gate 0", "Gate A", "Gate B", "Gate C", "Gate D", "Gate E", "Gate F"])
        for gate in result["gate_results"].values():
            for check in gate["checks"]:
                self.assertEqual(set(check), {"metric", "baseline", "variant", "threshold", "operator", "computed_value", "status", "passed"})

    def test_gate_zero_failure_blocks_later_gate_evaluation(self):
        variant = passing_variant()
        variant["validity"]["invariants"]["t0_replay_100"] = False
        result = evaluate_variant_release_gates(baseline(), variant, "T1", gate_config())
        self.assertFalse(result["experiment_valid"])
        self.assertEqual(list(result["gate_results"]), ["Gate 0"])
        self.assertEqual(result["release_gate_status"], "not_evaluated")

    def test_baseline_zero_relative_growth_is_not_applicable(self):
        base = baseline()
        variant = passing_variant()
        base["metrics_by_iou"]["0.50"]["fp"] = 0
        variant["metrics_by_iou"]["0.50"]["fp"] = 2
        result = evaluate_variant_release_gates(base, variant, "T1", gate_config())
        checks = result["gate_results"]["Gate D"]["checks"]
        relative = next(item for item in checks if item["metric"] == "FP@0.50_relative_growth")
        self.assertEqual(relative["status"], "not_applicable")
        self.assertIsNone(relative["computed_value"])
        self.assertTrue(relative["passed"])
        self.assertIsNone(absolute_relative_delta(0, 2)["relative_delta"])

    def test_t_off_runs_validity_but_can_never_qualify(self):
        result = evaluate_variant_release_gates(baseline(), {"validity": validity()}, "T_off", gate_config())
        self.assertTrue(result["experiment_valid"])
        self.assertFalse(result["release_candidate_qualified"])
        self.assertIn("Gate G", result["gate_results"])


class V154SelectionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.qualified = {"T1": {"release_candidate_qualified": True}, "T2": {"release_candidate_qualified": True}}

    def test_pareto_dominant_candidate_is_selected(self):
        metrics = {"T1": selection_metric("T1"), "T2": selection_metric("T2", f1_0_50=0.50, tp_0_50=20, f1_0_25=0.60, tp_0_25=42, fp_0_50=12, effective_car_detection_growth=11, strict_background_candidate_after_nms=12)}
        result = select_release_candidate(self.qualified, metrics, gate_config())
        self.assertEqual(result["selected_release_candidate"], "T1")
        self.assertEqual(result["selection_path"], "pareto_dominance")

    def test_performance_equivalent_uses_cost_tiebreak(self):
        metrics = {
            "T1": selection_metric("T1", fp_0_50=12, strict_background_candidate_after_nms=10),
            "T2": selection_metric("T2", fp_0_50=11, strict_background_candidate_after_nms=11),
        }
        result = select_release_candidate(self.qualified, metrics, gate_config())
        self.assertEqual(result["selected_release_candidate"], "T2")
        self.assertTrue(result["performance_equivalent"])

    def test_non_equivalent_non_pareto_uses_frozen_lexicographic_order(self):
        metrics = {
            "T1": selection_metric("T1", f1_0_50=0.52, tp_0_50=20, fp_0_50=20),
            "T2": selection_metric("T2", f1_0_50=0.51, tp_0_50=25, fp_0_50=10),
        }
        result = select_release_candidate(self.qualified, metrics, gate_config())
        self.assertEqual(result["selected_release_candidate"], "T1")
        self.assertEqual(result["selection_path"], "non_pareto_non_equivalent_lexicographic")

    def test_no_candidate_falls_back_to_t0(self):
        failed = {"T1": {"release_candidate_qualified": False}, "T2": {"release_candidate_qualified": False}}
        result = select_release_candidate(failed, {}, gate_config())
        self.assertIsNone(result["selected_release_candidate"])
        self.assertEqual(result["safe_configuration"], "T0")


class V154ArtifactSchemaTest(unittest.TestCase):
    def test_phase1_skeleton_contains_no_formal_results(self):
        identity = load_json("configs/experiments/v15_4/pre_run_identity.json")
        artifact = build_intensity_filter_ablation_skeleton("schedule.json", "a" * 64, "gate.json", "b" * 64, identity)
        self.assertEqual(artifact["schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(list(artifact["variants"]), ["T0", "T1", "T2", "T_off"])
        self.assertFalse(artifact["formal_results_observed"])
        self.assertTrue(all(item["status"] == "NOT_RUN" for item in artifact["variants"].values()))


if __name__ == "__main__":
    unittest.main()
