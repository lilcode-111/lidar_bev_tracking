import unittest
import json
from unittest.mock import patch

from bev_tracking.v15_4_formal import V154FormalRunError, build_matrix_identity_audit, require_passed_t0_gate
from bev_tracking.v15_4_low_memory_audit import finalize_release_artifact
from bev_tracking.v15_4_materialization import load_json


def report(points, matched, after):
    value = {
        "source": {"source_run_id": "v15_4_formal_t0_aaaaaaaaaaaa"},
        "summary": {"num_frames": 100, "candidate_generation_totals": {"effective_car_detection_count": 10, "car_candidate_count_before_nms": 12}},
        "source_point_universes": [{"frame_id": "000001", "universes": {
            "global_post_intensity": {"source_point_indices": points},
            "positive_gt_post_intensity": {"source_point_indices": points[:1]},
            "annotation_excluded_background_post_intensity": {"source_point_indices": points[1:]},
        }}],
        "gt_candidate_records": [{"frame_id": "000001", "gt_id": "gt_1",
            "matched_by_iou": {"0.50": matched, "0.25": matched}, "cluster_ids": ["c"],
            "car_detection_ids_before_nms": ["d"], "car_detection_ids_after_nms": after,
            "best_iou_after_nms": 0.6 if matched else 0.2}],
        "source_point_identity_records": [{"frame_id": "000001", "gt_id": "gt_1", "post_intensity_diagnostic_pca": {"iou": 0.3 if matched else 0.1}}],
        "strict_background_lineage": {"records": []},
        "frames": [{"failure_evidence": {"summary": {"metrics_by_iou": {
            "0.50": {"tp": int(matched), "fp": 1, "fn": int(not matched), "neutralized_detections": 0},
            "0.25": {"tp": int(matched), "fp": 1, "fn": int(not matched), "neutralized_detections": 0},
        }}}}],
    }
    return value


class V154Phase2Day3Test(unittest.TestCase):
    def test_matrix_requires_passed_t0_gate(self):
        with self.assertRaises(V154FormalRunError):
            require_passed_t0_gate({"schema_version": "15.4-t0-replay-gate-v1", "status": "FAIL"})

    def test_matrix_builds_monotonicity_and_regression_audits(self):
        reports = {
            "T0": report([1], True, ["d"]),
            "T1": report([1, 2], True, ["d"]),
            "T2": report([1, 2, 3], False, []),
            "T_off": report([1, 2, 3, 4], False, []),
        }
        audit = build_matrix_identity_audit(reports)
        self.assertEqual(audit["source_point_monotonicity"]["status"], "PASS")
        self.assertEqual(audit["comparisons_vs_T0"]["T2"]["tp_regression"]["by_iou"]["0.50"]["TP_regressed_GT"]["count"], 1)

    def test_matrix_rejects_non_monotonic_source_identity(self):
        reports = {"T0": report([1], True, ["d"]), "T1": report([2], True, ["d"]), "T2": report([2], True, ["d"]), "T_off": report([2], True, ["d"])}
        with self.assertRaises(ValueError):
            build_matrix_identity_audit(reports)

    def test_finalize_is_json_serializable_and_uses_frozen_gates(self):
        reports = {"T0": report([1], True, ["d"]), "T1": report([1, 2], True, ["d"]), "T2": report([1, 2, 3], True, ["d"]), "T_off": report([1, 2, 3, 4], True, ["d"])}
        for name, value in reports.items():
            value["source"]["source_run_id"] = f"v15_4_formal_{name.lower()}_aaaaaaaaaaaa"
        audit = build_matrix_identity_audit(reports)
        audit["audit_recovery_commit"] = "b" * 40
        objects = {"audit": audit, **reports}
        gate = load_json("configs/experiments/v15_4/v15_4_release_gate.json")
        schedule = load_json("configs/experiments/v15_4/threshold_schedule.json")
        with patch("bev_tracking.v15_4_low_memory_audit.load_json", side_effect=lambda path: objects[str(path)]):
            artifact = finalize_release_artifact({name: name for name in reports}, "audit", gate, schedule, "a" * 40, "b" * 40)
        json.dumps(artifact)
        self.assertEqual(artifact["schema_version"], "15.4-intensity-filter-ablation-v1")
        self.assertIn("release_gate_results", artifact)


if __name__ == "__main__":
    unittest.main()
