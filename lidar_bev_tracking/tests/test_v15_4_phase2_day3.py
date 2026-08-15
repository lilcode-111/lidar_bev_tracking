import unittest

from bev_tracking.v15_4_formal import V154FormalRunError, build_matrix_identity_audit, require_passed_t0_gate


def report(points, matched, after):
    return {
        "source_point_universes": [{"frame_id": "000001", "universes": {
            "global_post_intensity": {"source_point_indices": points},
            "positive_gt_post_intensity": {"source_point_indices": points[:1]},
            "annotation_excluded_background_post_intensity": {"source_point_indices": points[1:]},
        }}],
        "gt_candidate_records": [{"frame_id": "000001", "gt_id": "gt_1",
            "matched_by_iou": {"0.50": matched, "0.25": matched}, "cluster_ids": ["c"],
            "car_detection_ids_before_nms": ["d"], "car_detection_ids_after_nms": after,
            "best_iou_after_nms": 0.6 if matched else 0.2}],
    }


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


if __name__ == "__main__":
    unittest.main()
