import copy
import unittest

import numpy as np

from bev_tracking.failure_evidence import (
    build_candidate_identity_records,
    build_source_point_identity_records,
)
from bev_tracking.v15_4_materialization import (
    V154MaterializationError,
    build_effective_config_matrix,
    build_t0_reference_artifacts,
    canonical_identity_sha256,
    load_json,
    validate_source_point_monotonicity,
    validate_t0_replay,
)


def schedule():
    return load_json("configs/experiments/v15_4/threshold_schedule.json")


def minimal_base_config():
    return {
        "data": {"root": "data"},
        "detector": {"eps": 0.6, "min_points": 20, "z_min": -0.9, "intensity_min": 0.38},
        "evaluation": {"iou_threshold": 0.5},
    }


class V154Day2ConfigAndPointIdentityTest(unittest.TestCase):
    def test_effective_configs_differ_only_by_intensity(self):
        matrix = build_effective_config_matrix(minimal_base_config(), schedule())
        self.assertEqual([matrix[name]["detector"]["intensity_min"] for name in matrix], [0.38, 0.30, 0.15, 0.0])
        self.assertEqual(matrix["T0"]["detector"]["z_min"], matrix["T_off"]["detector"]["z_min"])

    def test_source_point_monotonicity_uses_index_identity(self):
        values = {
            "T0": {"global": [("000001", 1)], "gt": [("000001", 1)], "background": []},
            "T1": {"global": [("000001", 1), ("000001", 2)], "gt": [("000001", 1)], "background": [("000001", 2)]},
            "T2": {"global": [("000001", 1), ("000001", 2), ("000001", 3)], "gt": [("000001", 1), ("000001", 3)], "background": [("000001", 2)]},
            "T_off": {"global": [("000001", 1), ("000001", 2), ("000001", 3), ("000001", 4)], "gt": [("000001", 1), ("000001", 3)], "background": [("000001", 2), ("000001", 4)]},
        }
        result = validate_source_point_monotonicity(values)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["coordinate_row_dedup_used"])

    def test_missing_source_index_fails_monotonicity(self):
        values = {name: {"global": [("000001", 1)]} for name in ("T0", "T1", "T2", "T_off")}
        values["T1"]["global"] = [("000001", 2)]
        with self.assertRaises(V154MaterializationError):
            validate_source_point_monotonicity(values)

    def test_candidate_lineage_has_stable_identity(self):
        clusters = [np.asarray([[1.0, 2.0, 0.0, 0.5]], dtype=np.float32)]
        raw = [{"id": "1", "class_name": "car"}]
        records = build_candidate_identity_records("7", clusters, raw, raw)
        self.assertEqual(records[0]["detection_identity"], "000007:1")
        self.assertTrue(records[0]["after_nms"])
        self.assertEqual(len(records[0]["stable_cluster_signature"]), 64)

    def test_requested_gt_source_indices_are_raw_row_indices(self):
        points = np.asarray([[1, 0, 0, .4], [1, 0, 0, .4], [8, 8, 0, .4]], dtype=np.float32)
        indices = np.asarray([0, 1, 2], dtype=np.int64)
        stages = {name: points for name in ("raw", "roi", "z_filter", "intensity_filter")}
        source = {name: indices for name in stages}
        gt = {"id": "gt_1", "x": 1.0, "y": 0.0, "z": 0.0, "length": 4.0, "width": 2.0, "height": 2.0, "yaw": 0.0}
        records = build_source_point_identity_records(
            frame_id="1", gt_boxes=[gt], stages=stages, stage_source_indices=source,
            requested_gt_keys=[["000001", "gt_1"]],
        )
        self.assertEqual(records[0]["stages"]["intensity_filter"]["source_point_indices"], [0, 1])


class V154Day2T0ReferenceTest(unittest.TestCase):
    def make_inputs(self):
        delta = [[f"{index:06d}", "gt_1"] for index in range(22)]
        identity = {
            "formal_100": {"num_frames": 100, "manifest_sha256": "m" * 64},
            "delta_22": {"ordered_identity_list": delta, "ordered_identity_sha256": canonical_identity_sha256(delta)},
        }
        source_records = []
        historical_records = []
        gt_records = []
        frames = []
        for index in range(100):
            frame_id = f"{index:06d}"
            metrics = {
                "0.50": {"tp": int(index == 0), "fp": int(index == 0), "fn": 0, "neutralized_detections": 0},
                "0.25": {"tp": int(index == 0), "fp": int(index == 0), "fn": 0, "neutralized_detections": 0},
            }
            frames.append({"failure_evidence": {"summary": {"metrics_by_iou": metrics}}})
            gt_records.append({"frame_id": frame_id, "gt_id": "gt_1", "matched_by_iou": {"0.50": index == 0, "0.25": index == 0}})
            if index < 22:
                source_records.append({
                    "frame_id": frame_id, "gt_id": "gt_1",
                    "stages": {"intensity_filter": {"count": 3, "source_point_indices": [1, 2, 3]}},
                    "post_intensity_diagnostic_pca": {"iou": 0.25},
                })
                historical_records.append({
                    "frame_id": frame_id, "gt_id": "gt_1",
                    "stage_point_counts": {"intensity_filter": 3}, "O3": {"iou": 0.25},
                })
        report = {
            "summary": {"num_frames": 100, "geometry_failed_frames": 0},
            "source": {"requested_frame_ids": [f"{index:06d}" for index in range(100)]},
            "source_point_identity_records": source_records,
            "gt_candidate_records": gt_records,
            "candidate_identity_records": [{"detection_identity": "000000:1", "is_car_candidate_before_nms": True, "after_nms": True}],
            "frames": frames,
        }
        expected_metric = {"tp": 1, "fp": 1, "fn": 0, "neutralized_detections": 0, "precision": 0.5, "recall": 1.0, "f1": 2 / 3}
        declaration = {"metrics_by_iou": {"0.50": expected_metric, "0.25": expected_metric}}
        historical = {"corrected_point_retention_day3": {"delta_records": historical_records}}
        return report, historical, identity, declaration

    def test_builds_complete_t0_references_without_non_t0_results(self):
        t0_25, t0_100 = build_t0_reference_artifacts(*self.make_inputs())
        self.assertEqual(t0_25["record_count"], 22)
        self.assertEqual(t0_25["historical_match_status"], "PASS")
        self.assertEqual(t0_100["num_frames"], 100)
        self.assertEqual(t0_100["candidate_identity_set"], ["000000:1"])

    def test_geometry_sanity_failure_is_recorded_but_does_not_block_reference(self):
        inputs = list(self.make_inputs())
        inputs[0]["summary"]["geometry_failed_frames"] = 3
        _, t0_100 = build_t0_reference_artifacts(*inputs)
        audit = t0_100["geometry_sanity_audit"]
        self.assertEqual(audit["failed_frames"], 3)
        self.assertEqual(audit["passed_frames"], 97)
        self.assertEqual(audit["role"], "non_blocking_diagnostic")
        self.assertEqual(audit["status"], "OBSERVED_WITH_FAILURES")
        self.assertFalse(audit["blocks_t0_reference"])

    def test_t0_replay_uses_tolerance_only_for_float_fields(self):
        t0_25, _ = build_t0_reference_artifacts(*self.make_inputs())
        replay = copy.deepcopy(t0_25)
        replay["records"][0]["O3_iou"] += 1e-9
        self.assertEqual(validate_t0_replay(t0_25, replay)["status"], "PASS")
        replay["records"][0]["post_intensity_point_count"] += 1
        with self.assertRaises(V154MaterializationError):
            validate_t0_replay(t0_25, replay)

    def test_t0_replay_reports_schema_mismatch_once(self):
        reference = {"schema_version": "15.4-t0-25-reference-v1", "value": 1}
        replay = {"schema_version": "15.4-t0-25-reference-v2", "value": 1}
        with self.assertRaises(V154MaterializationError) as context:
            validate_t0_replay(reference, replay)
        self.assertEqual(
            str(context.exception),
            "T0 replay mismatch: schema_version",
        )


if __name__ == "__main__":
    unittest.main()
