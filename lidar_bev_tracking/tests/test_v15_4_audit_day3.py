import unittest

import numpy as np

from bev_tracking.v15_4_audit import (
    build_candidate_regression_audit,
    build_frame_source_point_universes,
    build_strict_background_lineage,
    build_tp_regression_audit,
    classify_regression_reasons,
    extract_monotonicity_universes,
)


def box(box_id, class_name, x, y):
    return {
        "id": box_id,
        "class_name": class_name,
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "length": 2.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": 0.0,
    }


def gt_record(
    frame_id,
    gt_id,
    *,
    matched_050,
    matched_025=None,
    clusters=None,
    before=None,
    after=None,
    best_iou=0.0,
):
    return {
        "frame_id": str(frame_id).zfill(6),
        "gt_id": gt_id,
        "matched_by_iou": {
            "0.50": bool(matched_050),
            "0.25": bool(matched_050 if matched_025 is None else matched_025),
        },
        "cluster_ids": list(clusters or []),
        "car_detection_ids_before_nms": list(before or []),
        "car_detection_ids_after_nms": list(after or []),
        "best_iou_after_nms": float(best_iou),
    }


class V154SourceUniverseTest(unittest.TestCase):
    def test_annotation_excluded_background_preserves_source_index_identity(self):
        points = np.asarray(
            [
                [5.0, 5.0, 0.0, 0.4],
                [5.0, 5.0, 0.0, 0.4],
                [0.0, 0.0, 0.0, 0.4],
                [8.0, 8.0, 0.0, 0.4],
            ],
            dtype=np.float32,
        )
        indices = np.asarray([10, 11, 12, 13], dtype=np.int64)
        stages = {"intensity_filter": points}
        source = {"intensity_filter": indices}
        annotations = [
            box("gt_1", "car", 0.0, 0.0),
            box("gt_2", "dontcare", 8.0, 8.0),
        ]
        result = build_frame_source_point_universes(
            frame_id="1", stages=stages, stage_source_indices=source, gt_boxes=annotations
        )
        universes = result["universes"]
        self.assertEqual(universes["global_post_intensity"]["source_point_indices"], [10, 11, 12, 13])
        self.assertEqual(universes["positive_gt_post_intensity"]["source_point_indices"], [12])
        self.assertEqual(
            universes["annotation_excluded_background_post_intensity"]["source_point_indices"],
            [10, 11, 13],
        )
        self.assertFalse(result["coordinate_row_dedup_used"])

    def test_extracts_three_monotonicity_universes_with_frame_identity(self):
        frame = {
            "frame_id": "000001",
            "universes": {
                "global_post_intensity": {"source_point_indices": [1, 2]},
                "positive_gt_post_intensity": {"source_point_indices": [1]},
                "annotation_excluded_background_post_intensity": {"source_point_indices": [2]},
            },
        }
        result = extract_monotonicity_universes({"source_point_universes": [frame]})
        self.assertEqual(result["global"], [("000001", 1), ("000001", 2)])
        self.assertEqual(result["positive_gt"], [("000001", 1)])
        self.assertEqual(result["annotation_excluded_background"], [("000001", 2)])


class V154StrictBackgroundLineageTest(unittest.TestCase):
    def test_lineage_excludes_positive_and_other_annotation_overlap(self):
        clusters = [
            np.asarray([[5.0, 5.0, 0.0, 0.4]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.0, 0.4]], dtype=np.float32),
            np.asarray([[2.0, 2.0, 0.0, 0.4]], dtype=np.float32),
        ]
        detections = [
            {"id": "1", "class_name": "car", "x": 5.0, "y": 5.0},
            {"id": "2", "class_name": "car", "x": 0.0, "y": 0.0},
            {"id": "3", "class_name": "car", "x": 2.0, "y": 2.0},
        ]
        evaluation = {
            "matches": [],
            "neutralized_detections": [],
            "false_positives": [{"det_id": "1", "det_index": 0}],
            "auxiliary": {"0.25": {"false_positives": [{"det_id": "1", "det_index": 0}]}},
        }
        result = build_strict_background_lineage(
            frame_id="1",
            clusters=clusters,
            raw_detections=detections,
            detections_after_nms=[detections[0]],
            gt_boxes=[box("gt_1", "car", 0, 0), box("gt_2", "pedestrian", 2, 2)],
            evaluation=evaluation,
        )
        self.assertEqual(result["record_count"], 1)
        record = result["records"][0]
        self.assertEqual(record["detection_identity"], "000001:1")
        self.assertTrue(record["final_car_candidate"])
        self.assertTrue(record["FP@0.50"])
        self.assertTrue(record["FP@0.25"])


class V154RegressionAuditTest(unittest.TestCase):
    def setUp(self):
        self.t0 = {
            "gt_candidate_records": [
                gt_record("1", "gt_1", matched_050=True, clusters=["c"], before=["d"], after=["d"], best_iou=0.7),
                gt_record("1", "gt_2", matched_050=False, clusters=[], best_iou=0.0),
            ]
        }

    def test_tp_audit_outputs_identity_lists_and_hashes(self):
        variant = {
            "gt_candidate_records": [
                gt_record("1", "gt_1", matched_050=False, clusters=["c"], before=["d"], after=["d"], best_iou=0.4),
                gt_record("1", "gt_2", matched_050=True, clusters=["c2"], before=["d2"], after=["d2"], best_iou=0.6),
            ]
        }
        audit = build_tp_regression_audit(self.t0, variant)
        primary = audit["by_iou"]["0.50"]
        self.assertEqual(primary["TP_regressed_GT"]["identity_list"], [["000001", "gt_1"]])
        self.assertEqual(primary["TP_improved_GT"]["identity_list"], [["000001", "gt_2"]])
        self.assertEqual(len(primary["TP_regressed_GT"]["identity_sha256"]), 64)

    def test_candidate_audit_uses_associated_after_nms_car_candidate(self):
        variant = {
            "gt_candidate_records": [
                gt_record("1", "gt_1", matched_050=False, clusters=["c"], before=["d"], after=[]),
                gt_record("1", "gt_2", matched_050=False, clusters=["c2"], before=["d2"], after=["d2"]),
            ]
        }
        audit = build_candidate_regression_audit(self.t0, variant)
        self.assertEqual(audit["candidate_regressed_GT"]["identity_list"], [["000001", "gt_1"]])
        self.assertEqual(audit["candidate_improved_GT"]["identity_list"], [["000001", "gt_2"]])

    def test_regression_reason_covers_all_frozen_categories(self):
        t0_records = []
        variant_records = []
        cases = [
            ("gt_1", [], [], [], 0.0, "NO_ASSOCIATED_CLUSTER"),
            ("gt_2", ["c"], [], [], 0.0, "REJECTED_BY_CAR_CLASSIFIER"),
            ("gt_3", ["c"], ["d"], [], 0.0, "REMOVED_BY_NMS"),
            ("gt_4", ["c"], ["d"], ["d"], 0.4, "IOU_REGRESSION"),
            ("gt_5", ["c"], ["d"], ["d"], 0.6, "EVALUATION_COMPETITION"),
        ]
        for gt_id, clusters, before, after, best_iou, _ in cases:
            t0_records.append(gt_record("1", gt_id, matched_050=True, clusters=["c"], before=["d"], after=["d"], best_iou=0.7))
            variant_records.append(gt_record("1", gt_id, matched_050=False, clusters=clusters, before=before, after=after, best_iou=best_iou))
        audit = classify_regression_reasons(
            {"gt_candidate_records": t0_records},
            {"gt_candidate_records": variant_records},
            "0.50",
        )
        self.assertEqual([item["reason"] for item in audit["records"]], [item[-1] for item in cases])
        self.assertEqual(audit["unexplained_count"], 0)


if __name__ == "__main__":
    unittest.main()
