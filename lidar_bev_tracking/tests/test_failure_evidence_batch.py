import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.failure_evidence import box_geometry_delta, box_yaw_error
from bev_tracking.failure_evidence_batch import (
    aggregate_diagnostic_reports,
    load_diagnostic_frame_ids,
    load_diagnostic_manifest,
)


class FailureEvidenceGeometryTest(unittest.TestCase):
    def test_box_geometry_delta_reports_center_size_and_pi_periodic_yaw(self):
        gt = {"x": 10.0, "y": 2.0, "length": 4.0, "width": 2.0, "yaw": 0.1}
        detection = {"x": 11.0, "y": 0.0, "length": 5.0, "width": 1.5, "yaw": 0.1 + 3.141592653589793}

        delta = box_geometry_delta(gt, detection)

        self.assertAlmostEqual(delta["dx_m"], 1.0)
        self.assertAlmostEqual(delta["dy_m"], -2.0)
        self.assertAlmostEqual(delta["center_error_m"], 5 ** 0.5)
        self.assertAlmostEqual(delta["length_error_m"], 1.0)
        self.assertAlmostEqual(delta["width_error_m"], -0.5)
        self.assertAlmostEqual(delta["yaw_error_rad"], 0.0)
        self.assertAlmostEqual(box_yaw_error(0.2, 0.1), 0.1)


class FailureEvidenceBatchTest(unittest.TestCase):
    def test_manifest_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.txt"
            path.write_text("000001\n1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_diagnostic_frame_ids(path)

    def test_manifest_metadata_freezes_hash_count_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.txt"
            path.write_text("000001\n000003\n", encoding="utf-8")

            manifest = load_diagnostic_manifest(path)

            self.assertEqual(manifest["num_frames"], 2)
            self.assertEqual(manifest["frame_ids"], ["000001", "000003"])
            self.assertEqual(len(manifest["sha256"]), 64)

    def test_manifest_rejects_unsorted_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.txt"
            path.write_text("000003\n000001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_diagnostic_manifest(path)

    def test_batch_aggregates_unique_primary_reasons_and_geometry(self):
        frame_reports = [
            {
                "frame_id": "000001",
                "geometry_sanity": {
                    "passed": True,
                    "tolerances": {"center_error_m": 0.001, "yaw_error_rad": 0.002},
                    "boxes": [
                        {
                            "gt_id": "gt_1",
                            "class_name": "car",
                            "center_round_trip_error_m": 0.0002,
                            "yaw_round_trip_error_rad": 0.0003,
                        },
                        {
                            "gt_id": "gt_2",
                            "class_name": "car",
                            "center_round_trip_error_m": 0.0004,
                            "yaw_round_trip_error_rad": 0.0001,
                        },
                    ],
                },
                "yaw_semantics": {
                    "passed": True,
                    "tolerances": {
                        "yaw_semantic_error_rad": 0.0001,
                        "corner_alignment_error_m": 0.001,
                    },
                    "boxes": [
                        {
                            "gt_id": "gt_1",
                            "class_name": "car",
                            "yaw_semantic_error_rad": 0.0,
                            "corner_alignment_error_m": 0.0005,
                        }
                    ],
                },
                "failure_evidence": {
                    "summary": {"num_positive_gt": 2, "num_false_negatives": 2},
                    "failure_evidence": [
                        {
                            "primary_reason": "removed_by_intensity_filter",
                            "supporting_flags": ["insufficient_points_for_clustering"],
                            "geometry_delta_after_nms": {},
                        },
                        {
                            "primary_reason": "final_iou_below_threshold",
                            "supporting_flags": [],
                            "geometry_delta_after_nms": {
                                "center_error_m": 1.0,
                                "length_error_m": -0.5,
                                "width_error_m": 0.2,
                                "yaw_error_rad": 0.1,
                            },
                        },
                    ],
                },
            }
        ]

        manifest = {"path": "configs/test.txt", "sha256": "abc", "num_frames": 1, "frame_ids": ["000001"]}
        report = aggregate_diagnostic_reports(
            frame_reports,
            "data/kitti",
            ["000001"],
            manifest_metadata=manifest,
        )

        self.assertEqual(report["summary"]["num_false_negatives"], 2)
        self.assertEqual(report["summary"]["primary_reason_counts"]["removed_by_intensity_filter"], 1)
        self.assertEqual(report["summary"]["low_iou_geometry"]["count"], 1)
        self.assertEqual(report["summary"]["low_iou_geometry"]["center_error_m"]["mean_abs"], 1.0)
        center = report["summary"]["geometry_errors"]["center_round_trip_error_m"]
        self.assertAlmostEqual(center["mean"], 0.0003)
        self.assertEqual(center["max"], 0.0004)
        self.assertEqual(center["worst_object"]["gt_id"], "gt_2")
        corner = report["summary"]["geometry_errors"]["corner_alignment_error_m"]
        self.assertEqual(corner["tolerance"], 0.001)
        self.assertTrue(corner["passed"])
        self.assertEqual(report["source"]["diagnostic_manifest"], manifest)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
