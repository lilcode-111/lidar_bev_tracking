import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.failure_evidence import box_geometry_delta, box_yaw_error
from bev_tracking.failure_evidence_batch import aggregate_diagnostic_reports, load_diagnostic_frame_ids


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

    def test_batch_aggregates_unique_primary_reasons_and_geometry(self):
        frame_reports = [
            {
                "frame_id": "000001",
                "geometry_sanity": {"passed": True},
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

        report = aggregate_diagnostic_reports(frame_reports, "data/kitti", ["000001"])

        self.assertEqual(report["summary"]["num_false_negatives"], 2)
        self.assertEqual(report["summary"]["primary_reason_counts"]["removed_by_intensity_filter"], 1)
        self.assertEqual(report["summary"]["low_iou_geometry"]["count"], 1)
        self.assertEqual(report["summary"]["low_iou_geometry"]["center_error_m"]["mean_abs"], 1.0)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
