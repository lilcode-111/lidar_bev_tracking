import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import ErrorCode, ErrorStage, FrameStatus
from bev_tracking.report_writer import write_batch_report
from bev_tracking.result_types import FrameError, FrameMetrics, FrameResult


def frame(frame_id, status=FrameStatus.SUCCESS, error=None):
    metrics = {}
    if status == FrameStatus.SUCCESS:
        metrics = {
            "0.50": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
            "0.25": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
        }
    return FrameResult(
        frame_id=frame_id,
        status=status,
        metrics_by_iou=metrics,
        error=error,
        num_points=10 if status == FrameStatus.SUCCESS else None,
        num_positive_gt=1 if status == FrameStatus.SUCCESS else None,
        num_raw_detections=1 if status == FrameStatus.SUCCESS else None,
        num_detections_after_nms=1 if status == FrameStatus.SUCCESS else None,
    )


class ReportWriterTest(unittest.TestCase):
    def test_report_writer_creates_run_directory_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = build_batch_result(
                frame_results=[
                    frame("000000"),
                    frame(
                        "000001",
                        status=FrameStatus.SKIPPED,
                        error=FrameError(ErrorCode.MISSING_BIN, ErrorStage.INPUT_CHECK, "missing bin"),
                    ),
                ],
                frame_ids=["000000", "000001"],
            )

            final_batch, paths = write_batch_report(
                batch,
                output_root=Path(tmp) / "kitti_batch_eval",
                config_effective={"data": {"frame_ids": ["000000", "000001"]}},
                command="unit-test",
            )

            self.assertTrue(paths["summary_json"].exists())
            self.assertTrue(paths["frames_csv"].exists())
            self.assertTrue(paths["config_input"].exists())
            self.assertTrue(paths["config_effective"].exists())
            self.assertTrue(paths["git_metadata"].exists())
            self.assertTrue(paths["frame_manifest"].exists())
            self.assertTrue((paths["per_frame_report_dir"] / "000000.json").exists())
            self.assertTrue((paths["per_frame_report_dir"] / "000001.json").exists())

            summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["batch_status"], final_batch.status.value)
            self.assertEqual(summary["frames"]["requested"], 2)
            self.assertEqual(summary["frames"]["metric_valid"], 1)
            self.assertEqual(summary["errors"]["counts_by_code"]["missing_bin"], 1)
            self.assertEqual(summary["metrics_by_iou"]["0.50"]["tp"], 1)

            csv_text = paths["frames_csv"].read_text(encoding="utf-8")
            self.assertIn("tp_iou_0_50", csv_text)
            self.assertIn("missing_bin", csv_text)


if __name__ == "__main__":
    unittest.main()
