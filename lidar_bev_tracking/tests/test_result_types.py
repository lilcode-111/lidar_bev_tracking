import json
import unittest
from pathlib import Path

from bev_tracking.error_codes import BatchStatus, ErrorCode, ErrorStage, FrameStatus
from bev_tracking.result_types import BatchResult, FrameError, FrameMetrics, FrameResult


class ResultTypesTest(unittest.TestCase):
    def test_frame_metrics_to_dict_is_json_serializable(self):
        metrics = FrameMetrics(
            tp=1,
            fp=2,
            fn=3,
            precision=0.25,
            recall=0.5,
            f1=0.333333,
            neutralized_detections=4,
            per_class={"car": {"tp": 1, "fp": 2, "fn": 3, "precision": 0.25, "recall": 0.5, "f1": 0.333333}},
        )

        output = metrics.to_dict()

        self.assertEqual(output["tp"], 1)
        self.assertEqual(output["neutralized_detections"], 4)
        self.assertEqual(output["per_class"]["car"]["tp"], 1)
        json.dumps(output)

    def test_frame_error_to_dict_uses_stable_values(self):
        error = FrameError(
            error_code=ErrorCode.MISSING_LABEL,
            error_stage=ErrorStage.INPUT_CHECK,
            error_message="missing label file",
            exception_type="FileNotFoundError",
            input_path=Path("data/kitti/training/label_2/000001.txt"),
        )

        output = error.to_dict()

        self.assertEqual(output["error_code"], "missing_label")
        self.assertEqual(output["error_stage"], "input_check")
        self.assertEqual(output["input_path"], "data/kitti/training/label_2/000001.txt")
        json.dumps(output)

    def test_success_frame_result_keeps_metrics_and_derives_metric_valid(self):
        result = FrameResult(
            frame_id="1",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)},
            num_points=100,
            artifacts={"frame_report_path": Path("outputs/frames/000001.json")},
        )

        output = result.to_dict()

        self.assertTrue(output["metric_valid"])
        self.assertEqual(output["frame_id"], "000001")
        self.assertEqual(output["metrics_by_iou"]["0.50"]["tp"], 1)
        self.assertEqual(output["frame_report_path"], "outputs/frames/000001.json")
        json.dumps(output)

    def test_partial_success_is_metric_valid(self):
        result = FrameResult(frame_id="000002", status=FrameStatus.PARTIAL_SUCCESS)

        output = result.to_dict()

        self.assertTrue(output["metric_valid"])
        self.assertEqual(output["status"], "partial_success")

    def test_skipped_and_failed_frame_results_clear_metrics(self):
        skipped = FrameResult(
            frame_id="000003",
            status=FrameStatus.SKIPPED,
            metrics_by_iou={"0.50": FrameMetrics(tp=9, fp=9, fn=9)},
            error=FrameError(ErrorCode.MISSING_BIN, ErrorStage.INPUT_CHECK, "missing bin"),
        )
        failed = FrameResult(
            frame_id="000004",
            status=FrameStatus.FAILED,
            metrics_by_iou={"0.50": FrameMetrics(tp=9, fp=9, fn=9)},
            error=FrameError(ErrorCode.LABEL_PARSE_FAILED, ErrorStage.LABEL_PARSE, "bad label"),
        )

        self.assertFalse(skipped.metric_valid)
        self.assertFalse(failed.metric_valid)
        self.assertEqual(skipped.to_dict()["metrics_by_iou"], {})
        self.assertEqual(failed.to_dict()["metrics_by_iou"], {})
        json.dumps(skipped.to_dict())
        json.dumps(failed.to_dict())

    def test_batch_result_to_dict_is_json_serializable(self):
        frame = FrameResult(
            frame_id="000000",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)},
        )
        batch = BatchResult(
            run_id="20260713T000000Z_kitti_car_batch_test",
            status=BatchStatus.SUCCESS,
            frame_results=[frame],
            frame_counts={"requested": 1, "metric_valid": 1, "success": 1, "partial_success": 0, "skipped": 0, "failed": 0},
            metrics_by_iou={"0.50": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)},
            artifacts={"summary_json": Path("outputs/kitti_batch_eval/run/summary.json")},
        )

        output = batch.to_dict()

        self.assertEqual(output["status"], "success")
        self.assertEqual(output["frame_results"][0]["status"], "success")
        self.assertEqual(output["metrics_by_iou"]["0.50"]["tp"], 1)
        self.assertEqual(output["artifacts"]["summary_json"], "outputs/kitti_batch_eval/run/summary.json")
        json.dumps(output)


if __name__ == "__main__":
    unittest.main()
