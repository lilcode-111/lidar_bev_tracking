import unittest

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import BatchStatus, ErrorCode, ErrorStage, FrameStatus
from bev_tracking.result_types import FrameError, FrameMetrics, FrameResult


def frame(frame_id, status, error=None):
    metrics_by_iou = {}
    if status in {FrameStatus.SUCCESS, FrameStatus.PARTIAL_SUCCESS}:
        metrics_by_iou = {
            "0.50": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
            "0.25": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
        }
    return FrameResult(frame_id=frame_id, status=status, metrics_by_iou=metrics_by_iou, error=error, num_points=10)


class BatchStatusCountsTest(unittest.TestCase):
    def test_all_success_frames_make_success_batch(self):
        batch = build_batch_result(
            frame_results=[frame("000000", FrameStatus.SUCCESS), frame("000001", FrameStatus.SUCCESS)],
            frame_ids=["000000", "000001"],
        )

        self.assertEqual(batch.status, BatchStatus.SUCCESS)
        self.assertEqual(batch.frame_counts["requested"], 2)
        self.assertEqual(batch.frame_counts["success"], 2)
        self.assertEqual(batch.frame_counts["metric_valid"], 2)
        self.assertEqual(batch.frame_counts["excluded_from_metrics"], 0)

    def test_mixed_status_frames_make_partial_success_batch(self):
        batch = build_batch_result(
            frame_results=[
                frame("000000", FrameStatus.SUCCESS),
                frame(
                    "000001",
                    FrameStatus.SKIPPED,
                    FrameError(ErrorCode.MISSING_BIN, ErrorStage.INPUT_CHECK, "missing bin"),
                ),
                frame(
                    "000002",
                    FrameStatus.FAILED,
                    FrameError(ErrorCode.UNEXPECTED_FRAME_ERROR, ErrorStage.UNKNOWN, "boom"),
                ),
            ],
            frame_ids=["000000", "000001", "000002"],
        )

        self.assertEqual(batch.status, BatchStatus.PARTIAL_SUCCESS)
        self.assertEqual(batch.frame_counts["requested"], 3)
        self.assertEqual(batch.frame_counts["success"], 1)
        self.assertEqual(batch.frame_counts["skipped"], 1)
        self.assertEqual(batch.frame_counts["failed"], 1)
        self.assertEqual(batch.frame_counts["metric_valid"], 1)
        self.assertEqual(batch.frame_counts["excluded_from_metrics"], 2)
        self.assertEqual(batch.error_counts["missing_bin"], 1)
        self.assertEqual(batch.error_counts["unexpected_frame_error"], 1)

    def test_no_metric_valid_frames_make_failed_batch(self):
        batch = build_batch_result(
            frame_results=[
                frame(
                    "000000",
                    FrameStatus.SKIPPED,
                    FrameError(ErrorCode.MISSING_LABEL, ErrorStage.INPUT_CHECK, "missing label"),
                )
            ],
            frame_ids=["000000"],
        )

        self.assertEqual(batch.status, BatchStatus.FAILED)
        self.assertEqual(batch.frame_counts["metric_valid"], 0)
        self.assertEqual(batch.frame_counts["excluded_from_metrics"], 1)

    def test_frame_count_invariants_hold(self):
        batch = build_batch_result(
            frame_results=[
                frame("000000", FrameStatus.SUCCESS),
                frame("000001", FrameStatus.PARTIAL_SUCCESS),
                frame("000002", FrameStatus.SKIPPED, FrameError(ErrorCode.MISSING_CALIB, ErrorStage.INPUT_CHECK, "missing calib")),
                frame("000003", FrameStatus.FAILED, FrameError(ErrorCode.DETECTOR_FAILED, ErrorStage.DETECTION, "detector failed")),
            ],
            frame_ids=["000000", "000001", "000002", "000003"],
        )
        counts = batch.frame_counts

        self.assertEqual(
            counts["success"] + counts["partial_success"] + counts["skipped"] + counts["failed"],
            counts["requested"],
        )
        self.assertEqual(counts["success"] + counts["partial_success"], counts["metric_valid"])
        self.assertEqual(counts["skipped"] + counts["failed"], counts["excluded_from_metrics"])


if __name__ == "__main__":
    unittest.main()
