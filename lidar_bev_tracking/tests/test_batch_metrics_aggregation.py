import unittest

from bev_tracking.batch_pipeline import build_batch_result, summarize_batch_reports
from bev_tracking.error_codes import ErrorCode, ErrorStage, FrameStatus
from bev_tracking.result_types import FrameError, FrameMetrics, FrameResult


def valid_frame(frame_id, tp, fp, fn, neutralized=0):
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SUCCESS,
        metrics_by_iou={
            "0.50": FrameMetrics(
                tp=tp,
                fp=fp,
                fn=fn,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                neutralized_detections=neutralized,
                per_class={"car": {"tp": tp, "fp": fp, "fn": fn}},
            ),
            "0.25": FrameMetrics(
                tp=tp + 1,
                fp=fp,
                fn=fn,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                neutralized_detections=neutralized + 1,
                per_class={"car": {"tp": tp + 1, "fp": fp, "fn": fn}},
            ),
        },
        num_points=100,
        num_positive_gt=tp + fn,
        num_raw_detections=tp + fp,
        num_detections_after_nms=tp + fp,
    )


class BatchMetricsAggregationTest(unittest.TestCase):
    def test_batch_metrics_aggregate_only_metric_valid_frames(self):
        skipped = FrameResult(
            frame_id="000002",
            status=FrameStatus.SKIPPED,
            metrics_by_iou={"0.50": FrameMetrics(tp=99, fp=99, fn=99)},
            error=FrameError(ErrorCode.MISSING_BIN, ErrorStage.INPUT_CHECK, "missing bin"),
        )

        batch = build_batch_result(
            frame_results=[
                valid_frame("000000", tp=1, fp=1, fn=0, neutralized=1),
                valid_frame("000001", tp=2, fp=0, fn=1, neutralized=0),
                skipped,
            ],
            frame_ids=["000000", "000001", "000002"],
        )

        metrics_050 = batch.metrics_by_iou["0.50"]
        self.assertEqual(metrics_050.tp, 3)
        self.assertEqual(metrics_050.fp, 1)
        self.assertEqual(metrics_050.fn, 1)
        self.assertEqual(metrics_050.neutralized_detections, 1)
        self.assertEqual(metrics_050.precision, 3 / 4)
        self.assertEqual(metrics_050.recall, 3 / 4)
        self.assertEqual(metrics_050.f1, 6 / 8)
        self.assertEqual(metrics_050.per_class["car"]["tp"], 3)
        self.assertEqual(metrics_050.per_class["car"]["fp"], 1)
        self.assertEqual(metrics_050.per_class["car"]["fn"], 1)
        self.assertEqual(metrics_050.per_class["car"]["f1"], 6 / 8)

        metrics_025 = batch.metrics_by_iou["0.25"]
        self.assertEqual(metrics_025.tp, 5)
        self.assertEqual(metrics_025.fp, 1)
        self.assertEqual(metrics_025.fn, 1)
        self.assertEqual(metrics_025.neutralized_detections, 3)
        self.assertEqual(metrics_025.per_class["car"]["tp"], 5)
        self.assertEqual(batch.totals["num_points"], 200)

    def test_legacy_reports_without_metric_valid_are_treated_as_valid(self):
        frame_reports = [
            {
                "frame_id": "000000",
                "num_points": 1,
                "num_gt_boxes": 1,
                "num_detections_after_nms": 1,
                "iou_threshold": 0.5,
                "metrics": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "per_class": {}},
                "auxiliary": {
                    "0.25": {
                        "metrics": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "per_class": {}}
                    }
                },
                "report_path": "",
            }
        ]

        summary = summarize_batch_reports(
            frame_reports=frame_reports,
            data_root="data/kitti",
            eps=0.6,
            min_points=20,
            oriented=False,
            nms_iou_threshold=0.3,
            eval_iou_threshold=0.5,
            auxiliary_iou_thresholds=[0.25],
        )

        self.assertEqual(summary["metrics_by_iou"]["0.50"]["tp"], 1)
        self.assertEqual(summary["metrics_by_iou"]["0.25"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
