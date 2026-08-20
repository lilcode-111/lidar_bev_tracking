import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking import batch_pipeline
from bev_tracking.batch_pipeline import build_batch_result, run_kitti_batch_report_from_config
from bev_tracking.error_codes import FrameStatus
from bev_tracking.result_types import FrameMetrics, FrameResult


def success_frame(frame_id):
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SUCCESS,
        metrics_by_iou={
            "0.50": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
            "0.25": FrameMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0),
        },
        num_points=10,
        num_positive_gt=1,
        num_raw_detections=1,
        num_detections_after_nms=1,
    )


class BatchReportIntegrationTest(unittest.TestCase):
    def test_config_entrypoint_writes_full_batch_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "kitti_eval_batch.yaml"
            config_path.write_text("data:\n  frame_ids:\n    - '000000'\n", encoding="utf-8")
            config = {
                "data": {"root": "data/kitti", "frame_ids": ["000000"]},
                "detector": {
                    "eps": 0.6,
                    "min_points": 20,
                    "oriented": True,
                    "gesr_enabled": True,
                    "gesr_reason_attribution": True,
                },
                "nms": {"iou_threshold": 0.3},
                "evaluation": {"iou_threshold": 0.5, "auxiliary_iou_thresholds": [0.25]},
                "outputs": {"batch_report_root": str(Path(tmp) / "runs")},
            }
            captured = {}
            original = batch_pipeline.run_kitti_batch_result

            def fake_batch_result(**kwargs):
                captured.update(kwargs)
                return build_batch_result(
                    frame_results=[success_frame("000000")],
                    data_root=kwargs["data_root"],
                    frame_ids=kwargs["frame_ids"],
                    eps=kwargs["eps"],
                    min_points=kwargs["min_points"],
                    oriented=kwargs["oriented"],
                    nms_iou_threshold=kwargs["nms_iou_threshold"],
                    eval_iou_threshold=kwargs["eval_iou_threshold"],
                    auxiliary_iou_thresholds=kwargs["auxiliary_iou_thresholds"],
                )

            try:
                batch_pipeline.run_kitti_batch_result = fake_batch_result
                final_batch, paths = run_kitti_batch_report_from_config(
                    config,
                    config_input_path=config_path,
                    command="unit-test command",
                )
            finally:
                batch_pipeline.run_kitti_batch_result = original

            self.assertEqual(captured["frame_ids"], ["000000"])
            self.assertTrue(captured["oriented"])
            self.assertTrue(captured["gesr_enabled"])
            self.assertTrue(captured["gesr_reason_attribution"])
            self.assertEqual(final_batch.frame_counts["metric_valid"], 1)
            self.assertTrue(paths["summary_json"].exists())
            self.assertTrue(paths["frames_csv"].exists())
            self.assertTrue((paths["per_frame_report_dir"] / "000000.json").exists())

            summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["command"], "unit-test command")
            self.assertEqual(summary["metrics_by_iou"]["0.50"]["tp"], 1)
            self.assertIn("frame_ids", paths["config_input"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
