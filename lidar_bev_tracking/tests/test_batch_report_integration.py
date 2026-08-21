import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_batch_loop_reports_each_frame_progress(self):
        progress = []
        with patch.object(batch_pipeline, "precheck_kitti_frame_inputs", return_value=None), patch.object(
            batch_pipeline,
            "run_kitti_frame_evaluation",
            side_effect=lambda **kwargs: success_frame(kwargs["frame_id"]),
        ):
            results = batch_pipeline.run_kitti_batch_frame_results(
                frame_ids=["1", "2"],
                progress_callback=lambda index, total, frame_id: progress.append((index, total, frame_id)),
            )

        self.assertEqual(progress, [(1, 2, "000001"), (2, 2, "000002")])
        self.assertEqual([result.frame_id for result in results], ["000001", "000002"])

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
                    "gesr_evidence_level": "compact",
                },
                "nms": {"iou_threshold": 0.3},
                "evaluation": {"iou_threshold": 0.5, "auxiliary_iou_thresholds": [0.25]},
                "outputs": {"batch_report_root": str(Path(tmp) / "runs")},
            }
            captured = {}
            progress_callback = lambda index, total, frame_id: None
            original = batch_pipeline.run_kitti_batch_result

            def fake_batch_result(**kwargs):
                captured.update(kwargs)
                streamed_frame = success_frame("000000")
                streamed_frame.artifacts["large_payload"] = ["discarded-after-write"]
                streamed_frame = kwargs["frame_result_callback"](streamed_frame)
                return build_batch_result(
                    frame_results=[streamed_frame],
                    data_root=kwargs["data_root"],
                    frame_ids=kwargs["frame_ids"],
                    eps=kwargs["eps"],
                    min_points=kwargs["min_points"],
                    oriented=kwargs["oriented"],
                    nms_iou_threshold=kwargs["nms_iou_threshold"],
                    eval_iou_threshold=kwargs["eval_iou_threshold"],
                    auxiliary_iou_thresholds=kwargs["auxiliary_iou_thresholds"],
                    gesr_evidence_level=kwargs["gesr_evidence_level"],
                )

            try:
                batch_pipeline.run_kitti_batch_result = fake_batch_result
                final_batch, paths = run_kitti_batch_report_from_config(
                    config,
                    config_input_path=config_path,
                    command="unit-test command",
                    progress_callback=progress_callback,
                )
            finally:
                batch_pipeline.run_kitti_batch_result = original

            self.assertEqual(captured["frame_ids"], ["000000"])
            self.assertTrue(captured["oriented"])
            self.assertTrue(captured["gesr_enabled"])
            self.assertTrue(captured["gesr_reason_attribution"])
            self.assertEqual(captured["gesr_evidence_level"], "compact")
            self.assertIs(captured["progress_callback"], progress_callback)
            self.assertNotIn("large_payload", final_batch.frame_results[0].artifacts)
            self.assertEqual(final_batch.frame_counts["metric_valid"], 1)
            self.assertTrue(paths["summary_json"].exists())
            self.assertTrue(paths["frames_csv"].exists())
            self.assertTrue((paths["per_frame_report_dir"] / "000000.json").exists())
            frame_payload = json.loads(
                (paths["per_frame_report_dir"] / "000000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                frame_payload["artifacts"]["large_payload"],
                ["discarded-after-write"],
            )

            summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["command"], "unit-test command")
            self.assertEqual(summary["metrics_by_iou"]["0.50"]["tp"], 1)
            self.assertIn("frame_ids", paths["config_input"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
