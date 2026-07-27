import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import FrameStatus
from bev_tracking.failure_analysis import analyze_failure_cases_from_run_directory
from bev_tracking.report_writer import write_batch_report, write_failure_cases_report
from bev_tracking.result_types import FrameMetrics, FrameResult


def metrics(tp, fp, fn):
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return FrameMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        per_class={
            "car": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        },
    )


class FailureAnalysisIntegrationTest(unittest.TestCase):
    def test_disk_run_is_validated_analyzed_and_written(self):
        frame = FrameResult(
            frame_id="000000",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={
                "0.50": metrics(tp=0, fp=1, fn=2),
                "0.25": metrics(tp=1, fp=1, fn=1),
            },
            num_points=100,
            num_labels_raw=2,
            num_positive_gt=2,
            num_raw_detections=1,
            num_car_detections_before_nms=1,
            num_detections_after_nms=1,
        )
        batch = build_batch_result(frame_results=[frame], frame_ids=[frame.frame_id])

        with tempfile.TemporaryDirectory() as tmp:
            final_batch, paths = write_batch_report(
                batch,
                output_root=Path(tmp) / "runs",
                config_effective={"data": {"frame_ids": [frame.frame_id]}},
                command="failure-analysis-integration-test",
            )
            run_directory = paths["summary_json"].parent

            analysis = analyze_failure_cases_from_run_directory(run_directory, top_k=5)
            payload, output_path = write_failure_cases_report(
                run_directory,
                analysis,
                generated_at="2026-07-23T00:00:00Z",
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved, payload)
            self.assertEqual(saved["source"]["mode"], "disk")
            self.assertEqual(saved["source"]["run_id"], final_batch.run_id)
            self.assertEqual(saved["source"]["batch_status"], final_batch.status.value)
            self.assertEqual(saved["validation"]["source_contract"]["status"], "passed")
            self.assertEqual(saved["validation"]["source_consistency"]["status"], "passed")
            self.assertEqual(saved["config"]["primary_iou"], 0.5)
            self.assertEqual(saved["config"]["required_iou_keys"], ["0.50", "0.25"])
            self.assertIn("0.50", saved["failure_cases"][0]["metrics_by_iou"])
            self.assertIn("0.25", saved["failure_cases"][0]["metrics_by_iou"])


if __name__ == "__main__":
    unittest.main()
