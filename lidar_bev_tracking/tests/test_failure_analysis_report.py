import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from bev_tracking import report_writer
from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import ErrorCode, FrameStatus
from bev_tracking.failure_analysis import generate_failure_cases_from_run_directory
from bev_tracking.report_writer import ReportWriteError, write_batch_report, write_failure_cases_report
from bev_tracking.result_types import FrameMetrics, FrameResult
from scripts.run_failure_analysis import main


def metric(tp, fp, fn):
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
        per_class={"car": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}},
    )


def create_run(root):
    frame = FrameResult(
        frame_id="000000",
        status=FrameStatus.SUCCESS,
        metrics_by_iou={"0.50": metric(0, 1, 2), "0.25": metric(1, 1, 1)},
        num_points=100,
        num_labels_raw=2,
        num_positive_gt=2,
        num_raw_detections=1,
        num_car_detections_before_nms=1,
        num_detections_after_nms=1,
    )
    batch = build_batch_result(frame_results=[frame], frame_ids=[frame.frame_id])
    return write_batch_report(
        batch,
        output_root=Path(root) / "runs",
        config_effective={"data": {"frame_ids": [frame.frame_id]}},
        command="failure-analysis-report-test",
    )


class FailureAnalysisReportTest(unittest.TestCase):
    def test_failure_cases_report_contains_source_selection_and_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_batch, paths = create_run(tmp)
            run_directory = paths["summary_json"].parent
            cases = generate_failure_cases_from_run_directory(run_directory, top_k=2)

            payload, output_path = write_failure_cases_report(run_directory, cases, top_k=2)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], "14.0")
            self.assertEqual(saved["source"]["run_id"], final_batch.run_id)
            self.assertEqual(saved["selection"]["top_k"], 2)
            self.assertEqual(saved["summary"]["total_failure_cases"], len(cases))
            self.assertEqual(saved["summary"]["counts_by_category"]["most_false_negatives"], 1)
            self.assertEqual(saved["summary"]["counts_by_category"]["most_false_positives"], 1)
            self.assertEqual(saved["failure_cases"][0]["source_run_id"], final_batch.run_id)

    def test_cli_generates_failure_cases_json_from_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = create_run(tmp)
            run_directory = paths["summary_json"].parent

            with redirect_stdout(io.StringIO()):
                exit_code = main(["--run-dir", str(run_directory), "--top-k", "1"])

            self.assertEqual(exit_code, 0)
            output_path = run_directory / "failure_cases.json"
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["selection"]["top_k"], 1)

    def test_atomic_write_failure_keeps_existing_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = create_run(tmp)
            run_directory = paths["summary_json"].parent
            cases = generate_failure_cases_from_run_directory(run_directory, top_k=1)
            _, output_path = write_failure_cases_report(run_directory, cases, top_k=1)
            original_text = output_path.read_text(encoding="utf-8")
            original_replace = report_writer.os.replace

            def fail_failure_report_replace(source, destination):
                if Path(destination) == output_path:
                    raise PermissionError("failure report blocked")
                return original_replace(source, destination)

            try:
                report_writer.os.replace = fail_failure_report_replace
                with self.assertRaises(ReportWriteError) as ctx:
                    write_failure_cases_report(run_directory, cases, top_k=1)
            finally:
                report_writer.os.replace = original_replace

            self.assertEqual(ctx.exception.error_code, ErrorCode.FAILURE_CASES_WRITE_FAILED)
            self.assertEqual(output_path.read_text(encoding="utf-8"), original_text)
            self.assertFalse((run_directory / ".failure_cases.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
