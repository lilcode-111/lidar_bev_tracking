import csv
import json
import tempfile
import unittest
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import ErrorCode, ErrorStage, FrameStatus
from bev_tracking.failure_analysis import (
    SourceContractError,
    generate_failure_cases,
    generate_failure_cases_from_run_directory,
    load_batch_result_from_run_directory,
)
from bev_tracking.report_writer import write_batch_report
from bev_tracking.result_types import FrameError, FrameMetrics, FrameResult


def metrics(tp, fp, fn, neutralized=0):
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
        neutralized_detections=neutralized,
        per_class={"car": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}},
    )


def success_frame(frame_id, tp, fp, fn, neutralized=0):
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SUCCESS,
        metrics_by_iou={
            "0.50": metrics(tp, fp, fn, neutralized),
            "0.25": metrics(tp + fn, fp, 0, neutralized),
        },
        num_points=100,
        num_labels_raw=tp + fn,
        num_positive_gt=tp + fn,
        num_raw_detections=tp + fp + neutralized,
        num_car_detections_before_nms=tp + fp + neutralized,
        num_detections_after_nms=tp + fp + neutralized,
    )


def skipped_frame(frame_id):
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SKIPPED,
        error=FrameError(ErrorCode.MISSING_BIN, ErrorStage.INPUT_CHECK, "missing bin"),
    )


def write_sample_run(root):
    frames = [
        success_frame("000000", tp=1, fp=2, fn=1),
        success_frame("000001", tp=0, fp=0, fn=3),
        skipped_frame("000002"),
    ]
    batch = build_batch_result(frame_results=frames, frame_ids=[frame.frame_id for frame in frames])
    return write_batch_report(
        batch,
        output_root=Path(root) / "runs",
        config_effective={"data": {"frame_ids": [frame.frame_id for frame in frames]}},
        command="failure-analysis-disk-test",
    )


class FailureAnalysisDiskTest(unittest.TestCase):
    def test_disk_mode_matches_in_memory_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_batch, paths = write_sample_run(tmp)
            run_directory = paths["summary_json"].parent

            loaded = load_batch_result_from_run_directory(run_directory)
            memory_cases = [case.to_dict() for case in generate_failure_cases(final_batch, top_k=5)]
            disk_cases = [case.to_dict() for case in generate_failure_cases_from_run_directory(run_directory, top_k=5)]

            self.assertEqual([frame.frame_id for frame in loaded.frame_results], ["000000", "000001", "000002"])
            self.assertEqual(loaded.frame_counts, final_batch.frame_counts)
            self.assertEqual(disk_cases, memory_cases)

    def test_status_mismatch_across_reports_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = write_sample_run(tmp)
            manifest = json.loads(paths["frame_manifest"].read_text(encoding="utf-8"))
            manifest["frames"][0]["status"] = "failed"
            paths["frame_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(SourceContractError, "status mismatch"):
                load_batch_result_from_run_directory(paths["summary_json"].parent)

    def test_primary_metric_mismatch_in_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = write_sample_run(tmp)
            with open(paths["frames_csv"], "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
                fieldnames = list(rows[0])
            rows[0]["fn_iou_0_50"] = "99"
            with open(paths["frames_csv"], "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(SourceContractError, "primary metric mismatch"):
                load_batch_result_from_run_directory(paths["summary_json"].parent)

    def test_frame_set_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = write_sample_run(tmp)
            (paths["per_frame_report_dir"] / "000001.json").unlink()

            with self.assertRaisesRegex(SourceContractError, "file set"):
                load_batch_result_from_run_directory(paths["summary_json"].parent)

        with tempfile.TemporaryDirectory() as tmp:
            _, paths = write_sample_run(tmp)
            with open(paths["frames_csv"], "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
                fieldnames = list(rows[0])
            rows[1]["frame_id"] = rows[0]["frame_id"]
            with open(paths["frames_csv"], "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(SourceContractError, "duplicate frame_id"):
                load_batch_result_from_run_directory(paths["summary_json"].parent)

    def test_summary_counts_and_metrics_are_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = write_sample_run(tmp)
            summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
            summary["metrics_by_iou"]["0.50"]["tp"] = 100
            paths["summary_json"].write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(SourceContractError, "summary primary metric mismatch"):
                load_batch_result_from_run_directory(paths["summary_json"].parent)


if __name__ == "__main__":
    unittest.main()
