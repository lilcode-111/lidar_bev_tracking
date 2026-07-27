import copy
import json
import unittest
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.failure_analysis import analyze_failure_cases
from bev_tracking.report_writer import build_failure_cases_report
from bev_tracking.result_types import FailureCategory, FrameMetrics, FrameResult
from bev_tracking.error_codes import FrameStatus


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


def frame(frame_id, tp, fp, fn, neutralized=0):
    positive_gt = tp + fn
    effective_detections = tp + fp + neutralized
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SUCCESS,
        metrics_by_iou={
            "0.50": metrics(tp, fp, fn, neutralized),
            "0.25": metrics(positive_gt, fp, 0, neutralized),
        },
        num_points=100,
        num_labels_raw=positive_gt,
        num_positive_gt=positive_gt,
        num_raw_detections=effective_detections,
        num_car_detections_before_nms=effective_detections,
        num_detections_after_nms=effective_detections,
    )


def analyze(frames, top_k=1):
    batch = build_batch_result(
        frame_results=frames,
        frame_ids=[item.frame_id for item in frames],
    )
    return analyze_failure_cases(batch, top_k=top_k)


class FailureCasesWriterTest(unittest.TestCase):
    def test_each_category_reports_eligible_and_selected_counts(self):
        analysis = analyze(
            [
                frame("000000", tp=0, fp=2, fn=3),
                frame("000001", tp=0, fp=0, fn=2),
                frame("000002", tp=2, fp=0, fn=0),
            ],
            top_k=1,
        )

        payload = build_failure_cases_report(
            Path("outputs/run"),
            analysis,
            generated_at="2026-07-23T00:00:00Z",
        )

        expected_order = [category.value for category in FailureCategory]
        self.assertEqual(payload["generated_categories"], expected_order)
        for category_name in expected_order:
            category = payload["categories"][category_name]
            self.assertGreaterEqual(category["eligible_count"], category["selected_count"])
            self.assertLessEqual(category["selected_count"], 1)
            self.assertEqual(category["selected_count"], len(category["cases"]))
            self.assertEqual(
                payload["category_definitions"][category_name]["sort_rule"],
                category["sort_rule"],
            )

        self.assertEqual(payload["categories"]["most_false_negatives"]["eligible_count"], 2)
        self.assertEqual(payload["categories"]["zero_detection_with_gt"]["eligible_count"], 1)
        self.assertEqual(payload["summary"]["total_case_entries"], 5)
        self.assertLessEqual(
            payload["summary"]["unique_failure_frames"],
            payload["summary"]["total_case_entries"],
        )
        json.dumps(payload)

    def test_empty_categories_are_emitted_with_empty_case_lists(self):
        analysis = analyze([frame("000000", tp=1, fp=0, fn=0)], top_k=5)

        payload = build_failure_cases_report(
            Path("outputs/run"),
            analysis,
            generated_at="2026-07-23T00:00:00Z",
        )

        for category_name in [
            "most_false_negatives",
            "most_false_positives",
            "lowest_recall",
            "zero_detection_with_gt",
        ]:
            category = payload["categories"][category_name]
            self.assertEqual(category["eligible_count"], 0)
            self.assertEqual(category["selected_count"], 0)
            self.assertEqual(category["cases"], [])

    def test_business_payload_is_stable_except_generated_at(self):
        analysis = analyze(
            [
                frame("000002", tp=0, fp=1, fn=2),
                frame("000001", tp=0, fp=1, fn=2),
            ],
            top_k=2,
        )

        first = build_failure_cases_report(
            Path("outputs/run"),
            analysis,
            generated_at="2026-07-23T00:00:00Z",
        )
        second = build_failure_cases_report(
            Path("outputs/run"),
            analysis,
            generated_at="2026-07-23T00:01:00Z",
        )
        first_business = copy.deepcopy(first)
        second_business = copy.deepcopy(second)
        first_business["generation"].pop("generated_at")
        second_business["generation"].pop("generated_at")

        self.assertEqual(first_business, second_business)
        self.assertNotEqual(first["generation"]["generated_at"], second["generation"]["generated_at"])


if __name__ == "__main__":
    unittest.main()
