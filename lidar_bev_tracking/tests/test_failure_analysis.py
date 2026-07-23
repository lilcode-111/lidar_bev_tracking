import copy
import json
import unittest
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import FrameStatus
from bev_tracking.failure_analysis import (
    DuplicateFrameIdError,
    SourceContractError,
    effective_car_detection_count,
    generate_failure_cases,
    validate_top_k,
)
from bev_tracking.result_types import FailureCategory, FailureCase, FrameMetrics, FrameResult


def frame(
    frame_id,
    tp=0,
    fp=0,
    fn=0,
    recall=None,
    neutralized=0,
    positive_gt=None,
    detections_after_nms=0,
    status=FrameStatus.SUCCESS,
):
    metrics = {}
    if status in {FrameStatus.SUCCESS, FrameStatus.PARTIAL_SUCCESS}:
        precision = None if tp + fp == 0 else tp / (tp + fp)
        effective_recall = recall if recall is not None else (None if tp + fn == 0 else tp / (tp + fn))
        f1 = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)

        def frame_metrics():
            return FrameMetrics(
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=effective_recall,
                f1=f1,
                neutralized_detections=neutralized,
                per_class={"car": {"tp": tp, "fp": fp, "fn": fn}},
            )

        metrics = {"0.50": frame_metrics(), "0.25": frame_metrics()}
    return FrameResult(
        frame_id=frame_id,
        status=status,
        metrics_by_iou=metrics,
        num_positive_gt=positive_gt if positive_gt is not None else tp + fn,
        num_raw_detections=detections_after_nms,
        num_car_detections_before_nms=detections_after_nms,
        num_detections_after_nms=detections_after_nms,
        artifacts={
            "frame_report_path": Path(f"outputs/kitti_batch_eval/run/frames/{str(frame_id).zfill(6)}.json"),
            "source_run_id": "run_001",
            "source_run_directory": Path("outputs/kitti_batch_eval/run"),
        },
    )


def batch(frames):
    return build_batch_result(frame_results=frames, frame_ids=[item.frame_id for item in frames])


def cases_by_category(cases, category):
    return [case for case in cases if case.category == FailureCategory(category)]


class FailureAnalysisTest(unittest.TestCase):
    def test_failure_case_to_dict_is_json_serializable(self):
        case = FailureCase(
            category=FailureCategory.MOST_FALSE_NEGATIVES,
            rank=1,
            frame_id="1",
            reason_code="high_false_negative_count",
            source_status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": {"tp": 1, "fp": 0, "fn": 2}},
        )

        output = case.to_dict()

        self.assertEqual(output["category"], "most_false_negatives")
        self.assertEqual(output["frame_id"], "000001")
        json.dumps(output)

    def test_top_k_validation_rejects_invalid_values(self):
        self.assertEqual(validate_top_k(5), 5)
        for value in [True, False, 0, -1, 1.5, "5", None]:
            with self.assertRaises(ValueError):
                validate_top_k(value)

    def test_five_categories_and_stable_sorting(self):
        result = batch(
            [
                frame("000003", tp=0, fp=3, fn=5, recall=0.0, neutralized=0, positive_gt=5, detections_after_nms=99),
                frame("000001", tp=1, fp=3, fn=2, recall=1 / 3, neutralized=1, positive_gt=3, detections_after_nms=2),
                frame("000002", tp=2, fp=1, fn=2, recall=0.5, neutralized=0, positive_gt=4, detections_after_nms=10),
                frame("000004", tp=0, fp=0, fn=4, recall=0.0, neutralized=0, positive_gt=4, detections_after_nms=8),
            ]
        )

        cases = generate_failure_cases(result, top_k=2)

        fn_cases = cases_by_category(cases, FailureCategory.MOST_FALSE_NEGATIVES)
        self.assertEqual([case.frame_id for case in fn_cases], ["000003", "000004"])

        fp_cases = cases_by_category(cases, FailureCategory.MOST_FALSE_POSITIVES)
        self.assertEqual([case.frame_id for case in fp_cases], ["000003", "000001"])

        recall_cases = cases_by_category(cases, FailureCategory.LOWEST_RECALL)
        self.assertEqual([case.frame_id for case in recall_cases], ["000003", "000004"])

        zero_cases = cases_by_category(cases, FailureCategory.ZERO_DETECTION_WITH_GT)
        self.assertEqual([case.frame_id for case in zero_cases], ["000004"])

        high_det_cases = cases_by_category(cases, FailureCategory.HIGHEST_EFFECTIVE_CAR_DETECTIONS)
        self.assertTrue(all(case.diagnostic_only for case in high_det_cases))
        self.assertEqual([case.frame_id for case in high_det_cases], ["000001", "000003"])

    def test_effective_car_detection_count_uses_metrics_not_nms_count(self):
        metrics = FrameMetrics(tp=1, fp=2, fn=0, neutralized_detections=3)

        self.assertEqual(effective_car_detection_count(metrics), 6)

        result = batch([frame("000000", tp=1, fp=2, fn=0, recall=1.0, neutralized=3, detections_after_nms=99)])
        cases = generate_failure_cases(result, top_k=5)
        high_det = cases_by_category(cases, FailureCategory.HIGHEST_EFFECTIVE_CAR_DETECTIONS)[0]

        self.assertEqual(high_det.effective_car_detection_count, 6)
        self.assertEqual(high_det.detection_counts["num_detections_after_nms"], 99)

    def test_recall_null_is_excluded_from_lowest_recall(self):
        result = batch(
            [
                frame("000000", tp=0, fp=0, fn=0, recall=None, positive_gt=0),
                frame("000001", tp=1, fp=0, fn=1, recall=0.5, positive_gt=2),
            ]
        )

        cases = generate_failure_cases(result, top_k=5)
        recall_cases = cases_by_category(cases, FailureCategory.LOWEST_RECALL)

        self.assertEqual([case.frame_id for case in recall_cases], ["000001"])

    def test_skipped_and_failed_are_never_eligible_but_partial_success_is(self):
        result = batch(
            [
                frame("000000", tp=0, fp=0, fn=4, recall=0.0, positive_gt=4, status=FrameStatus.PARTIAL_SUCCESS),
                frame("000001", tp=0, fp=0, fn=99, status=FrameStatus.SKIPPED),
                frame("000002", tp=0, fp=0, fn=99, status=FrameStatus.FAILED),
            ]
        )

        cases = generate_failure_cases(result, top_k=5)
        all_frame_ids = {case.frame_id for case in cases}

        self.assertEqual(all_frame_ids, {"000000"})

    def test_same_frame_can_enter_multiple_categories(self):
        result = batch([frame("000000", tp=0, fp=4, fn=3, recall=0.0, positive_gt=3)])

        cases = generate_failure_cases(result, top_k=5)
        categories = {case.category for case in cases if case.frame_id == "000000"}

        self.assertIn(FailureCategory.MOST_FALSE_NEGATIVES, categories)
        self.assertIn(FailureCategory.MOST_FALSE_POSITIVES, categories)
        self.assertIn(FailureCategory.LOWEST_RECALL, categories)
        self.assertIn(FailureCategory.HIGHEST_EFFECTIVE_CAR_DETECTIONS, categories)

    def test_source_contract_errors_on_missing_primary_metrics_and_duplicate_frame_id(self):
        with self.assertRaises(SourceContractError):
            generate_failure_cases(batch([FrameResult(frame_id="000000", status=FrameStatus.SUCCESS)]))

        with self.assertRaises(DuplicateFrameIdError):
            generate_failure_cases(batch([frame("000000", tp=1), frame("000000", tp=2)]))

    def test_generate_failure_cases_does_not_mutate_input(self):
        result = batch([frame("000000", tp=1, fp=1, fn=1, recall=0.5, positive_gt=2)])
        before = copy.deepcopy(result.to_dict())

        generate_failure_cases(result, top_k=5)

        self.assertEqual(result.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
