import unittest

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import ErrorCode, FrameStatus
from bev_tracking.failure_analysis import (
    InvalidFailureAnalysisConfigError,
    SourceContractError,
    generate_failure_cases,
    validate_top_k,
)
from bev_tracking.result_types import FailureCategory, FrameMetrics, FrameResult


def metrics(tp, fp, fn):
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return FrameMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def batch(frames):
    return build_batch_result(frame_results=frames, frame_ids=[frame.frame_id for frame in frames])


class FailureSelectionTest(unittest.TestCase):
    def test_selection_uses_primary_iou_but_case_keeps_both_iou_metrics(self):
        primary_worst = FrameResult(
            frame_id="000001",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": metrics(0, 0, 5), "0.25": metrics(5, 0, 0)},
            num_positive_gt=5,
        )
        auxiliary_worst = FrameResult(
            frame_id="000002",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": metrics(1, 0, 1), "0.25": metrics(0, 0, 100)},
            num_positive_gt=2,
        )

        cases = generate_failure_cases(batch([auxiliary_worst, primary_worst]), top_k=2)
        fn_cases = [case for case in cases if case.category == FailureCategory.MOST_FALSE_NEGATIVES]

        self.assertEqual(fn_cases[0].frame_id, "000001")
        self.assertEqual(list(fn_cases[0].metrics_by_iou), ["0.50", "0.25"])
        self.assertEqual(fn_cases[0].metrics_by_iou["0.50"]["fn"], 5)
        self.assertEqual(fn_cases[0].metrics_by_iou["0.25"]["fn"], 0)

    def test_missing_primary_or_auxiliary_iou_is_source_contract_error(self):
        primary_only = FrameResult(
            frame_id="000001",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": metrics(1, 0, 0)},
        )
        auxiliary_only = FrameResult(
            frame_id="000002",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.25": metrics(1, 0, 0)},
        )

        for frame in [primary_only, auxiliary_only]:
            with self.assertRaises(SourceContractError) as ctx:
                generate_failure_cases(batch([frame]))
            self.assertEqual(ctx.exception.error_code, ErrorCode.SOURCE_CONTRACT_ERROR)

    def test_ranking_values_explain_all_frozen_sort_fields(self):
        frame = FrameResult(
            frame_id="000001",
            status=FrameStatus.SUCCESS,
            metrics_by_iou={"0.50": metrics(1, 2, 3), "0.25": metrics(3, 1, 1)},
            num_positive_gt=4,
        )

        cases = generate_failure_cases(batch([frame]))
        case = next(case for case in cases if case.category == FailureCategory.MOST_FALSE_NEGATIVES)

        self.assertEqual(
            set(case.ranking_values),
            {"tp", "fp", "fn", "precision", "recall", "num_positive_gt", "neutralized_detections", "effective_car_detection_count"},
        )

    def test_invalid_top_k_has_stable_error_code(self):
        for value in [True, False, 0, -1, 1.5, "5", None]:
            with self.assertRaises(InvalidFailureAnalysisConfigError) as ctx:
                validate_top_k(value)
            self.assertEqual(ctx.exception.error_code, ErrorCode.INVALID_FAILURE_ANALYSIS_CONFIG)


if __name__ == "__main__":
    unittest.main()
