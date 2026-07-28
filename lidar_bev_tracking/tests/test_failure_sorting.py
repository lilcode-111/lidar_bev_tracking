import random
import unittest

from bev_tracking.batch_pipeline import build_batch_result
from bev_tracking.error_codes import FrameStatus
from bev_tracking.failure_analysis import generate_failure_cases
from bev_tracking.result_types import FailureCategory, FrameMetrics, FrameResult


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
        per_class={"car": {"tp": tp, "fp": fp, "fn": fn}},
    )


def frame(frame_id, tp, fp, fn, neutralized=0, positive_gt=None):
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SUCCESS,
        metrics_by_iou={
            "0.50": metrics(tp, fp, fn, neutralized),
            "0.25": metrics(tp, fp, fn, neutralized),
        },
        num_positive_gt=tp + fn if positive_gt is None else positive_gt,
    )


def category_ids(frames, category):
    batch = build_batch_result(frame_results=frames, frame_ids=[item.frame_id for item in frames])
    cases = generate_failure_cases(batch, top_k=20)
    return [case.frame_id for case in cases if case.category == category]


class FailureSortingTest(unittest.TestCase):
    def test_same_fn_uses_lower_recall_before_frame_id(self):
        frames = [
            frame("000001", tp=2, fp=0, fn=2),
            frame("000009", tp=1, fp=0, fn=2),
        ]

        ordered = category_ids(frames, FailureCategory.MOST_FALSE_NEGATIVES)

        self.assertEqual(ordered, ["000009", "000001"])

    def test_same_fp_uses_lower_precision_before_frame_id(self):
        frames = [
            frame("000001", tp=2, fp=2, fn=0),
            frame("000009", tp=0, fp=2, fn=0),
        ]

        ordered = category_ids(frames, FailureCategory.MOST_FALSE_POSITIVES)

        self.assertEqual(ordered, ["000009", "000001"])

    def test_same_recall_uses_more_false_negatives_before_frame_id(self):
        frames = [
            frame("000001", tp=1, fp=0, fn=1),
            frame("000009", tp=2, fp=0, fn=2),
        ]

        ordered = category_ids(frames, FailureCategory.LOWEST_RECALL)

        self.assertEqual(ordered, ["000009", "000001"])

    def test_high_detection_ranking_uses_all_frozen_tie_breaks(self):
        frames = [
            frame("000004", tp=2, fp=1, fn=0, neutralized=1),
            frame("000003", tp=1, fp=2, fn=0, neutralized=1),
            frame("000002", tp=1, fp=2, fn=0, neutralized=1),
            frame("000001", tp=1, fp=1, fn=0, neutralized=2),
        ]

        ordered = category_ids(frames, FailureCategory.HIGHEST_EFFECTIVE_CAR_DETECTIONS)

        self.assertEqual(ordered, ["000002", "000003", "000004", "000001"])

    def test_frame_id_is_final_tie_break_and_input_order_does_not_matter(self):
        original = [
            frame("000003", tp=1, fp=1, fn=1),
            frame("000001", tp=1, fp=1, fn=1),
            frame("000002", tp=1, fp=1, fn=1),
        ]
        shuffled = list(original)
        random.Random(7).shuffle(shuffled)

        expected = ["000001", "000002", "000003"]
        self.assertEqual(category_ids(original, FailureCategory.MOST_FALSE_NEGATIVES), expected)
        self.assertEqual(category_ids(shuffled, FailureCategory.MOST_FALSE_NEGATIVES), expected)


if __name__ == "__main__":
    unittest.main()
