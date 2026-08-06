import json
import unittest

from bev_tracking.candidate_conversion import derive_terminal_state, validate_terminal_assignments, validate_min_points
from bev_tracking.result_types import CandidateConversionEvidence, CandidateConversionState


def state(**overrides):
    values = {
        "filtered_point_count": 20,
        "min_points": 20,
        "cluster_ids": ["cluster_1"],
        "raw_detection_ids": ["cluster_1"],
        "car_detection_ids_before_nms": ["cluster_1"],
        "car_detection_ids_after_nms": ["cluster_1"],
        "best_iou_after_nms": 0.1,
        "matched_at_primary_iou": False,
    }
    values.update(overrides)
    return derive_terminal_state(**values)


class CandidateConversionDay1Test(unittest.TestCase):
    def test_terminal_precedence_follows_pipeline(self):
        self.assertEqual(state(filtered_point_count=0), CandidateConversionState.NO_FILTERED_POINTS)
        self.assertEqual(state(filtered_point_count=10, cluster_ids=[], raw_detection_ids=[]), CandidateConversionState.INSUFFICIENT_FILTERED_POINTS_FOR_ASSOCIATION)
        self.assertEqual(state(cluster_ids=[], raw_detection_ids=[]), CandidateConversionState.NO_ASSOCIATED_CLUSTER)
        self.assertEqual(state(raw_detection_ids=[]), CandidateConversionState.NO_ASSOCIATED_CLUSTER)
        self.assertEqual(state(car_detection_ids_before_nms=[]), CandidateConversionState.REJECTED_BY_CAR_CLASSIFIER)
        self.assertEqual(state(car_detection_ids_after_nms=[]), CandidateConversionState.REMOVED_BY_NMS)

    def test_iou_terminal_states_are_disjoint(self):
        self.assertEqual(state(best_iou_after_nms=0.24), CandidateConversionState.BOX_IOU_BELOW_0_25)
        self.assertEqual(state(best_iou_after_nms=0.25), CandidateConversionState.BOX_IOU_0_25_TO_0_50)
        self.assertEqual(state(best_iou_after_nms=0.49), CandidateConversionState.BOX_IOU_0_25_TO_0_50)
        self.assertEqual(state(best_iou_after_nms=0.5), CandidateConversionState.IOU_GE_0_50_BUT_UNMATCHED)
        self.assertEqual(state(best_iou_after_nms=0.8, matched_at_primary_iou=True), CandidateConversionState.MATCHED_AT_0_50)

    def test_terminal_assignments_conserve_gt_set(self):
        records = [
            CandidateConversionEvidence("1", "gt_1", CandidateConversionState.MATCHED_AT_0_50, "C1"),
            CandidateConversionEvidence("1", "gt_2", CandidateConversionState.REJECTED_BY_CAR_CLASSIFIER, "C1"),
        ]
        counts = validate_terminal_assignments(records, ["gt_1", "gt_2"])
        self.assertEqual(sum(counts.values()), 2)
        with self.assertRaises(ValueError):
            validate_terminal_assignments(records, ["gt_1"])

    def test_duplicate_gt_assignment_is_rejected(self):
        records = [
            CandidateConversionEvidence("1", "gt_1", CandidateConversionState.MATCHED_AT_0_50, "C0"),
            CandidateConversionEvidence("2", "gt_1", CandidateConversionState.MATCHED_AT_0_50, "C0"),
        ]
        with self.assertRaises(ValueError):
            validate_terminal_assignments(records, ["gt_1"])

    def test_evidence_is_json_serializable(self):
        evidence = CandidateConversionEvidence(
            frame_id="317",
            gt_id="gt_1",
            terminal_state=CandidateConversionState.BOX_IOU_0_25_TO_0_50,
            variant="C1",
            candidate_branches=[{"cluster_id": "cluster_1", "classification": "car"}],
        )
        payload = evidence.to_dict()
        self.assertEqual(payload["frame_id"], "000317")
        self.assertEqual(payload["terminal_state"], "box_iou_0_25_to_0_50")
        json.dumps(payload)

    def test_min_points_validation_rejects_bool_and_non_positive(self):
        for value in (True, 0, -1, 2.5, "20", None):
            with self.assertRaises(ValueError):
                validate_min_points(value)
        self.assertEqual(validate_min_points(20), 20)


if __name__ == "__main__":
    unittest.main()
