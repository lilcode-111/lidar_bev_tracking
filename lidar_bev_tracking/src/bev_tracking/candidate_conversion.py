"""Protocol helpers for the v15.3 Car-candidate conversion diagnosis.

This module only defines attribution semantics. It does not run clustering,
classification, NMS, or evaluation.
"""

from collections import Counter
from numbers import Integral

from bev_tracking.result_types import CandidateConversionState


PRIMARY_IOU = 0.50
AUXILIARY_IOU = 0.25


def derive_terminal_state(
    *,
    filtered_point_count,
    min_points,
    cluster_ids,
    raw_detection_ids,
    car_detection_ids_before_nms,
    car_detection_ids_after_nms,
    best_iou_after_nms,
    matched_at_primary_iou,
):
    """Assign exactly one terminal state using the frozen pipeline order."""
    if filtered_point_count == 0:
        return CandidateConversionState.NO_FILTERED_POINTS
    if filtered_point_count < min_points and not cluster_ids:
        return CandidateConversionState.INSUFFICIENT_FILTERED_POINTS_FOR_ASSOCIATION
    if not cluster_ids:
        return CandidateConversionState.NO_ASSOCIATED_CLUSTER
    if not raw_detection_ids:
        return CandidateConversionState.NO_ASSOCIATED_CLUSTER
    if not car_detection_ids_before_nms:
        return CandidateConversionState.REJECTED_BY_CAR_CLASSIFIER
    if not car_detection_ids_after_nms:
        return CandidateConversionState.REMOVED_BY_NMS
    if matched_at_primary_iou:
        return CandidateConversionState.MATCHED_AT_0_50
    if best_iou_after_nms < AUXILIARY_IOU:
        return CandidateConversionState.BOX_IOU_BELOW_0_25
    if best_iou_after_nms < PRIMARY_IOU:
        return CandidateConversionState.BOX_IOU_0_25_TO_0_50
    return CandidateConversionState.IOU_GE_0_50_BUT_UNMATCHED


def validate_terminal_assignments(records, expected_gt_ids):
    """Validate uniqueness and conservation of GT terminal assignments."""
    expected = {str(gt_id) for gt_id in expected_gt_ids}
    actual_ids = [str(record.gt_id) for record in records]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("duplicate GT terminal assignment")
    if set(actual_ids) != expected:
        raise ValueError("GT terminal assignments do not conserve positive GT set")
    counts = Counter(record.terminal_state.value for record in records)
    return {state.value: int(counts.get(state.value, 0)) for state in CandidateConversionState}


def validate_min_points(min_points):
    if isinstance(min_points, bool) or not isinstance(min_points, Integral) or min_points <= 0:
        raise ValueError("min_points must be a positive integer")
    return int(min_points)
