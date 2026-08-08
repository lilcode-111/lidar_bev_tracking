"""Protocol helpers for the v15.3 Car-candidate conversion diagnosis.

This module only defines attribution semantics. It does not run clustering,
classification, NMS, or evaluation.
"""

from collections import Counter
import math
from numbers import Integral

import numpy as np

from bev_tracking.eval_policy import is_positive_detection
from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.result_types import CandidateConversionEvidence, CandidateConversionState


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


def gt_cluster_association_indices(filtered_points, gt_box, clusters):
    """Return all associated cluster indices using the frozen v15.2 gate."""
    gt_point_count = int(points_in_oriented_3d_box(filtered_points, gt_box).sum())
    if gt_point_count < 3:
        return gt_point_count, []
    minimum = max(3, int(math.ceil(0.10 * gt_point_count)))
    associated = []
    for index, cluster in enumerate(clusters):
        if int(points_in_oriented_3d_box(cluster, gt_box).sum()) >= minimum:
            associated.append(index)
    return gt_point_count, associated


def _axis_features(cluster):
    x_span = float(max(cluster[:, 0].max() - cluster[:, 0].min(), 0.1))
    y_span = float(max(cluster[:, 1].max() - cluster[:, 1].min(), 0.1))
    z_span = float(max(cluster[:, 2].max() - cluster[:, 2].min(), 0.0))
    return {
        "num_points": int(len(cluster)),
        "axis_length": x_span,
        "axis_width": y_span,
        "height_span": z_span,
        "range_xy_m": float(np.hypot(float(cluster[:, 0].mean()), float(cluster[:, 1].mean()))),
        "point_density_xy": float(len(cluster) / max(x_span * y_span, 1e-6)),
    }


def _branch_record(cluster_index, cluster, raw_detection, kept_ids, gt_box):
    detection_id = str(raw_detection.get("id", f"cluster_{cluster_index + 1}"))
    is_kept = detection_id in kept_ids
    return {
        "cluster_id": f"cluster_{cluster_index + 1}",
        "raw_detection_id": detection_id,
        "class_name": str(raw_detection.get("class_name", "")),
        "is_car_candidate": bool(is_positive_detection(raw_detection)),
        "is_after_nms": bool(is_kept),
        "cluster_features": _axis_features(cluster),
        "pca_length": float(raw_detection.get("length", 0.0)),
        "pca_width": float(raw_detection.get("width", 0.0)),
        "candidate_iou": float(bev_iou(gt_box, raw_detection)),
    }


def build_candidate_conversion_evidence(
    *,
    frame_id,
    gt_box,
    stages,
    clusters,
    raw_detections,
    detections_after_nms,
    evaluation,
    variant,
    min_points,
    source_run_id=None,
):
    """Build one GT's complete C0/C1 candidate-conversion lineage."""
    validate_min_points(min_points)
    filtered_points = stages["intensity_filter"]
    filtered_count, associated_indices = gt_cluster_association_indices(filtered_points, gt_box, clusters)
    associated_raw = [raw_detections[index] for index in associated_indices]
    kept_ids = {str(det.get("id")) for det in detections_after_nms}
    car_before = [det for det in associated_raw if is_positive_detection(det)]
    car_after = [det for det in car_before if str(det.get("id")) in kept_ids]
    branches = [
        _branch_record(index, clusters[index], raw_detections[index], kept_ids, gt_box)
        for index in associated_indices
    ]
    associated_after_ids = {str(det.get("id")) for det in car_after}
    associated_after = [det for det in car_after if str(det.get("id")) in associated_after_ids]
    best_before = max((float(bev_iou(gt_box, det)) for det in car_before), default=0.0)
    best_after = max((float(bev_iou(gt_box, det)) for det in associated_after), default=0.0)
    matched_ids = {
        str(item.get("det_id"))
        for item in evaluation.get("matches", [])
        if str(item.get("gt_id")) == str(gt_box.get("id"))
    }
    matched = bool(matched_ids & associated_after_ids)
    terminal_state = derive_terminal_state(
        filtered_point_count=filtered_count,
        min_points=min_points,
        cluster_ids=[f"cluster_{index + 1}" for index in associated_indices],
        raw_detection_ids=[str(det.get("id")) for det in associated_raw],
        car_detection_ids_before_nms=[str(det.get("id")) for det in car_before],
        car_detection_ids_after_nms=[str(det.get("id")) for det in car_after],
        best_iou_after_nms=best_after,
        matched_at_primary_iou=matched,
    )
    frame_id = str(frame_id).zfill(6)
    range_xy = float(np.hypot(float(gt_box["x"]), float(gt_box["y"])))
    distance_bin = "near_0_15" if range_xy < 15.0 else "mid_15_30" if range_xy < 30.0 else "far_30_inf"
    return CandidateConversionEvidence(
        frame_id=frame_id,
        gt_id=gt_box["id"],
        terminal_state=terminal_state,
        variant=variant,
        distance_bin=distance_bin,
        stage_point_counts={
            name: int(points_in_oriented_3d_box(stage_points, gt_box).sum())
            for name, stage_points in stages.items()
        },
        cluster_ids=[f"cluster_{index + 1}" for index in associated_indices],
        raw_detection_ids=[str(det.get("id")) for det in associated_raw],
        car_detection_ids_before_nms=[str(det.get("id")) for det in car_before],
        car_detection_ids_after_nms=[str(det.get("id")) for det in car_after],
        candidate_branches=branches,
        best_iou_before_nms=best_before,
        best_iou_after_nms=best_after,
        matched_at_primary_iou=matched,
        source_run_id=source_run_id,
    )


def build_candidate_conversion_report(
    *,
    frame_id,
    gt_boxes,
    stages,
    clusters,
    raw_detections,
    detections_after_nms,
    evaluation,
    variant,
    min_points,
    source_run_id=None,
):
    """Build evidence for every positive GT and summarize terminal states."""
    positive_gt = [box for box in gt_boxes if str(box.get("class_name", "")).lower() == "car"]
    records = [
        build_candidate_conversion_evidence(
            frame_id=frame_id,
            gt_box=gt_box,
            stages=stages,
            clusters=clusters,
            raw_detections=raw_detections,
            detections_after_nms=detections_after_nms,
            evaluation=evaluation,
            variant=variant,
            min_points=min_points,
            source_run_id=source_run_id,
        )
        for gt_box in positive_gt
    ]
    counts = validate_terminal_assignments(records, [box["id"] for box in positive_gt])
    return {
        "frame_id": str(frame_id).zfill(6),
        "variant": str(variant),
        "num_positive_gt": len(positive_gt),
        "terminal_state_counts": counts,
        "evidence": [record.to_dict() for record in records],
    }
