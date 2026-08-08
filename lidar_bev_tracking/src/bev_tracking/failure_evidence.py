from collections import Counter
import hashlib

import numpy as np

from bev_tracking.clustering_detector import (
    cluster_to_box,
    cluster_to_oriented_box,
    euclidean_cluster,
    split_obstacle_filter_stages,
)
from bev_tracking.eval_policy import classify_gt_box, is_positive_detection
from bev_tracking.evaluation import evaluate_detections
from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.nms import nms_bev
from bev_tracking.result_types import FailureEvidence, FailureReason, FilterStageCounts


FAILURE_EVIDENCE_SCHEMA_VERSION = "15.1"
GT_COVERAGE_IOU_THRESHOLDS = (0.10, 0.15, 0.25, 0.50)


def build_failure_evidence_report(
    points,
    gt_boxes,
    frame_id,
    eps=0.6,
    min_points=20,
    oriented=True,
    z_min=-0.9,
    intensity_min=0.38,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    source_run_id=None,
    clustering_policy=None,
    candidate_variant=None,
):
    frame_id = str(frame_id).zfill(6)
    stages = split_obstacle_filter_stages(
        points,
        z_min=z_min,
        intensity_min=intensity_min,
    )
    if clustering_policy is None:
        clusters = euclidean_cluster(stages["intensity_filter"], eps=eps, min_points=min_points)
    else:
        from bev_tracking.adaptive_clustering import cluster_points

        clusters = cluster_points(stages["intensity_filter"], clustering_policy)
    box_fn = cluster_to_oriented_box if oriented else cluster_to_box
    raw_detections = [box_fn(cluster, index + 1) for index, cluster in enumerate(clusters)]
    detections_after_nms = nms_bev(raw_detections, iou_threshold=nms_iou_threshold)
    evaluation = evaluate_detections(
        detections_after_nms,
        gt_boxes,
        iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
    )

    positive_gt = [box for box in gt_boxes if classify_gt_box(box) == "positive"]
    gt_by_id = {box["id"]: box for box in positive_gt}
    false_negative_ids = [item["gt_id"] for item in evaluation["false_negatives"]]
    cluster_gt_ids = [
        {
            gt["id"]
            for gt in positive_gt
            if points_in_oriented_3d_box(cluster, gt).any()
        }
        for cluster in clusters
    ]

    evidence = [
        build_gt_failure_evidence(
            frame_id=frame_id,
            gt_box=gt_by_id[gt_id],
            stages=stages,
            cluster_gt_ids=cluster_gt_ids,
            raw_detections=raw_detections,
            detections_after_nms=detections_after_nms,
            min_points=min_points,
            eval_iou_threshold=eval_iou_threshold,
            source_run_id=source_run_id,
        )
        for gt_id in false_negative_ids
    ]
    reason_counts = Counter(item.primary_reason.value for item in evidence)
    metrics_by_iou = evaluation_metrics_by_iou(evaluation)
    stage_point_counts = {stage_name: int(len(stage_points)) for stage_name, stage_points in stages.items()}
    car_detections_before_nms = [det for det in raw_detections if is_positive_detection(det)]
    car_detections_after_nms = [det for det in detections_after_nms if is_positive_detection(det)]
    matched_gt_ids_by_iou = evaluation_matched_gt_ids_by_iou(evaluation)
    gt_candidate_records = [
        build_gt_candidate_record(
            frame_id=frame_id,
            gt_box=gt_box,
            stages=stages,
            cluster_gt_ids=cluster_gt_ids,
            raw_detections=raw_detections,
            detections_after_nms=detections_after_nms,
            matched_gt_ids_by_iou=matched_gt_ids_by_iou,
            primary_iou_key=f"{eval_iou_threshold:.2f}",
        )
        for gt_box in positive_gt
    ]
    candidate_coverage = summarize_gt_candidate_records(gt_candidate_records)
    primary_metrics = metrics_by_iou[f"{eval_iou_threshold:.2f}"]
    candidate_generation = {
        "cluster_count": int(len(clusters)),
        "raw_detection_count": int(len(raw_detections)),
        "car_candidate_count_before_nms": int(len(car_detections_before_nms)),
        "non_car_candidate_count": int(len(raw_detections) - len(car_detections_before_nms)),
        "nms_suppressed_count": int(len(raw_detections) - len(detections_after_nms)),
        "car_nms_suppressed_count": int(len(car_detections_before_nms) - len(car_detections_after_nms)),
        "final_car_detection_count": int(len(car_detections_after_nms)),
        "neutralized_detection_count": int(primary_metrics["neutralized_detections"]),
        "effective_car_detection_count": int(primary_metrics["effective_car_detection_count"]),
    }
    candidate_conversion = None
    if candidate_variant is not None:
        from bev_tracking.candidate_conversion import build_candidate_conversion_report

        candidate_conversion = build_candidate_conversion_report(
            frame_id=frame_id,
            gt_boxes=gt_boxes,
            stages=stages,
            clusters=clusters,
            raw_detections=raw_detections,
            detections_after_nms=detections_after_nms,
            evaluation=evaluation,
            variant=candidate_variant,
            min_points=min_points,
            source_run_id=source_run_id,
        )

    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "frame_id": frame_id,
        "parameters": {
            "eps": float(eps),
            "min_points": int(min_points),
            "oriented": bool(oriented),
            "z_min": float(z_min),
            "intensity_min": float(intensity_min),
            "nms_iou_threshold": float(nms_iou_threshold),
            "eval_iou_threshold": float(eval_iou_threshold),
            "auxiliary_iou_thresholds": [float(threshold) for threshold in auxiliary_iou_thresholds],
        },
        "summary": {
            "stage_point_counts": stage_point_counts,
            "filtered_point_hash": array_hash(stages["intensity_filter"]),
            "cluster_signatures": [array_hash(cluster) for cluster in clusters],
            "z_to_intensity_identical": bool(np.array_equal(stages["z_filter"], stages["intensity_filter"])),
            "num_positive_gt": int(len(positive_gt)),
            "num_false_negatives": int(len(false_negative_ids)),
            "num_clusters": int(len(clusters)),
            "num_raw_detections": int(len(raw_detections)),
            "num_car_detections_before_nms": int(len(car_detections_before_nms)),
            "num_detections_after_nms": int(len(detections_after_nms)),
            "metrics_by_iou": metrics_by_iou,
            "candidate_generation": candidate_generation,
            "candidate_coverage": candidate_coverage,
            "primary_reason_counts": dict(sorted(reason_counts.items())),
        },
        "gt_candidate_records": gt_candidate_records,
        "failure_evidence": [item.to_dict() for item in evidence],
        "candidate_conversion": candidate_conversion,
    }


def array_hash(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def evaluation_matched_gt_ids_by_iou(evaluation):
    outputs = {
        f'{evaluation["iou_threshold"]:.2f}': {
            str(item["gt_id"])
            for item in evaluation.get("matches", [])
        }
    }
    for iou_key, auxiliary in evaluation.get("auxiliary", {}).items():
        outputs[iou_key] = {
            str(item["gt_id"])
            for item in auxiliary.get("matches", [])
        }
    return outputs


def build_gt_candidate_record(
    frame_id,
    gt_box,
    stages,
    cluster_gt_ids,
    raw_detections,
    detections_after_nms,
    matched_gt_ids_by_iou,
    primary_iou_key,
):
    stage_counts = gt_filter_stage_counts(stages, gt_box)
    associated_indices = [
        index
        for index, gt_ids in enumerate(cluster_gt_ids)
        if gt_box["id"] in gt_ids
    ]
    associated_raw = [raw_detections[index] for index in associated_indices]
    associated_car_before_nms = [det for det in associated_raw if is_positive_detection(det)]
    kept_ids = {str(det["id"]) for det in detections_after_nms}
    associated_car_after_nms = [
        det
        for det in associated_car_before_nms
        if str(det["id"]) in kept_ids
    ]
    positive_before_nms = [det for det in raw_detections if is_positive_detection(det)]
    positive_after_nms = [det for det in detections_after_nms if is_positive_detection(det)]
    best_iou_before_nms, _ = best_iou_candidate(gt_box, positive_before_nms)
    best_iou_after_nms, _ = best_iou_candidate(gt_box, positive_after_nms)
    matched_by_iou = {
        iou_key: str(gt_box["id"]) in matched_gt_ids
        for iou_key, matched_gt_ids in sorted(matched_gt_ids_by_iou.items(), reverse=True)
    }

    if not associated_indices:
        candidate_outcome = "no_cluster"
    elif not associated_raw:
        candidate_outcome = "cluster_without_raw_detection"
    elif not associated_car_before_nms:
        candidate_outcome = "raw_detection_rejected_as_non_car"
    elif not associated_car_after_nms:
        candidate_outcome = "car_candidate_removed_by_nms"
    elif matched_by_iou.get(primary_iou_key, False):
        candidate_outcome = "matched_at_primary_iou"
    else:
        candidate_outcome = "final_car_candidate_below_primary_iou"

    range_xy_m = float(np.hypot(float(gt_box["x"]), float(gt_box["y"])))
    return {
        "frame_id": str(frame_id).zfill(6),
        "gt_id": str(gt_box["id"]),
        "range_xy_m": range_xy_m,
        "distance_bin": distance_bin(range_xy_m),
        "stage_point_counts": stage_counts.to_dict(),
        "cluster_ids": [f"cluster_{index + 1}" for index in associated_indices],
        "raw_detection_ids": [str(det["id"]) for det in associated_raw],
        "car_detection_ids_before_nms": [str(det["id"]) for det in associated_car_before_nms],
        "car_detection_ids_after_nms": [str(det["id"]) for det in associated_car_after_nms],
        "best_iou_before_nms": float(best_iou_before_nms),
        "best_iou_after_nms": float(best_iou_after_nms),
        "matched_by_iou": matched_by_iou,
        "candidate_outcome": candidate_outcome,
    }


def summarize_gt_candidate_records(records):
    counts = {
        "num_positive_gt": int(len(records)),
        "gt_with_cluster": sum(bool(item["cluster_ids"]) for item in records),
        "gt_with_raw_detection": sum(bool(item["raw_detection_ids"]) for item in records),
        "gt_with_car_detection_before_nms": sum(
            bool(item["car_detection_ids_before_nms"])
            for item in records
        ),
        "gt_with_car_detection_after_nms": sum(
            bool(item["car_detection_ids_after_nms"])
            for item in records
        ),
    }
    for threshold in GT_COVERAGE_IOU_THRESHOLDS:
        field = f"gt_with_best_iou_ge_{threshold:.2f}".replace(".", "_")
        counts[field] = sum(float(item["best_iou_after_nms"]) >= threshold for item in records)

    denominator = counts["num_positive_gt"]
    ratios = {
        field: float(value / denominator) if denominator else None
        for field, value in counts.items()
        if field != "num_positive_gt"
    }
    outcome_counts = Counter(item["candidate_outcome"] for item in records)
    zero_car_candidate_gt_count = sum(
        not item["car_detection_ids_after_nms"]
        for item in records
    )
    return {
        "counts": {field: int(value) for field, value in counts.items()},
        "ratios": ratios,
        "zero_detection_with_gt_eligible_count": int(zero_car_candidate_gt_count),
        "zero_car_candidate_gt_count": int(zero_car_candidate_gt_count),
        "candidate_outcome_counts": dict(sorted(outcome_counts.items())),
    }


def gt_filter_stage_counts(stages, gt_box):
    return FilterStageCounts(
        **{
            stage_name: int(points_in_oriented_3d_box(stage_points, gt_box).sum())
            for stage_name, stage_points in stages.items()
        }
    )


def distance_bin(range_xy_m):
    if range_xy_m < 15.0:
        return "near_0_15"
    if range_xy_m < 30.0:
        return "mid_15_30"
    return "far_30_inf"


def evaluation_metrics_by_iou(evaluation):
    outputs = {
        f'{evaluation["iou_threshold"]:.2f}': evaluation_metrics(evaluation),
    }
    for iou_key, auxiliary in evaluation.get("auxiliary", {}).items():
        outputs[iou_key] = evaluation_metrics(auxiliary)
    return outputs


def evaluation_metrics(evaluation):
    metrics = dict(evaluation["metrics"])
    metrics["neutralized_detections"] = int(len(evaluation.get("neutralized_detections", [])))
    metrics["effective_car_detection_count"] = int(
        metrics["tp"] + metrics["fp"] + metrics["neutralized_detections"]
    )
    return metrics


def build_gt_failure_evidence(
    frame_id,
    gt_box,
    stages,
    cluster_gt_ids,
    raw_detections,
    detections_after_nms,
    min_points,
    eval_iou_threshold,
    source_run_id=None,
):
    stage_counts = gt_filter_stage_counts(stages, gt_box)
    associated_indices = [
        index
        for index, gt_ids in enumerate(cluster_gt_ids)
        if gt_box["id"] in gt_ids
    ]
    associated_cluster_ids = [f"cluster_{index + 1}" for index in associated_indices]
    associated_raw = [raw_detections[index] for index in associated_indices]
    associated_car = [det for det in associated_raw if is_positive_detection(det)]
    kept_ids = {det["id"] for det in detections_after_nms}
    associated_after_nms = [det for det in associated_car if det["id"] in kept_ids]

    positive_raw = [det for det in raw_detections if is_positive_detection(det)]
    positive_after_nms = [det for det in detections_after_nms if is_positive_detection(det)]
    best_iou_before_nms, best_detection_before_nms = best_iou_candidate(gt_box, positive_raw)
    best_iou_after_nms, best_detection_after_nms = best_iou_candidate(gt_box, positive_after_nms)

    conditions = failure_conditions(
        gt_box=gt_box,
        stage_counts=stage_counts,
        associated_indices=associated_indices,
        cluster_gt_ids=cluster_gt_ids,
        associated_raw=associated_raw,
        associated_car=associated_car,
        associated_after_nms=associated_after_nms,
        min_points=min_points,
        best_iou_after_nms=best_iou_after_nms,
        eval_iou_threshold=eval_iou_threshold,
    )
    primary_reason = conditions[0] if conditions else FailureReason.UNRESOLVED

    return FailureEvidence(
        frame_id=frame_id,
        gt_id=gt_box["id"],
        primary_reason=primary_reason,
        stage_point_counts=stage_counts,
        supporting_flags=conditions[1:],
        cluster_ids=associated_cluster_ids,
        raw_detection_ids=[det["id"] for det in associated_raw],
        car_detection_ids_before_nms=[det["id"] for det in associated_car],
        detection_ids_after_nms=[det["id"] for det in associated_after_nms],
        best_iou_before_nms=best_iou_before_nms,
        best_iou_after_nms=best_iou_after_nms,
        best_detection_before_nms=detection_snapshot(best_detection_before_nms),
        best_detection_after_nms=detection_snapshot(best_detection_after_nms),
        geometry_delta_after_nms=box_geometry_delta(gt_box, best_detection_after_nms),
        gt_box=gt_box,
        source_run_id=source_run_id,
    )


def failure_conditions(
    gt_box,
    stage_counts,
    associated_indices,
    cluster_gt_ids,
    associated_raw,
    associated_car,
    associated_after_nms,
    min_points,
    best_iou_after_nms,
    eval_iou_threshold,
):
    conditions = []
    if stage_counts.raw == 0:
        return [FailureReason.NO_RAW_POINTS_IN_GT]
    if stage_counts.roi == 0:
        return [FailureReason.REMOVED_BY_ROI]
    if stage_counts.z_filter == 0:
        return [FailureReason.REMOVED_BY_Z_FILTER]
    if stage_counts.intensity_filter == 0:
        return [FailureReason.REMOVED_BY_INTENSITY_FILTER]

    if not associated_indices:
        if stage_counts.raw >= min_points and stage_counts.roi < min_points:
            conditions.append(FailureReason.REMOVED_BY_ROI)
        elif stage_counts.roi >= min_points and stage_counts.z_filter < min_points:
            conditions.append(FailureReason.REMOVED_BY_Z_FILTER)
        elif stage_counts.z_filter >= min_points and stage_counts.intensity_filter < min_points:
            conditions.append(FailureReason.REMOVED_BY_INTENSITY_FILTER)
        else:
            conditions.append(FailureReason.INSUFFICIENT_POINTS_FOR_CLUSTERING)
    else:
        merged = any(len(cluster_gt_ids[index]) > 1 for index in associated_indices)
        if merged:
            conditions.append(FailureReason.CLUSTER_MERGING)
        if len(associated_indices) > 1:
            conditions.append(FailureReason.CLUSTER_FRAGMENTATION)
        if associated_raw and not associated_car:
            conditions.append(FailureReason.REJECTED_BY_CAR_CLASSIFICATION)
        if associated_car and not associated_after_nms:
            conditions.append(FailureReason.REMOVED_BY_NMS)

    if stage_counts.intensity_filter < min_points:
        conditions.append(FailureReason.INSUFFICIENT_POINTS_FOR_CLUSTERING)
    if 0.0 < best_iou_after_nms < eval_iou_threshold:
        conditions.append(FailureReason.FINAL_IOU_BELOW_THRESHOLD)

    return unique_reasons(conditions)


def unique_reasons(reasons):
    output = []
    for reason in reasons:
        if reason not in output:
            output.append(reason)
    return output


def best_iou_candidate(gt_box, detections):
    if not detections:
        return 0.0, None

    ranked = sorted(
        ((float(bev_iou(gt_box, det)), det) for det in detections),
        key=lambda item: (
            -item[0],
            int(item[1].get("det_index", 0)),
            str(item[1].get("id", "")),
        ),
    )
    return ranked[0]


def detection_snapshot(detection):
    if detection is None:
        return None
    fields = ["id", "class_name", "x", "y", "z", "length", "width", "yaw", "score", "num_points", "det_index"]
    return {field: detection[field] for field in fields if field in detection}


def box_geometry_delta(gt_box, detection):
    if detection is None:
        return {}

    dx = float(detection["x"] - gt_box["x"])
    dy = float(detection["y"] - gt_box["y"])
    return {
        "dx_m": dx,
        "dy_m": dy,
        "center_error_m": float(np.hypot(dx, dy)),
        "length_error_m": float(detection["length"] - gt_box["length"]),
        "width_error_m": float(detection["width"] - gt_box["width"]),
        "yaw_error_rad": box_yaw_error(detection["yaw"], gt_box["yaw"]),
    }


def box_yaw_error(yaw_a, yaw_b):
    return float(abs((float(yaw_a) - float(yaw_b) + np.pi / 2.0) % np.pi - np.pi / 2.0))
