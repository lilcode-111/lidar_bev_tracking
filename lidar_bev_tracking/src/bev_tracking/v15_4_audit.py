import hashlib

import numpy as np

from bev_tracking.eval_policy import classify_gt_box, is_positive_detection, normalize_class_name
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.v15_4_materialization import canonical_identity_sha256


AUDIT_SCHEMA_VERSION = "15.4-regression-background-audit-day3-v1"
ANNOTATION_EXCLUSION_CLASSES = {
    "car",
    "van",
    "truck",
    "pedestrian",
    "cyclist",
    "person_sitting",
    "tram",
    "misc",
}
REGRESSION_REASONS = (
    "NO_ASSOCIATED_CLUSTER",
    "REJECTED_BY_CAR_CLASSIFIER",
    "REMOVED_BY_NMS",
    "IOU_REGRESSION",
    "EVALUATION_COMPETITION",
)


class V154AuditError(ValueError):
    pass


def build_frame_source_point_universes(
    *, frame_id, stages, stage_source_indices, gt_boxes
):
    """Build threshold-dependent point universes from raw LiDAR row identity."""
    frame_id = str(frame_id).zfill(6)
    points = np.asarray(stages["intensity_filter"])
    source_indices = np.asarray(stage_source_indices["intensity_filter"], dtype=np.int64)
    if len(points) != len(source_indices):
        raise V154AuditError("intensity points and source indices are misaligned")

    positive_boxes = [box for box in gt_boxes if classify_gt_box(box) == "positive"]
    exclusion_boxes = [
        box
        for box in gt_boxes
        if normalize_class_name(box.get("class_name")) in ANNOTATION_EXCLUSION_CLASSES
    ]
    positive_mask = _union_box_mask(points, positive_boxes)
    annotation_mask = _union_box_mask(points, exclusion_boxes)
    global_indices = source_indices.tolist()
    positive_indices = source_indices[positive_mask].tolist()
    background_indices = source_indices[~annotation_mask].tolist()
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "frame_id": frame_id,
        "point_identity": "(frame_id, raw_lidar_point_index)",
        "coordinate_row_dedup_used": False,
        "annotation_exclusion_classes": sorted(ANNOTATION_EXCLUSION_CLASSES),
        "universes": {
            "global_post_intensity": _point_set_record(global_indices),
            "positive_gt_post_intensity": _point_set_record(positive_indices),
            "annotation_excluded_background_post_intensity": _point_set_record(background_indices),
        },
    }


def build_strict_background_lineage(
    *, frame_id, clusters, raw_detections, detections_after_nms, gt_boxes, evaluation
):
    """Follow 15.3.1 strict-background clusters through classifier, NMS and evaluation."""
    if len(clusters) != len(raw_detections):
        raise V154AuditError("cluster and raw detection counts differ")
    frame_id = str(frame_id).zfill(6)
    annotation_boxes = [
        box
        for box in gt_boxes
        if normalize_class_name(box.get("class_name")) in ANNOTATION_EXCLUSION_CLASSES
    ]
    positive_boxes = [box for box in gt_boxes if classify_gt_box(box) == "positive"]
    kept_ids = {str(item["id"]) for item in detections_after_nms}
    primary_fp = _evaluation_det_ids(evaluation.get("false_positives", []))
    auxiliary_fp = {
        key: _evaluation_det_ids(value.get("false_positives", []))
        for key, value in evaluation.get("auxiliary", {}).items()
    }
    det_indices = _evaluation_det_indices(evaluation)

    records = []
    for index, (cluster, detection) in enumerate(zip(clusters, raw_detections), start=1):
        positive_association = any(
            points_in_oriented_3d_box(cluster, box).any() for box in positive_boxes
        )
        overlapping_annotations = sorted(
            {
                normalize_class_name(box.get("class_name"))
                for box in annotation_boxes
                if points_in_oriented_3d_box(cluster, box).any()
            }
        )
        if positive_association or overlapping_annotations:
            continue
        raw_id = str(detection["id"])
        is_car = bool(is_positive_detection(detection))
        after_nms = raw_id in kept_ids
        records.append(
            {
                "frame_id": frame_id,
                "cluster_id": f"cluster_{index}",
                "stable_cluster_signature": _array_hash(cluster),
                "raw_detection_id": raw_id,
                "detection_identity": f"{frame_id}:{raw_id}",
                "det_index": det_indices.get(raw_id),
                "class_name": normalize_class_name(detection.get("class_name")),
                "strict_background": True,
                "positive_car_gt_association": False,
                "annotation_overlap_categories": [],
                "raw_detection": True,
                "car_candidate_before_nms": is_car,
                "after_nms": after_nms,
                "final_car_candidate": bool(is_car and after_nms),
                "FP@0.50": bool(raw_id in primary_fp),
                "FP@0.25": bool(raw_id in auxiliary_fp.get("0.25", set())),
            }
        )
    identities = [item["detection_identity"] for item in records]
    if len(identities) != len(set(identities)):
        raise V154AuditError("strict-background lineage contains duplicate detection identity")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "frame_id": frame_id,
        "unit_of_analysis": "unique_detection_identity",
        "definition_source": "15.3.1 strict-background semantics",
        "record_count": len(records),
        "identity_sha256": canonical_identity_sha256(sorted(identities)),
        "records": records,
    }


def build_tp_regression_audit(t0_report, variant_report, iou_keys=("0.50", "0.25")):
    t0 = _index_gt_records(t0_report)
    variant = _index_gt_records(variant_report)
    if set(t0) != set(variant):
        raise V154AuditError("T0 and variant positive GT identity sets differ")
    output = {}
    for iou_key in iou_keys:
        t0_tp = {key for key, item in t0.items() if bool(item["matched_by_iou"][iou_key])}
        variant_tp = {key for key, item in variant.items() if bool(item["matched_by_iou"][iou_key])}
        output[iou_key] = {
            "TP_regressed_GT": _identity_set_record(t0_tp - variant_tp),
            "TP_improved_GT": _identity_set_record(variant_tp - t0_tp),
            "TP_unchanged_GT": _identity_set_record(t0_tp & variant_tp),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "gt_identity": "(frame_id, gt_id)",
        "by_iou": output,
    }


def build_candidate_regression_audit(t0_report, variant_report):
    t0 = _index_gt_records(t0_report)
    variant = _index_gt_records(variant_report)
    if set(t0) != set(variant):
        raise V154AuditError("T0 and variant positive GT identity sets differ")
    t0_positive = {key for key, item in t0.items() if item["car_detection_ids_after_nms"]}
    variant_positive = {key for key, item in variant.items() if item["car_detection_ids_after_nms"]}
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "gt_identity": "(frame_id, gt_id)",
        "candidate_regressed_GT": _identity_set_record(t0_positive - variant_positive),
        "candidate_improved_GT": _identity_set_record(variant_positive - t0_positive),
        "candidate_unchanged_GT": _identity_set_record(t0_positive & variant_positive),
    }


def classify_regression_reasons(t0_report, variant_report, iou_key):
    t0 = _index_gt_records(t0_report)
    variant = _index_gt_records(variant_report)
    if set(t0) != set(variant):
        raise V154AuditError("T0 and variant positive GT identity sets differ")
    records = []
    for key in sorted(t0):
        if not t0[key]["matched_by_iou"][iou_key] or variant[key]["matched_by_iou"][iou_key]:
            continue
        item = variant[key]
        if not item["cluster_ids"]:
            reason = "NO_ASSOCIATED_CLUSTER"
        elif not item["car_detection_ids_before_nms"]:
            reason = "REJECTED_BY_CAR_CLASSIFIER"
        elif not item["car_detection_ids_after_nms"]:
            reason = "REMOVED_BY_NMS"
        elif float(item["best_iou_after_nms"]) >= float(iou_key):
            reason = "EVALUATION_COMPETITION"
        else:
            reason = "IOU_REGRESSION"
        records.append({"frame_id": key[0], "gt_id": key[1], "reason": reason})
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "iou": str(iou_key),
        "allowed_reasons": list(REGRESSION_REASONS),
        "unexplained_count": 0,
        "records": records,
    }


def extract_monotonicity_universes(report):
    output = {
        "global": [],
        "positive_gt": [],
        "annotation_excluded_background": [],
    }
    frames = report.get("source_point_universes", [])
    for frame in frames:
        frame_id = str(frame["frame_id"]).zfill(6)
        mapping = {
            "global": "global_post_intensity",
            "positive_gt": "positive_gt_post_intensity",
            "annotation_excluded_background": "annotation_excluded_background_post_intensity",
        }
        for output_name, source_name in mapping.items():
            output[output_name].extend(
                (frame_id, int(index))
                for index in frame["universes"][source_name]["source_point_indices"]
            )
    return output


def _index_gt_records(report):
    records = report.get("gt_candidate_records")
    if not isinstance(records, list):
        raise V154AuditError("report is missing gt_candidate_records")
    output = {}
    for item in records:
        key = (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
        if key in output:
            raise V154AuditError(f"duplicate GT identity: {key}")
        output[key] = item
    return output


def _identity_set_record(values):
    ordered = [[frame_id, gt_id] for frame_id, gt_id in sorted(values)]
    return {
        "count": len(ordered),
        "identity_list": ordered,
        "identity_sha256": canonical_identity_sha256(ordered),
    }


def _point_set_record(indices):
    ordered = [int(value) for value in indices]
    if len(ordered) != len(set(ordered)):
        raise V154AuditError("source point universe contains duplicate raw indices")
    return {
        "count": len(ordered),
        "source_point_indices": ordered,
        "source_point_indices_sha256": canonical_identity_sha256(ordered),
    }


def _union_box_mask(points, boxes):
    mask = np.zeros(len(points), dtype=bool)
    for box in boxes:
        mask |= points_in_oriented_3d_box(points, box)
    return mask


def _evaluation_det_ids(items):
    return {str(item.get("det_id")) for item in items}


def _evaluation_det_indices(evaluation):
    output = {}
    categories = ("matches", "neutralized_detections", "false_positives")
    for category in categories:
        for item in evaluation.get(category, []):
            output[str(item.get("det_id"))] = int(item.get("det_index", 0))
    return output


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
