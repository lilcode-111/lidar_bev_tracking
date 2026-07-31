from collections import Counter
from pathlib import Path

from bev_tracking.failure_evidence import build_failure_evidence_report
from bev_tracking.geometry_sanity import (
    DEFAULT_CENTER_TOLERANCE_M,
    DEFAULT_YAW_TOLERANCE_RAD,
    build_geometry_sanity_report,
)
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib


FAILURE_EVIDENCE_BATCH_SCHEMA_VERSION = "15.0"


def load_diagnostic_frame_ids(manifest_path):
    manifest_path = Path(manifest_path)
    frame_ids = [
        str(line.strip()).zfill(6)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frame_ids:
        raise ValueError("diagnostic frame manifest must not be empty")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("diagnostic frame manifest contains duplicate frame_id")
    return frame_ids


def run_kitti_diagnostic_failure_evidence(
    data_root,
    frame_ids,
    eps=0.6,
    min_points=20,
    oriented=True,
    z_min=-0.9,
    intensity_min=0.38,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    center_tolerance_m=DEFAULT_CENTER_TOLERANCE_M,
    yaw_tolerance_rad=DEFAULT_YAW_TOLERANCE_RAD,
    source_run_id=None,
    progress_callback=None,
):
    normalized_ids = [str(frame_id).zfill(6) for frame_id in frame_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("diagnostic frame_ids contain duplicates")

    frame_reports = []
    total = len(normalized_ids)
    for index, frame_id in enumerate(normalized_ids, start=1):
        if progress_callback is not None:
            progress_callback(index, total, frame_id)

        velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
        calib_path = resolve_kitti_calib_path(data_root, frame_id)
        points = load_kitti_point_cloud(velodyne_path)
        labels = load_kitti_labels(label_path)
        calib = load_kitti_calib(calib_path)
        gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)

        geometry_report = build_geometry_sanity_report(
            points,
            labels,
            calib,
            frame_id=frame_id,
            center_tolerance_m=center_tolerance_m,
            yaw_tolerance_rad=yaw_tolerance_rad,
        )
        evidence_report = build_failure_evidence_report(
            points,
            gt_boxes,
            frame_id=frame_id,
            eps=eps,
            min_points=min_points,
            oriented=oriented,
            z_min=z_min,
            intensity_min=intensity_min,
            nms_iou_threshold=nms_iou_threshold,
            eval_iou_threshold=eval_iou_threshold,
            source_run_id=source_run_id,
        )
        frame_reports.append(
            {
                "frame_id": frame_id,
                "geometry_sanity": geometry_report,
                "failure_evidence": evidence_report,
            }
        )

    return aggregate_diagnostic_reports(
        frame_reports,
        data_root=data_root,
        requested_frame_ids=normalized_ids,
        source_run_id=source_run_id,
    )


def aggregate_diagnostic_reports(frame_reports, data_root, requested_frame_ids, source_run_id=None):
    primary_reasons = Counter()
    supporting_flags = Counter()
    total_positive_gt = 0
    total_false_negatives = 0
    geometry_passed_frames = 0
    low_iou_deltas = []

    for frame_report in frame_reports:
        geometry_report = frame_report["geometry_sanity"]
        evidence_report = frame_report["failure_evidence"]
        geometry_passed_frames += int(bool(geometry_report["passed"]))
        total_positive_gt += int(evidence_report["summary"]["num_positive_gt"])
        total_false_negatives += int(evidence_report["summary"]["num_false_negatives"])

        for item in evidence_report["failure_evidence"]:
            primary_reasons[item["primary_reason"]] += 1
            supporting_flags.update(item.get("supporting_flags", []))
            if item["primary_reason"] == "final_iou_below_threshold" and item.get("geometry_delta_after_nms"):
                low_iou_deltas.append(item["geometry_delta_after_nms"])

    return {
        "schema_version": FAILURE_EVIDENCE_BATCH_SCHEMA_VERSION,
        "source": {
            "data_root": str(data_root),
            "source_run_id": source_run_id,
            "requested_frame_ids": list(requested_frame_ids),
        },
        "summary": {
            "num_frames": int(len(frame_reports)),
            "geometry_passed_frames": int(geometry_passed_frames),
            "geometry_failed_frames": int(len(frame_reports) - geometry_passed_frames),
            "num_positive_gt": int(total_positive_gt),
            "num_false_negatives": int(total_false_negatives),
            "primary_reason_counts": dict(sorted(primary_reasons.items())),
            "supporting_flag_counts": dict(sorted(supporting_flags.items())),
            "low_iou_geometry": summarize_geometry_deltas(low_iou_deltas),
        },
        "frames": frame_reports,
    }


def summarize_geometry_deltas(deltas):
    fields = ["center_error_m", "length_error_m", "width_error_m", "yaw_error_rad"]
    summary = {"count": int(len(deltas))}
    for field in fields:
        values = [float(item[field]) for item in deltas if field in item]
        summary[field] = {
            "mean_abs": float(sum(abs(value) for value in values) / len(values)) if values else None,
            "max_abs": float(max((abs(value) for value in values), default=0.0)) if values else None,
        }
    return summary
