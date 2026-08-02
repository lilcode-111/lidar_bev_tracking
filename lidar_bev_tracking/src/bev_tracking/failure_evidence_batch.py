from collections import Counter
from pathlib import Path

from bev_tracking.experiment_gate import sha256_file
from bev_tracking.failure_evidence import (
    build_failure_evidence_report,
    summarize_gt_candidate_records,
)
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
from bev_tracking.kitti_yaw_validation import (
    DEFAULT_CORNER_TOLERANCE_M,
    DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD,
    build_kitti_yaw_semantic_report,
)


FAILURE_EVIDENCE_BATCH_SCHEMA_VERSION = "15.1"


def load_diagnostic_frame_ids(manifest_path):
    return load_diagnostic_manifest(manifest_path)["frame_ids"]


def load_diagnostic_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    raw_ids = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(frame_id) != 6 or not frame_id.isdigit() for frame_id in raw_ids):
        raise ValueError("diagnostic frame manifest must contain six-digit frame ids")
    frame_ids = [str(frame_id).zfill(6) for frame_id in raw_ids]
    if not frame_ids:
        raise ValueError("diagnostic frame manifest must not be empty")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("diagnostic frame manifest contains duplicate frame_id")
    if frame_ids != sorted(frame_ids):
        raise ValueError("diagnostic frame manifest must be sorted")
    return {
        "path": manifest_path.as_posix(),
        "sha256": sha256_file(manifest_path),
        "num_frames": int(len(frame_ids)),
        "frame_ids": frame_ids,
    }


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
    auxiliary_iou_thresholds=(0.25,),
    center_tolerance_m=DEFAULT_CENTER_TOLERANCE_M,
    yaw_tolerance_rad=DEFAULT_YAW_TOLERANCE_RAD,
    yaw_semantic_tolerance_rad=DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD,
    corner_tolerance_m=DEFAULT_CORNER_TOLERANCE_M,
    manifest_metadata=None,
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
        yaw_semantic_report = build_kitti_yaw_semantic_report(
            labels,
            calib,
            frame_id=frame_id,
            yaw_tolerance_rad=yaw_semantic_tolerance_rad,
            corner_tolerance_m=corner_tolerance_m,
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
            auxiliary_iou_thresholds=auxiliary_iou_thresholds,
            source_run_id=source_run_id,
        )
        frame_reports.append(
            {
                "frame_id": frame_id,
                "geometry_sanity": geometry_report,
                "yaw_semantics": yaw_semantic_report,
                "failure_evidence": evidence_report,
            }
        )

    return aggregate_diagnostic_reports(
        frame_reports,
        data_root=data_root,
        requested_frame_ids=normalized_ids,
        manifest_metadata=manifest_metadata,
        source_run_id=source_run_id,
    )


def aggregate_diagnostic_reports(
    frame_reports,
    data_root,
    requested_frame_ids,
    manifest_metadata=None,
    source_run_id=None,
):
    primary_reasons = Counter()
    supporting_flags = Counter()
    total_positive_gt = 0
    total_false_negatives = 0
    geometry_passed_frames = 0
    low_iou_deltas = []
    geometry_measurements = {
        "center_round_trip_error_m": [],
        "yaw_round_trip_error_rad": [],
        "yaw_semantic_error_rad": [],
        "corner_alignment_error_m": [],
    }
    geometry_tolerances = {}
    gt_candidate_records = []
    stage_point_totals = {"raw": 0, "roi": 0, "z_filter": 0, "intensity_filter": 0}
    candidate_generation_totals = {
        "cluster_count": 0,
        "raw_detection_count": 0,
        "car_candidate_count_before_nms": 0,
        "non_car_candidate_count": 0,
        "nms_suppressed_count": 0,
        "car_nms_suppressed_count": 0,
        "final_car_detection_count": 0,
        "neutralized_detection_count": 0,
        "effective_car_detection_count": 0,
    }

    for frame_report in frame_reports:
        geometry_report = frame_report["geometry_sanity"]
        yaw_semantic_report = frame_report["yaw_semantics"]
        evidence_report = frame_report["failure_evidence"]
        geometry_passed_frames += int(bool(geometry_report["passed"] and yaw_semantic_report["passed"]))
        total_positive_gt += int(evidence_report["summary"]["num_positive_gt"])
        total_false_negatives += int(evidence_report["summary"]["num_false_negatives"])
        gt_candidate_records.extend(evidence_report.get("gt_candidate_records", []))

        for stage_name, value in evidence_report["summary"].get("stage_point_counts", {}).items():
            stage_point_totals[stage_name] += int(value)
        for field, value in evidence_report["summary"].get("candidate_generation", {}).items():
            candidate_generation_totals[field] += int(value)

        collect_box_measurements(
            geometry_measurements,
            frame_report["frame_id"],
            geometry_report.get("boxes", []),
            ("center_round_trip_error_m", "yaw_round_trip_error_rad"),
        )
        collect_box_measurements(
            geometry_measurements,
            frame_report["frame_id"],
            yaw_semantic_report.get("boxes", []),
            ("yaw_semantic_error_rad", "corner_alignment_error_m"),
        )
        geometry_tolerances.update(
            {
                "center_round_trip_error_m": geometry_report["tolerances"]["center_error_m"],
                "yaw_round_trip_error_rad": geometry_report["tolerances"]["yaw_error_rad"],
                "yaw_semantic_error_rad": yaw_semantic_report["tolerances"]["yaw_semantic_error_rad"],
                "corner_alignment_error_m": yaw_semantic_report["tolerances"]["corner_alignment_error_m"],
            }
        )

        for item in evidence_report["failure_evidence"]:
            primary_reasons[item["primary_reason"]] += 1
            supporting_flags.update(item.get("supporting_flags", []))
            if item["primary_reason"] == "final_iou_below_threshold" and item.get("geometry_delta_after_nms"):
                low_iou_deltas.append(item["geometry_delta_after_nms"])

    candidate_coverage = summarize_gt_candidate_records(gt_candidate_records)
    if gt_candidate_records and candidate_coverage["counts"]["num_positive_gt"] != total_positive_gt:
        raise ValueError("GT candidate record count does not match positive GT count")

    return {
        "schema_version": FAILURE_EVIDENCE_BATCH_SCHEMA_VERSION,
        "source": {
            "data_root": str(data_root),
            "source_run_id": source_run_id,
            "requested_frame_ids": list(requested_frame_ids),
            "diagnostic_manifest": manifest_metadata,
        },
        "summary": {
            "num_frames": int(len(frame_reports)),
            "geometry_passed_frames": int(geometry_passed_frames),
            "geometry_failed_frames": int(len(frame_reports) - geometry_passed_frames),
            "geometry_errors": summarize_geometry_measurements(
                geometry_measurements,
                geometry_tolerances,
            ),
            "num_positive_gt": int(total_positive_gt),
            "num_false_negatives": int(total_false_negatives),
            "stage_point_counts": stage_point_totals,
            "candidate_generation_totals": candidate_generation_totals,
            "candidate_coverage": candidate_coverage,
            "primary_reason_counts": dict(sorted(primary_reasons.items())),
            "supporting_flag_counts": dict(sorted(supporting_flags.items())),
            "low_iou_geometry": summarize_geometry_deltas(low_iou_deltas),
        },
        "gt_candidate_records": gt_candidate_records,
        "frames": frame_reports,
    }


def collect_box_measurements(target, frame_id, boxes, fields):
    for box in boxes:
        for field in fields:
            target[field].append(
                {
                    "frame_id": str(frame_id).zfill(6),
                    "gt_id": str(box.get("gt_id", "")),
                    "class_name": str(box.get("class_name", "")),
                    "value": float(box[field]),
                }
            )


def summarize_geometry_measurements(measurements, tolerances):
    summary = {}
    for name, items in measurements.items():
        tolerance = float(tolerances[name])
        ordered = sorted(
            items,
            key=lambda item: (-item["value"], item["frame_id"], item["gt_id"]),
        )
        values = [item["value"] for item in items]
        maximum = max(values, default=0.0)
        summary[name] = {
            "count": int(len(values)),
            "mean": float(sum(values) / len(values)) if values else None,
            "max": float(maximum),
            "tolerance": tolerance,
            "passed": bool(maximum <= tolerance),
            "worst_object": ordered[0] if ordered else None,
        }
    return summary


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
