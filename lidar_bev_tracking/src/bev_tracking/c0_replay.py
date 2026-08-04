"""C0 exact replay gate for the v15.2 unified clustering entry point."""

from collections import Counter

from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.failure_evidence import build_failure_evidence_report


C0_REPLAY_SCHEMA_VERSION = "15.2-c0"


def compare_c0_reports(legacy_report, unified_report):
    mismatches = []
    legacy_summary = legacy_report["summary"]
    unified_summary = unified_report["summary"]
    fields = (
        "stage_point_counts",
        "filtered_point_hash",
        "cluster_signatures",
        "z_to_intensity_identical",
        "num_positive_gt",
        "num_false_negatives",
        "num_clusters",
        "num_raw_detections",
        "num_car_detections_before_nms",
        "num_detections_after_nms",
        "metrics_by_iou",
        "candidate_generation",
        "candidate_coverage",
        "primary_reason_counts",
    )
    for field in fields:
        if legacy_summary.get(field) != unified_summary.get(field):
            mismatches.append(f"summary.{field}")
    if legacy_report.get("gt_candidate_records") != unified_report.get("gt_candidate_records"):
        mismatches.append("gt_candidate_records")
    if legacy_report.get("failure_evidence") != unified_report.get("failure_evidence"):
        mismatches.append("failure_evidence")
    return mismatches


def run_c0_frame_replay(points, gt_boxes, frame_id, **kwargs):
    policy = ClusteringPolicy(
        mode="fixed",
        eps=float(kwargs.get("eps", 0.6)),
        min_points=int(kwargs.get("min_points", 20)),
    )
    legacy_report = build_failure_evidence_report(
        points,
        gt_boxes,
        frame_id=frame_id,
        **kwargs,
    )
    unified_report = build_failure_evidence_report(
        points,
        gt_boxes,
        frame_id=frame_id,
        clustering_policy=policy,
        **kwargs,
    )
    mismatches = compare_c0_reports(legacy_report, unified_report)
    return {
        "frame_id": str(frame_id).zfill(6),
        "passed": not mismatches,
        "mismatches": mismatches,
        "legacy_summary": legacy_report["summary"],
        "unified_summary": unified_report["summary"],
    }


def aggregate_c0_replay(frame_reports, manifest_metadata=None, source_run_id=None):
    mismatch_counts = Counter()
    for frame in frame_reports:
        mismatch_counts.update(frame["mismatches"])
    return {
        "schema_version": C0_REPLAY_SCHEMA_VERSION,
        "source": {
            "source_run_id": source_run_id,
            "diagnostic_manifest": manifest_metadata,
            "requested_frame_ids": [frame["frame_id"] for frame in frame_reports],
        },
        "summary": {
            "num_frames": len(frame_reports),
            "passed_frames": sum(frame["passed"] for frame in frame_reports),
            "failed_frames": sum(not frame["passed"] for frame in frame_reports),
            "replay_mismatch_count": sum(len(frame["mismatches"]) for frame in frame_reports),
            "mismatch_counts": dict(sorted(mismatch_counts.items())),
        },
        "frames": frame_reports,
    }
