"""Read-only data contract for the v15.3.2 point-retention oracle ladder."""

import numpy as np

POINT_RETENTION_SCHEMA_VERSION = "15.3.2-point-retention-day1"
STAGE_ORDER = ("raw", "roi", "z_filter", "intensity_filter")
PCA_STATUS_VALUES = ("valid", "insufficient_points", "degenerate_geometry")
PCA_MIN_POINTS = 3
MATERIAL_GAIN_IOU = 0.10


def pca_input_status(points):
    """Classify whether an XY point set can support a two-dimensional PCA box."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape N x 2 or wider")
    if len(points) < PCA_MIN_POINTS:
        return "insufficient_points"
    centered_xy = points[:, :2].astype(np.float64) - points[:, :2].mean(axis=0)
    if np.linalg.matrix_rank(centered_xy) < 2:
        return "degenerate_geometry"
    return "valid"


def validate_stage_point_counts(counts):
    if set(counts) != set(STAGE_ORDER):
        raise ValueError("stage point counts must define raw, roi, z_filter, intensity_filter")
    values = [int(counts[stage]) for stage in STAGE_ORDER]
    if any(value < 0 for value in values):
        raise ValueError("stage point counts must be non-negative")
    if any(left < right for left, right in zip(values, values[1:])):
        raise ValueError("GT stage point counts must be monotonically non-increasing")
    return True


def build_point_retention_day1(reports_by_variant, base_variant="C0", candidate_variant="C1"):
    """Freeze delta-22 and P1-control keys against canonical 15.3.1 evidence."""
    if base_variant not in reports_by_variant or candidate_variant not in reports_by_variant:
        raise ValueError("point-retention diagnostic requires C0 and C1")

    def index_candidate_evidence(reports, variant):
        indexed = {}
        for report in reports:
            evidence = (report.get("candidate_conversion") or {}).get("evidence")
            if not isinstance(evidence, list):
                raise ValueError(f"missing candidate conversion evidence for {variant}")
            for item in evidence:
                key = (str(item.get("frame_id")).zfill(6), str(item.get("gt_id")))
                if key in indexed:
                    raise ValueError(f"duplicate candidate evidence for {variant}: {key}")
                indexed[key] = item
        return indexed

    base = index_candidate_evidence(reports_by_variant[base_variant], base_variant)
    candidate = index_candidate_evidence(reports_by_variant[candidate_variant], candidate_variant)
    if set(base) != set(candidate):
        raise ValueError("C0 and C1 canonical GT sets differ")
    delta_keys = sorted(
        key for key in candidate
        if candidate[key].get("cluster_ids") and not base[key].get("cluster_ids")
    )

    p1_control_keys = set()
    for report in reports_by_variant[candidate_variant]:
        groups = (report.get("cluster_separability") or {}).get("records", [])
        for cluster_record in groups:
            if cluster_record.get("group") == "P1":
                p1_control_keys.update(
                    (str(report["frame_id"]).zfill(6), str(gt_id))
                    for gt_id in cluster_record.get("associated_gt_ids", [])
                )

    canonical_records = {}
    for key, item in candidate.items():
        counts = item.get("stage_point_counts")
        if not isinstance(counts, dict):
            raise ValueError(f"missing canonical stage_point_counts for C1: {key}")
        counts = {stage: int(counts[stage]) for stage in STAGE_ORDER}
        validate_stage_point_counts(counts)
        canonical_records[key] = {
            "frame_id": key[0],
            "gt_id": key[1],
            "variant": candidate_variant,
            "distance_bin": item.get("distance_bin"),
            "stage_point_counts": counts,
            "removed_point_counts": {
                "roi": counts["raw"] - counts["roi"],
                "z_filter": counts["roi"] - counts["z_filter"],
                "intensity_filter": counts["z_filter"] - counts["intensity_filter"],
            },
            "source_of_truth": "candidate_conversion.evidence.stage_point_counts",
        }

    def key_payload(key):
        return {"frame_id": key[0], "gt_id": key[1]}

    return {
        "schema_version": POINT_RETENTION_SCHEMA_VERSION,
        "stage_count_source_of_truth": "candidate_conversion.evidence.stage_point_counts",
        "base_variant": base_variant,
        "candidate_variant": candidate_variant,
        "oracle_ladder": {
            "O1": "best single associated-cluster PCA oracle",
            "O2": "unique-point union of all associated clusters",
            "O2_gt_clipped": "associated union restricted to the GT box",
            "O3": "post-intensity GT points",
            "O4": "post-z/pre-intensity GT points",
            "O5": "post-ROI/pre-z GT points",
            "O6": "raw GT points",
        },
        "pca_input_contract": {
            "minimum_points": PCA_MIN_POINTS,
            "valid_statuses": list(PCA_STATUS_VALUES),
            "rank_requirement_xy": 2,
        },
        "attribution_contract": {
            "material_gain_iou": MATERIAL_GAIN_IOU,
            "preserve_signed_deltas": True,
            "labels": [
                "CLUSTER_FRAGMENTATION_LIMITED",
                "CLUSTER_FORMATION_LIMITED",
                "INTENSITY_FILTER_LIMITED",
                "Z_FILTER_LIMITED",
                "ROI_FILTER_LIMITED",
                "RAW_GEOMETRY_OBSERVABILITY_LIMITED",
                "MIXED",
                "UNRESOLVED",
            ],
        },
        "delta_gt_count": int(len(delta_keys)),
        "delta_gt_keys": [key_payload(key) for key in delta_keys],
        "p1_control_gt_count": int(len(p1_control_keys)),
        "p1_control_gt_keys": [key_payload(key) for key in sorted(p1_control_keys)],
        "delta_records": [canonical_records[key] for key in delta_keys],
        "p1_control_records": [canonical_records[key] for key in sorted(p1_control_keys)],
    }
