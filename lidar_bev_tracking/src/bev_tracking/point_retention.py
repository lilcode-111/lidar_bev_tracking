"""Read-only data contract for the v15.3.2 point-retention oracle ladder."""

import numpy as np

from bev_tracking.eval_policy import classify_gt_box
from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.oriented_box import estimate_oriented_box_xy

POINT_RETENTION_SCHEMA_VERSION = "15.3.2-point-retention-day1"
FRAGMENT_ORACLE_SCHEMA_VERSION = "15.3.2-fragment-oracles-day2"
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


def build_pca_oracle(points, gt_box):
    """Fit the frozen PCA box when valid; never synthesize evidence for bad inputs."""
    points = np.asarray(points)
    status = pca_input_status(points)
    result = {
        "status": status,
        "num_points": int(len(points)),
        "box": None,
        "iou": None,
    }
    if status != "valid":
        return result
    x, y, length, width, yaw = estimate_oriented_box_xy(points)
    box = {
        "x": float(x), "y": float(y), "length": float(length),
        "width": float(width), "yaw": float(yaw),
    }
    result["box"] = box
    result["iou"] = float(bev_iou(gt_box, box))
    return result


def build_frame_fragment_oracles(
    *, frame_id, variant, gt_boxes, clusters, raw_detections, candidate_conversion
):
    """Build O1/O2/O2_gt_clipped from canonical associations without reclustering."""
    if len(clusters) != len(raw_detections):
        raise ValueError("clusters and raw_detections must have the same length")
    frame_id = str(frame_id).zfill(6)
    gt_by_id = {
        str(box["id"]): box for box in gt_boxes if classify_gt_box(box) == "positive"
    }
    evidence = candidate_conversion.get("evidence") if isinstance(candidate_conversion, dict) else None
    if not isinstance(evidence, list):
        raise ValueError("fragment oracles require canonical candidate conversion evidence")
    evidence_by_gt = {str(item["gt_id"]): item for item in evidence}
    if set(evidence_by_gt) != set(gt_by_id):
        raise ValueError("fragment oracle GT set differs from canonical candidate evidence")
    cluster_by_id = {
        str(detection.get("id", f"cluster_{index + 1}")): (cluster, detection)
        for index, (cluster, detection) in enumerate(zip(clusters, raw_detections))
    }

    records = []
    for gt_id, canonical in evidence_by_gt.items():
        cluster_ids = [str(value) for value in canonical.get("cluster_ids", [])]
        if not cluster_ids:
            continue
        gt_box = gt_by_id[gt_id]
        branch_iou = {
            str(item.get("cluster_id")): float(item.get("candidate_iou", 0.0))
            for item in canonical.get("candidate_branches", [])
        }
        fragments = []
        fragment_points = []
        for cluster_id in cluster_ids:
            if cluster_id not in cluster_by_id:
                raise ValueError(f"canonical associated cluster is missing: {cluster_id}")
            cluster, detection = cluster_by_id[cluster_id]
            status = pca_input_status(cluster)
            iou = float(bev_iou(gt_box, detection)) if status == "valid" else None
            if status == "valid":
                if cluster_id not in branch_iou:
                    raise ValueError(f"missing canonical branch IoU: {cluster_id}")
                if not np.isclose(iou, branch_iou[cluster_id], rtol=0.0, atol=1e-9):
                    raise ValueError(f"O1 differs from canonical candidate IoU: {cluster_id}")
            fragments.append(
                {
                    "cluster_id": cluster_id,
                    "status": status,
                    "num_points": int(len(cluster)),
                    "iou": iou,
                    "canonical_candidate_iou": branch_iou.get(cluster_id),
                }
            )
            fragment_points.append(cluster)

        valid_fragments = [item for item in fragments if item["status"] == "valid"]
        selected = max(valid_fragments, key=lambda item: (item["iou"], item["cluster_id"])) \
            if valid_fragments else None
        o1 = {
            "status": "valid" if selected else fragments[0]["status"],
            "selected_cluster_id": selected["cluster_id"] if selected else None,
            "num_points": selected["num_points"] if selected else 0,
            "iou": selected["iou"] if selected else None,
        }
        stacked = np.vstack(fragment_points)
        union = np.unique(stacked, axis=0)
        gt_clipped = union[points_in_oriented_3d_box(union, gt_box)]
        o2 = build_pca_oracle(union, gt_box)
        o2_gt_clipped = build_pca_oracle(gt_clipped, gt_box)
        records.append(
            {
                "frame_id": frame_id,
                "gt_id": gt_id,
                "variant": str(variant),
                "distance_bin": canonical.get("distance_bin"),
                "associated_cluster_count": int(len(cluster_ids)),
                "associated_cluster_ids": cluster_ids,
                "associated_cluster_point_count": int(sum(len(points) for points in fragment_points)),
                "associated_union_point_count": int(len(union)),
                "duplicate_union_point_count": int(len(stacked) - len(union)),
                "associated_union_gt_clipped_point_count": int(len(gt_clipped)),
                "fragment_oracles": fragments,
                "O1": o1,
                "O2": o2,
                "O2_gt_clipped": o2_gt_clipped,
                "signed_deltas": {
                    "O2_minus_O1": (
                        float(o2["iou"] - o1["iou"])
                        if o2["iou"] is not None and o1["iou"] is not None else None
                    ),
                    "O2_gt_clipped_minus_O2": (
                        float(o2_gt_clipped["iou"] - o2["iou"])
                        if o2_gt_clipped["iou"] is not None and o2["iou"] is not None else None
                    ),
                },
            }
        )
    return {
        "schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
        "frame_id": frame_id,
        "variant": str(variant),
        "associated_gt_count": int(len(records)),
        "records": records,
    }


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


def build_point_retention_day2(day1, reports_by_variant):
    """Select delta and P1-control fragment oracles from one completed Day 1 result."""
    if not isinstance(day1, dict) or day1.get("schema_version") != POINT_RETENTION_SCHEMA_VERSION:
        raise ValueError("Day 2 requires a valid point-retention Day 1 result")
    candidate_variant = day1["candidate_variant"]
    if candidate_variant not in reports_by_variant:
        raise ValueError("missing Day 1 candidate variant reports")
    indexed = {}
    for report in reports_by_variant[candidate_variant]:
        payload = report.get("fragment_recoverability") or {}
        if payload.get("schema_version") != FRAGMENT_ORACLE_SCHEMA_VERSION:
            raise ValueError(f"missing fragment oracle evidence: {report.get('frame_id')}")
        for item in payload.get("records", []):
            key = (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
            if key in indexed:
                raise ValueError(f"duplicate fragment oracle evidence: {key}")
            indexed[key] = item

    def payload_keys(name):
        return [(str(item["frame_id"]).zfill(6), str(item["gt_id"])) for item in day1[name]]

    delta_keys = payload_keys("delta_gt_keys")
    control_keys = payload_keys("p1_control_gt_keys")
    missing = sorted((set(delta_keys) | set(control_keys)) - set(indexed))
    if missing:
        raise ValueError(f"missing selected fragment oracle evidence: {missing}")
    return {
        "schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
        "source_day1_schema_version": POINT_RETENTION_SCHEMA_VERSION,
        "candidate_variant": candidate_variant,
        "delta_gt_count": int(len(delta_keys)),
        "p1_control_gt_count": int(len(control_keys)),
        "delta_records": [indexed[key] for key in delta_keys],
        "p1_control_records": [indexed[key] for key in control_keys],
    }
