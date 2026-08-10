"""Read-only P1/P2/N grouping for the v15.3.1 separability diagnostic."""

from collections import Counter

import numpy as np

from bev_tracking.candidate_conversion import gt_cluster_association_indices
from bev_tracking.clustering_policy import distance_bin_name
from bev_tracking.eval_policy import classify_gt_box, is_positive_detection
from bev_tracking.geometry_sanity import points_in_oriented_3d_box


CLUSTER_GROUP_SCHEMA_VERSION = "15.3.1-cluster-groups-day1"
CLUSTER_GROUPS = ("P1", "P2", "N")


def build_frame_cluster_groups(
    *, frame_id, variant, filtered_points, clusters, raw_detections, gt_boxes
):
    """Partition each cluster once; retain GT associations for later read-only features."""
    if len(clusters) != len(raw_detections):
        raise ValueError("clusters and raw_detections must have the same length")
    frame_id = str(frame_id).zfill(6)
    positive_gt = [box for box in gt_boxes if classify_gt_box(box) == "positive"]
    associations = {index: [] for index in range(len(clusters))}
    for gt_box in positive_gt:
        gt_point_count, cluster_indices = gt_cluster_association_indices(
            filtered_points, gt_box, clusters
        )
        gt_range = float(np.hypot(float(gt_box["x"]), float(gt_box["y"])))
        for cluster_index in cluster_indices:
            associations[cluster_index].append(
                {
                    "gt_id": str(gt_box["id"]),
                    "gt_distance_bin": distance_bin_name(gt_range),
                    "gt_filtered_point_count": int(gt_point_count),
                    "cluster_points_in_gt": int(
                        points_in_oriented_3d_box(clusters[cluster_index], gt_box).sum()
                    ),
                }
            )

    records = []
    for index, (cluster, detection) in enumerate(zip(clusters, raw_detections)):
        cluster_id = str(detection.get("id", f"cluster_{index + 1}"))
        linked = sorted(associations[index], key=lambda item: item["gt_id"])
        if linked:
            group = "P1" if is_positive_detection(detection) else "P2"
        else:
            group = "N"
        cluster_range = float(
            np.hypot(float(cluster[:, 0].mean()), float(cluster[:, 1].mean()))
        )
        overlap_categories = sorted(
            {
                classify_gt_box(gt_box)
                for gt_box in gt_boxes
                if points_in_oriented_3d_box(cluster, gt_box).any()
            }
        )
        records.append(
            {
                "frame_id": frame_id,
                "variant": str(variant),
                "cluster_id": cluster_id,
                "raw_detection_id": cluster_id,
                "group": group,
                "cluster_distance_bin": distance_bin_name(cluster_range),
                "associated_gt": linked,
                "associated_gt_ids": [item["gt_id"] for item in linked],
                "annotation_overlap_categories": overlap_categories,
                "is_strict_background": bool(group == "N" and not overlap_categories),
                "is_delta_22": False,
                "delta_22_gt_ids": [],
            }
        )

    validate_cluster_group_partition(records, len(clusters))
    counts = Counter(record["group"] for record in records)
    return {
        "schema_version": CLUSTER_GROUP_SCHEMA_VERSION,
        "unit_of_analysis": "unique_cluster",
        "group_definitions": {
            "P1": "positive-Car-GT-associated cluster accepted as a Car candidate before NMS",
            "P2": "positive-Car-GT-associated cluster rejected by the Car classifier",
            "N": "cluster without a positive-Car-GT association; annotation overlap is retained",
        },
        "frame_id": frame_id,
        "variant": str(variant),
        "cluster_count": int(len(clusters)),
        "group_counts": {group: int(counts.get(group, 0)) for group in CLUSTER_GROUPS},
        "records": records,
    }


def validate_cluster_group_partition(records, expected_cluster_count):
    cluster_ids = [str(record.get("cluster_id")) for record in records]
    if len(records) != int(expected_cluster_count):
        raise ValueError("cluster grouping does not conserve the cluster set")
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster grouping contains duplicate cluster ids")
    invalid = sorted({record.get("group") for record in records} - set(CLUSTER_GROUPS))
    if invalid:
        raise ValueError(f"invalid cluster groups: {invalid}")
    for record in records:
        associated = bool(record.get("associated_gt_ids"))
        if (record["group"] in {"P1", "P2"}) != associated:
            raise ValueError("P1/P2 must be GT-associated and N must be unassociated")
    return True


def build_cluster_group_day1(reports_by_variant, base_variant="C0", candidate_variant="C1"):
    """Aggregate frame groups and mark the C0-to-C1 newly associated P2 cohort."""
    if base_variant not in reports_by_variant or candidate_variant not in reports_by_variant:
        raise ValueError("cluster grouping requires base and candidate variants")

    def index_gt_evidence(reports, variant):
        indexed = {}
        for report in reports:
            evidence = (report.get("candidate_conversion") or {}).get("evidence")
            if not isinstance(evidence, list):
                raise ValueError(f"missing candidate conversion evidence for {variant}")
            for item in evidence:
                key = (str(item.get("frame_id")).zfill(6), str(item.get("gt_id")))
                if key in indexed:
                    raise ValueError(f"duplicate GT evidence for {variant}: {key}")
                indexed[key] = item
        return indexed

    base = index_gt_evidence(reports_by_variant[base_variant], base_variant)
    candidate = index_gt_evidence(reports_by_variant[candidate_variant], candidate_variant)
    if set(base) != set(candidate):
        raise ValueError("base and candidate GT evidence sets differ")
    delta_keys = {
        key for key in candidate
        if candidate[key].get("cluster_ids") and not base[key].get("cluster_ids")
    }

    records_by_variant = {}
    counts_by_variant = {}
    for variant, reports in reports_by_variant.items():
        records = []
        for report in reports:
            payload = report.get("cluster_separability") or {}
            if payload.get("schema_version") != CLUSTER_GROUP_SCHEMA_VERSION:
                raise ValueError(f"missing cluster grouping for {variant}: {report.get('frame_id')}")
            for source in payload.get("records", []):
                item = dict(source)
                delta_ids = sorted(
                    gt_id for gt_id in item.get("associated_gt_ids", [])
                    if (item["frame_id"], str(gt_id)) in delta_keys
                ) if variant == candidate_variant else []
                item["delta_22_gt_ids"] = delta_ids
                item["is_delta_22"] = bool(item.get("group") == "P2" and delta_ids)
                records.append(item)
        counts = Counter(item["group"] for item in records)
        records_by_variant[variant] = records
        counts_by_variant[variant] = {
            group: int(counts.get(group, 0)) for group in CLUSTER_GROUPS
        }

    candidate_records = records_by_variant[candidate_variant]
    marked_gt_ids = {
        (item["frame_id"], gt_id)
        for item in candidate_records if item["is_delta_22"]
        for gt_id in item["delta_22_gt_ids"]
    }
    unexpected = sorted(delta_keys - marked_gt_ids)
    return {
        "schema_version": CLUSTER_GROUP_SCHEMA_VERSION,
        "base_variant": base_variant,
        "candidate_variant": candidate_variant,
        "group_counts_by_variant": counts_by_variant,
        "delta_associated_gt_count": int(len(delta_keys)),
        "delta_p2_gt_count": int(len(marked_gt_ids)),
        "delta_p2_cluster_count": int(
            sum(item["is_delta_22"] for item in candidate_records)
        ),
        "delta_gt_without_p2_cluster": [
            {"frame_id": frame_id, "gt_id": gt_id} for frame_id, gt_id in unexpected
        ],
        "records_by_variant": records_by_variant,
    }
