"""Read-only P1/P2/N grouping for the v15.3.1 separability diagnostic."""

from collections import Counter

import numpy as np

from bev_tracking.candidate_conversion import gt_cluster_association_indices
from bev_tracking.clustering_policy import DISTANCE_BIN_NAMES, distance_bin_name
from bev_tracking.eval_policy import classify_gt_box, is_positive_detection
from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import points_in_oriented_3d_box


CLUSTER_GROUP_SCHEMA_VERSION = "15.3.1-cluster-groups-day1"
CLUSTER_FEATURE_SCHEMA_VERSION = "15.3.1-cluster-features-day2"
CLUSTER_DIAGNOSTIC_SCHEMA_VERSION = "15.3.1-cluster-separability-day3"
CLUSTER_GROUPS = ("P1", "P2", "N")
COMPARISON_FEATURES = (
    "num_points",
    "axis_length",
    "axis_width",
    "height_span",
    "point_density_xy",
    "pca_length",
    "pca_width",
    "gt_coverage_ratio",
    "cluster_purity_ratio",
    "pca_iou_to_target_gt",
)


def _cluster_features(cluster, detection):
    axis_length = float(max(cluster[:, 0].max() - cluster[:, 0].min(), 0.1))
    axis_width = float(max(cluster[:, 1].max() - cluster[:, 1].min(), 0.1))
    height_span = float(max(cluster[:, 2].max() - cluster[:, 2].min(), 0.0))
    return {
        "num_points": int(len(cluster)),
        "axis_length": axis_length,
        "axis_width": axis_width,
        "height_span": height_span,
        "point_density_xy": float(len(cluster) / max(axis_length * axis_width, 1e-6)),
        "pca_length": float(detection.get("length", 0.0)),
        "pca_width": float(detection.get("width", 0.0)),
    }


def build_frame_cluster_groups(
    *, frame_id, variant, filtered_points, clusters, raw_detections, gt_boxes
):
    """Partition each cluster once; retain GT associations for later read-only features."""
    if len(clusters) != len(raw_detections):
        raise ValueError("clusters and raw_detections must have the same length")
    frame_id = str(frame_id).zfill(6)
    positive_gt = [box for box in gt_boxes if classify_gt_box(box) == "positive"]
    positive_gt_by_id = {str(box["id"]): box for box in positive_gt}
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
        linked = []
        for source in associations[index]:
            item = dict(source)
            overlap = int(item["cluster_points_in_gt"])
            item["gt_coverage_ratio"] = float(
                overlap / item["gt_filtered_point_count"]
            ) if item["gt_filtered_point_count"] else None
            item["cluster_purity_ratio"] = float(overlap / len(cluster)) if len(cluster) else None
            item["pca_box_iou"] = float(
                bev_iou(positive_gt_by_id[item["gt_id"]], detection)
            )
            linked.append(item)
        linked = sorted(linked, key=lambda item: item["gt_id"])
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
        target = sorted(
            linked,
            key=lambda item: (-int(item["cluster_points_in_gt"]), item["gt_id"]),
        )[0] if linked else None
        features = _cluster_features(cluster, detection)
        features.update(
            {
                "target_gt_id": target["gt_id"] if target else None,
                "gt_coverage_ratio": target["gt_coverage_ratio"] if target else None,
                "cluster_purity_ratio": target["cluster_purity_ratio"] if target else None,
                "pca_iou_to_target_gt": target["pca_box_iou"] if target else None,
                "oracle_car_iou": target["pca_box_iou"] if group == "P2" else None,
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
                "feature_schema_version": CLUSTER_FEATURE_SCHEMA_VERSION,
                "features": features,
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


def validate_cluster_feature_records(records):
    numeric_fields = (
        "num_points", "axis_length", "axis_width", "height_span",
        "point_density_xy", "pca_length", "pca_width",
    )
    for record in records:
        if record.get("feature_schema_version") != CLUSTER_FEATURE_SCHEMA_VERSION:
            raise ValueError("missing Day 2 cluster feature schema")
        features = record.get("features") or {}
        for field in numeric_fields:
            value = features.get(field)
            if value is None or not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"invalid cluster feature: {field}")
        associated = record.get("group") in {"P1", "P2"}
        for field in ("gt_coverage_ratio", "cluster_purity_ratio"):
            value = features.get(field)
            if associated and (value is None or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"invalid associated-cluster ratio: {field}")
            if not associated and value is not None:
                raise ValueError(f"N must not define {field}")
        oracle = features.get("oracle_car_iou")
        if record.get("group") == "P2":
            if oracle is None or not 0.0 <= float(oracle) <= 1.0:
                raise ValueError("P2 must define a valid read-only oracle IoU")
        elif oracle is not None:
            raise ValueError("only P2 may define oracle_car_iou")
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


def _oracle_summary(records):
    values = [float(item["features"]["oracle_car_iou"]) for item in records]
    return {
        "count": int(len(values)),
        "iou_ge_0_25_count": int(sum(value >= 0.25 for value in values)),
        "iou_ge_0_50_count": int(sum(value >= 0.50 for value in values)),
        "mean": float(sum(values) / len(values)) if values else None,
        "max": max(values) if values else None,
    }


def build_cluster_feature_day2(grouped):
    """Consume one Day 1 result; validate features and summarize P2 oracle IoU."""
    if not isinstance(grouped, dict) or grouped.get("schema_version") != CLUSTER_GROUP_SCHEMA_VERSION:
        raise ValueError("Day 2 requires a valid Day 1 cluster-group result")
    for records in grouped["records_by_variant"].values():
        validate_cluster_feature_records(records)
    candidate_variant = grouped["candidate_variant"]
    candidate_records = grouped["records_by_variant"][candidate_variant]
    p2 = [item for item in candidate_records if item["group"] == "P2"]
    delta_p2 = [item for item in p2 if item["is_delta_22"]]
    return {
        "schema_version": CLUSTER_FEATURE_SCHEMA_VERSION,
        "source_group_schema_version": CLUSTER_GROUP_SCHEMA_VERSION,
        "base_variant": grouped["base_variant"],
        "candidate_variant": candidate_variant,
        "group_counts_by_variant": grouped["group_counts_by_variant"],
        "delta_associated_gt_count": grouped["delta_associated_gt_count"],
        "delta_p2_gt_count": grouped["delta_p2_gt_count"],
        "delta_p2_cluster_count": grouped["delta_p2_cluster_count"],
        "delta_gt_without_p2_cluster": grouped["delta_gt_without_p2_cluster"],
        "feature_definitions": {
            "gt_coverage_ratio": "cluster points inside target GT / filtered points inside target GT",
            "cluster_purity_ratio": "cluster points inside target GT / cluster points",
            "oracle_car_iou": "P2 original PCA box versus target GT; no pipeline mutation",
        },
        "candidate_p2_oracle": _oracle_summary(p2),
        "delta_22_p2_oracle": _oracle_summary(delta_p2),
    }


def _distribution(values):
    numeric = np.asarray(values, dtype=np.float64)
    if not len(numeric):
        return {
            "count": 0, "min": None, "p25": None, "median": None,
            "mean": None, "p75": None, "max": None,
        }
    p25, median, p75 = np.percentile(numeric, [25, 50, 75])
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "p25": float(p25),
        "median": float(median),
        "mean": float(numeric.mean()),
        "p75": float(p75),
        "max": float(numeric.max()),
    }


def _feature_distributions(records):
    outputs = {}
    for distance_bin in (*DISTANCE_BIN_NAMES, "total"):
        selected = records if distance_bin == "total" else [
            item for item in records if item["cluster_distance_bin"] == distance_bin
        ]
        outputs[distance_bin] = {
            "cluster_count": int(len(selected)),
            "features": {
                field: _distribution(
                    [item["features"][field] for item in selected if item["features"].get(field) is not None]
                )
                for field in COMPARISON_FEATURES
            },
        }
    return outputs


def build_cluster_separability_day3(grouped, featured):
    """Build the compact final comparison without recomputing Day 1 or Day 2."""
    if not isinstance(grouped, dict) or grouped.get("schema_version") != CLUSTER_GROUP_SCHEMA_VERSION:
        raise ValueError("Day 3 requires a valid Day 1 cluster-group result")
    if not isinstance(featured, dict) or featured.get("schema_version") != CLUSTER_FEATURE_SCHEMA_VERSION:
        raise ValueError("Day 3 requires a valid Day 2 feature result")
    candidate_variant = grouped["candidate_variant"]
    if featured.get("candidate_variant") != candidate_variant:
        raise ValueError("Day 1 and Day 2 candidate variants differ")
    records = grouped["records_by_variant"][candidate_variant]
    validate_cluster_feature_records(records)

    p1 = [item for item in records if item["group"] == "P1"]
    p2 = [item for item in records if item["group"] == "P2"]
    delta_p2 = [item for item in p2 if item["is_delta_22"]]
    associated_context = {
        (item["frame_id"], item["cluster_distance_bin"]) for item in (*p1, *p2)
    }
    delta_context = {
        (item["frame_id"], item["cluster_distance_bin"]) for item in delta_p2
    }
    strict_n = [
        item for item in records
        if item["group"] == "N"
        and item["is_strict_background"]
        and (item["frame_id"], item["cluster_distance_bin"]) in associated_context
    ]
    delta_strict_n = [
        item for item in strict_n
        if (item["frame_id"], item["cluster_distance_bin"]) in delta_context
    ]

    delta_records = [
        {
            "frame_id": item["frame_id"],
            "cluster_id": item["cluster_id"],
            "delta_22_gt_ids": list(item["delta_22_gt_ids"]),
            "cluster_distance_bin": item["cluster_distance_bin"],
            "features": dict(item["features"]),
        }
        for item in delta_p2
    ]
    return {
        "schema_version": CLUSTER_DIAGNOSTIC_SCHEMA_VERSION,
        "source_group_schema_version": CLUSTER_GROUP_SCHEMA_VERSION,
        "source_feature_schema_version": CLUSTER_FEATURE_SCHEMA_VERSION,
        "candidate_variant": candidate_variant,
        "selection": {
            "P1": "all candidate-variant P1 clusters",
            "P2": "all candidate-variant P2 clusters",
            "N": "unique strict-background N clusters in the same frame and distance bin as P1/P2",
            "delta_22_N": "unique strict-background N clusters in the same frame and distance bin as delta-22 P2",
        },
        "set_counts": {
            "P1": int(len(p1)),
            "P2": int(len(p2)),
            "N": int(len(strict_n)),
            "delta_22_P2": int(len(delta_p2)),
            "delta_22_N": int(len(delta_strict_n)),
        },
        "all_candidate_context": {
            "P1": _feature_distributions(p1),
            "P2": _feature_distributions(p2),
            "N": _feature_distributions(strict_n),
        },
        "delta_22_context": {
            "P2": _feature_distributions(delta_p2),
            "N": _feature_distributions(delta_strict_n),
        },
        "oracle": {
            "all_P2": dict(featured["candidate_p2_oracle"]),
            "delta_22_P2": dict(featured["delta_22_p2_oracle"]),
        },
        "delta_22_records": delta_records,
        "decision_question": (
            "Are the delta-22 P2 clusters detection-worthy classifier false rejects, "
            "or incomplete vehicle fragments?"
        ),
        "frozen_pipeline": [
            "clustering", "2.0m_classifier_gate", "500_point_classifier_gate",
            "nms", "evaluation",
        ],
    }
