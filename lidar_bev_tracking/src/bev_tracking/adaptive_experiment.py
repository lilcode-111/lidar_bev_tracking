"""Variant registration and GT/cluster merging gates for v15.2."""

from dataclasses import dataclass
import math

from bev_tracking.adaptive_clustering import cluster_points
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.clustering_detector import split_obstacle_filter_stages
from bev_tracking.eval_policy import classify_gt_box
from bev_tracking.failure_evidence import build_failure_evidence_report
from bev_tracking.geometry_sanity import points_in_oriented_3d_box


VARIANT_ORDER = ("C0", "C1", "C2", "C3")
MIN_GT_POINTS_FOR_MERGING = 3
MIN_CLUSTER_POINTS_IN_GT = 3
MIN_GT_CLUSTER_FRACTION = 0.10


@dataclass(frozen=True)
class VariantSpec:
    name: str
    policy: ClusteringPolicy
    intensity_min: float = 0.38
    z_min: float = -0.9

    def to_dict(self):
        payload = {
            "name": self.name,
            "mode": self.policy.mode,
            "eps": float(self.policy.eps),
            "min_points": int(self.policy.min_points),
            "global_max_eps": float(self.policy.global_max_eps),
            "intensity_min": float(self.intensity_min),
            "z_min": float(self.z_min),
        }
        if self.policy.distance_params is not None:
            payload["distance_params"] = self.policy.distance_params
        return payload


def validate_variant_specs(specs):
    if not isinstance(specs, dict) or tuple(specs) != VARIANT_ORDER:
        raise ValueError("variant specs must contain C0, C1, C2, C3 in order")
    c0 = specs["C0"]
    if c0.policy.mode != "fixed" or c0.policy.eps != 0.6 or c0.policy.min_points != 20:
        raise ValueError("C0 must use fixed eps=0.6 and min_points=20")
    for name, spec in specs.items():
        if spec.intensity_min != 0.38 or spec.z_min != -0.9:
            raise ValueError(f"{name} changes frozen filter parameters")
    return True


def preregister_variant_specs(specs):
    validate_variant_specs(specs)
    return {
        "schema_version": "15.2-variants",
        "variant_order": list(VARIANT_ORDER),
        "frozen_fields": ["intensity_min", "z_min", "nms", "evaluation_policy"],
        "variants": {name: specs[name].to_dict() for name in VARIANT_ORDER},
        "selection_rule": "pre_registered_before_results; no post-hoc parameter changes",
    }


def variant_specs_from_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("variants"), dict):
        raise ValueError("config must contain a variants mapping")
    specs = {}
    for name in VARIANT_ORDER:
        raw = config["variants"].get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"missing variant config: {name}")
        mode = raw.get("mode", "fixed")
        if mode == "adaptive":
            policy = ClusteringPolicy(
                mode="adaptive",
                global_max_eps=float(raw.get("global_max_eps", 0.85)),
                distance_params=raw.get("distance_params"),
            )
        else:
            policy = ClusteringPolicy(
                mode="fixed",
                eps=float(raw.get("eps", 0.6)),
                min_points=int(raw.get("min_points", 20)),
                global_max_eps=float(raw.get("global_max_eps", 0.85)),
            )
        specs[name] = VariantSpec(
            name=name,
            policy=policy,
            intensity_min=float(raw.get("intensity_min", 0.38)),
            z_min=float(raw.get("z_min", -0.9)),
        )
    validate_variant_specs(specs)
    return specs


def positive_gt_boxes(gt_boxes):
    return [box for box in gt_boxes if classify_gt_box(box) == "positive"]


def build_gt_cluster_associations(filtered_points, clusters, gt_boxes):
    records = []
    for gt_box in positive_gt_boxes(gt_boxes):
        gt_mask = points_in_oriented_3d_box(filtered_points, gt_box)
        gt_point_count = int(gt_mask.sum())
        eligible = gt_point_count >= MIN_GT_POINTS_FOR_MERGING
        minimum_association = max(
            MIN_CLUSTER_POINTS_IN_GT,
            int(math.ceil(MIN_GT_CLUSTER_FRACTION * gt_point_count)),
        )
        associated_cluster_indices = []
        for cluster_index, cluster in enumerate(clusters):
            inside_count = int(points_in_oriented_3d_box(cluster, gt_box).sum())
            if eligible and inside_count >= minimum_association:
                associated_cluster_indices.append(cluster_index)
        records.append(
            {
                "gt_id": str(gt_box["id"]),
                "num_points_in_gt": gt_point_count,
                "eligible": bool(eligible),
                "minimum_cluster_points": int(minimum_association),
                "associated_cluster_indices": associated_cluster_indices,
            }
        )
    return records


def summarize_merging(records, clusters):
    eligible_records = [record for record in records if record["eligible"]]
    cluster_to_gt = {index: [] for index in range(len(clusters))}
    for record in eligible_records:
        for cluster_index in record["associated_cluster_indices"]:
            cluster_to_gt[cluster_index].append(record["gt_id"])

    merged_clusters = {
        str(cluster_index): sorted(gt_ids)
        for cluster_index, gt_ids in cluster_to_gt.items()
        if len(gt_ids) >= 2
    }
    merged_gt_ids = sorted({gt_id for gt_ids in merged_clusters.values() for gt_id in gt_ids})
    eligible_count = len(eligible_records)
    return {
        "eligible_positive_gt_count": int(eligible_count),
        "merged_positive_gt_count": int(len(merged_gt_ids)),
        "merging_rate": float(len(merged_gt_ids) / eligible_count) if eligible_count else None,
        "merged_cluster_count": int(len(merged_clusters)),
        "merged_clusters": merged_clusters,
    }


def run_variant_frame(points, gt_boxes, frame_id, variant, **kwargs):
    if not isinstance(variant, VariantSpec):
        raise TypeError("variant must be a VariantSpec")
    stages = split_obstacle_filter_stages(
        points,
        z_min=variant.z_min,
        intensity_min=variant.intensity_min,
    )
    parameters = variant.policy.params_for_range(0.0)
    clusters = cluster_points(stages["intensity_filter"], variant.policy)
    report = build_failure_evidence_report(
        points,
        gt_boxes,
        frame_id=frame_id,
        eps=parameters["eps"],
        min_points=parameters["min_points"],
        z_min=variant.z_min,
        intensity_min=variant.intensity_min,
        clustering_policy=variant.policy,
        **kwargs,
    )
    associations = build_gt_cluster_associations(stages["intensity_filter"], clusters, gt_boxes)
    report["summary"]["merging"] = summarize_merging(associations, clusters)
    report["gt_cluster_associations"] = associations
    report["variant"] = variant.to_dict()
    return report


def compare_eligible_gt_sets(reports_by_variant):
    sets = {
        name: {
            (str(report["frame_id"]).zfill(6), record["gt_id"])
            for report in reports
            for record in report.get("gt_cluster_associations", [])
            if record["eligible"]
        }
        for name, reports in reports_by_variant.items()
    }
    reference = sets["C0"]
    mismatches = {
        name: sorted(reference.symmetric_difference(current))
        for name, current in sets.items()
        if current != reference
    }
    return {"passed": not mismatches, "eligible_gt_by_variant": sets, "mismatches": mismatches}


def aggregate_variant_reports(variant_name, reports):
    if not reports:
        raise ValueError(f"variant has no frame reports: {variant_name}")
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "neutralized_detections": 0,
        "effective_car_detection_count": 0,
        "num_positive_gt": 0,
        "zero_detection_with_gt": 0,
        "merged_positive_gt_count": 0,
        "eligible_positive_gt_count": 0,
    }
    auxiliary = {"tp": 0, "fp": 0, "fn": 0, "neutralized_detections": 0}
    for report in reports:
        summary = report["summary"]
        primary = summary["metrics_by_iou"]["0.50"]
        totals["tp"] += int(primary["tp"])
        totals["fp"] += int(primary["fp"])
        totals["fn"] += int(primary["fn"])
        totals["neutralized_detections"] += int(primary["neutralized_detections"])
        totals["effective_car_detection_count"] += int(primary["effective_car_detection_count"])
        totals["num_positive_gt"] += int(summary["num_positive_gt"])
        totals["zero_detection_with_gt"] += int(
            summary["candidate_coverage"]["zero_detection_with_gt_eligible_count"]
        )
        merging = summary.get("merging", {})
        totals["merged_positive_gt_count"] += int(merging.get("merged_positive_gt_count", 0))
        totals["eligible_positive_gt_count"] += int(merging.get("eligible_positive_gt_count", 0))

        metrics = summary["metrics_by_iou"]["0.25"]
        for field in auxiliary:
            auxiliary[field] += int(metrics[field])

    totals["precision"] = _safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
    totals["recall"] = _safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
    totals["f1"] = _safe_f1(totals["tp"], totals["fp"], totals["fn"])
    totals["merging_rate"] = _safe_ratio(
        totals["merged_positive_gt_count"], totals["eligible_positive_gt_count"]
    )
    return {
        "variant": variant_name,
        "num_frames": len(reports),
        "primary_iou_0_50": totals,
        "auxiliary_iou_0_25": auxiliary,
    }


def validate_variant_batch_results(reports_by_variant):
    if tuple(reports_by_variant) != VARIANT_ORDER:
        raise ValueError("variant reports must contain C0, C1, C2, C3 in order")
    eligible_gate = compare_eligible_gt_sets(reports_by_variant)
    if not eligible_gate["passed"]:
        raise ValueError(f"eligible positive GT set mismatch: {eligible_gate['mismatches']}")
    return {
        "passed": True,
        "eligible_gt_gate": eligible_gate,
        "frame_counts": {name: len(reports) for name, reports in reports_by_variant.items()},
    }


def rank_variant_summaries(summaries):
    return sorted(
        summaries,
        key=lambda item: (
            -int(item["primary_iou_0_50"]["tp"]),
            int(item["primary_iou_0_50"]["fp"]),
            float(item["primary_iou_0_50"]["merging_rate"] or float("inf")),
            str(item["variant"]),
        ),
    )


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _safe_f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else None
