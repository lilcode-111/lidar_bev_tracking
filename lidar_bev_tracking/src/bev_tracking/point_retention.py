"""Read-only data contract for the v15.3.2 point-retention oracle ladder."""

import numpy as np

from bev_tracking.eval_policy import classify_gt_box
from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.oriented_box import estimate_oriented_box_xy

POINT_RETENTION_SCHEMA_VERSION = "15.3.2-point-retention-day1"
FRAGMENT_ORACLE_SCHEMA_VERSION = "15.3.2-fragment-oracles-day2"
STAGE_ORACLE_SCHEMA_VERSION = "15.3.2-stage-oracles-day3"
ROOT_CAUSE_SCHEMA_VERSION = "15.3.2-root-cause-day4"
STAGE_ORDER = ("raw", "roi", "z_filter", "intensity_filter")
PCA_STATUS_VALUES = ("valid", "insufficient_points", "degenerate_geometry")
PCA_MIN_POINTS = 3
MATERIAL_GAIN_IOU = 0.10
RECOVERABLE_IOU = 0.25


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
    *, frame_id, variant, gt_boxes, clusters, raw_detections, candidate_conversion,
    raw_points, cluster_source_point_indices
):
    """Build O1/O2 from canonical associations using raw-LiDAR point identity."""
    if len(clusters) != len(raw_detections):
        raise ValueError("clusters and raw_detections must have the same length")
    if len(clusters) != len(cluster_source_point_indices):
        raise ValueError("every cluster must have source point indices")
    raw_points = np.asarray(raw_points)
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
    cluster_by_id = {}
    for index, (cluster, detection, source_indices) in enumerate(
        zip(clusters, raw_detections, cluster_source_point_indices)
    ):
        source_indices = np.asarray(source_indices, dtype=np.int64)
        if source_indices.ndim != 1 or len(source_indices) != len(cluster):
            raise ValueError("cluster source point index count differs from cluster point count")
        if np.any(source_indices < 0) or np.any(source_indices >= len(raw_points)):
            raise ValueError("cluster source point index is outside raw LiDAR bounds")
        if not np.array_equal(raw_points[source_indices], cluster):
            raise ValueError("cluster points differ from raw LiDAR source point indices")
        cluster_id = str(detection.get("id", f"cluster_{index + 1}"))
        cluster_by_id[cluster_id] = (cluster, detection, source_indices)

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
        fragment_source_indices = []
        for cluster_id in cluster_ids:
            if cluster_id not in cluster_by_id:
                raise ValueError(f"canonical associated cluster is missing: {cluster_id}")
            cluster, detection, source_indices = cluster_by_id[cluster_id]
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
                    "source_point_indices": source_indices.tolist(),
                }
            )
            fragment_points.append(cluster)
            fragment_source_indices.append(source_indices)

        valid_fragments = [item for item in fragments if item["status"] == "valid"]
        selected = max(valid_fragments, key=lambda item: (item["iou"], item["cluster_id"])) \
            if valid_fragments else None
        o1 = {
            "status": "valid" if selected else fragments[0]["status"],
            "selected_cluster_id": selected["cluster_id"] if selected else None,
            "num_points": selected["num_points"] if selected else 0,
            "iou": selected["iou"] if selected else None,
        }
        stacked_source_indices = np.concatenate(fragment_source_indices)
        union_source_indices = np.unique(stacked_source_indices)
        union = raw_points[union_source_indices]
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
                "duplicate_union_point_count": int(
                    len(stacked_source_indices) - len(union_source_indices)
                ),
                "point_identity": "raw_lidar_point_index",
                "associated_union_source_point_indices": union_source_indices.tolist(),
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
        "point_identity": "raw_lidar_point_index",
        "source_index_alignment_gate_passed": True,
        "associated_gt_count": int(len(records)),
        "records": records,
    }


def build_frame_stage_oracles(*, frame_id, variant, stages, gt_boxes, candidate_conversion):
    """Build O3-O6 once from stage coordinates and gate counts against canonical evidence."""
    missing_stages = [stage for stage in STAGE_ORDER if stage not in stages]
    if missing_stages:
        raise ValueError(f"missing frozen pipeline stages: {missing_stages}")
    frame_id = str(frame_id).zfill(6)
    gt_by_id = {
        str(box["id"]): box for box in gt_boxes if classify_gt_box(box) == "positive"
    }
    evidence = candidate_conversion.get("evidence") if isinstance(candidate_conversion, dict) else None
    if not isinstance(evidence, list):
        raise ValueError("stage oracles require canonical candidate conversion evidence")
    evidence_by_gt = {str(item["gt_id"]): item for item in evidence}
    if set(evidence_by_gt) != set(gt_by_id):
        raise ValueError("stage oracle GT set differs from canonical candidate evidence")

    stage_to_oracle = {
        "intensity_filter": "O3",
        "z_filter": "O4",
        "roi": "O5",
        "raw": "O6",
    }
    records = []
    for gt_id, canonical in evidence_by_gt.items():
        gt_box = gt_by_id[gt_id]
        canonical_counts = canonical.get("stage_point_counts")
        if not isinstance(canonical_counts, dict):
            raise ValueError(f"missing canonical stage counts: {frame_id} {gt_id}")
        validate_stage_point_counts(canonical_counts)
        oracles = {}
        for stage in STAGE_ORDER:
            stage_points = stages[stage]
            selected = stage_points[points_in_oriented_3d_box(stage_points, gt_box)]
            expected_count = int(canonical_counts[stage])
            if len(selected) != expected_count:
                raise ValueError(
                    f"stage point count differs from canonical evidence: "
                    f"{frame_id} {gt_id} {stage} {len(selected)} != {expected_count}"
                )
            oracle = build_pca_oracle(selected, gt_box)
            oracle["source_stage"] = stage
            oracle["canonical_count_match"] = True
            oracles[stage_to_oracle[stage]] = oracle
        records.append(
            {
                "frame_id": frame_id,
                "gt_id": gt_id,
                "variant": str(variant),
                "distance_bin": canonical.get("distance_bin"),
                **oracles,
                "signed_deltas": {
                    "O4_minus_O3": _signed_oracle_delta(oracles["O4"], oracles["O3"]),
                    "O5_minus_O4": _signed_oracle_delta(oracles["O5"], oracles["O4"]),
                    "O6_minus_O5": _signed_oracle_delta(oracles["O6"], oracles["O5"]),
                },
            }
        )
    return {
        "schema_version": STAGE_ORACLE_SCHEMA_VERSION,
        "frame_id": frame_id,
        "variant": str(variant),
        "canonical_count_gate_passed": True,
        "num_positive_gt": int(len(records)),
        "records": records,
    }


def _signed_oracle_delta(later_source, earlier_source):
    if later_source.get("iou") is None or earlier_source.get("iou") is None:
        return None
    return float(later_source["iou"] - earlier_source["iou"])


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


def build_point_retention_day3(day1, day2, reports_by_variant):
    """Join O1/O2 with O3-O6 once and preserve every signed ladder change."""
    if not isinstance(day1, dict) or day1.get("schema_version") != POINT_RETENTION_SCHEMA_VERSION:
        raise ValueError("Day 3 requires a valid point-retention Day 1 result")
    if not isinstance(day2, dict) or day2.get("schema_version") != FRAGMENT_ORACLE_SCHEMA_VERSION:
        raise ValueError("Day 3 requires a valid fragment-oracle Day 2 result")
    candidate_variant = day1["candidate_variant"]
    if day2.get("candidate_variant") != candidate_variant:
        raise ValueError("Day 1 and Day 2 candidate variants differ")
    indexed_stage = {}
    for report in reports_by_variant.get(candidate_variant, []):
        payload = report.get("stage_recoverability") or {}
        if payload.get("schema_version") != STAGE_ORACLE_SCHEMA_VERSION:
            raise ValueError(f"missing stage oracle evidence: {report.get('frame_id')}")
        if not payload.get("canonical_count_gate_passed"):
            raise ValueError(f"stage oracle count gate failed: {report.get('frame_id')}")
        for item in payload.get("records", []):
            key = (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
            if key in indexed_stage:
                raise ValueError(f"duplicate stage oracle evidence: {key}")
            indexed_stage[key] = item

    def index_selected(records):
        return {
            (str(item["frame_id"]).zfill(6), str(item["gt_id"])): item
            for item in records
        }

    day1_delta = index_selected(day1["delta_records"])
    day1_control = index_selected(day1["p1_control_records"])
    day2_delta = index_selected(day2["delta_records"])
    day2_control = index_selected(day2["p1_control_records"])
    if set(day1_delta) != set(day2_delta) or set(day1_control) != set(day2_control):
        raise ValueError("Day 1 and Day 2 selected GT sets differ")

    def join(keys, stage_counts, fragments):
        outputs = []
        for key in sorted(keys):
            if key not in indexed_stage:
                raise ValueError(f"missing selected stage oracle evidence: {key}")
            canonical = stage_counts[key]
            fragment = fragments[key]
            stage = indexed_stage[key]
            counts = canonical["stage_point_counts"]
            for oracle_name, count_name in (("O3", "intensity_filter"), ("O4", "z_filter"), ("O5", "roi"), ("O6", "raw")):
                if int(stage[oracle_name]["num_points"]) != int(counts[count_name]):
                    raise ValueError(f"joined oracle count differs from Day 1: {key} {oracle_name}")
            outputs.append(
                {
                    "frame_id": key[0],
                    "gt_id": key[1],
                    "distance_bin": canonical.get("distance_bin"),
                    "stage_point_counts": dict(counts),
                    "associated_cluster_point_count": fragment["associated_cluster_point_count"],
                    "associated_union_point_count": fragment["associated_union_point_count"],
                    "associated_union_gt_clipped_point_count": fragment["associated_union_gt_clipped_point_count"],
                    "O1": dict(fragment["O1"]),
                    "O2": dict(fragment["O2"]),
                    "O2_gt_clipped": dict(fragment["O2_gt_clipped"]),
                    "O3": dict(stage["O3"]),
                    "O4": dict(stage["O4"]),
                    "O5": dict(stage["O5"]),
                    "O6": dict(stage["O6"]),
                    "signed_deltas": {
                        "O2_minus_O1": fragment["signed_deltas"]["O2_minus_O1"],
                        "O2_gt_clipped_minus_O2": fragment["signed_deltas"]["O2_gt_clipped_minus_O2"],
                        "O3_minus_O2": _signed_oracle_delta(stage["O3"], fragment["O2"]),
                        "O3_minus_O2_gt_clipped": _signed_oracle_delta(stage["O3"], fragment["O2_gt_clipped"]),
                        "O4_minus_O3": stage["signed_deltas"]["O4_minus_O3"],
                        "O5_minus_O4": stage["signed_deltas"]["O5_minus_O4"],
                        "O6_minus_O5": stage["signed_deltas"]["O6_minus_O5"],
                    },
                }
            )
        return outputs

    return {
        "schema_version": STAGE_ORACLE_SCHEMA_VERSION,
        "source_day1_schema_version": POINT_RETENTION_SCHEMA_VERSION,
        "source_day2_schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
        "candidate_variant": candidate_variant,
        "canonical_count_gate_passed": True,
        "delta_gt_count": int(len(day1_delta)),
        "p1_control_gt_count": int(len(day1_control)),
        "delta_records": join(day1_delta, day1_delta, day2_delta),
        "p1_control_records": join(day1_control, day1_control, day2_control),
    }


def attribute_point_retention_record(record, material_gain_iou=MATERIAL_GAIN_IOU):
    """Apply the preregistered ladder without changing or filling missing evidence."""
    if material_gain_iou <= 0.0:
        raise ValueError("material gain threshold must be positive")
    required_oracles = ("O1", "O2", "O2_gt_clipped", "O3", "O4", "O5", "O6")
    if any(not isinstance(record.get(name), dict) for name in required_oracles):
        raise ValueError("root-cause attribution requires the complete O1-O6 ladder")
    deltas = record.get("signed_deltas")
    if not isinstance(deltas, dict):
        raise ValueError("root-cause attribution requires signed ladder deltas")

    signal_sources = {
        "CLUSTER_FRAGMENTATION_LIMITED": ("O2_minus_O1",),
        "CLUSTER_FORMATION_LIMITED": (
            "O2_gt_clipped_minus_O2", "O3_minus_O2_gt_clipped",
        ),
        "INTENSITY_FILTER_LIMITED": ("O4_minus_O3",),
        "Z_FILTER_LIMITED": ("O5_minus_O4",),
        "ROI_FILTER_LIMITED": ("O6_minus_O5",),
    }
    material_sources = {
        label: [
            name for name in names
            if deltas.get(name) is not None and float(deltas[name]) >= material_gain_iou
        ]
        for label, names in signal_sources.items()
    }
    material_signals = [label for label, sources in material_sources.items() if sources]
    raw_iou = record["O6"].get("iou")
    raw_recoverable = raw_iou is not None and float(raw_iou) >= RECOVERABLE_IOU
    if len(material_signals) > 1:
        root_cause = "MIXED"
    elif len(material_signals) == 1:
        root_cause = material_signals[0]
    elif not raw_recoverable:
        root_cause = "RAW_GEOMETRY_OBSERVABILITY_LIMITED"
    else:
        root_cause = "UNRESOLVED"
    return {
        "root_cause": root_cause,
        "material_signals": material_signals,
        "material_signal_sources": {
            label: sources for label, sources in material_sources.items() if sources
        },
        "material_gain_iou": float(material_gain_iou),
        "raw_recoverable_iou": float(RECOVERABLE_IOU),
        "raw_recoverable": bool(raw_recoverable),
    }


def _numeric_summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _summarize_oracle(records, oracle_name):
    values = [record[oracle_name].get("iou") for record in records]
    summary = _numeric_summary(values)
    valid_values = [float(value) for value in values if value is not None]
    summary.update({
        "null_count": int(len(values) - len(valid_values)),
        "iou_ge_0_25_count": int(sum(value >= 0.25 for value in valid_values)),
        "iou_ge_0_50_count": int(sum(value >= 0.50 for value in valid_values)),
    })
    return summary


def _summarize_delta(records, delta_name):
    values = [record["signed_deltas"].get(delta_name) for record in records]
    summary = _numeric_summary(values)
    valid_values = [float(value) for value in values if value is not None]
    summary.update({
        "null_count": int(len(values) - len(valid_values)),
        "positive_count": int(sum(value > 0.0 for value in valid_values)),
        "negative_count": int(sum(value < 0.0 for value in valid_values)),
        "material_gain_count": int(sum(value >= MATERIAL_GAIN_IOU for value in valid_values)),
    })
    return summary


def _summarize_point_retention_cohort(records):
    root_cause_counts = {}
    for record in records:
        label = record["attribution"]["root_cause"]
        root_cause_counts[label] = root_cause_counts.get(label, 0) + 1
    ordered_counts = dict(sorted(root_cause_counts.items()))
    if ordered_counts:
        ranked = sorted(ordered_counts.items(), key=lambda item: (-item[1], item[0]))
        unique_dominant = len(ranked) == 1 or ranked[0][1] > ranked[1][1]
        dominant_label = ranked[0][0] if unique_dominant else None
        dominant_count = ranked[0][1] if unique_dominant else 0
    else:
        dominant_label, dominant_count = None, 0
    count = len(records)
    count_fields = (
        "associated_cluster_point_count", "associated_union_point_count",
        "associated_union_gt_clipped_point_count",
    )
    stage_fields = {
        "raw": "raw", "post_roi": "roi", "post_z": "z_filter",
        "post_intensity": "intensity_filter",
    }
    distance_breakdown = {}
    for record in records:
        distance_bin = record.get("distance_bin")
        label = record["attribution"]["root_cause"]
        bucket = distance_breakdown.setdefault(distance_bin, {"count": 0, "root_cause_counts": {}})
        bucket["count"] += 1
        bucket["root_cause_counts"][label] = bucket["root_cause_counts"].get(label, 0) + 1
    for bucket in distance_breakdown.values():
        bucket["root_cause_counts"] = dict(sorted(bucket["root_cause_counts"].items()))
    delta_names = (
        "O2_minus_O1", "O2_gt_clipped_minus_O2", "O3_minus_O2",
        "O3_minus_O2_gt_clipped", "O4_minus_O3", "O5_minus_O4", "O6_minus_O5",
    )
    return {
        "gt_count": int(count),
        "root_cause_counts": ordered_counts,
        "dominant_root_cause": dominant_label,
        "dominant_root_cause_count": int(dominant_count),
        "dominant_root_cause_ratio": float(dominant_count / count) if count and dominant_label else None,
        "majority_reached": bool(count and dominant_label and dominant_count > count / 2.0),
        "oracle_iou": {
            name: _summarize_oracle(records, name)
            for name in ("O1", "O2", "O2_gt_clipped", "O3", "O4", "O5", "O6")
        },
        "signed_deltas": {name: _summarize_delta(records, name) for name in delta_names},
        "point_counts": {
            **{field: _numeric_summary(record[field] for record in records) for field in count_fields},
            **{
                label: _numeric_summary(record["stage_point_counts"][field] for record in records)
                for label, field in stage_fields.items()
            },
        },
        "distance_breakdown": dict(sorted(distance_breakdown.items())),
    }


def build_point_retention_day4(day1, day3):
    """Attribute and aggregate the frozen delta/P1 ladders for final review."""
    if not isinstance(day1, dict) or day1.get("schema_version") != POINT_RETENTION_SCHEMA_VERSION:
        raise ValueError("Day 4 requires a valid point-retention Day 1 result")
    if not isinstance(day3, dict) or day3.get("schema_version") != STAGE_ORACLE_SCHEMA_VERSION:
        raise ValueError("Day 4 requires a valid point-retention Day 3 result")
    if day3.get("candidate_variant") != day1.get("candidate_variant"):
        raise ValueError("Day 1 and Day 3 candidate variants differ")
    contract = day1.get("attribution_contract") or {}
    if float(contract.get("material_gain_iou", -1.0)) != MATERIAL_GAIN_IOU:
        raise ValueError("Day 1 material gain threshold differs from Day 4")
    expected_labels = set(contract.get("labels", []))

    def attribute(records, expected_count, cohort_name):
        if len(records) != int(expected_count):
            raise ValueError(f"{cohort_name} record count differs from Day 3 metadata")
        seen = set()
        outputs = []
        for source in records:
            key = (str(source["frame_id"]).zfill(6), str(source["gt_id"]))
            if key in seen:
                raise ValueError(f"duplicate Day 4 GT record: {key}")
            seen.add(key)
            output = dict(source)
            output["attribution"] = attribute_point_retention_record(source)
            if output["attribution"]["root_cause"] not in expected_labels:
                raise ValueError("Day 4 root cause is absent from Day 1 contract")
            outputs.append(output)
        return outputs

    delta_records = attribute(day3.get("delta_records", []), day3.get("delta_gt_count", -1), "delta")
    control_records = attribute(
        day3.get("p1_control_records", []), day3.get("p1_control_gt_count", -1), "P1 control"
    )
    if len(delta_records) != int(day1.get("delta_gt_count", -1)):
        raise ValueError("Day 4 delta cohort differs from frozen Day 1 cohort")
    if len(control_records) != int(day1.get("p1_control_gt_count", -1)):
        raise ValueError("Day 4 P1 cohort differs from frozen Day 1 cohort")
    return {
        "schema_version": ROOT_CAUSE_SCHEMA_VERSION,
        "source_day1_schema_version": POINT_RETENTION_SCHEMA_VERSION,
        "source_day3_schema_version": STAGE_ORACLE_SCHEMA_VERSION,
        "candidate_variant": day1["candidate_variant"],
        "attribution_policy": {
            "material_gain_iou": float(MATERIAL_GAIN_IOU),
            "raw_recoverable_iou": float(RECOVERABLE_IOU),
            "multiple_material_signals": "MIXED",
            "raw_limited_only_without_material_stage_signal": True,
            "cluster_formation_delta_sources": [
                "O2_gt_clipped_minus_O2", "O3_minus_O2_gt_clipped",
            ],
        },
        "delta_gt_count": int(len(delta_records)),
        "p1_control_gt_count": int(len(control_records)),
        "delta_records": delta_records,
        "p1_control_records": control_records,
        "delta_summary": _summarize_point_retention_cohort(delta_records),
        "p1_control_summary": _summarize_point_retention_cohort(control_records),
    }
