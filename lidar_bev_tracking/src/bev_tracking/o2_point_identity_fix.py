"""Narrow v15.3.2 O2 correction using raw-LiDAR source point identity."""

import copy
import hashlib
from pathlib import Path

import numpy as np

from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.point_retention import (
    MATERIAL_GAIN_IOU,
    ROOT_CAUSE_SCHEMA_VERSION,
    STAGE_ORACLE_SCHEMA_VERSION,
    build_pca_oracle,
    build_point_retention_day4,
)
from bev_tracking.report_writer import atomic_write_text


O2_FIX_SCHEMA_VERSION = "15.3.2-o2-point-identity-fix"
ROOT_CAUSE_LABELS = (
    "INTENSITY_FILTER_LIMITED", "MIXED", "Z_FILTER_LIMITED",
    "RAW_GEOMETRY_OBSERVABILITY_LIMITED",
)


def correct_fragment_record(source, *, gt_box, raw_points, clusters, cluster_source_indices):
    """Replace coordinate-row O2 with unique raw source-index O2."""
    raw_points = np.asarray(raw_points)
    cluster_ids = [str(value) for value in source.get("associated_cluster_ids", [])]
    if not cluster_ids:
        raise ValueError("corrected fragment record requires associated cluster ids")
    fragments_by_id = {
        str(item["cluster_id"]): item for item in source.get("fragment_oracles", [])
    }
    selected_indices = []
    corrected_fragments = []
    for cluster_id in cluster_ids:
        if not cluster_id.startswith("cluster_"):
            raise ValueError(f"invalid canonical cluster id: {cluster_id}")
        position = int(cluster_id.split("_", 1)[1]) - 1
        if position < 0 or position >= len(clusters):
            raise ValueError(f"canonical cluster is absent in provenance replay: {cluster_id}")
        cluster = np.asarray(clusters[position])
        source_indices = np.asarray(cluster_source_indices[position], dtype=np.int64)
        if len(cluster) != len(source_indices):
            raise ValueError(f"source index count differs for {cluster_id}")
        if not np.array_equal(raw_points[source_indices], cluster):
            raise ValueError(f"source indices do not reconstruct {cluster_id}")
        old_fragment = fragments_by_id.get(cluster_id)
        if old_fragment is None:
            raise ValueError(f"missing frozen fragment evidence: {cluster_id}")
        if int(old_fragment["num_points"]) != len(cluster):
            raise ValueError(f"cluster point count changed during provenance replay: {cluster_id}")
        replay = build_pca_oracle(cluster, gt_box)
        if replay["iou"] is not None and old_fragment.get("iou") is not None:
            if not np.isclose(replay["iou"], old_fragment["iou"], rtol=0.0, atol=1e-9):
                raise ValueError(f"O1 fragment IoU changed during provenance replay: {cluster_id}")
        corrected = dict(old_fragment)
        corrected["source_point_indices"] = source_indices.tolist()
        corrected_fragments.append(corrected)
        selected_indices.append(source_indices)

    stacked_indices = np.concatenate(selected_indices)
    union_indices = np.unique(stacked_indices)
    union_points = raw_points[union_indices]
    clipped_points = union_points[points_in_oriented_3d_box(union_points, gt_box)]
    o2 = build_pca_oracle(union_points, gt_box)
    o2_clipped = build_pca_oracle(clipped_points, gt_box)
    output = copy.deepcopy(source)
    output.update({
        "associated_cluster_point_count": int(len(stacked_indices)),
        "associated_union_point_count": int(len(union_indices)),
        "duplicate_union_point_count": int(len(stacked_indices) - len(union_indices)),
        "associated_union_gt_clipped_point_count": int(len(clipped_points)),
        "point_identity": "raw_lidar_point_index",
        "associated_union_source_point_indices": union_indices.tolist(),
        "fragment_oracles": corrected_fragments,
        "O2": o2,
        "O2_gt_clipped": o2_clipped,
        "signed_deltas": {
            "O2_minus_O1": _iou_delta(o2, source["O1"]),
            "O2_gt_clipped_minus_O2": _iou_delta(o2_clipped, o2),
        },
    })
    return output


def rebuild_day3_with_corrected_o2(day3, corrected_day2):
    """Keep O1/O3-O6 frozen and replace only O2-derived Day 3 evidence."""
    if day3.get("schema_version") != STAGE_ORACLE_SCHEMA_VERSION:
        raise ValueError("O2 fix requires a valid Day 3 source")

    def rebuild(source_records, corrected_records):
        corrections = _index(corrected_records)
        if set(_index(source_records)) != set(corrections):
            raise ValueError("corrected O2 GT set differs from frozen Day 3 cohort")
        outputs = []
        for source in source_records:
            key = (str(source["frame_id"]).zfill(6), str(source["gt_id"]))
            fragment = corrections[key]
            output = copy.deepcopy(source)
            output.update({
                "associated_cluster_point_count": fragment["associated_cluster_point_count"],
                "associated_union_point_count": fragment["associated_union_point_count"],
                "associated_union_gt_clipped_point_count": fragment["associated_union_gt_clipped_point_count"],
                "point_identity": "raw_lidar_point_index",
                "O2": copy.deepcopy(fragment["O2"]),
                "O2_gt_clipped": copy.deepcopy(fragment["O2_gt_clipped"]),
            })
            output["signed_deltas"].update({
                "O2_minus_O1": fragment["signed_deltas"]["O2_minus_O1"],
                "O2_gt_clipped_minus_O2": fragment["signed_deltas"]["O2_gt_clipped_minus_O2"],
                "O3_minus_O2": _iou_delta(output["O3"], output["O2"]),
                "O3_minus_O2_gt_clipped": _iou_delta(output["O3"], output["O2_gt_clipped"]),
            })
            outputs.append(output)
        return outputs

    output = copy.deepcopy(day3)
    output["point_identity"] = "raw_lidar_point_index"
    output["delta_records"] = rebuild(day3["delta_records"], corrected_day2["delta_records"])
    output["p1_control_records"] = rebuild(
        day3["p1_control_records"], corrected_day2["p1_control_records"]
    )
    return output


def build_o2_fix_result(source_report, corrected_day2, corrected_day3):
    """Build corrected Day 4 and the requested before/after closure evidence."""
    day1 = source_report["point_retention_day1"]
    before_day3 = source_report["point_retention_day3"]
    before_day4 = source_report["point_retention_day4"]
    if before_day4.get("schema_version") != ROOT_CAUSE_SCHEMA_VERSION:
        raise ValueError("O2 fix requires a valid Day 4 source")
    corrected_day4 = build_point_retention_day4(day1, corrected_day3)
    before_records = _index(before_day3["delta_records"])
    after_records = _index(corrected_day3["delta_records"])
    if set(before_records) != set(after_records) or len(after_records) != 22:
        raise ValueError("O2 fix must preserve the frozen delta-22 cohort")

    per_gt = []
    for key in sorted(before_records):
        before, after = before_records[key], after_records[key]
        per_gt.append({
            "frame_id": key[0], "gt_id": key[1],
            "O2_iou_before": before["O2"].get("iou"),
            "O2_iou_after": after["O2"].get("iou"),
            "O2_gt_clipped_iou_before": before["O2_gt_clipped"].get("iou"),
            "O2_gt_clipped_iou_after": after["O2_gt_clipped"].get("iou"),
            "O2_minus_O1_before": before["signed_deltas"].get("O2_minus_O1"),
            "O2_minus_O1_after": after["signed_deltas"].get("O2_minus_O1"),
            "O3_minus_O2_gt_clipped_before": before["signed_deltas"].get("O3_minus_O2_gt_clipped"),
            "O3_minus_O2_gt_clipped_after": after["signed_deltas"].get("O3_minus_O2_gt_clipped"),
        })

    before_attr = _index(before_day4["delta_records"])
    after_attr = _index(corrected_day4["delta_records"])
    original_mixed = [
        key for key, item in before_attr.items()
        if item["attribution"]["material_signals"]
        == ["CLUSTER_FORMATION_LIMITED", "INTENSITY_FILTER_LIMITED"]
    ]
    mixed_review = [{
        "frame_id": key[0], "gt_id": key[1],
        "before_root_cause": before_attr[key]["attribution"]["root_cause"],
        "before_material_signals": before_attr[key]["attribution"]["material_signals"],
        "after_root_cause": after_attr[key]["attribution"]["root_cause"],
        "after_material_signals": after_attr[key]["attribution"]["material_signals"],
        "unchanged": before_attr[key]["attribution"] == after_attr[key]["attribution"],
    } for key in sorted(original_mixed)]

    before_o4 = _material_count(before_day3["delta_records"], "O4_minus_O3")
    after_o4 = _material_count(corrected_day3["delta_records"], "O4_minus_O3")
    if before_o4 != 20 or after_o4 != 20:
        raise ValueError("core O4-O3 material result changed or no longer equals 20/22")
    return {
        "schema_version": O2_FIX_SCHEMA_VERSION,
        "scope": "diagnostic_only_o2_point_identity_correction",
        "gates": {
            "delta_22_preserved": True,
            "o1_frozen": True,
            "o3_o6_frozen": True,
            "canonical_point_count_gate_passed": bool(
                corrected_day3["canonical_count_gate_passed"]
            ),
            "core_o4_minus_o3_material_count_preserved": True,
        },
        "corrected_point_retention_day2": corrected_day2,
        "corrected_point_retention_day3": corrected_day3,
        "corrected_point_retention_day4": corrected_day4,
        "before_after": {
            "delta_22_O2_iou": per_gt,
            "O2_minus_O1_material_count": {
                "before": _material_count(before_day3["delta_records"], "O2_minus_O1"),
                "after": _material_count(corrected_day3["delta_records"], "O2_minus_O1"),
            },
            "O3_minus_O2_gt_clipped_material_count": {
                "before": _material_count(
                    before_day3["delta_records"], "O3_minus_O2_gt_clipped"
                ),
                "after": _material_count(
                    corrected_day3["delta_records"], "O3_minus_O2_gt_clipped"
                ),
            },
            "original_cluster_formation_plus_intensity_samples": mixed_review,
            "root_cause_counts": {
                label: {
                    "before": int(
                        before_day4["delta_summary"]["root_cause_counts"].get(label, 0)
                    ),
                    "after": int(
                        corrected_day4["delta_summary"]["root_cause_counts"].get(label, 0)
                    ),
                }
                for label in ROOT_CAUSE_LABELS
            },
            "O4_minus_O3_material_count": {"before": before_o4, "after": after_o4},
        },
    }


def write_sha256_sidecar(path):
    """Write `<hash>  <filename>` beside a completed diagnostic JSON."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = Path(f"{path}.sha256")
    atomic_write_text(sidecar, f"{digest}  {path.name}\n")
    return digest, sidecar


def _iou_delta(later, earlier):
    if later.get("iou") is None or earlier.get("iou") is None:
        return None
    return float(later["iou"] - earlier["iou"])


def _index(records):
    output = {}
    for item in records:
        key = (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
        if key in output:
            raise ValueError(f"duplicate GT record: {key}")
        output[key] = item
    return output


def _material_count(records, delta_name):
    return int(sum(
        item["signed_deltas"].get(delta_name) is not None
        and float(item["signed_deltas"][delta_name]) >= MATERIAL_GAIN_IOU
        for item in records
    ))
