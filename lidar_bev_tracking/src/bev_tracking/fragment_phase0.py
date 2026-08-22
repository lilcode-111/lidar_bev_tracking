"""Read-only Fragment-Level Geometry Recovery Phase-0 analysis."""

from collections import deque
import csv
import json
import math
from pathlib import Path

import numpy as np

from bev_tracking.clustering_detector import split_obstacle_filter_stages_with_indices
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.gesr_v1 import (
    CANDIDATE_INTENSITY_MIN,
    CONNECTIVITY_RADIUS_M,
    NUMERICAL_DTYPE,
    SEED_INTENSITY_MIN,
    build_seed_components_optimized,
    deterministic_pca_2d,
)
from bev_tracking.gesr_v1_failure_analysis import (
    compact_oracle,
    gt_point_indices,
    load_json,
    select_t2_material_recovery_targets,
)
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.point_retention import build_pca_oracle
from bev_tracking.v15_4_audit import ANNOTATION_EXCLUSION_CLASSES


MATERIAL_GAIN = 0.10
AXIS_ORDER = ("SEED_MAJOR", "SEED_MINOR")
SIDE_ORDER = ("NEG", "POS")


class FragmentPhase0Error(ValueError):
    pass


def _grid_cell(point_xy):
    return tuple(np.floor(np.asarray(point_xy) / CONNECTIVITY_RADIUS_M).astype(np.int64))


def build_candidate_fragments(points, source_indices):
    """Build radius-connected components once on the complete candidate pool."""
    points = np.asarray(points, dtype=NUMERICAL_DTYPE)
    indices = np.asarray(source_indices, dtype=np.int64)
    order = np.argsort(indices, kind="stable")
    points, indices = points[order], indices[order]
    grid = {}
    for position, point in enumerate(points):
        grid.setdefault(_grid_cell(point[:2]), []).append(position)
    radius2 = CONNECTIVITY_RADIUS_M * CONNECTIVITY_RADIUS_M
    visited = np.zeros(len(points), dtype=bool)
    fragments = []
    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            cx, cy = _grid_cell(points[current, :2])
            candidates = []
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    candidates.extend(grid.get((gx, gy), ()))
            for other in sorted(candidates):
                if visited[other]:
                    continue
                delta = points[other, :2] - points[current, :2]
                if float(delta @ delta) <= radius2:
                    visited[other] = True
                    queue.append(other)
        member_indices = np.sort(indices[np.asarray(members, dtype=np.int64)])
        fragments.append(build_fragment_record(points, indices, member_indices))
    return sorted(fragments, key=lambda item: item["runtime_id"])


def build_fragment_record(sorted_points, sorted_indices, member_source_indices):
    positions = np.searchsorted(sorted_indices, member_source_indices)
    member_points = sorted_points[positions]
    count = len(member_points)
    if count == 1:
        center = member_points[0, :2]
        endpoints = {"NEG": center, "POS": center}
        axis = None
        geometry_valid = False
        major_span = 0.0
        diagnostic = {"lambda1": None, "lambda2": None, "minor_span": 0.0}
    else:
        geometry = deterministic_pca_2d(member_points[:, :2])
        center = np.asarray(geometry.center)
        axis = np.asarray(geometry.major)
        endpoints = {
            "NEG": center + geometry.u_min * axis,
            "POS": center + geometry.u_max * axis,
        }
        geometry_valid = bool(geometry.valid)
        major_span = float(geometry.u_max - geometry.u_min)
        diagnostic = {
            "lambda1": geometry.lambda1,
            "lambda2": geometry.lambda2,
            "minor_span": float(geometry.v_max - geometry.v_min),
        }
    return {
        "runtime_id": int(member_source_indices[0]),
        "source_indices": member_source_indices,
        "point_count": int(count),
        "fragment_type": "SINGLETON" if count == 1 else "STRUCTURED",
        "center": np.asarray(center, dtype=np.float64),
        "major_axis": None if axis is None else np.asarray(axis, dtype=np.float64),
        "major_axis_valid": bool(geometry_valid),
        "major_span": major_span,
        "endpoints": endpoints,
        "minor_span": diagnostic["minor_span"],
        "lambda1": diagnostic["lambda1"],
        "lambda2": diagnostic["lambda2"],
        "linearity": (
            None
            if diagnostic["lambda1"] in (None, 0.0)
            else float(1.0 - diagnostic["lambda2"] / diagnostic["lambda1"])
        ),
        "range": float(np.linalg.norm(center)),
        "intensity_min": float(member_points[:, 3].min()),
        "intensity_mean": float(member_points[:, 3].mean()),
        "intensity_max": float(member_points[:, 3].max()),
    }


def best_endpoint_relation(fragment, valid_seed_components):
    candidates = []
    for component in valid_seed_components:
        geometry = component.geometry
        center = np.asarray(geometry.center)
        major = np.asarray(geometry.major)
        minor = np.asarray(geometry.minor)
        for axis_index, axis_name in enumerate(AXIS_ORDER):
            axis = major if axis_name == "SEED_MAJOR" else minor
            perpendicular = minor if axis_name == "SEED_MAJOR" else -major
            limits = (
                (geometry.u_min, geometry.u_max)
                if axis_name == "SEED_MAJOR"
                else (geometry.v_min, geometry.v_max)
            )
            for side_index, seed_side in enumerate(SIDE_ORDER):
                limit = limits[side_index]
                seed_endpoint = center + limit * axis
                outward = -axis if seed_side == "NEG" else axis
                for fragment_side_index, fragment_side in enumerate(SIDE_ORDER):
                    fragment_endpoint = np.asarray(fragment["endpoints"][fragment_side])
                    displacement = fragment_endpoint - seed_endpoint
                    forward = float(displacement @ outward)
                    if forward < 0.0:
                        continue
                    gap = float(np.linalg.norm(displacement))
                    lateral = float(abs(displacement @ perpendicular))
                    orientation_valid = bool(fragment["major_axis_valid"])
                    orientation = (
                        float(
                            math.acos(
                                float(
                                    np.clip(
                                        abs(fragment["major_axis"] @ axis), 0.0, 1.0
                                    )
                                )
                            )
                        )
                        if orientation_valid
                        else None
                    )
                    key = (
                        gap,
                        lateral,
                        0 if orientation_valid else 1,
                        orientation if orientation is not None else float("inf"),
                        component.runtime_id,
                        axis_index,
                        side_index,
                        fragment_side_index,
                    )
                    candidates.append(
                        (
                            key,
                            {
                                "seed_component_runtime_id": component.runtime_id,
                                "best_continuation_axis": axis_name,
                                "seed_outward_side": seed_side,
                                "fragment_endpoint_side": fragment_side,
                                "forward_projection": forward,
                                "endpoint_gap": gap,
                                "lateral_offset": lateral,
                                "orientation_difference": orientation,
                                "orientation_valid": orientation_valid,
                            },
                        )
                    )
    if not candidates:
        return {"has_computable_seed_relation": False}
    return {"has_computable_seed_relation": True, **min(candidates, key=lambda item: item[0])[1]}


def union_box_mask(points, boxes):
    mask = np.zeros(len(points), dtype=bool)
    for box in boxes:
        mask |= points_in_oriented_3d_box(points, box)
    return mask


def analyze_frame(data_root, frame_id, targets):
    velodyne, labels_path = resolve_kitti_paths(data_root, frame_id)
    raw = load_kitti_point_cloud(velodyne)
    labels = load_kitti_labels(labels_path)
    calib = load_kitti_calib(resolve_kitti_calib_path(data_root, frame_id))
    boxes = kitti_labels_to_lidar_boxes(labels, calib)
    boxes_by_id = {str(box["id"]): box for box in boxes}
    stages, stage_indices = split_obstacle_filter_stages_with_indices(
        raw, z_min=-0.9, intensity_min=0.38
    )
    _, t2_stage_indices = split_obstacle_filter_stages_with_indices(
        raw, z_min=-0.9, intensity_min=0.15
    )
    z_points = stages["z_filter"]
    z_indices = stage_indices["z_filter"]
    intensities = np.asarray(z_points[:, 3], dtype=NUMERICAL_DTYPE)
    candidate_mask = (
        (intensities >= CANDIDATE_INTENSITY_MIN)
        & (intensities < SEED_INTENSITY_MIN)
    )
    seed_mask = intensities >= SEED_INTENSITY_MIN
    fragments = build_candidate_fragments(z_points[candidate_mask], z_indices[candidate_mask])
    seed_components = build_seed_components_optimized(
        z_points[seed_mask], z_indices[seed_mask]
    )
    valid_seed_components = [item for item in seed_components if item.geometry.valid]
    annotation_boxes = [
        box
        for box in boxes
        if str(box.get("class_name", "")).strip().lower() in ANNOTATION_EXCLUSION_CLASSES
    ]
    for fragment in fragments:
        fragment_points = raw[fragment["source_indices"]]
        fragment["strict_background"] = not bool(
            union_box_mask(fragment_points, annotation_boxes).any()
        )
        fragment["relation"] = best_endpoint_relation(fragment, valid_seed_components)

    pairs = []
    cooperative = []
    for target in targets:
        gt_id = target["gt_id"]
        gt_box = boxes_by_id[gt_id]
        t0_gt = gt_point_indices(raw, stage_indices["intensity_filter"], gt_box)
        t2_gt = gt_point_indices(raw, t2_stage_indices["intensity_filter"], gt_box)
        t2_added_gt = np.setdiff1d(t2_gt, t0_gt, assume_unique=True)
        t0_oracle = compact_oracle(raw, t0_gt, gt_box)
        associated_fragments = []
        for fragment in fragments:
            fragment_points = raw[fragment["source_indices"]]
            inside_count = int(points_in_oriented_3d_box(fragment_points, gt_box).sum())
            recovery_point_count = int(
                len(
                    np.intersect1d(
                        fragment["source_indices"], t2_added_gt, assume_unique=True
                    )
                )
            )
            if recovery_point_count == 0:
                continue
            associated_fragments.append(fragment)
            cf_indices = np.union1d(t0_gt, fragment["source_indices"])
            oracle = build_pca_oracle(raw[cf_indices], gt_box)
            delta = (
                None
                if oracle["iou"] is None or t0_oracle["iou"] is None
                else float(oracle["iou"] - t0_oracle["iou"])
            )
            pairs.append(
                {
                    "frame_id": frame_id,
                    "gt_id": gt_id,
                    "fragment_runtime_id": fragment["runtime_id"],
                    "fragment_type": fragment["fragment_type"],
                    "fragment_point_count": fragment["point_count"],
                    "GT_inside_point_count": inside_count,
                    "T2_added_GT_point_count": recovery_point_count,
                    "non_GT_point_count": fragment["point_count"] - inside_count,
                    "oracle_fragment_purity": float(inside_count / fragment["point_count"]),
                    "T0_iou": t0_oracle["iou"],
                    "counterfactual_iou": oracle["iou"],
                    "fragment_delta_iou": delta,
                    "material_positive": delta is not None and delta >= MATERIAL_GAIN,
                }
            )
        material = [item for item in pairs if item["frame_id"] == frame_id and item["gt_id"] == gt_id and item["material_positive"]]
        cooperative_flag = False
        union_delta = None
        if not material and associated_fragments:
            union_fragment_indices = np.unique(
                np.concatenate([item["source_indices"] for item in associated_fragments])
            )
            union_oracle = build_pca_oracle(raw[np.union1d(t0_gt, union_fragment_indices)], gt_box)
            union_delta = (
                None
                if union_oracle["iou"] is None or t0_oracle["iou"] is None
                else float(union_oracle["iou"] - t0_oracle["iou"])
            )
            cooperative_flag = union_delta is not None and union_delta >= MATERIAL_GAIN
        cooperative.append(
            {
                "frame_id": frame_id,
                "gt_id": gt_id,
                "cooperative_recovery": cooperative_flag,
                "union_delta_iou": union_delta,
            }
        )
    return fragments, pairs, cooperative


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(values) == 0:
        return {key: None for key in ("valid_N", "min", "P25", "P50", "P75", "max")}
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return {
        "valid_N": int(len(values)),
        "min": float(values.min()),
        "P25": float(p25),
        "P50": float(p50),
        "P75": float(p75),
        "max": float(values.max()),
    }


def roc_auc_lower_is_positive(positive, negative):
    positive = [float(value) for value in positive if value is not None]
    negative = [float(value) for value in negative if value is not None]
    if not positive or not negative:
        return None
    ordered_negative = np.sort(np.asarray(negative, dtype=np.float64))
    wins = 0
    ties = 0
    for value in positive:
        left = int(np.searchsorted(ordered_negative, value, side="left"))
        right = int(np.searchsorted(ordered_negative, value, side="right"))
        wins += len(ordered_negative) - right
        ties += right - left
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def fragment_public_row(frame_id, fragment):
    relation = fragment["relation"]
    return {
        "frame_id": frame_id,
        "fragment_runtime_id": fragment["runtime_id"],
        "fragment_type": fragment["fragment_type"],
        "point_count": fragment["point_count"],
        "major_axis_valid": fragment["major_axis_valid"],
        "major_span": fragment["major_span"],
        "minor_span": fragment["minor_span"],
        "linearity": fragment["linearity"],
        "range": fragment["range"],
        "intensity_min": fragment["intensity_min"],
        "intensity_mean": fragment["intensity_mean"],
        "intensity_max": fragment["intensity_max"],
        "strict_background": fragment["strict_background"],
        **relation,
    }


def build_summary(fragment_rows, pairs, cooperative, target_count):
    recovery_ids = {(item["frame_id"], item["fragment_runtime_id"]) for item in pairs}
    material_pairs = [item for item in pairs if item["material_positive"]]
    material_ids = {
        (item["frame_id"], item["fragment_runtime_id"]) for item in material_pairs
    }
    material_fragments = [
        item for item in fragment_rows if (item["frame_id"], item["fragment_runtime_id"]) in material_ids
    ]
    backgrounds = [item for item in fragment_rows if item["strict_background"]]
    comparison = {}
    for field in ("endpoint_gap", "lateral_offset", "orientation_difference"):
        positive_values = [item.get(field) for item in material_fragments]
        background_values = [item.get(field) for item in backgrounds]
        comparison[field] = {
            "positive": distribution(positive_values),
            "strict_background": distribution(background_values),
            "ROC_AUC_lower_value_is_positive": roc_auc_lower_is_positive(
                positive_values, background_values
            ),
        }
    positive_gt = {(item["frame_id"], item["gt_id"]) for item in material_pairs}
    purity = [item["oracle_fragment_purity"] for item in material_pairs]
    no_relation = [item for item in fragment_rows if not item["has_computable_seed_relation"]]
    return {
        "candidate_fragment_count": len(fragment_rows),
        "recovery_associated_unique_fragment_count": len(recovery_ids),
        "recovery_associated_fragment_GT_pair_count": len(pairs),
        "material_positive_unique_fragment_count": len(material_ids),
        "material_positive_fragment_GT_pair_count": len(material_pairs),
        "singleton_material_positive_unique_fragment_count": sum(
            item["fragment_type"] == "SINGLETON" for item in material_fragments
        ),
        "structured_material_positive_unique_fragment_count": sum(
            item["fragment_type"] == "STRUCTURED" for item in material_fragments
        ),
        "fragment_formation_GT_coverage": {
            "count": len(positive_gt), "denominator": target_count
        },
        "cooperative_recovery_GT_count": sum(
            item["cooperative_recovery"] for item in cooperative
        ),
        "strict_background_fragment_count": len(backgrounds),
        "NO_SEED_RELATION_AVAILABLE_count": len(no_relation),
        "NO_SEED_RELATION_AVAILABLE_by_label": {
            "material_positive": sum(
                (item["frame_id"], item["fragment_runtime_id"]) in material_ids
                for item in no_relation
            ),
            "strict_background": sum(item["strict_background"] for item in no_relation),
        },
        "positive_vs_background": comparison,
        "material_positive_oracle_fragment_purity": distribution(purity),
    }


def run_fragment_phase0(data_root, gate_result_path, progress_callback=None):
    gate_result = load_json(gate_result_path)
    targets = select_t2_material_recovery_targets(gate_result)
    by_frame = {}
    for target in targets:
        by_frame.setdefault(target["frame_id"], []).append(target)
    fragment_rows, pairs, cooperative = [], [], []
    for index, frame_id in enumerate(sorted(by_frame), start=1):
        if progress_callback:
            progress_callback(index, len(by_frame), frame_id, by_frame[frame_id])
        fragments, frame_pairs, frame_cooperative = analyze_frame(
            data_root, frame_id, by_frame[frame_id]
        )
        fragment_rows.extend(fragment_public_row(frame_id, item) for item in fragments)
        pairs.extend(frame_pairs)
        cooperative.extend(frame_cooperative)
    summary = build_summary(fragment_rows, pairs, cooperative, len(targets))
    material_pairs = [item for item in pairs if item["material_positive"]]
    fragment_lookup = {
        (item["frame_id"], item["fragment_runtime_id"]): item
        for item in fragment_rows
    }
    positive_cases = sorted(
        material_pairs,
        key=lambda item: (-item["fragment_delta_iou"], item["frame_id"], item["gt_id"]),
    )[:5]
    positive_cases = [
        {
            **item,
            "fragment_geometry": fragment_lookup[
                (item["frame_id"], item["fragment_runtime_id"])
            ],
        }
        for item in positive_cases
    ]
    background_cases = sorted(
        [item for item in fragment_rows if item["strict_background"] and item.get("endpoint_gap") is not None],
        key=lambda item: (item["endpoint_gap"], item["frame_id"], item["fragment_runtime_id"]),
    )[:5]
    return {
        "schema_version": "fragment-level-geometry-recovery-phase0-v1",
        "scope": {"target_GT_count": len(targets), "unique_frame_count": len(by_frame)},
        "semantics": {
            "one_shared_runtime_graph_per_frame": True,
            "candidate_intensity": "0.15 <= intensity < 0.38",
            "fragment_radius_m": CONNECTIVITY_RADIUS_M,
            "positive_distribution_unit": "unique_material_positive_fragment",
            "background_distribution_unit": "unique_strict_background_fragment",
            "ROC_AUC_score_direction": "lower_geometry_value_is_more_positive",
            "offline_oracle_labels_not_runtime_features": True,
        },
        "source_phase2_execution_commit": gate_result.get("actual_commit"),
        "runtime_algorithm_implemented": False,
        "learning_performed": False,
        "summary": summary,
        "representative_cases": {
            "material_positive_fragment_GT_pairs": positive_cases,
            "strict_background_fragments": background_cases,
        },
        "fragment_rows": fragment_rows,
        "fragment_GT_pairs": pairs,
        "cooperative_records": cooperative,
    }


def write_phase0_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    fragments_path = output_dir / "fragments.csv"
    pairs_path = output_dir / "fragment_gt_pairs.csv"
    summary_payload = {key: value for key, value in result.items() if key not in {"fragment_rows", "fragment_GT_pairs"}}
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    write_csv(fragments_path, result["fragment_rows"])
    write_csv(pairs_path, result["fragment_GT_pairs"])
    return {"summary": summary_path, "fragments": fragments_path, "pairs": pairs_path}


def write_csv(path, rows):
    path = Path(path)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
