"""Read-only 15.5 Phase-0 geometry-critical selectivity analysis."""

import gc
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.point_retention import build_pca_oracle
from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_materialization import load_json, raw_file_sha256
from bev_tracking.v15_5_seed_support import (
    DAY2_SCHEMA_VERSION,
    DAY3_SCHEMA_VERSION,
    POINT_IDENTITY,
    R_SEED_M,
    SCHEMA_VERSION,
    V155SeedSupportError,
    canonical_record_sha256,
    load_verified_metric_cache,
    nearest_percentile,
    oracle_iou_delta,
)


SCHEMA_VERSION_GEOMETRY_CRITICAL = "15.5-phase0-geometry-critical-selectivity-v1"
MATERIAL_RECOVERY_DELTA = 0.10
RUNTIME_FEATURES = (
    "intensity",
    "range_xy",
    "d_seed",
    "n_seed_0p6",
    "local_neighbor_count_0p6",
    "nearest_neighbor_distance",
)
OFFLINE_ORACLE_FEATURES = (
    "projection_on_seed_pca_axis_1",
    "projection_on_seed_pca_axis_2",
    "distance_to_current_axis_extent",
    "distance_to_axis_1_extent",
    "distance_to_axis_2_extent",
)


def build_geometry_critical_analysis(
    *,
    repo_root=".",
    day1_path="outputs/seed_support_selectivity/v15_5_phase0_day1.json",
    day2_path="outputs/seed_support_selectivity/v15_5_phase0_day2.json",
    day3_path="outputs/seed_support_selectivity/v15_5_phase0_day3.json",
    output_path="outputs/seed_support_selectivity/v15_5_phase0_geometry_critical.json",
):
    root = Path(repo_root)
    paths = {"day1": root / day1_path, "day2": root / day2_path, "day3": root / day3_path}
    day1, day2, day3 = (load_json(paths[name]) for name in ("day1", "day2", "day3"))
    validate_frozen_inputs(day1, day2, day3, paths, day1_path, day2_path)

    metric_context, far_stability, cache_identity = load_hm_metric_context(day1, root)
    t0_records, t2_records, t0_global = load_formal_geometry_context(day1, root)
    day3_keys = {
        (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
        for item in day3["delta_22"]["records"]
    }
    if set(t0_records) != day3_keys or set(t2_records) != day3_keys or len(day3_keys) != 22:
        raise V155SeedSupportError("geometry-critical delta-22 identities differ from Day3")

    raw_by_frame = {}
    boxes_by_frame = {}
    retained_global_by_frame = {}
    trees_by_frame = {}
    far_records = []
    marginal_records = []
    data_root = root / day1["source"]["data_root"]
    for number, key in enumerate(sorted(day3_keys), start=1):
        frame_id, gt_id = key
        if frame_id not in raw_by_frame:
            raw, boxes = load_frame_geometry(data_root, frame_id)
            raw_by_frame[frame_id], boxes_by_frame[frame_id] = raw, boxes
            supported_hm = supported_hm_indices(metric_context[frame_id])
            retained = np.union1d(t0_global[frame_id], supported_hm)
            retained_global_by_frame[frame_id] = retained
            trees_by_frame[frame_id] = cKDTree(raw[retained, :2].astype(np.float64)) if len(retained) else None
        if gt_id not in boxes_by_frame[frame_id]:
            raise V155SeedSupportError(f"delta-22 GT box is missing: {key}")
        raw = raw_by_frame[frame_id]
        gt_box = boxes_by_frame[frame_id][gt_id]
        t0_indices = stage_indices(t0_records[key], len(raw), key)
        t2_indices = stage_indices(t2_records[key], len(raw), key)
        hm_indices = np.setdiff1d(t2_indices, t0_indices, assume_unique=True)
        metrics = select_frame_hm_metrics(metric_context[frame_id], hm_indices, key)
        supported_mask = metrics["d_seed"] <= R_SEED_M
        original_seed_supported = np.union1d(t0_indices, hm_indices[supported_mask])
        far_mask = metrics["range_xy"] >= 30.0
        far_uniform = np.union1d(t0_indices, hm_indices[far_mask])
        far_seed = np.union1d(t0_indices, hm_indices[far_mask & supported_mask])

        base_oracle = build_pca_oracle(raw[original_seed_supported], gt_box)
        far_record = build_far_gt_record(
            key, raw, gt_box, t0_indices, t2_indices, far_uniform, far_seed
        )
        far_records.append(far_record)

        dropped_positions = np.flatnonzero(~supported_mask)
        for position in dropped_positions:
            point_index = int(hm_indices[position])
            added_oracle = build_pca_oracle(
                raw[np.union1d(original_seed_supported, [point_index])], gt_box
            )
            gain = oracle_iou_delta(added_oracle, base_oracle)
            if gain is None:
                raise V155SeedSupportError(f"single-point marginal PCA is invalid: {key} {point_index}")
            offline = offline_pca_extent_features(raw[original_seed_supported, :2], raw[point_index, :2])
            tree = trees_by_frame[frame_id]
            if tree is None:
                neighbor_count, nearest = 0, None
            else:
                point_xy = raw[point_index, :2].astype(np.float64)
                neighbor_count = int(tree.query_ball_point(point_xy, r=R_SEED_M, return_length=True))
                nearest_value = float(tree.query(point_xy, k=1, workers=1)[0])
                nearest = nearest_value if np.isfinite(nearest_value) else None
            marginal_records.append({
                "frame_id": frame_id,
                "gt_id": gt_id,
                "raw_lidar_point_index": point_index,
                "band": str(metrics["band"][position]),
                "G_p": float(gain),
                "base_iou": base_oracle["iou"],
                "iou_after_adding_point": added_oracle["iou"],
                "runtime_features": {
                    "intensity": float(raw[point_index, 3]),
                    "range_xy": float(metrics["range_xy"][position]),
                    "d_seed": float(metrics["d_seed"][position]),
                    "n_seed_0p6": int(metrics["n_seed_0p6"][position]),
                    "local_neighbor_count_0p6": neighbor_count,
                    "nearest_neighbor_distance": nearest,
                },
                "offline_oracle_geometry_features": offline,
            })
        print(f"[Geometry-critical {number:02d}/{len(day3_keys):02d}] {frame_id}:{gt_id}")

    expected_dropped = sum(
        int(item["point_counts"]["H_M_available"]) - int(item["point_counts"]["H_M_seed_supported"])
        for item in day3["delta_22"]["records"]
    )
    if expected_dropped != 146 or len(marginal_records) != expected_dropped:
        raise V155SeedSupportError(
            f"dropped delta-22 point conservation failed: expected={expected_dropped} observed={len(marginal_records)}"
        )
    identities = [
        (item["frame_id"], item["gt_id"], item["raw_lidar_point_index"])
        for item in marginal_records
    ]
    if len(identities) != len(set(identities)):
        raise V155SeedSupportError("dropped GT-point association identity is duplicated")

    range_distribution = build_recovery_range_distribution(day3, boxes_by_frame)
    far_comparison = build_far_comparison_summary(far_stability, far_records)
    marginal_summary = summarize_marginal_gain(marginal_records)
    output = {
        "schema_version": SCHEMA_VERSION_GEOMETRY_CRITICAL,
        "analysis_role": "read_only_geometry_critical_selectivity",
        "source": {
            "day1": {"path": day1_path, "sha256": raw_file_sha256(paths["day1"])},
            "day2": {"path": day2_path, "sha256": raw_file_sha256(paths["day2"])},
            "day3": {"path": day3_path, "sha256": raw_file_sha256(paths["day3"])},
            "formal_comparison_commit": day1["formal_comparison_commit"],
            "formal_reports": {name: day1["source"]["formal_reports"][name] for name in ("T0", "T2")},
            "H_M_metric_cache_identity_sha256": canonical_record_sha256(sorted(cache_identity)),
        },
        "frozen_contracts": {
            "point_identity": POINT_IDENTITY,
            "dropped_analysis_identity": "(frame_id, gt_id, raw_lidar_point_index)",
            "coordinate_row_dedup_used": False,
            "range_segments": {"near": "[0,15)", "mid": "[15,30)", "far": "[30,+inf)"},
            "GT_range_source": "GT center range_xy",
            "far_point_source": "individual point range_xy >= 30m",
            "r_seed_m": R_SEED_M,
            "r_seed_search": False,
            "neighbor_reference": "current global T0 plus globally seed-supported H/M points",
            "neighbor_space": "XY",
            "neighbor_radius_m": R_SEED_M,
            "neighbor_radius_or_K_search": False,
            "single_point_base_is_fixed": True,
            "single_point_gain_is_non_additive": True,
            "high_G_threshold_defined": False,
            "PCA_extent_role": "offline_oracle_explanation_only_not_runtime_rule_evidence",
            "GT_oracle_leakage": False,
        },
        "far_H_M_frame_stability": far_stability,
        "T2_material_recovery_GT_range_distribution": range_distribution,
        "far_uniform_vs_far_seed": far_comparison,
        "dropped_vehicle_points": {
            "expected_count": 146,
            "observed_count": len(marginal_records),
            "records": sorted(marginal_records, key=lambda item: (item["frame_id"], item["gt_id"], item["raw_lidar_point_index"])),
            "summary": marginal_summary,
        },
        "direction_assessment": {
            "Range_Conditioned": "EVIDENCE_READY_PENDING_ALGORITHM_REVIEW",
            "Geometry_Extension": "EVIDENCE_READY_PENDING_ALGORITHM_REVIEW",
        },
        "analysis_complete": True,
        "algorithm_implementation_authorized": False,
        "GT_oracle_leakage": False,
        "r_seed_search": False,
        "range_boundary_search": False,
        "neighbor_radius_or_K_search": False,
        "formal_pipeline_rerun": False,
        "formal_results_modified": False,
    }
    atomic_write_json(root / output_path, output)
    return output


def validate_frozen_inputs(day1, day2, day3, paths, day1_path, day2_path):
    if day1.get("schema_version") != SCHEMA_VERSION or day1.get("day1_complete") is not True:
        raise V155SeedSupportError("Day1 is not complete")
    if day2.get("schema_version") != DAY2_SCHEMA_VERSION or day2.get("day2_complete") is not True:
        raise V155SeedSupportError("Day2 is not complete")
    if day3.get("schema_version") != DAY3_SCHEMA_VERSION or day3.get("phase0_complete") is not True:
        raise V155SeedSupportError("Day3 is not complete")
    if day2["source"]["day1"] != {"path": day1_path, "sha256": raw_file_sha256(paths["day1"])}:
        raise V155SeedSupportError("Day2 does not reference the supplied Day1 identity")
    if day3["source"]["day1"] != {"path": day1_path, "sha256": raw_file_sha256(paths["day1"])}:
        raise V155SeedSupportError("Day3 does not reference the supplied Day1 identity")
    if day3["source"]["day2"] != {"path": day2_path, "sha256": raw_file_sha256(paths["day2"])}:
        raise V155SeedSupportError("Day3 does not reference the supplied Day2 identity")
    if any(item.get("formal_pipeline_rerun") is not False for item in (day1, day2, day3)):
        raise V155SeedSupportError("formal pipeline rerun contract changed")
    if day3.get("gt_oracle_leakage") is not False:
        raise V155SeedSupportError("Day3 GT leakage contract changed")


def load_hm_metric_context(day1, root):
    frame_ids = [str(item["frame_id"]).zfill(6) for item in day1["per_frame_counts"]]
    frame_counts = {
        frame: {
            band: {
                category: {"point_count": 0, "seed_supported_count": 0}
                for category in ("vehicle", "background")
            }
            for band in ("H", "M")
        }
        for frame in frame_ids
    }
    context = {frame: {} for frame in frame_ids}
    identities = []
    processed = 0
    for record in day1["point_metric_cache"]["records"]:
        if record["band"] not in ("H", "M"):
            continue
        if record["category"] not in ("global", "vehicle", "background"):
            continue
        metrics = load_verified_metric_cache(root, record)
        frame_id, band, category = str(record["frame_id"]).zfill(6), record["band"], record["category"]
        identities.append([record["path"], record["sha256"]])
        processed += 1
        if category == "global":
            context[frame_id][band] = metrics
            continue
        far = metrics["range_xy"] >= 30.0
        frame_counts[frame_id][band][category] = {
            "point_count": int(np.count_nonzero(far)),
            "seed_supported_count": int(np.count_nonzero(far & (metrics["d_seed"] <= R_SEED_M))),
        }
    if processed != len(frame_ids) * 2 * 3:
        raise V155SeedSupportError("H/M metric cache coverage is incomplete")
    if any(set(context[frame]) != {"H", "M"} for frame in frame_ids):
        raise V155SeedSupportError("global H/M metric context is incomplete")
    return context, build_far_frame_stability(frame_counts), identities


def build_far_frame_stability(frame_counts):
    output = {}
    for band in ("H", "M"):
        records = []
        for frame_id in sorted(frame_counts):
            categories = frame_counts[frame_id][band]
            vehicle, background = categories["vehicle"], categories["background"]
            vrr = ratio(vehicle["seed_supported_count"], vehicle["point_count"])
            brr = ratio(background["seed_supported_count"], background["point_count"])
            comparable = vrr is not None and brr is not None
            records.append({
                "frame_id": frame_id,
                "vehicle": {**vehicle, "VRR": vrr},
                "background": {**background, "BRR": brr},
                "comparable": comparable,
                "VRR_minus_BRR": vrr - brr if comparable else None,
            })
        comparable = [item for item in records if item["comparable"]]
        differences = np.asarray([item["VRR_minus_BRR"] for item in comparable], dtype=np.float64)
        vehicle_count = sum(item["vehicle"]["point_count"] for item in records)
        vehicle_rescue = sum(item["vehicle"]["seed_supported_count"] for item in records)
        background_count = sum(item["background"]["point_count"] for item in records)
        background_rescue = sum(item["background"]["seed_supported_count"] for item in records)
        output[band] = {
            "aggregate": {
                "vehicle_point_count": vehicle_count,
                "vehicle_rescue_count": vehicle_rescue,
                "VRR": ratio(vehicle_rescue, vehicle_count),
                "background_point_count": background_count,
                "background_rescue_count": background_rescue,
                "BRR": ratio(background_rescue, background_count),
            },
            "frame_stability": {
                "frame_count": len(records),
                "comparable_frame_count": len(comparable),
                "non_comparable_frame_count": len(records) - len(comparable),
                "positive_frame_count": int(np.count_nonzero(differences > 0)),
                "negative_frame_count": int(np.count_nonzero(differences < 0)),
                "tie_frame_count": int(np.count_nonzero(differences == 0)),
                "median_VRR_minus_BRR": float(np.median(differences)) if len(differences) else None,
                "vehicle_rescue_concentration": concentration(records, "vehicle"),
                "background_rescue_concentration": concentration(records, "background"),
            },
            "per_frame": records,
        }
    return output


def concentration(records, category):
    ranked = sorted(
        (
            {"frame_id": item["frame_id"], "rescue_count": item[category]["seed_supported_count"]}
            for item in records
        ),
        key=lambda item: (-item["rescue_count"], item["frame_id"]),
    )
    total = sum(item["rescue_count"] for item in ranked)
    return {
        "total_rescue_count": total,
        "top1_frame_share": ranked[0]["rescue_count"] / total if total else None,
        "top5_frame_share": sum(item["rescue_count"] for item in ranked[:5]) / total if total else None,
        "top1_frame": ranked[:1],
        "top5_frames": ranked[:5],
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def load_formal_geometry_context(day1, root):
    output_records = {}
    t0_global = None
    for variant in ("T0", "T2"):
        artifact = day1["source"]["formal_reports"][variant]
        path = root / artifact["path"]
        if raw_file_sha256(path) != artifact["sha256"]:
            raise V155SeedSupportError(f"{variant} formal report hash mismatch")
        report = load_json(path)
        records = {}
        for item in report.get("source_point_identity_records", []):
            key = (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
            if key in records:
                raise V155SeedSupportError(f"duplicate {variant} delta-22 identity: {key}")
            records[key] = item
        output_records[variant] = records
        if variant == "T0":
            t0_global = {}
            for frame in report.get("source_point_universes", []):
                frame_id = str(frame["frame_id"]).zfill(6)
                values = np.asarray(
                    frame["universes"]["global_post_intensity"]["source_point_indices"], dtype=np.int64
                )
                if len(values) != len(np.unique(values)):
                    raise V155SeedSupportError(f"duplicate T0 global point identity: {frame_id}")
                t0_global[frame_id] = np.sort(values)
        del report
        gc.collect()
    return output_records["T0"], output_records["T2"], t0_global


def load_frame_geometry(data_root, frame_id):
    velodyne, labels_path = resolve_kitti_paths(data_root, frame_id)
    raw = load_kitti_point_cloud(velodyne)
    labels = load_kitti_labels(labels_path)
    calib = load_kitti_calib(resolve_kitti_calib_path(data_root, frame_id))
    boxes = {str(box["id"]): box for box in kitti_labels_to_lidar_boxes(labels, calib)}
    return raw, boxes


def supported_hm_indices(frame_context):
    arrays = [
        metrics["source_point_indices"][metrics["d_seed"] <= R_SEED_M]
        for metrics in frame_context.values()
    ]
    values = np.sort(np.concatenate(arrays)) if arrays else np.asarray([], dtype=np.int64)
    if len(values) != len(np.unique(values)):
        raise V155SeedSupportError("H/M supported point identities overlap")
    return values


def stage_indices(record, raw_count, key):
    stage = record["stages"]["intensity_filter"]
    values = np.sort(np.asarray(stage["source_point_indices"], dtype=np.int64))
    if len(values) != int(stage["count"]) or len(values) != len(np.unique(values)):
        raise V155SeedSupportError(f"invalid stage identity: {key}")
    if len(values) and (values[0] < 0 or values[-1] >= raw_count):
        raise V155SeedSupportError(f"stage identity out of raw point range: {key}")
    return values


def select_frame_hm_metrics(frame_context, hm_indices, key):
    combined = {}
    for name in ("source_point_indices", "range_xy", "d_seed", "n_seed_0p6"):
        combined[name] = np.concatenate([frame_context[band][name] for band in ("H", "M")])
    combined["band"] = np.concatenate([
        np.full(len(frame_context[band]["source_point_indices"]), band, dtype="U1")
        for band in ("H", "M")
    ])
    order = np.argsort(combined["source_point_indices"])
    combined = {name: values[order] for name, values in combined.items()}
    source = combined["source_point_indices"]
    positions = np.searchsorted(source, hm_indices)
    if np.any(positions >= len(source)) or not np.array_equal(source[positions], hm_indices):
        raise V155SeedSupportError(f"delta-22 H/M identity is not in the global cache: {key}")
    return {name: values[positions] for name, values in combined.items() if name != "source_point_indices"}


def build_far_gt_record(key, raw, gt_box, t0_indices, t2_indices, far_uniform, far_seed):
    indices = {"T0": t0_indices, "R_far_seed": far_seed, "R_far_uniform": far_uniform, "T2": t2_indices}
    representations = {name: build_pca_oracle(raw[values], gt_box) for name, values in indices.items()}
    output = {
        "frame_id": key[0],
        "gt_id": key[1],
        "GT_range_xy": float(np.hypot(gt_box["x"], gt_box["y"])),
        "GT_range_segment": range_segment_from_value(float(np.hypot(gt_box["x"], gt_box["y"]))),
        "point_counts": {name: int(len(values)) for name, values in indices.items()},
        "representations": representations,
        "iou_delta_vs_T0": {},
        "material_recovery_vs_T0": {},
    }
    for name in ("R_far_seed", "R_far_uniform", "T2"):
        delta = oracle_iou_delta(representations[name], representations["T0"])
        output["iou_delta_vs_T0"][name] = delta
        output["material_recovery_vs_T0"][name] = delta is not None and delta >= MATERIAL_RECOVERY_DELTA
    return output


def range_segment_from_value(value):
    if value < 0:
        raise V155SeedSupportError("range cannot be negative")
    if value < 15.0:
        return "near"
    if value < 30.0:
        return "mid"
    return "far"


def build_recovery_range_distribution(day3, boxes_by_frame):
    summary = day3["delta_22"]["summary"]
    t2 = {tuple(item) for item in summary["T2"]["material_recovery_vs_T0_identity_list"]}
    seed = {tuple(item) for item in summary["Seed_Supported"]["material_recovery_vs_T0_identity_list"]}
    retained = t2 & seed
    return {
        "T2_material_recovery": distribute_gt_keys_by_range(t2, boxes_by_frame),
        "Seed_Supported_retained_T2_recovery": distribute_gt_keys_by_range(retained, boxes_by_frame),
    }


def distribute_gt_keys_by_range(keys, boxes_by_frame):
    output = {segment: [] for segment in ("near", "mid", "far")}
    for frame_id, gt_id in sorted(keys):
        box = boxes_by_frame[frame_id][gt_id]
        distance = float(np.hypot(box["x"], box["y"]))
        output[range_segment_from_value(distance)].append({
            "frame_id": frame_id, "gt_id": gt_id, "GT_range_xy": distance
        })
    return {
        "count": len(keys),
        "by_range": {
            segment: {"count": len(records), "records": records}
            for segment, records in output.items()
        },
    }


def build_far_comparison_summary(far_stability, records):
    point_counts = {}
    for mode, field in (("R_far_uniform", "point_count"), ("R_far_seed", "rescue_count")):
        point_counts[mode] = {
            category: sum(
                far_stability[band]["aggregate"][f"{category}_{field}"]
                for band in ("H", "M")
            )
            for category in ("vehicle", "background")
        }
    modes = {}
    for mode in ("T0", "R_far_seed", "R_far_uniform", "T2"):
        oracles = [item["representations"][mode] for item in records]
        ious = [float(item["iou"]) for item in oracles if item["iou"] is not None]
        material = [] if mode == "T0" else [
            [item["frame_id"], item["gt_id"]]
            for item in records if item["material_recovery_vs_T0"][mode]
        ]
        modes[mode] = {
            "valid_pca_count": len(ious),
            "delta_22_pca_iou_median": float(np.median(ious)) if ious else None,
            "iou_ge_0_25_count": sum(value >= 0.25 for value in ious),
            "iou_ge_0_50_count": sum(value >= 0.50 for value in ious),
            "material_recovery_count": len(material) if mode != "T0" else None,
            "material_recovery_identity_list": material if mode != "T0" else None,
        }
        if mode in point_counts:
            modes[mode]["vehicle_rescue_count"] = point_counts[mode]["vehicle"]
            modes[mode]["background_rescue_count"] = point_counts[mode]["background"]
    return {"modes": modes, "delta_22_records": records}


def offline_pca_extent_features(base_xy, point_xy):
    base = np.asarray(base_xy, dtype=np.float64)
    point = np.asarray(point_xy, dtype=np.float64)
    centered = base - np.mean(base, axis=0)
    if len(base) < 3 or np.linalg.matrix_rank(centered) < 2:
        raise V155SeedSupportError("offline PCA extent requires valid two-dimensional base geometry")
    _, vectors = np.linalg.eigh(centered.T @ centered)
    axes = vectors[:, ::-1]
    for index in range(2):
        pivot = int(np.argmax(np.abs(axes[:, index])))
        if axes[pivot, index] < 0:
            axes[:, index] *= -1
    base_projection = centered @ axes
    point_projection = (point - np.mean(base, axis=0)) @ axes
    minimum, maximum = np.min(base_projection, axis=0), np.max(base_projection, axis=0)
    extension = np.maximum(np.maximum(minimum - point_projection, point_projection - maximum), 0.0)
    return {
        "projection_on_seed_pca_axis_1": float(point_projection[0]),
        "projection_on_seed_pca_axis_2": float(point_projection[1]),
        "distance_to_current_axis_extent": float(np.linalg.norm(extension)),
        "distance_to_axis_1_extent": float(extension[0]),
        "distance_to_axis_2_extent": float(extension[1]),
        "axis_1_current_extent": [float(minimum[0]), float(maximum[0])],
        "axis_2_current_extent": [float(minimum[1]), float(maximum[1])],
        "role": "offline_oracle_explanation_only",
    }


def summarize_marginal_gain(records):
    gains = np.asarray([item["G_p"] for item in records], dtype=np.float64)
    distribution = {
        "count": len(records),
        "min": float(np.min(gains)),
        "P25": nearest_percentile(gains, 25),
        "P50": nearest_percentile(gains, 50),
        "P75": nearest_percentile(gains, 75),
        "P90": nearest_percentile(gains, 90),
        "max": float(np.max(gains)),
        "positive_count": int(np.count_nonzero(gains > 0)),
        "zero_count": int(np.count_nonzero(gains == 0)),
        "negative_count": int(np.count_nonzero(gains < 0)),
        "high_G_threshold_defined": False,
    }
    runtime_correlations = feature_correlations(records, "runtime_features", RUNTIME_FEATURES)
    offline_correlations = feature_correlations(
        records, "offline_oracle_geometry_features", OFFLINE_ORACLE_FEATURES
    )
    ordered = sorted(records, key=lambda item: (-item["G_p"], item["frame_id"], item["gt_id"], item["raw_lidar_point_index"]))
    top = [{
        "frame_id": item["frame_id"],
        "gt_id": item["gt_id"],
        "raw_lidar_point_index": item["raw_lidar_point_index"],
        "band": item["band"],
        "G_p": item["G_p"],
        "runtime_features": item["runtime_features"],
        "offline_oracle_geometry_features": item["offline_oracle_geometry_features"],
    } for item in ordered[:10]]
    ranking = [
        {
            "rank": rank,
            "frame_id": item["frame_id"],
            "gt_id": item["gt_id"],
            "raw_lidar_point_index": item["raw_lidar_point_index"],
            "band": item["band"],
            "G_p": item["G_p"],
        }
        for rank, item in enumerate(ordered, start=1)
    ]
    return {
        "G_p_distribution": distribution,
        "runtime_feature_spearman_correlation": runtime_correlations,
        "offline_oracle_extent_spearman_correlation": offline_correlations,
        "G_p_ranking": ranking,
        "top_G_cases_descriptive_only": top,
    }


def feature_correlations(records, group, feature_names):
    gains = np.asarray([item["G_p"] for item in records], dtype=np.float64)
    output = {}
    for feature in feature_names:
        values = np.asarray([
            np.nan if item[group].get(feature) is None else float(item[group][feature])
            for item in records
        ], dtype=np.float64)
        valid = np.isfinite(gains) & np.isfinite(values)
        if np.count_nonzero(valid) < 2 or len(np.unique(values[valid])) < 2:
            rho = None
        else:
            result = spearmanr(values[valid], gains[valid])
            rho = float(result.statistic) if np.isfinite(result.statistic) else None
        output[feature] = {"paired_count": int(np.count_nonzero(valid)), "spearman_rho": rho}
    return output
