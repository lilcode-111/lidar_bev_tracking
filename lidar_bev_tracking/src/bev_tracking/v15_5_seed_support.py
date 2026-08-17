import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bev_tracking.kitti import load_kitti_labels, load_kitti_point_cloud, resolve_kitti_calib_path, resolve_kitti_paths
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.point_retention import build_pca_oracle
from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_materialization import load_json, raw_file_sha256


SCHEMA_VERSION = "15.5-phase0-seed-support-day1-v1"
DAY2_SCHEMA_VERSION = "15.5-phase0-seed-support-day2-v1"
DAY3_SCHEMA_VERSION = "15.5-phase0-seed-support-day3-v1"
POINT_IDENTITY = "(frame_id, raw_lidar_point_index)"
R_SEED_M = 0.60
VARIANTS = ("T0", "T1", "T2", "T_off")
UNIVERSES = {
    "global": "global_post_intensity",
    "vehicle": "positive_gt_post_intensity",
    "background": "annotation_excluded_background_post_intensity",
}
BANDS = {
    "H": {"included": "T1", "excluded": "T0", "intensity_min": 0.30, "intensity_max_exclusive": 0.38},
    "M": {"included": "T2", "excluded": "T1", "intensity_min": 0.15, "intensity_max_exclusive": 0.30},
    "L": {"included": "T_off", "excluded": "T2", "intensity_min": 0.00, "intensity_max_exclusive": 0.15},
}
REPORT_REGISTRY_IDS = {
    "T0": "t0_formal_report",
    "T1": "t1_formal_report",
    "T2": "t2_formal_report",
    "T_off": "t_off_formal_report",
}


class V155SeedSupportError(ValueError):
    pass


def build_phase0_day3(
    *,
    repo_root=".",
    day1_path="outputs/seed_support_selectivity/v15_5_phase0_day1.json",
    day2_path="outputs/seed_support_selectivity/v15_5_phase0_day2.json",
    output_path="outputs/seed_support_selectivity/v15_5_phase0_day3.json",
):
    """Compare delta-22 T0, globally seed-supported H/M, and T2 representations."""
    root = Path(repo_root)
    day1_file, day2_file = root / day1_path, root / day2_path
    day1, day2 = load_json(day1_file), load_json(day2_file)
    validate_day1_for_day2(day1)
    validate_day2_for_day3(day1, day2, day1_path, day1_file)

    source_records = {}
    for variant in ("T0", "T2"):
        artifact = day1["source"]["formal_reports"][variant]
        report_path = root / artifact["path"]
        if raw_file_sha256(report_path) != artifact["sha256"]:
            raise V155SeedSupportError(f"{variant} formal report hash mismatch")
        source_records[variant] = load_delta22_source_records(report_path, variant)
    if set(source_records["T0"]) != set(source_records["T2"]):
        raise V155SeedSupportError("T0 and T2 delta-22 GT identities differ")
    if len(source_records["T0"]) != 22:
        raise V155SeedSupportError("Day3 requires exactly the frozen delta-22 GT identities")

    supported_by_frame, support_cache_identity = load_global_hm_support(day1, root)
    gt_boxes_by_frame = {}
    raw_points_by_frame = {}
    records = []
    keys = sorted(source_records["T0"])
    for number, key in enumerate(keys, start=1):
        frame_id, gt_id = key
        if frame_id not in raw_points_by_frame:
            data_root = root / day1["source"]["data_root"]
            velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
            raw_points_by_frame[frame_id] = load_kitti_point_cloud(velodyne_path)
            labels = load_kitti_labels(label_path)
            calib = load_kitti_calib(resolve_kitti_calib_path(data_root, frame_id))
            gt_boxes_by_frame[frame_id] = {
                str(box["id"]): box for box in kitti_labels_to_lidar_boxes(labels, calib)
            }
        if gt_id not in gt_boxes_by_frame[frame_id]:
            raise V155SeedSupportError(f"frozen GT identity is missing from KITTI labels: {key}")
        record = build_delta22_representation_record(
            key=key,
            raw_points=raw_points_by_frame[frame_id],
            gt_box=gt_boxes_by_frame[frame_id][gt_id],
            t0_record=source_records["T0"][key],
            t2_record=source_records["T2"][key],
            globally_supported_hm=supported_by_frame.get(frame_id, np.asarray([], dtype=np.int64)),
        )
        records.append(record)
        print(f"[Day3 {number:02d}/{len(keys):02d}] {frame_id}:{gt_id}")

    summary = summarize_delta22_representations(records)
    output = {
        "schema_version": DAY3_SCHEMA_VERSION,
        "analysis_role": "read_only_delta_22_representation_check",
        "source": {
            "day1": {"path": day1_path, "sha256": raw_file_sha256(day1_file)},
            "day2": {"path": day2_path, "sha256": raw_file_sha256(day2_file)},
            "formal_comparison_commit": day1["formal_comparison_commit"],
            "formal_reports": {name: day1["source"]["formal_reports"][name] for name in ("T0", "T2")},
            "global_H_M_support_cache_identity_sha256": canonical_record_sha256(sorted(support_cache_identity)),
        },
        "contracts": {
            "point_identity": POINT_IDENTITY,
            "coordinate_row_dedup_used": False,
            "seed_supported_definition": "T0 + ((H union M) where globally_computed_d_seed <= 0.60m)",
            "gt_use": "evaluation_only_after_global_seed_support_decision",
            "gt_oracle_leakage": False,
            "r_seed_m": R_SEED_M,
            "r_seed_search_allowed": False,
            "material_recovery_iou_delta_min": 0.10,
        },
        "delta_22": {"count": len(records), "records": records, "summary": summary},
        "phase0_observation": {
            "VRR_higher_than_BRR": day2["descriptive_observation"]["VRR_higher_than_BRR"],
            "seed_supported_T2_material_recovery_retention": summary["recovery_retention_vs_T2"],
            "decision_status": "EVIDENCE_READY_PENDING_ALGORITHM_REVIEW",
        },
        "day3_complete": True,
        "phase0_complete": True,
        "gt_oracle_leakage": False,
        "formal_pipeline_rerun": False,
        "formal_results_modified": False,
    }
    atomic_write_json(root / output_path, output)
    return output


def validate_day2_for_day3(day1, day2, day1_path, day1_file):
    if day2.get("schema_version") != DAY2_SCHEMA_VERSION or day2.get("day2_complete") is not True:
        raise V155SeedSupportError("Day2 artifact is not complete")
    if day2.get("source", {}).get("day1", {}).get("path") != day1_path:
        raise V155SeedSupportError("Day2 references a different Day1 artifact")
    if day2["source"]["day1"].get("sha256") != raw_file_sha256(day1_file):
        raise V155SeedSupportError("Day2 Day1 hash is stale")
    if day2.get("gt_oracle_leakage") is not False or day2.get("formal_pipeline_rerun") is not False:
        raise V155SeedSupportError("Day2 violates the read-only Phase-0 contract")
    if day2.get("source", {}).get("formal_comparison_commit") != day1.get("formal_comparison_commit"):
        raise V155SeedSupportError("Day1 and Day2 formal commit identities differ")


def load_delta22_source_records(report_path, variant):
    report = load_json(report_path)
    records = report.get("source_point_identity_records")
    if not isinstance(records, list):
        raise V155SeedSupportError(f"{variant} report is missing source point identity records")
    output = {}
    for item in records:
        key = (str(item["frame_id"]).zfill(6), str(item["gt_id"]))
        if key in output:
            raise V155SeedSupportError(f"duplicate delta-22 identity in {variant}: {key}")
        if item.get("point_identity") != "raw_lidar_point_index":
            raise V155SeedSupportError(f"{variant} delta-22 point identity contract changed")
        output[key] = item
    del report
    gc.collect()
    return output


def load_global_hm_support(day1, root):
    by_frame = {}
    identities = []
    records = [
        record for record in day1["point_metric_cache"]["records"]
        if record["category"] == "global" and record["band"] in ("H", "M")
    ]
    expected_frames = {str(item["frame_id"]).zfill(6) for item in day1["per_frame_counts"]}
    if len(records) != len(expected_frames) * 2:
        raise V155SeedSupportError("global H/M support cache coverage is incomplete")
    for record in records:
        metrics = load_verified_metric_cache(root, record)
        supported = np.asarray(metrics["source_point_indices"][metrics["d_seed"] <= R_SEED_M], dtype=np.int64)
        frame_id = str(record["frame_id"]).zfill(6)
        by_frame.setdefault(frame_id, []).append(supported)
        identities.append([record["path"], record["sha256"]])
    output = {}
    for frame_id in expected_frames:
        arrays = by_frame.get(frame_id, [])
        values = np.sort(np.concatenate(arrays)) if arrays else np.asarray([], dtype=np.int64)
        if len(values) != len(np.unique(values)):
            raise V155SeedSupportError(f"supported H/M point identities overlap: {frame_id}")
        output[frame_id] = values
    return output, identities


def build_delta22_representation_record(*, key, raw_points, gt_box, t0_record, t2_record, globally_supported_hm):
    raw_points = np.asarray(raw_points)
    t0_indices = _validated_stage_indices(t0_record, "intensity_filter", len(raw_points), key)
    t2_indices = _validated_stage_indices(t2_record, "intensity_filter", len(raw_points), key)
    missing = np.setdiff1d(t0_indices, t2_indices, assume_unique=True)
    if len(missing):
        raise V155SeedSupportError(f"T0 GT points are not a T2 subset: {key}")
    hm_indices = np.setdiff1d(t2_indices, t0_indices, assume_unique=True)
    supported_added = np.intersect1d(
        hm_indices, np.asarray(globally_supported_hm, dtype=np.int64), assume_unique=True
    )
    seed_supported_indices = np.union1d(t0_indices, supported_added)
    if len(seed_supported_indices) != len(t0_indices) + len(supported_added):
        raise V155SeedSupportError(f"seed-supported source identities overlap T0: {key}")

    oracles = {
        "T0": build_pca_oracle(raw_points[t0_indices], gt_box),
        "Seed_Supported": build_pca_oracle(raw_points[seed_supported_indices], gt_box),
        "T2": build_pca_oracle(raw_points[t2_indices], gt_box),
    }
    _validate_recomputed_oracle(oracles["T0"], t0_record.get("post_intensity_diagnostic_pca"), "T0", key)
    _validate_recomputed_oracle(oracles["T2"], t2_record.get("post_intensity_diagnostic_pca"), "T2", key)
    seed_delta = oracle_iou_delta(oracles["Seed_Supported"], oracles["T0"])
    t2_delta = oracle_iou_delta(oracles["T2"], oracles["T0"])
    return {
        "frame_id": key[0],
        "gt_id": key[1],
        "point_counts": {
            "T0": int(len(t0_indices)),
            "H_M_available": int(len(hm_indices)),
            "H_M_seed_supported": int(len(supported_added)),
            "Seed_Supported": int(len(seed_supported_indices)),
            "T2": int(len(t2_indices)),
        },
        "representations": oracles,
        "iou_delta_vs_T0": {"Seed_Supported": seed_delta, "T2": t2_delta},
        "material_recovery_vs_T0": {
            "Seed_Supported": seed_delta is not None and seed_delta >= 0.10,
            "T2": t2_delta is not None and t2_delta >= 0.10,
        },
    }


def _validated_stage_indices(record, stage, raw_count, key):
    values = np.asarray(record["stages"][stage]["source_point_indices"], dtype=np.int64)
    if len(values) != int(record["stages"][stage]["count"]):
        raise V155SeedSupportError(f"stage point count mismatch: {key}")
    if len(values) != len(np.unique(values)):
        raise V155SeedSupportError(f"duplicate source point identity: {key}")
    values = np.sort(values)
    if len(values) and (values[0] < 0 or values[-1] >= raw_count):
        raise V155SeedSupportError(f"source point identity is out of range: {key}")
    return values


def _validate_recomputed_oracle(recomputed, frozen, variant, key, tolerance=1e-8):
    if not isinstance(frozen, dict):
        raise V155SeedSupportError(f"{variant} frozen PCA oracle is missing: {key}")
    if recomputed["status"] != frozen.get("status") or recomputed["num_points"] != int(frozen.get("num_points", -1)):
        raise V155SeedSupportError(f"{variant} frozen PCA status/count mismatch: {key}")
    left, right = recomputed["iou"], frozen.get("iou")
    if (left is None) != (right is None) or (left is not None and not np.isclose(left, right, atol=tolerance, rtol=0.0)):
        raise V155SeedSupportError(f"{variant} frozen PCA IoU mismatch: {key}")


def oracle_iou_delta(later, baseline):
    if later.get("iou") is None or baseline.get("iou") is None:
        return None
    return float(later["iou"] - baseline["iou"])


def summarize_delta22_representations(records):
    modes = ("T0", "Seed_Supported", "T2")
    summary = {}
    for mode in modes:
        oracles = [item["representations"][mode] for item in records]
        ious = [float(item["iou"]) for item in oracles if item["iou"] is not None]
        point_counts = [int(item["point_counts"][mode]) for item in records]
        material = [] if mode == "T0" else [
            [item["frame_id"], item["gt_id"]]
            for item in records if item["material_recovery_vs_T0"][mode]
        ]
        summary[mode] = {
            "gt_count": len(records),
            "valid_pca_count": len(ious),
            "point_count_median": float(np.median(point_counts)) if point_counts else None,
            "iou_median": float(np.median(ious)) if ious else None,
            "iou_ge_0_25_count": sum(value >= 0.25 for value in ious),
            "iou_ge_0_50_count": sum(value >= 0.50 for value in ious),
            "material_recovery_vs_T0_count": len(material) if mode != "T0" else None,
            "material_recovery_vs_T0_identity_list": material if mode != "T0" else None,
        }
    seed_keys = {tuple(value) for value in summary["Seed_Supported"]["material_recovery_vs_T0_identity_list"]}
    t2_keys = {tuple(value) for value in summary["T2"]["material_recovery_vs_T0_identity_list"]}
    retained = sorted(seed_keys & t2_keys)
    summary["recovery_retention_vs_T2"] = {
        "T2_material_recovery_count": len(t2_keys),
        "Seed_Supported_material_recovery_count": len(seed_keys),
        "retained_T2_material_recovery_count": len(retained),
        "retained_T2_material_recovery_identity_list": [list(key) for key in retained],
        "retention_ratio": len(retained) / len(t2_keys) if t2_keys else None,
    }
    return summary


def build_phase0_day2(
    *,
    repo_root=".",
    day1_path="outputs/seed_support_selectivity/v15_5_phase0_day1.json",
    output_path="outputs/seed_support_selectivity/v15_5_phase0_day2.json",
):
    root = Path(repo_root)
    day1_file = root / day1_path
    day1 = load_json(day1_file)
    validate_day1_for_day2(day1)
    frame_ids = [str(item["frame_id"]).zfill(6) for item in day1["per_frame_counts"]]
    if len(frame_ids) != 100 or len(set(frame_ids)) != 100:
        raise V155SeedSupportError("Day2 requires the frozen 100 unique frame identities")

    accumulator = _empty_metric_accumulator()
    frame_counts = {
        frame_id: {
            "frame_id": frame_id,
            "by_band": {
                band: {
                    category: {"point_count": 0, "seed_supported_count": 0}
                    for category in ("vehicle", "background")
                }
                for band in BANDS
            },
        }
        for frame_id in frame_ids
    }
    cache_records = day1["point_metric_cache"]["records"]
    expected_records = len(frame_ids) * len(BANDS) * len(UNIVERSES)
    if len(cache_records) != expected_records:
        raise V155SeedSupportError("Day1 point-metric cache record count is incomplete")
    verified_cache_identity = []
    processed = 0
    for record in cache_records:
        if record["category"] == "global":
            continue
        if record["category"] not in ("vehicle", "background") or record["band"] not in BANDS:
            raise V155SeedSupportError("unexpected Day1 cache category or band")
        metrics = load_verified_metric_cache(root, record)
        band, category = record["band"], record["category"]
        frame_id = str(record["frame_id"]).zfill(6)
        if frame_id not in frame_counts:
            raise V155SeedSupportError(f"cache references unknown frame: {frame_id}")
        supported = metrics["d_seed"] <= R_SEED_M
        point_count = int(len(metrics["d_seed"]))
        rescue_count = int(np.count_nonzero(supported))
        if point_count != int(record["point_count"]) or rescue_count != int(record["seed_supported_count"]):
            raise V155SeedSupportError(f"Day1 cache count mismatch: {record['path']}")
        frame_counts[frame_id]["by_band"][band][category] = {
            "point_count": point_count,
            "seed_supported_count": rescue_count,
        }
        _append_metric_values(accumulator[band]["all"][category], metrics)
        segments = segment_masks(metrics["range_xy"])
        for segment, mask in segments.items():
            _append_metric_values(
                accumulator[band][segment][category],
                {name: values[mask] for name, values in metrics.items()},
            )
        verified_cache_identity.append([record["path"], record["sha256"]])
        processed += 1

    if processed != len(frame_ids) * len(BANDS) * 2:
        raise V155SeedSupportError("vehicle/background Day1 cache coverage is incomplete")
    by_band = {
        band: build_selectivity_record(
            summarize_metric_values(accumulator[band]["all"]["vehicle"]),
            summarize_metric_values(accumulator[band]["all"]["background"]),
        )
        for band in BANDS
    }
    by_band_range = {
        band: {
            segment: build_selectivity_record(
                summarize_metric_values(accumulator[band][segment]["vehicle"]),
                summarize_metric_values(accumulator[band][segment]["background"]),
            )
            for segment in ("near", "mid", "far")
        }
        for band in BANDS
    }
    validate_day2_conservation(day1, by_band, by_band_range)
    primary = combine_selectivity(accumulator, ("H", "M"))
    all_low = combine_selectivity(accumulator, ("H", "M", "L"))
    per_frame, stability = build_frame_stability(frame_counts)
    output = {
        "schema_version": DAY2_SCHEMA_VERSION,
        "analysis_role": "read_only_phase0_descriptive",
        "source": {
            "day1": {"path": day1_path, "sha256": raw_file_sha256(day1_file)},
            "formal_comparison_commit": day1["formal_comparison_commit"],
            "formal_reports": day1["source"]["formal_reports"],
            "cache_record_count_declared": len(cache_records),
            "cache_record_count_verified": processed,
            "cache_identity_sha256": canonical_record_sha256(sorted(verified_cache_identity)),
        },
        "contracts": {
            "point_identity": POINT_IDENTITY,
            "coordinate_row_dedup_used": False,
            "r_seed_m": R_SEED_M,
            "r_seed_search_allowed": False,
            "support_rule": "d_seed <= 0.60m",
            "distance_segments": {
                "near": "0 <= range_xy < 15",
                "mid": "15 <= range_xy < 30",
                "far": "range_xy >= 30",
            },
            "percentile_method": "nearest",
            "gt_oracle_leakage": False,
            "decision_threshold_for_primary_frozen": False,
        },
        "by_band": by_band,
        "by_band_and_range": by_band_range,
        "primary_H_plus_M": primary,
        "all_low_intensity_H_M_L": all_low,
        "per_frame": per_frame,
        "frame_stability": stability,
        "descriptive_observation": {
            "VRR_higher_than_BRR": primary["VRR"] is not None and primary["BRR"] is not None and primary["VRR"] > primary["BRR"],
            "interpretation_status": "DESCRIPTIVE_ONLY_NO_FROZEN_DECISION_THRESHOLD",
        },
        "day2_complete": True,
        "delta_22_representation_complete": False,
        "gt_oracle_leakage": False,
        "formal_pipeline_rerun": False,
        "formal_results_modified": False,
    }
    atomic_write_json(root / output_path, output)
    return output


def validate_day1_for_day2(day1):
    if day1.get("schema_version") != SCHEMA_VERSION or day1.get("day1_complete") is not True:
        raise V155SeedSupportError("Day1 artifact is not complete")
    identity = day1.get("identity_validation", {})
    if identity.get("status") != "PASS" or identity.get("source_point_monotonicity") != "PASS":
        raise V155SeedSupportError("Day1 source identity validation did not PASS")
    contracts = day1.get("contracts", {})
    if contracts.get("point_identity") != POINT_IDENTITY or contracts.get("coordinate_row_dedup_used") is not False:
        raise V155SeedSupportError("Day1 point identity contract changed")
    if float(contracts.get("r_seed_m", -1)) != R_SEED_M or contracts.get("r_seed_search_allowed") is not False:
        raise V155SeedSupportError("Day1 r_seed contract changed")
    if contracts.get("gt_oracle_leakage") is not False or day1.get("formal_pipeline_rerun") is not False:
        raise V155SeedSupportError("Day1 violates read-only or GT leakage contract")


def load_verified_metric_cache(root, record):
    path = root / record["path"]
    if not path.is_file() or raw_file_sha256(path) != record["sha256"]:
        raise V155SeedSupportError(f"Day1 metric cache hash mismatch: {record['path']}")
    with np.load(path, allow_pickle=False) as payload:
        required = ("source_point_indices", "range_xy", "d_seed", "n_seed_0p6")
        if set(payload.files) != set(required):
            raise V155SeedSupportError(f"Day1 metric cache fields changed: {record['path']}")
        output = {name: np.asarray(payload[name]) for name in required}
    lengths = {len(values) for values in output.values()}
    if len(lengths) != 1:
        raise V155SeedSupportError(f"Day1 metric cache arrays are misaligned: {record['path']}")
    identities = output["source_point_indices"].astype(np.int64, copy=False)
    if len(identities) != len(np.unique(identities)):
        raise V155SeedSupportError(f"Day1 metric cache contains duplicate identity: {record['path']}")
    if np.any(output["range_xy"] < 0) or np.any(output["d_seed"] < 0) or np.any(output["n_seed_0p6"] < 0):
        raise V155SeedSupportError(f"Day1 metric cache contains invalid metric values: {record['path']}")
    return output


def segment_masks(range_xy):
    values = np.asarray(range_xy, dtype=np.float64)
    return {
        "near": (values >= 0.0) & (values < 15.0),
        "mid": (values >= 15.0) & (values < 30.0),
        "far": values >= 30.0,
    }


def summarize_metric_values(values):
    d_seed = _concatenate(values["d_seed"], dtype=np.float64)
    n_seed = _concatenate(values["n_seed_0p6"], dtype=np.int64)
    count = int(len(d_seed))
    supported = int(np.count_nonzero(d_seed <= R_SEED_M))
    return {
        "point_count": count,
        "seed_supported_count": supported,
        "seed_supported_ratio": supported / count if count else None,
        "d_seed_percentiles_m": {
            name: nearest_percentile(d_seed, percentile)
            for name, percentile in (("P25", 25), ("P50", 50), ("P75", 75), ("P90", 90))
        },
        "no_finite_seed_count": int(np.count_nonzero(~np.isfinite(d_seed))),
        "n_seed_0p6_explanation": {
            "mean": float(np.mean(n_seed)) if count else None,
            "P50": nearest_percentile(n_seed, 50),
            "P90": nearest_percentile(n_seed, 90),
        },
    }


def build_selectivity_record(vehicle, background):
    vrr = vehicle["seed_supported_ratio"]
    brr = background["seed_supported_ratio"]
    return {
        "vehicle": vehicle,
        "background": background,
        "VRR": vrr,
        "BRR": brr,
        "VRR_minus_BRR": vrr - brr if vrr is not None and brr is not None else None,
        "VRR_div_BRR": vrr / brr if vrr is not None and brr not in (None, 0.0) else None,
    }


def combine_selectivity(accumulator, bands):
    combined = {
        category: {
            name: [array for band in bands for array in accumulator[band]["all"][category][name]]
            for name in ("d_seed", "n_seed_0p6")
        }
        for category in ("vehicle", "background")
    }
    return build_selectivity_record(
        summarize_metric_values(combined["vehicle"]),
        summarize_metric_values(combined["background"]),
    )


def build_frame_stability(frame_counts):
    records = []
    for frame_id in sorted(frame_counts):
        item = frame_counts[frame_id]
        primary = {}
        for category in ("vehicle", "background"):
            count = sum(item["by_band"][band][category]["point_count"] for band in ("H", "M"))
            rescued = sum(item["by_band"][band][category]["seed_supported_count"] for band in ("H", "M"))
            primary[category] = {
                "point_count": count,
                "seed_supported_count": rescued,
                "rescue_ratio": rescued / count if count else None,
            }
        vrr, brr = primary["vehicle"]["rescue_ratio"], primary["background"]["rescue_ratio"]
        records.append({
            "frame_id": frame_id,
            "by_band": item["by_band"],
            "primary_H_plus_M": {
                **primary,
                "VRR_minus_BRR": vrr - brr if vrr is not None and brr is not None else None,
            },
        })
    comparable = [record for record in records if record["primary_H_plus_M"]["VRR_minus_BRR"] is not None]
    differences = np.asarray([record["primary_H_plus_M"]["VRR_minus_BRR"] for record in comparable], dtype=np.float64)
    vehicle_rescues = np.asarray([
        record["primary_H_plus_M"]["vehicle"]["seed_supported_count"] for record in records
    ], dtype=np.int64)
    total_rescue = int(np.sum(vehicle_rescues))
    ordered = np.sort(vehicle_rescues)[::-1]
    stability = {
        "frame_count": len(records),
        "frames_with_vehicle_points": sum(record["primary_H_plus_M"]["vehicle"]["point_count"] > 0 for record in records),
        "frames_with_vehicle_rescue": int(np.count_nonzero(vehicle_rescues)),
        "comparable_frame_count": len(comparable),
        "frames_VRR_gt_BRR": int(np.count_nonzero(differences > 0)),
        "frames_VRR_eq_BRR": int(np.count_nonzero(differences == 0)),
        "frames_VRR_lt_BRR": int(np.count_nonzero(differences < 0)),
        "frame_VRR_minus_BRR_median": float(np.median(differences)) if len(differences) else None,
        "max_frame_vehicle_rescue_share": float(ordered[0] / total_rescue) if total_rescue else None,
        "top5_frame_vehicle_rescue_share": float(np.sum(ordered[:5]) / total_rescue) if total_rescue else None,
    }
    return records, stability


def validate_day2_conservation(day1, by_band, by_band_range):
    for band in BANDS:
        for category in ("vehicle", "background"):
            expected = day1["band_totals"][band][category]
            observed = by_band[band][category]
            if int(expected["point_count"]) != observed["point_count"] or int(expected["seed_supported_count"]) != observed["seed_supported_count"]:
                raise V155SeedSupportError(f"Day2 does not conserve Day1 totals: {band} {category}")
            segment_count = sum(by_band_range[band][segment][category]["point_count"] for segment in ("near", "mid", "far"))
            segment_rescue = sum(by_band_range[band][segment][category]["seed_supported_count"] for segment in ("near", "mid", "far"))
            if segment_count != observed["point_count"] or segment_rescue != observed["seed_supported_count"]:
                raise V155SeedSupportError(f"Day2 range segments do not conserve totals: {band} {category}")


def nearest_percentile(values, percentile):
    values = np.asarray(values)
    if not len(values):
        return None
    result = np.percentile(values, percentile, method="nearest")
    return float(result) if np.isfinite(result) else None


def canonical_record_sha256(value):
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _empty_metric_accumulator():
    return {
        band: {
            segment: {
                category: {"d_seed": [], "n_seed_0p6": []}
                for category in ("vehicle", "background")
            }
            for segment in ("all", "near", "mid", "far")
        }
        for band in BANDS
    }


def _append_metric_values(target, metrics):
    target["d_seed"].append(np.asarray(metrics["d_seed"], dtype=np.float64))
    target["n_seed_0p6"].append(np.asarray(metrics["n_seed_0p6"], dtype=np.int64))


def _concatenate(values, dtype):
    arrays = [np.asarray(value, dtype=dtype) for value in values if len(value)]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=dtype)


def build_phase0_day1(
    *,
    repo_root=".",
    closure_path="docs/v15_4_closure.json",
    evidence_path="docs/v15_4_closure_evidence.json",
    cache_dir="outputs/seed_support_selectivity/day1_cache",
    output_path="outputs/seed_support_selectivity/v15_5_phase0_day1.json",
):
    root = Path(repo_root)
    closure = load_json(root / closure_path)
    evidence = load_json(root / evidence_path)
    _validate_closed_source(root, closure, evidence, evidence_path)
    registry = evidence["evidence_registry"]
    cache_root = root / cache_dir
    source_indices = {}
    source_artifacts = {}
    data_roots = set()
    source_run_ids = {}
    formal_commit = closure["canonical_artifacts"]["experiment_result"]["formal_comparison_commit"]

    for variant in VARIANTS:
        registry_id = REPORT_REGISTRY_IDS[variant]
        record = registry[registry_id]
        report_path = root / record["path"]
        if raw_file_sha256(report_path) != record["sha256"]:
            raise V155SeedSupportError(f"15.4 source artifact hash mismatch: {registry_id}")
        index = materialize_variant_identity_cache(
            report_path=report_path,
            cache_dir=cache_root / "source_identity" / variant,
            variant=variant,
        )
        if not index["source_run_id"].endswith(formal_commit[:12]):
            raise V155SeedSupportError(f"{variant} source run commit identity mismatch")
        source_indices[variant] = index
        source_run_ids[variant] = index["source_run_id"]
        data_roots.add(index["data_root"])
        source_artifacts[variant] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "registry_id": registry_id,
        }
    if len(data_roots) != 1:
        raise V155SeedSupportError("formal variants use different data roots")
    data_root = next(iter(data_roots))
    frame_ids = validate_variant_frame_identity(source_indices)

    totals = {
        band: {
            category: {"point_count": 0, "seed_supported_count": 0}
            for category in ("global", "vehicle", "background")
        }
        for band in BANDS
    }
    per_frame = []
    cache_records = []
    intensity_audit = {band: {"min": None, "max": None, "violation_count": 0} for band in BANDS}
    for frame_number, frame_id in enumerate(frame_ids, start=1):
        raw_points = load_kitti_point_cloud(resolve_kitti_paths(root / data_root, frame_id)[0])
        frame_sets = load_frame_universes(source_indices, frame_id)
        validate_frame_monotonicity(frame_sets, frame_id)
        seed_indices = frame_sets["T0"]["global"]
        _validate_raw_indices(seed_indices, len(raw_points), frame_id)
        if len(seed_indices) and np.any(raw_points[seed_indices, 3] < 0.38):
            raise V155SeedSupportError(f"{frame_id} T0 seed identity contains intensity below 0.38")
        seed_xy = raw_points[seed_indices, :2]
        frame_record = {"frame_id": frame_id, "seed_count": int(len(seed_indices)), "bands": {}}
        for category in UNIVERSES:
            validate_band_partition(frame_sets, category=category, frame_id=frame_id)

        for band_name, contract in BANDS.items():
            band_indices = {
                category: build_band_identity(
                    frame_sets[contract["included"]][category],
                    frame_sets[contract["excluded"]][category],
                )
                for category in UNIVERSES
            }
            if not set(band_indices["vehicle"]).issubset(band_indices["global"]):
                raise V155SeedSupportError(f"{frame_id} {band_name} vehicle identity is not a global subset")
            if not set(band_indices["background"]).issubset(band_indices["global"]):
                raise V155SeedSupportError(f"{frame_id} {band_name} background identity is not a global subset")
            if np.intersect1d(band_indices["vehicle"], band_indices["background"]).size:
                raise V155SeedSupportError(f"{frame_id} {band_name} vehicle/background identities overlap")
            _validate_raw_indices(band_indices["global"], len(raw_points), frame_id)
            intensities = raw_points[band_indices["global"], 3]
            update_intensity_audit(intensity_audit[band_name], intensities, contract)

            global_metrics = compute_seed_metrics(raw_points, band_indices["global"], seed_xy)
            frame_record["bands"][band_name] = {}
            for category in ("global", "vehicle", "background"):
                metrics = select_identity_metrics(
                    global_metrics,
                    band_indices["global"],
                    band_indices[category],
                )
                relative = f"point_metrics/{frame_id}_{band_name}_{category}.npz"
                cache_path = cache_root / relative
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, **metrics)
                count = int(len(metrics["source_point_indices"]))
                rescued = int(np.count_nonzero(metrics["d_seed"] <= R_SEED_M))
                record = {
                    "frame_id": frame_id,
                    "band": band_name,
                    "category": category,
                    "path": (Path(cache_dir) / relative).as_posix(),
                    "sha256": raw_file_sha256(cache_path),
                    "point_count": count,
                    "seed_supported_count": rescued,
                }
                cache_records.append(record)
                frame_record["bands"][band_name][category] = {
                    "point_count": count,
                    "seed_supported_count": rescued,
                }
                totals[band_name][category]["point_count"] += count
                totals[band_name][category]["seed_supported_count"] += rescued
        per_frame.append(frame_record)
        print(f"[Day1 {frame_number:03d}/{len(frame_ids):03d}] {frame_id}")

    for band_name, audit in intensity_audit.items():
        if audit["violation_count"]:
            raise V155SeedSupportError(f"{band_name} source identity violates its frozen intensity interval")
    output = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "read_only_phase0",
        "formal_comparison_commit": formal_commit,
        "source": {
            "closure": {"path": closure_path, "sha256": raw_file_sha256(root / closure_path)},
            "closure_evidence": {"path": evidence_path, "sha256": raw_file_sha256(root / evidence_path)},
            "formal_reports": source_artifacts,
            "source_run_ids": source_run_ids,
            "data_root": data_root,
        },
        "contracts": {
            "point_identity": POINT_IDENTITY,
            "coordinate_row_dedup_used": False,
            "band_identity_operator": "source_identity_set_difference",
            "bands": BANDS,
            "seed_identity": "T0.global_post_intensity",
            "seed_intensity_min": 0.38,
            "r_seed_m": R_SEED_M,
            "r_seed_search_allowed": False,
            "range_segments": {
                "near": "0 <= range_xy < 15",
                "mid": "15 <= range_xy < 30",
                "far": "range_xy >= 30",
            },
            "gt_oracle_leakage": False,
            "support_computed_before_category_selection": True,
        },
        "identity_validation": {
            "status": "PASS",
            "num_frames": len(frame_ids),
            "variant_frame_identity_equal": True,
            "source_point_monotonicity": "PASS",
            "band_partition_disjoint": True,
            "band_partition_union": "T_off_minus_T0",
            "intensity_interval_audit": intensity_audit,
        },
        "band_totals": totals,
        "per_frame_counts": per_frame,
        "point_metric_cache": {
            "format": "numpy_npz",
            "fields": ["source_point_indices", "range_xy", "d_seed", "n_seed_0p6"],
            "records": cache_records,
        },
        "day1_complete": True,
        "near_mid_far_analysis_complete": False,
        "delta_22_representation_complete": False,
        "formal_pipeline_rerun": False,
        "formal_results_modified": False,
    }
    atomic_write_json(root / output_path, output)
    return output


def materialize_variant_identity_cache(*, report_path, cache_dir, variant):
    report = load_json(report_path)
    frames = report.get("source_point_universes", [])
    if int(report.get("summary", {}).get("num_frames", -1)) != 100 or len(frames) != 100:
        raise V155SeedSupportError(f"{variant} formal report is not the frozen 100-frame run")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "variant": variant,
        "source_run_id": report["source"]["source_run_id"],
        "data_root": report["source"]["data_root"],
        "frames": {},
    }
    for frame in frames:
        frame_id = str(frame["frame_id"]).zfill(6)
        if frame_id in index["frames"]:
            raise V155SeedSupportError(f"duplicate frame identity in {variant}: {frame_id}")
        index["frames"][frame_id] = {}
        for category, source_name in UNIVERSES.items():
            values = np.asarray(frame["universes"][source_name]["source_point_indices"], dtype=np.int64)
            if len(values) != len(np.unique(values)):
                raise V155SeedSupportError(f"duplicate source point identity: {variant} {frame_id} {category}")
            values = np.sort(values)
            path = cache_dir / f"{frame_id}_{category}.npy"
            np.save(path, values, allow_pickle=False)
            index["frames"][frame_id][category] = path.as_posix()
    del report
    gc.collect()
    return index


def validate_variant_frame_identity(indices):
    frames = set(indices["T0"]["frames"])
    if len(frames) != 100 or any(set(indices[name]["frames"]) != frames for name in VARIANTS):
        raise V155SeedSupportError("formal variant frame identities differ")
    return sorted(frames)


def load_frame_universes(indices, frame_id):
    return {
        variant: {
            category: np.load(indices[variant]["frames"][frame_id][category], allow_pickle=False)
            for category in UNIVERSES
        }
        for variant in VARIANTS
    }


def validate_frame_monotonicity(frame_sets, frame_id):
    for left, right in zip(VARIANTS, VARIANTS[1:]):
        for category in UNIVERSES:
            missing = np.setdiff1d(frame_sets[left][category], frame_sets[right][category], assume_unique=True)
            if len(missing):
                raise V155SeedSupportError(
                    f"source identity monotonicity failed: {frame_id} {left}->{right} {category}"
                )


def build_band_identity(included_indices, excluded_indices):
    return np.setdiff1d(included_indices, excluded_indices, assume_unique=True)


def validate_band_partition(frame_sets, *, category, frame_id="unknown"):
    bands = [
        build_band_identity(
            frame_sets[contract["included"]][category],
            frame_sets[contract["excluded"]][category],
        )
        for contract in BANDS.values()
    ]
    if any(np.intersect1d(bands[left], bands[right]).size for left in range(3) for right in range(left + 1, 3)):
        raise V155SeedSupportError(f"{frame_id} {category} band identities overlap")
    combined = np.sort(np.concatenate(bands)) if any(len(values) for values in bands) else np.asarray([], dtype=np.int64)
    expected = np.setdiff1d(frame_sets["T_off"][category], frame_sets["T0"][category], assume_unique=True)
    if not np.array_equal(combined, expected):
        raise V155SeedSupportError(f"{frame_id} {category} bands do not partition T_off minus T0")
    return {name: values for name, values in zip(BANDS, bands)}


def compute_seed_metrics(raw_points, query_indices, seed_xy):
    query_indices = np.asarray(query_indices, dtype=np.int64)
    query_xy = np.asarray(raw_points[query_indices, :2], dtype=np.float64)
    range_xy = np.linalg.norm(query_xy, axis=1)
    if not len(query_indices):
        d_seed = np.asarray([], dtype=np.float64)
        n_seed = np.asarray([], dtype=np.int32)
    elif not len(seed_xy):
        d_seed = np.full(len(query_indices), np.inf, dtype=np.float64)
        n_seed = np.zeros(len(query_indices), dtype=np.int32)
    else:
        tree = cKDTree(np.asarray(seed_xy, dtype=np.float64))
        d_seed = np.asarray(tree.query(query_xy, k=1, workers=1)[0], dtype=np.float64)
        n_seed = np.asarray(
            tree.query_ball_point(query_xy, r=R_SEED_M, return_length=True, workers=1),
            dtype=np.int32,
        )
    return {
        "source_point_indices": query_indices,
        "range_xy": range_xy,
        "d_seed": d_seed,
        "n_seed_0p6": n_seed,
    }


def select_identity_metrics(global_metrics, global_indices, selected_indices):
    global_indices = np.asarray(global_indices, dtype=np.int64)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if not len(selected_indices):
        positions = np.asarray([], dtype=np.int64)
    else:
        positions = np.searchsorted(global_indices, selected_indices)
        if np.any(positions >= len(global_indices)) or not np.array_equal(global_indices[positions], selected_indices):
            raise V155SeedSupportError("category point identity is not a global-band subset")
    return {name: np.asarray(values)[positions] for name, values in global_metrics.items()}


def update_intensity_audit(audit, intensities, contract):
    raw_values = np.asarray(intensities)
    if not len(raw_values):
        return
    lower = np.asarray(contract["intensity_min"], dtype=raw_values.dtype).item()
    upper = np.asarray(contract["intensity_max_exclusive"], dtype=raw_values.dtype).item()
    values = raw_values.astype(np.float64, copy=False)
    current_min, current_max = float(np.min(values)), float(np.max(values))
    audit["min"] = current_min if audit["min"] is None else min(audit["min"], current_min)
    audit["max"] = current_max if audit["max"] is None else max(audit["max"], current_max)
    violations = (raw_values < lower) | (raw_values >= upper)
    audit["violation_count"] += int(np.count_nonzero(violations))


def range_segment(range_xy):
    value = float(range_xy)
    if value < 0:
        raise V155SeedSupportError("range_xy cannot be negative")
    if value < 15.0:
        return "near"
    if value < 30.0:
        return "mid"
    return "far"


def _validate_closed_source(root, closure, evidence, evidence_path):
    if closure.get("status") != "CLOSED" or closure.get("archive_status") != "COMPLETE":
        raise V155SeedSupportError("15.4 product archive is not COMPLETE")
    if closure.get("product_review", {}).get("status") != "APPROVE_WITH_MINOR_CHANGES":
        raise V155SeedSupportError("15.4 product sign-off is missing")
    if closure.get("formal_results_modified") is not False or closure.get("formal_100_rerun") is not False:
        raise V155SeedSupportError("15.4 closure does not preserve formal results")
    if evidence.get("status") != "CLOSED":
        raise V155SeedSupportError("15.4 closure evidence is not closed")
    expected = closure["canonical_artifacts"]["closure_evidence"]["sha256"]
    if raw_file_sha256(root / evidence_path) != expected:
        raise V155SeedSupportError("15.4 closure evidence identity mismatch")


def _validate_raw_indices(indices, point_count, frame_id):
    values = np.asarray(indices, dtype=np.int64)
    if len(values) and (int(values[0]) < 0 or int(values[-1]) >= point_count):
        raise V155SeedSupportError(f"raw LiDAR point index out of range: {frame_id}")
