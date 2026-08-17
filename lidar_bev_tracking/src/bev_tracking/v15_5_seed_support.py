import gc
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bev_tracking.kitti import load_kitti_point_cloud, resolve_kitti_paths
from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_materialization import load_json, raw_file_sha256


SCHEMA_VERSION = "15.5-phase0-seed-support-day1-v1"
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
