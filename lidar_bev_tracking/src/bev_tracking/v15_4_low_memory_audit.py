import gc
from pathlib import Path

import numpy as np

from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_audit import build_candidate_regression_audit, build_tp_regression_audit, classify_regression_reasons
from bev_tracking.v15_4_materialization import load_json


UNIVERSE_MAPPING = {
    "global": "global_post_intensity",
    "positive_gt": "positive_gt_post_intensity",
    "annotation_excluded_background": "annotation_excluded_background_post_intensity",
}


def materialize_compact_cache(report_path, cache_dir):
    report = load_json(report_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cache_dir / "gt_records.json", {"gt_candidate_records": report["gt_candidate_records"]})
    index = {"source_run_id": report["source"]["source_run_id"], "frames": {}}
    for frame in report["source_point_universes"]:
        frame_id = str(frame["frame_id"]).zfill(6)
        index["frames"][frame_id] = {}
        for name, source_name in UNIVERSE_MAPPING.items():
            relative = f"{frame_id}_{name}.npy"
            values = np.asarray(frame["universes"][source_name]["source_point_indices"], dtype=np.int64)
            if len(values) != len(np.unique(values)):
                raise ValueError(f"duplicate point identity in {frame_id} {name}")
            np.save(cache_dir / relative, values, allow_pickle=False)
            index["frames"][frame_id][name] = relative
    atomic_write_json(cache_dir / "index.json", index)
    del report
    gc.collect()
    return index


def validate_compact_monotonicity(cache_dirs):
    variants = ("T0", "T1", "T2", "T_off")
    indices = {name: load_json(Path(cache_dirs[name]) / "index.json") for name in variants}
    comparisons = []
    for left_name, right_name in zip(variants, variants[1:]):
        left_index, right_index = indices[left_name], indices[right_name]
        if set(left_index["frames"]) != set(right_index["frames"]):
            raise ValueError("compact audit frame identities differ")
        for frame_id in sorted(left_index["frames"]):
            for universe in UNIVERSE_MAPPING:
                left = np.load(Path(cache_dirs[left_name]) / left_index["frames"][frame_id][universe], allow_pickle=False)
                right = np.load(Path(cache_dirs[right_name]) / right_index["frames"][frame_id][universe], allow_pickle=False)
                missing = np.setdiff1d(left, right, assume_unique=True)
                if len(missing):
                    raise ValueError(f"source-point monotonicity failed: {left_name}->{right_name} {frame_id} {universe} missing={int(missing[0])}")
                comparisons.append({"left": left_name, "right": right_name, "frame_id": frame_id, "universe": universe, "left_count": int(len(left)), "right_count": int(len(right)), "passed": True})
    return {"status": "PASS", "point_identity": "(frame_id, raw_lidar_point_index)", "coordinate_row_dedup_used": False, "comparisons": comparisons}


def build_low_memory_matrix_audit(report_paths, cache_root, formal_comparison_commit):
    cache_root = Path(cache_root)
    cache_dirs = {name: cache_root / name for name in ("T0", "T1", "T2", "T_off")}
    for name, path in report_paths.items():
        index = materialize_compact_cache(path, cache_dirs[name])
        expected = f"_{formal_comparison_commit[:12]}"
        if not index["source_run_id"].endswith(expected):
            raise ValueError(f"{name} report was not produced by formal comparison commit")
    monotonicity = validate_compact_monotonicity(cache_dirs)
    t0 = load_json(cache_dirs["T0"] / "gt_records.json")
    comparisons = {}
    for name in ("T1", "T2", "T_off"):
        variant = load_json(cache_dirs[name] / "gt_records.json")
        comparisons[name] = {
            "tp_regression": build_tp_regression_audit(t0, variant),
            "candidate_regression": build_candidate_regression_audit(t0, variant),
            "regression_reasons": {key: classify_regression_reasons(t0, variant, key) for key in ("0.50", "0.25")},
        }
        del variant
        gc.collect()
    return {"schema_version": "15.4-formal-matrix-identity-audit-v1", "formal_comparison_commit": formal_comparison_commit, "audit_mode": "low_memory_per_report_per_frame", "source_point_monotonicity": monotonicity, "comparisons_vs_T0": comparisons}
