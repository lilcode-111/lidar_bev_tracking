"""Read-only Phase-1.1 OOF representation-capacity failure analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bev_tracking.clustering_detector import split_obstacle_filter_stages_with_indices
from bev_tracking.fragment_learning_dataset import MODEL_FEATURE_FIELDS
from bev_tracking.fragment_learning_training import FRAGMENT_TYPE_ENCODING, _git_identity
from bev_tracking.fragment_phase0 import build_candidate_fragments, best_endpoint_relation
from bev_tracking.gesr_v1 import (
    CANDIDATE_INTENSITY_MIN,
    NUMERICAL_DTYPE,
    SEED_INTENSITY_MIN,
    build_seed_components_optimized,
)
from bev_tracking.kitti import load_kitti_point_cloud, resolve_kitti_paths


ANALYSIS_SCHEMA_VERSION = "learned-fragment-relation-phase1-1-v1"
SCORE_THRESHOLD = 0.50
GOOD_FOLDS = {1, 5}
BAD_FOLDS = {2, 3}
NEUTRAL_FOLDS = {4}
TOP_MATCH_COUNT = 12
SMD_MODERATE = 0.50
SMD_LARGE = 0.80


class FragmentLearningFailureAnalysisError(ValueError):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _feature_matrix(rows):
    matrix = np.empty((len(rows), len(MODEL_FEATURE_FIELDS)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, field in enumerate(MODEL_FEATURE_FIELDS):
            if field == "fragment_type":
                try:
                    value = FRAGMENT_TYPE_ENCODING[row[field]]
                except KeyError as exc:
                    raise FragmentLearningFailureAnalysisError(
                        f"unknown fragment_type {row[field]}"
                    ) from exc
            else:
                value = float(row[field])
            matrix[row_index, column_index] = value
    if not np.isfinite(matrix).all():
        raise FragmentLearningFailureAnalysisError("non-finite frozen feature input")
    return matrix


def _quartiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"N": 0, "P25": None, "P50": None, "P75": None}
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return {"N": int(len(values)), "P25": float(p25), "P50": float(p50), "P75": float(p75)}


def _feature_distributions(matrix, masks):
    output = {}
    for group, mask in masks.items():
        output[group] = {
            field: _quartiles(matrix[mask, index])
            for index, field in enumerate(MODEL_FEATURE_FIELDS)
        }
        type_values = matrix[mask, 0]
        output[group]["fragment_type_counts"] = {
            "SINGLETON": int(np.sum(type_values == FRAGMENT_TYPE_ENCODING["SINGLETON"])),
            "STRUCTURED": int(np.sum(type_values == FRAGMENT_TYPE_ENCODING["STRUCTURED"])),
        }
    return output


def _smd(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0)
    difference = abs(float(a.mean()) - float(b.mean()))
    if pooled == 0.0:
        return 0.0 if difference == 0.0 else None
    return float(difference / pooled)


def _shift_summary(matrix, left_mask, right_mask):
    by_feature = {}
    for index, field in enumerate(MODEL_FEATURE_FIELDS):
        by_feature[field] = _smd(matrix[left_mask, index], matrix[right_mask, index])
    valid = {field: value for field, value in by_feature.items() if value is not None}
    ordered = sorted(valid.items(), key=lambda item: (-item[1], item[0]))
    return {
        "method": "absolute_standardized_mean_difference_with_pooled_SD",
        "left_N": int(left_mask.sum()),
        "right_N": int(right_mask.sum()),
        "moderate_or_larger_feature_count": sum(value >= SMD_MODERATE for value in valid.values()),
        "large_feature_count": sum(value >= SMD_LARGE for value in valid.values()),
        "maximum_SMD": None if not ordered else float(ordered[0][1]),
        "top_features": [
            {"feature": field, "absolute_SMD": float(value)}
            for field, value in ordered[:8]
        ],
        "by_feature": by_feature,
    }


def _frame_error_concentration(records):
    by_frame = defaultdict(lambda: {"FP": 0, "FP_score_mass": 0.0, "FN": 0, "P": 0, "TP": 0})
    for row in records:
        frame = row["frame_id"]
        label = row["label"]
        score = row["score"]
        positive = label == "POSITIVE"
        predicted = score >= SCORE_THRESHOLD
        by_frame[frame]["P"] += int(positive)
        by_frame[frame]["TP"] += int(positive and predicted)
        if not positive and predicted:
            by_frame[frame]["FP"] += 1
            by_frame[frame]["FP_score_mass"] += score
        if positive and not predicted:
            by_frame[frame]["FN"] += 1
    rows = [{"frame_id": frame, **values} for frame, values in sorted(by_frame.items())]
    total_fp = sum(row["FP"] for row in rows)
    total_mass = sum(row["FP_score_mass"] for row in rows)
    ranked = sorted(rows, key=lambda row: (-row["FP"], -row["FP_score_mass"], row["frame_id"]))
    concentration = {}
    for count in (1, 3, 5, 10):
        selected = ranked[:count]
        concentration[f"top_{count}"] = {
            "FP": sum(row["FP"] for row in selected),
            "FP_ratio": 0.0 if total_fp == 0 else sum(row["FP"] for row in selected) / total_fp,
            "FP_score_mass": sum(row["FP_score_mass"] for row in selected),
            "FP_score_mass_ratio": 0.0 if total_mass == 0 else sum(row["FP_score_mass"] for row in selected) / total_mass,
            "frames": [row["frame_id"] for row in selected],
        }
    positive_frames = [row for row in rows if row["P"] > 0]
    zero_recovery = [row for row in positive_frames if row["TP"] == 0]
    return {
        "definition": {"FP": "label != POSITIVE and M1_score >= 0.50", "FP_score_mass": "sum(M1_score for FP in frame)", "FN": "POSITIVE and M1_score < 0.50"},
        "total_FP": total_fp,
        "total_FP_score_mass": float(total_mass),
        "total_FN": sum(row["FN"] for row in rows),
        "positive_support_frame_count": len(positive_frames),
        "recovered_positive_frame_count": sum(row["TP"] >= 1 for row in positive_frames),
        "zero_recovery_frame_count": len(zero_recovery),
        "zero_recovery_frames": [row["frame_id"] for row in zero_recovery],
        "top_frame_concentration": concentration,
        "per_frame": rows,
    }


def _nearest_positive_pairs(matrix, records, n0_fp_mask):
    positive_indices = np.flatnonzero(np.asarray([row["label"] == "POSITIVE" for row in records]))
    fp_indices = np.flatnonzero(n0_fp_mask)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (matrix - means) / scales
    pairs = []
    for fp_index in fp_indices:
        distances = np.linalg.norm(standardized[positive_indices] - standardized[fp_index], axis=1)
        position = int(np.argmin(distances))
        positive_index = int(positive_indices[position])
        absolute_delta = np.abs(standardized[positive_index] - standardized[fp_index])
        largest = np.argsort(-absolute_delta, kind="stable")[:5]
        pairs.append(
            {
                "N0_FP_sample_row": int(records[fp_index]["sample_row"]),
                "N0_FP_frame_id": records[fp_index]["frame_id"],
                "N0_FP_fragment_identity": int(records[fp_index]["fragment_identity"]),
                "N0_FP_score": float(records[fp_index]["score"]),
                "positive_sample_row": int(records[positive_index]["sample_row"]),
                "positive_frame_id": records[positive_index]["frame_id"],
                "positive_fragment_identity": int(records[positive_index]["fragment_identity"]),
                "positive_score": float(records[positive_index]["score"]),
                "standardized_euclidean_distance": float(distances[position]),
                "largest_standardized_feature_deltas": [
                    {"feature": MODEL_FEATURE_FIELDS[int(index)], "absolute_delta": float(absolute_delta[index])}
                    for index in largest
                ],
            }
        )
    return sorted(pairs, key=lambda row: (row["standardized_euclidean_distance"], row["N0_FP_sample_row"]))


def _rebuild_frame_fragments(data_root, frame_id):
    velodyne_path, _ = resolve_kitti_paths(data_root, frame_id)
    raw = load_kitti_point_cloud(velodyne_path)
    stages, indices = split_obstacle_filter_stages_with_indices(
        raw, z_min=-0.9, intensity_min=SEED_INTENSITY_MIN
    )
    z_points = stages["z_filter"]
    z_indices = indices["z_filter"]
    intensity = np.asarray(z_points[:, 3], dtype=NUMERICAL_DTYPE)
    candidate = (intensity >= CANDIDATE_INTENSITY_MIN) & (intensity < SEED_INTENSITY_MIN)
    seed = intensity >= SEED_INTENSITY_MIN
    fragments = build_candidate_fragments(z_points[candidate], z_indices[candidate])
    components = build_seed_components_optimized(z_points[seed], z_indices[seed])
    valid_components = [component for component in components if component.geometry.valid]
    return raw, {int(item["runtime_id"]): item for item in fragments}, {
        int(component.runtime_id): component for component in valid_components
    }, valid_components


def _raw_fragment_summary(raw, fragment, valid_components, component_by_id):
    points = np.asarray(raw[fragment["source_indices"]], dtype=np.float64)
    relation = best_endpoint_relation(fragment, valid_components)
    seed_summary = None
    if relation.get("has_computable_seed_relation"):
        component = component_by_id[int(relation["seed_component_runtime_id"])]
        geometry = component.geometry
        seed_summary = {
            "runtime_id": int(component.runtime_id),
            "point_count": int(len(component.source_indices)),
            "center_xy": [float(value) for value in geometry.center],
            "major_span": float(geometry.u_max - geometry.u_min),
            "minor_span": float(geometry.v_max - geometry.v_min),
            "linearity": None if geometry.lambda1 == 0.0 else float(1.0 - geometry.lambda2 / geometry.lambda1),
        }
    xyz = points[:, :3]
    nearest = []
    if len(points) >= 2:
        nearest = cKDTree(points[:, :3]).query(points[:, :3], k=2)[0][:, 1]
    return {
        "point_count": int(len(points)),
        "source_index_min": int(np.min(fragment["source_indices"])),
        "source_index_max": int(np.max(fragment["source_indices"])),
        "xyz_min": [float(value) for value in xyz.min(axis=0)],
        "xyz_max": [float(value) for value in xyz.max(axis=0)],
        "z_quartiles": _quartiles(points[:, 2]),
        "nearest_neighbor_3d_quartiles": _quartiles(nearest),
        "relation": relation,
        "seed_component": seed_summary,
    }


def _attach_raw_geometry(pairs, data_root):
    selected = []
    seen_n0_frames = set()
    for pair in pairs:
        if pair["N0_FP_frame_id"] in seen_n0_frames:
            continue
        selected.append(pair)
        seen_n0_frames.add(pair["N0_FP_frame_id"])
        if len(selected) == TOP_MATCH_COUNT:
            break
    if len(selected) < TOP_MATCH_COUNT:
        selected_identities = {pair["N0_FP_sample_row"] for pair in selected}
        selected.extend(
            pair for pair in pairs
            if pair["N0_FP_sample_row"] not in selected_identities
        )
        selected = selected[:TOP_MATCH_COUNT]
    needed = defaultdict(set)
    for pair in selected:
        needed[pair["N0_FP_frame_id"]].add(pair["N0_FP_fragment_identity"])
        needed[pair["positive_frame_id"]].add(pair["positive_fragment_identity"])
    rebuilt = {}
    for frame_id in sorted(needed):
        raw, fragments, component_by_id, valid_components = _rebuild_frame_fragments(data_root, frame_id)
        missing = sorted(needed[frame_id] - set(fragments))
        if missing:
            raise FragmentLearningFailureAnalysisError(
                f"raw fragment identity replay failed in {frame_id}: {missing}"
            )
        rebuilt[frame_id] = (raw, fragments, component_by_id, valid_components)
    point_rows = []
    enriched = []
    for rank, pair in enumerate(selected, start=1):
        item = dict(pair)
        for role, frame_key, identity_key in (
            ("N0_FP", "N0_FP_frame_id", "N0_FP_fragment_identity"),
            ("POSITIVE", "positive_frame_id", "positive_fragment_identity"),
        ):
            frame_id = pair[frame_key]
            identity = pair[identity_key]
            raw, fragments, component_by_id, valid_components = rebuilt[frame_id]
            fragment = fragments[identity]
            item[f"{role}_raw_geometry"] = _raw_fragment_summary(
                raw, fragment, valid_components, component_by_id
            )
            center = np.asarray(fragment["center"], dtype=np.float64)
            major = fragment["major_axis"]
            minor = None if major is None else np.asarray([-major[1], major[0]])
            for source_index in fragment["source_indices"]:
                point = np.asarray(raw[int(source_index)], dtype=np.float64)
                delta_xy = point[:2] - center
                point_rows.append(
                    {
                        "pair_rank": rank, "role": role, "frame_id": frame_id,
                        "canonical_fragment_identity": identity,
                        "raw_source_index": int(source_index),
                        "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                        "intensity": float(point[3]),
                        "pca_u": 0.0 if major is None else float(delta_xy @ major),
                        "pca_v": 0.0 if minor is None else float(delta_xy @ minor),
                    }
                )
        enriched.append(item)
    return enriched, point_rows


def _decision_from_smd(summary):
    count = summary["moderate_or_larger_feature_count"]
    if count >= 3:
        return "YES"
    if count == 0:
        return "NO"
    return "INCONCLUSIVE"


def run_phase1_1_failure_analysis(output_dir, data_root):
    output_dir = Path(output_dir)
    result = json.loads((output_dir / "phase1_feasibility_result.json").read_text(encoding="utf-8"))
    if result.get("LEARNED_FRAGMENT_RELATION") != "NOT_SUPPORTED_BY_CURRENT_REPRESENTATION":
        raise FragmentLearningFailureAnalysisError("unexpected frozen Phase-1 conclusion")
    oof = _read_csv(output_dir / "phase1_oof_predictions.csv")
    x_rows = _read_csv(output_dir / "X_model.csv")
    split = json.loads(
        (output_dir / "fragment_learning_splits.json").read_text(encoding="utf-8")
    )
    if len(oof) != 3252 or len(x_rows) != len(oof):
        raise FragmentLearningFailureAnalysisError("frozen OOF/X_model count mismatch")
    oof = sorted(oof, key=lambda row: int(row["sample_row"]))
    if [int(row["sample_row"]) for row in oof] != list(range(len(oof))):
        raise FragmentLearningFailureAnalysisError("OOF sample identity is not contiguous")
    assignments = sorted(
        split.get("sample_assignments", []), key=lambda row: int(row["sample_row"])
    )
    if len(assignments) != len(oof):
        raise FragmentLearningFailureAnalysisError("frozen split/OOF count mismatch")
    for prediction, assignment in zip(oof, assignments):
        expected = (
            int(assignment["sample_row"]), str(assignment["frame_id"]).zfill(6),
            int(assignment["canonical_fragment_identity"]), assignment["label"],
            int(assignment["validation_fold"]),
        )
        actual = (
            int(prediction["sample_row"]), str(prediction["frame_id"]).zfill(6),
            int(prediction["canonical_fragment_identity"]), prediction["label"],
            int(prediction["validation_fold"]),
        )
        if actual != expected:
            raise FragmentLearningFailureAnalysisError(
                f"frozen split/OOF identity mismatch at sample {actual[0]}"
            )
    matrix = _feature_matrix(x_rows)
    records = [
        {
            "sample_row": int(row["sample_row"]),
            "frame_id": str(row["frame_id"]).zfill(6),
            "fragment_identity": int(row["canonical_fragment_identity"]),
            "label": row["label"],
            "fold": int(row["validation_fold"]),
            "score": float(row["M1_score"]),
        }
        for row in oof
    ]
    labels = np.asarray([row["label"] for row in records], dtype=object)
    folds = np.asarray([row["fold"] for row in records], dtype=np.int64)
    scores = np.asarray([row["score"] for row in records], dtype=np.float64)
    p_high = (labels == "POSITIVE") & (scores >= SCORE_THRESHOLD)
    p_miss = (labels == "POSITIVE") & (scores < SCORE_THRESHOLD)
    n0_fp = (labels == "N0") & (scores >= SCORE_THRESHOLD)
    n0_tn = (labels == "N0") & (scores < SCORE_THRESHOLD)
    masks = {"P_HIGH": p_high, "P_MISS": p_miss, "N0_FP": n0_fp}

    fp_clustering = _shift_summary(matrix, n0_fp, n0_tn)
    fold_shift = {}
    for label in ("POSITIVE", "N0"):
        good = np.isin(folds, list(GOOD_FOLDS)) & (labels == label)
        bad = np.isin(folds, list(BAD_FOLDS)) & (labels == label)
        neutral = np.isin(folds, list(NEUTRAL_FOLDS)) & (labels == label)
        fold_shift[label] = {
            "Good": _feature_distributions(matrix, {"values": good})["values"],
            "Bad": _feature_distributions(matrix, {"values": bad})["values"],
            "Neutral": _feature_distributions(matrix, {"values": neutral})["values"],
            "Good_vs_Bad_shift": _shift_summary(matrix, good, bad),
        }
    shift_counts = [fold_shift[label]["Good_vs_Bad_shift"]["moderate_or_larger_feature_count"] for label in ("POSITIVE", "N0")]
    if max(shift_counts) >= 3:
        shift_decision = "PRESENT"
    elif max(shift_counts) == 0:
        shift_decision = "NOT_OBVIOUS"
    else:
        shift_decision = "INCONCLUSIVE"

    pairs = _nearest_positive_pairs(matrix, records, n0_fp)
    representative_pairs, raw_point_rows = _attach_raw_geometry(pairs, data_root)
    pair_summary = {
        "matching_method": "nearest POSITIVE in full frozen 24-feature z-scored Euclidean space; many-to-one allowed",
        "matched_pair_count": len(pairs),
        "N0_FP_unique_frame_count": len({row["N0_FP_frame_id"] for row in pairs}),
        "matched_positive_support_frame_count": len({row["positive_frame_id"] for row in pairs}),
        "representative_pair_count": len(representative_pairs),
        "distance_quartiles": _quartiles([row["standardized_euclidean_distance"] for row in pairs]),
    }
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_execution_identity": result["execution_identity"],
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "read_only_inputs": ["phase1_feasibility_result.json", "phase1_oof_predictions.csv", "X_model.csv", "fragment_learning_splits.json", "selected_input_cache"],
        "definitions": {"P_HIGH": "POSITIVE and M1_score >= 0.50", "P_MISS": "POSITIVE and M1_score < 0.50", "N0_FP": "N0 and M1_score >= 0.50", "Good_folds": [1, 5], "Bad_folds": [2, 3], "Neutral_folds": [4]},
        "group_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "scene_concentration": _frame_error_concentration(records),
        "error_feature_distributions": _feature_distributions(matrix, masks),
        "N0_FP_vs_N0_TN_shift": fp_clustering,
        "FP_CLUSTERING_BY_EXISTING_FEATURE": _decision_from_smd(fp_clustering),
        "fold_distribution_shift": fold_shift,
        "BAD_FOLD_DISTRIBUTION_SHIFT": shift_decision,
        "matched_pairs_summary": pair_summary,
        "representative_matched_pairs": representative_pairs,
        "structural_review_status": "REQUIRES_READ_ONLY_REVIEW_OF_REPRESENTATIVE_MATCHED_PAIRS",
        "pending_final_questions": {
            "RAW_POINT_PATTERN_NECESSITY": "SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE",
            "REPRESENTATION_CAPACITY_DIAGNOSIS": "EXPLICIT_FEATURE_SET_INCOMPLETE / HANDCRAFTED_FRAGMENT_REPRESENTATION_NEAR_LIMIT / INCONCLUSIVE",
        },
        "MODEL_RETRAINED": False,
        "DATASET_REGENERATED": False,
        "SPLIT_REGENERATED": False,
        "PREDICTIONS_REGENERATED": False,
        "FIXED_100_EXECUTED": False,
    }
    result_path = output_dir / "phase1_1_failure_analysis.json"
    pair_path = output_dir / "phase1_1_matched_pairs.csv"
    point_path = output_dir / "phase1_1_raw_pair_points.csv"
    result_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("rank", "N0_FP_sample_row", "N0_FP_frame_id", "N0_FP_fragment_identity", "N0_FP_score", "positive_sample_row", "positive_frame_id", "positive_fragment_identity", "positive_score", "standardized_euclidean_distance")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, pair in enumerate(pairs, start=1):
            writer.writerow({"rank": rank, **{field: pair[field] for field in fields[1:]}})
    with point_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("pair_rank", "role", "frame_id", "canonical_fragment_identity", "raw_source_index", "x", "y", "z", "intensity", "pca_u", "pca_v")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_point_rows)
    return analysis, result_path, pair_path, point_path
