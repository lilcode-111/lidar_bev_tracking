"""Phase-1.3 read-only neighboring-fragment context mechanism analysis."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from bev_tracking.fragment_learning_failure_analysis import _rebuild_frame_fragments
from bev_tracking.fragment_learning_training import _git_identity
from bev_tracking.fragment_phase0 import best_endpoint_relation


SCHEMA_VERSION = "fragment-learning-phase1-3-neighbor-context-v1"
PHASE1_4_SCHEMA_VERSION = "fragment-learning-phase1-4-local-multi-fragment-v1"
MODERATE_SMD = 0.50
SMALL_SMD = 0.30
MIN_FOLD_DIRECTION_MATCHES = 4

FIELDS = (
    "nearest_neighbor_distance",
    "nearest_neighbor_outward_projection",
    "nearest_neighbor_lateral_projection",
    "nearest_neighbor_absolute_lateral_projection",
    "nearest_neighbor_outward_alignment_cosine",
)
PRIMARY_FIELDS = FIELDS[1:]
ISOLATION_FIELDS = (FIELDS[0],)


class NeighborContextAnalysisError(ValueError):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _distribution(values):
    valid = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if len(valid) == 0:
        return {"valid_N": 0, "min": None, "P25": None, "P50": None, "P75": None, "max": None}
    p25, p50, p75 = np.percentile(valid, [25, 50, 75])
    return {
        "valid_N": int(len(valid)), "min": float(valid.min()),
        "P25": float(p25), "P50": float(p50), "P75": float(p75),
        "max": float(valid.max()),
    }


def _effect(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(right) < 2:
        return {
            "left_N": len(left), "right_N": len(right), "signed_SMD": None,
            "absolute_SMD": None, "median_difference": None,
            "ROC_AUC_higher_value_is_POSITIVE": None, "direction": None,
        }
    pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
    difference = float(left.mean() - right.mean())
    smd = None if pooled == 0.0 and difference != 0.0 else (0.0 if pooled == 0.0 else difference / pooled)
    labels = np.r_[np.ones(len(left), dtype=np.int64), np.zeros(len(right), dtype=np.int64)]
    scores = np.r_[left, right]
    return {
        "left_N": len(left), "right_N": len(right),
        "signed_SMD": None if smd is None else float(smd),
        "absolute_SMD": None if smd is None else float(abs(smd)),
        "median_difference": float(np.median(left) - np.median(right)),
        "ROC_AUC_higher_value_is_POSITIVE": float(roc_auc_score(labels, scores)),
        "direction": None if difference == 0.0 else ("POSITIVE_GREATER" if difference > 0.0 else "POSITIVE_LOWER"),
    }


def ranked_fragment_relations(
    raw, target, fragments, component, seed_relation, *, neighbor_count
):
    """Rank distinct fragments by frozen contact distance and express them in seed PCA."""
    target_points = np.asarray(raw[target["source_indices"], :2], dtype=np.float64)
    ranked = []
    for neighbor_id in sorted(fragments):
        if neighbor_id == int(target["runtime_id"]):
            continue
        neighbor = fragments[neighbor_id]
        neighbor_points = np.asarray(raw[neighbor["source_indices"], :2], dtype=np.float64)
        distances, positions = cKDTree(neighbor_points).query(target_points, k=1)
        target_position = int(np.argmin(distances))
        neighbor_position = int(positions[target_position])
        key = (
            float(distances[target_position]), int(neighbor_id),
            int(target["source_indices"][target_position]),
            int(neighbor["source_indices"][neighbor_position]),
        )
        ranked.append((key, neighbor, target_position, neighbor_position))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise NeighborContextAnalysisError("target frame has no distinct neighboring fragment")
    geometry = component.geometry
    major = np.asarray(geometry.major, dtype=np.float64)
    minor = np.asarray(geometry.minor, dtype=np.float64)
    if seed_relation["best_continuation_axis"] == "SEED_MAJOR":
        axis, perpendicular = major, minor
    else:
        axis, perpendicular = minor, -major
    outward = -axis if seed_relation["seed_outward_side"] == "NEG" else axis
    output = []
    for rank, (_, neighbor, target_position, neighbor_position) in enumerate(
        ranked[:neighbor_count], start=1
    ):
        target_contact = target_points[target_position]
        neighbor_contact = np.asarray(
            raw[neighbor["source_indices"][neighbor_position], :2], dtype=np.float64
        )
        displacement = neighbor_contact - target_contact
        distance = float(np.linalg.norm(displacement))
        outward_projection = float(displacement @ outward)
        lateral_projection = float(displacement @ perpendicular)
        output.append({
            "neighbor_rank": rank,
            "neighbor_fragment_identity": int(neighbor["runtime_id"]),
            "neighbor_fragment_type": neighbor["fragment_type"],
            "neighbor_point_count": int(neighbor["point_count"]),
            "target_contact_source_index": int(target["source_indices"][target_position]),
            "neighbor_contact_source_index": int(neighbor["source_indices"][neighbor_position]),
            "neighbor_distance": distance,
            "neighbor_outward_projection": outward_projection,
            "neighbor_lateral_projection": lateral_projection,
            "neighbor_absolute_lateral_projection": abs(lateral_projection),
            "neighbor_outward_alignment_cosine": (
                0.0 if distance == 0.0 else float(outward_projection / distance)
            ),
        })
    return output


def nearest_fragment_relation(raw, target, fragments, component, seed_relation):
    """Backward-compatible Phase-1.3 nearest-neighbor relation."""
    item = ranked_fragment_relations(
        raw, target, fragments, component, seed_relation, neighbor_count=1
    )[0]
    return {
        "nearest_neighbor_fragment_identity": item["neighbor_fragment_identity"],
        "nearest_neighbor_fragment_type": item["neighbor_fragment_type"],
        "nearest_neighbor_point_count": item["neighbor_point_count"],
        "target_contact_source_index": item["target_contact_source_index"],
        "neighbor_contact_source_index": item["neighbor_contact_source_index"],
        "nearest_neighbor_distance": item["neighbor_distance"],
        "nearest_neighbor_outward_projection": item["neighbor_outward_projection"],
        "nearest_neighbor_lateral_projection": item["neighbor_lateral_projection"],
        "nearest_neighbor_absolute_lateral_projection": item["neighbor_absolute_lateral_projection"],
        "nearest_neighbor_outward_alignment_cosine": item["neighbor_outward_alignment_cosine"],
    }


def _load_frozen_targets(output_dir):
    output_dir = Path(output_dir)
    dataset_identities = defaultdict(set)
    labels = {}
    with (output_dir / "fragment_dataset.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = row["metadata_fields"]
            frame = str(metadata["frame_id"]).zfill(6)
            identity = int(metadata["canonical_fragment_identity"])
            dataset_identities[frame].add(identity)
            label = row["label_field"]["label"]
            if label in {"POSITIVE", "N1", "N0"}:
                labels[(frame, identity)] = label
    oof = {
        (str(row["frame_id"]).zfill(6), int(row["canonical_fragment_identity"])): row
        for row in _read_csv(output_dir / "lightgbm_v2_oof_predictions.csv")
    }
    margins = {
        (str(row["frame_id"]).zfill(6), int(row["canonical_fragment_identity"])): row
        for row in _read_csv(output_dir / "lightgbm_v2_margin_feature.csv")
    }
    if len(oof) != 3252 or len(margins) != 3252:
        raise NeighborContextAnalysisError("frozen OOF/margin count changed")
    targets = []
    for identity, label in labels.items():
        if label not in {"POSITIVE", "N1", "N0"}:
            continue
        prediction = oof.get(identity)
        margin = margins.get(identity)
        if prediction is None or margin is None or prediction["label"] != label:
            raise NeighborContextAnalysisError(f"frozen identity mismatch: {identity}")
        if label == "N0":
            target_group = "N0"
        elif label == "N1":
            target_group = "N1"
        else:
            target_group = "P_HIGH" if float(prediction["M1_score"]) >= 0.50 else "P_MISS"
        targets.append({
            "frame_id": identity[0], "fragment_identity": identity[1],
            "label": label, "group": target_group,
            "fold": int(prediction["validation_fold"]),
            "M1_score": float(prediction["M1_score"]),
            "M2_score": float(prediction["M2_score"]),
            "frozen_best_seed_component_runtime_id": int(margin["best_seed_component_runtime_id"]),
        })
    targets.sort(key=lambda row: (row["frame_id"], row["fragment_identity"]))
    counts = {label: sum(row["label"] == label for row in targets) for label in ("POSITIVE", "N1", "N0")}
    if counts != {"POSITIVE": 56, "N1": 121, "N0": 3075}:
        raise NeighborContextAnalysisError(f"frozen target counts changed: {counts}")
    return dataset_identities, targets


def replay_neighbor_records(output_dir, data_root, *, progress_callback=None):
    dataset_identities, targets = _load_frozen_targets(output_dir)
    by_frame = defaultdict(list)
    for target in targets:
        by_frame[target["frame_id"]].append(target)
    if len(by_frame) != 64 or set(by_frame) != set(dataset_identities):
        raise NeighborContextAnalysisError(
            "REPLAY_IDENTITY_MISMATCH: frozen manifest frame identity changed"
        )
    records = []
    for position, frame_id in enumerate(sorted(by_frame), start=1):
        if progress_callback:
            progress_callback(position, len(by_frame), frame_id)
        raw, fragments, component_by_id, valid_components = _rebuild_frame_fragments(data_root, frame_id)
        replay_ids = set(fragments)
        if replay_ids != dataset_identities[frame_id]:
            missing = sorted(dataset_identities[frame_id] - replay_ids)[:10]
            extra = sorted(replay_ids - dataset_identities[frame_id])[:10]
            raise NeighborContextAnalysisError(
                f"REPLAY_IDENTITY_MISMATCH: {frame_id}: missing={missing}, extra={extra}"
            )
        for target in by_frame[frame_id]:
            fragment = fragments[target["fragment_identity"]]
            relation = best_endpoint_relation(fragment, valid_components)
            if not relation.get("has_computable_seed_relation"):
                raise NeighborContextAnalysisError(
                    f"frozen best-seed relation missing: {frame_id}/{fragment['runtime_id']}"
                )
            seed_id = int(relation["seed_component_runtime_id"])
            if seed_id != target["frozen_best_seed_component_runtime_id"]:
                raise NeighborContextAnalysisError(
                    f"REPLAY_IDENTITY_MISMATCH: best seed changed: {frame_id}/{fragment['runtime_id']}"
                )
            spatial = nearest_fragment_relation(
                raw, fragment, fragments, component_by_id[seed_id], relation
            )
            records.append({
                **target, "fragment_type": fragment["fragment_type"],
                "target_point_count": int(fragment["point_count"]),
                "best_seed_component_runtime_id": seed_id,
                "best_continuation_axis": relation["best_continuation_axis"],
                "seed_outward_side": relation["seed_outward_side"],
                **spatial,
            })
    return records


def _group_distributions(records, groups, *, singleton_only=False):
    output = {}
    for group in groups:
        selected = [
            row for row in records
            if row["group"] == group and (not singleton_only or row["fragment_type"] == "SINGLETON")
        ]
        output[group] = {
            "sample_count": len(selected),
            "support_frame_count": len({row["frame_id"] for row in selected}),
            "distributions": {
                field: _distribution([row[field] for row in selected]) for field in FIELDS
            },
        }
    return output


def _field_effect(records, field, folds=None, singleton_only=False):
    selected = [
        row for row in records
        if row["label"] in {"POSITIVE", "N1"}
        and (folds is None or row["fold"] in folds)
        and (not singleton_only or row["fragment_type"] == "SINGLETON")
    ]
    return _effect(
        [row[field] for row in selected if row["label"] == "POSITIVE"],
        [row[field] for row in selected if row["label"] == "N1"],
    )


def _per_frame_direction(records, field, overall_direction):
    rows = []
    for frame_id in sorted({row["frame_id"] for row in records}):
        positive = [row[field] for row in records if row["frame_id"] == frame_id and row["label"] == "POSITIVE"]
        n1 = [row[field] for row in records if row["frame_id"] == frame_id and row["label"] == "N1"]
        if not positive or not n1:
            continue
        difference = float(np.median(positive) - np.median(n1))
        direction = None if difference == 0.0 else ("POSITIVE_GREATER" if difference > 0.0 else "POSITIVE_LOWER")
        rows.append({
            "frame_id": frame_id, "P": len(positive), "N1": len(n1),
            "median_difference": difference, "direction": direction,
            "matches_overall": direction == overall_direction,
        })
    return {
        "co_support_frame_count": len(rows),
        "matching_direction_frame_count": sum(row["matches_overall"] for row in rows),
        "rows": rows,
    }


def evaluate_neighbor_signal(records):
    fields = {}
    for field in FIELDS:
        overall = _field_effect(records, field)
        folds = {fold: _field_effect(records, field, {fold}) for fold in range(1, 6)}
        direction = overall["direction"]
        fold_matches = sum(item["direction"] == direction for item in folds.values())
        frame = _per_frame_direction(records, field, direction)
        supported = (
            overall["absolute_SMD"] is not None
            and overall["absolute_SMD"] >= MODERATE_SMD
            and fold_matches >= MIN_FOLD_DIRECTION_MATCHES
            and folds[1]["direction"] == direction
            and folds[5]["direction"] == direction
            and frame["matching_direction_frame_count"] >= 3
        )
        fields[field] = {
            "overall": overall, "folds": folds,
            "fold_direction_match_count": fold_matches,
            "Fold1_matches_overall": folds[1]["direction"] == direction,
            "Fold5_matches_overall": folds[5]["direction"] == direction,
            "cross_frame": frame, "supported": bool(supported),
        }

    def family_result(family_fields):
        supported_fields = [field for field in family_fields if fields[field]["supported"]]
        if supported_fields:
            status = "SUPPORTED"
        else:
            maximum = max(
                (fields[field]["overall"]["absolute_SMD"] or 0.0) for field in family_fields
            )
            status = "NOT_SUPPORTED" if maximum < SMALL_SMD else "INCONCLUSIVE"
        return {
            "status": status, "supported_fields": supported_fields,
            "strongest_fields": sorted(
                ({"field": field, "absolute_SMD": fields[field]["overall"]["absolute_SMD"]} for field in family_fields),
                key=lambda item: (-(item["absolute_SMD"] or -1.0), item["field"]),
            ),
        }

    primary = family_result(PRIMARY_FIELDS)
    isolation = family_result(ISOLATION_FIELDS)
    if primary["status"] == "SUPPORTED":
        neighbor = "SUPPORTED"
    elif primary["status"] == "NOT_SUPPORTED":
        neighbor = isolation["status"]
    else:
        neighbor = "INCONCLUSIVE"
    diagnosis = {
        "SUPPORTED": "NEIGHBOR_CONTEXT_SUPPORTED",
        "NOT_SUPPORTED": "NEIGHBOR_CONTEXT_NOT_SUPPORTED",
        "INCONCLUSIVE": "INCONCLUSIVE",
    }[neighbor]
    return {
        "predeclared_rule": {
            "overall_absolute_SMD_min": MODERATE_SMD,
            "matching_fold_direction_min": MIN_FOLD_DIRECTION_MATCHES,
            "Fold1_and_Fold5_must_match": True,
            "matching_co_support_frame_min": 3,
            "NOT_SUPPORTED_if_all_absolute_SMD_below": SMALL_SMD,
        },
        "fields": fields,
        "LOCAL_FRAGMENT_SUPPORT_COMPLEMENTARITY": primary["status"],
        "LOCAL_FRAGMENT_ISOLATION_CONTEXT": isolation["status"],
        "NEIGHBORING_FRAGMENT_CONTEXT_SIGNAL": neighbor,
        "REPRESENTATION_DIAGNOSIS": diagnosis,
        "families": {"support_complementarity": primary, "isolation": isolation},
    }


def _frame_concentration(records):
    target = [row for row in records if row["label"] in {"POSITIVE", "N1"}]
    counts = defaultdict(lambda: {"POSITIVE": 0, "N1": 0})
    for row in target:
        counts[row["frame_id"]][row["label"]] += 1
    ranked = sorted(
        ({"frame_id": frame, **values, "total": sum(values.values())} for frame, values in counts.items()),
        key=lambda row: (-row["total"], row["frame_id"]),
    )
    total = len(target)
    return {
        "per_frame": sorted(ranked, key=lambda row: row["frame_id"]),
        "top_frame_target_share": {
            f"top_{count}": {
                "count": sum(row["total"] for row in ranked[:count]),
                "ratio": float(sum(row["total"] for row in ranked[:count]) / total),
                "frames": [row["frame_id"] for row in ranked[:count]],
            }
            for count in (1, 3, 5)
        },
    }


def _representative_cases(records, signal):
    primary = list(PRIMARY_FIELDS)
    strongest = max(
        primary,
        key=lambda field: signal["fields"][field]["overall"]["absolute_SMD"] or -1.0,
    )
    output = {"selection_field": strongest}
    for label in ("POSITIVE", "N1"):
        selected = sorted(
            (row for row in records if row["label"] == label),
            key=lambda row: (row[strongest], row["frame_id"], row["fragment_identity"]),
        )
        positions = sorted({0, len(selected) // 2, len(selected) - 1})
        output[label] = [
            {key: selected[index][key] for key in (
                "frame_id", "fragment_identity", "fold", "fragment_type",
                "nearest_neighbor_fragment_identity", *FIELDS,
            )}
            for index in positions
        ]
    return output


def analyze_neighbor_context(output_dir, data_root, *, progress_callback=None):
    records = replay_neighbor_records(
        output_dir, data_root, progress_callback=progress_callback
    )
    signal = evaluate_neighbor_signal(records)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "execution_semantics": {
            "fragment_graph": "frozen deterministic replay with exact dataset identity check",
            "neighbor_distance": "minimum member-point XY distance",
            "spatial_reference": "target frozen best-seed PCA frame",
            "multiplicity_enabled": False,
            "neighbor_radius": None,
        },
        "counts": {
            "POSITIVE": sum(row["label"] == "POSITIVE" for row in records),
            "N1": sum(row["label"] == "N1" for row in records),
            "N0": sum(row["label"] == "N0" for row in records),
        },
        "group_distributions": _group_distributions(records, ("P_HIGH", "P_MISS", "N1", "N0")),
        "singleton_distributions": _group_distributions(records, ("P_HIGH", "P_MISS", "N1", "N0"), singleton_only=True),
        "signal_evaluation": signal,
        "frame_concentration": _frame_concentration(records),
        "representative_cases": _representative_cases(records, signal),
        "NEIGHBORING_FRAGMENT_CONTEXT_SIGNAL": signal["NEIGHBORING_FRAGMENT_CONTEXT_SIGNAL"],
        "LOCAL_FRAGMENT_SUPPORT_COMPLEMENTARITY": signal["LOCAL_FRAGMENT_SUPPORT_COMPLEMENTARITY"],
        "LOCAL_FRAGMENT_ISOLATION_CONTEXT": signal["LOCAL_FRAGMENT_ISOLATION_CONTEXT"],
        "REPRESENTATION_DIAGNOSIS": signal["REPRESENTATION_DIAGNOSIS"],
        "MODEL_RETRAINED": False,
        "FORMAL_FEATURE_ADDED": False,
        "FRAGMENT_GRAPH_CHANGED": False,
        "FORMAL_100_EXECUTED": False,
    }
    output_dir = Path(output_dir)
    result_path = output_dir / "phase1_3_neighbor_context_analysis.json"
    record_path = output_dir / "phase1_3_neighbor_context_records.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = (
        "frame_id", "fragment_identity", "label", "group", "fold",
        "fragment_type", "target_point_count", "best_seed_component_runtime_id",
        "best_continuation_axis", "seed_outward_side",
        "nearest_neighbor_fragment_identity", "nearest_neighbor_fragment_type",
        "nearest_neighbor_point_count", "target_contact_source_index",
        "neighbor_contact_source_index", *FIELDS,
    )
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row[field] for field in fields})
    return result, result_path, record_path


def replay_multi_fragment_records(output_dir, data_root, *, progress_callback=None):
    """Replay the frozen graph and retain exactly the two nearest fragments for P/N1."""
    dataset_identities, targets = _load_frozen_targets(output_dir)
    targets = [row for row in targets if row["label"] in {"POSITIVE", "N1"}]
    by_frame = defaultdict(list)
    for target in targets:
        by_frame[target["frame_id"]].append(target)
    if set(by_frame) - set(dataset_identities):
        raise NeighborContextAnalysisError(
            "REPLAY_IDENTITY_MISMATCH: target frame is absent from frozen dataset"
        )

    records = []
    all_frames = sorted(dataset_identities)
    for position, frame_id in enumerate(all_frames, start=1):
        if progress_callback:
            progress_callback(position, len(all_frames), frame_id)
        raw, fragments, component_by_id, valid_components = _rebuild_frame_fragments(
            data_root, frame_id
        )
        replay_ids = set(fragments)
        if replay_ids != dataset_identities[frame_id]:
            missing = sorted(dataset_identities[frame_id] - replay_ids)[:10]
            extra = sorted(replay_ids - dataset_identities[frame_id])[:10]
            raise NeighborContextAnalysisError(
                f"REPLAY_IDENTITY_MISMATCH: {frame_id}: missing={missing}, extra={extra}"
            )
        for target in by_frame.get(frame_id, ()):
            fragment = fragments[target["fragment_identity"]]
            relation = best_endpoint_relation(fragment, valid_components)
            if not relation.get("has_computable_seed_relation"):
                raise NeighborContextAnalysisError(
                    f"frozen best-seed relation missing: {frame_id}/{fragment['runtime_id']}"
                )
            seed_id = int(relation["seed_component_runtime_id"])
            if seed_id != target["frozen_best_seed_component_runtime_id"]:
                raise NeighborContextAnalysisError(
                    f"REPLAY_IDENTITY_MISMATCH: best seed changed: {frame_id}/{fragment['runtime_id']}"
                )
            neighbors = ranked_fragment_relations(
                raw, fragment, fragments, component_by_id[seed_id], relation,
                neighbor_count=2,
            )
            record = {
                **target,
                "fragment_type": fragment["fragment_type"],
                "target_point_count": int(fragment["point_count"]),
                "best_seed_component_runtime_id": seed_id,
                "best_continuation_axis": relation["best_continuation_axis"],
                "seed_outward_side": relation["seed_outward_side"],
                "neighbor_count_available": len(neighbors),
                "multi_neighbor_valid": len(neighbors) == 2,
            }
            for rank in (1, 2):
                item = neighbors[rank - 1] if len(neighbors) >= rank else None
                prefix = f"N{rank}_"
                for name in (
                    "neighbor_fragment_identity", "neighbor_fragment_type",
                    "neighbor_point_count", "target_contact_source_index",
                    "neighbor_contact_source_index", "neighbor_distance",
                    "neighbor_outward_projection", "neighbor_lateral_projection",
                    "neighbor_absolute_lateral_projection",
                    "neighbor_outward_alignment_cosine",
                ):
                    record[prefix + name] = None if item is None else item[name]
            record["L1"] = record["N1_neighbor_absolute_lateral_projection"]
            record["L2"] = record["N2_neighbor_absolute_lateral_projection"]
            record["multi_neighbor_lateral_envelope"] = (
                max(record["L1"], record["L2"])
                if record["multi_neighbor_valid"] else None
            )
            records.append(record)
    if len(records) != 177:
        raise NeighborContextAnalysisError(
            f"frozen POSITIVE/N1 target count changed: {len(records)}"
        )
    return records


def _phase1_4_subset(records, *, label=None, fold=None, singleton_only=False):
    return [
        row for row in records
        if (label is None or row["label"] == label)
        and (fold is None or row["fold"] == fold)
        and (not singleton_only or row["fragment_type"] == "SINGLETON")
    ]


def _phase1_4_stats(records, *, singleton_only=False):
    field = "multi_neighbor_lateral_envelope"
    groups = {}
    for label in ("POSITIVE", "N1"):
        selected = _phase1_4_subset(
            records, label=label, singleton_only=singleton_only
        )
        valid = [row[field] for row in selected if row[field] is not None]
        groups[label] = {
            "sample_N": len(selected),
            "valid_N": len(valid),
            "invalid_N": len(selected) - len(valid),
            "distribution": _distribution(valid),
        }
    effect = _effect(
        [row[field] for row in _phase1_4_subset(
            records, label="POSITIVE", singleton_only=singleton_only
        ) if row[field] is not None],
        [row[field] for row in _phase1_4_subset(
            records, label="N1", singleton_only=singleton_only
        ) if row[field] is not None],
    )
    return {"groups": groups, "effect": effect}


def analyze_local_multi_fragment_context(
    output_dir, data_root, *, progress_callback=None
):
    """Run the frozen K=2 Phase-1.4 composition analysis without model changes."""
    records = replay_multi_fragment_records(
        output_dir, data_root, progress_callback=progress_callback
    )
    overall = _phase1_4_stats(records)
    singleton = _phase1_4_stats(records, singleton_only=True)
    folds = {
        fold: _phase1_4_stats(
            [row for row in records if row["fold"] == fold]
        )
        for fold in range(1, 6)
    }
    direction = overall["effect"]["direction"]
    direction_matches = sum(
        item["effect"]["direction"] == direction for item in folds.values()
    )
    absolute_smd = overall["effect"]["absolute_SMD"]
    fold1_matches = folds[1]["effect"]["direction"] == direction
    fold5_matches = folds[5]["effect"]["direction"] == direction
    if (
        direction == "POSITIVE_LOWER"
        and absolute_smd is not None and absolute_smd >= MODERATE_SMD
        and direction_matches >= MIN_FOLD_DIRECTION_MATCHES
        and fold1_matches and fold5_matches
    ):
        status = "SUPPORTED"
    elif direction != "POSITIVE_LOWER" or (
        absolute_smd is not None and absolute_smd < SMALL_SMD
    ):
        status = "NOT_SUPPORTED"
    else:
        status = "INCONCLUSIVE"

    result = {
        "schema_version": PHASE1_4_SCHEMA_VERSION,
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "execution_semantics": {
            "fragment_graph": "frozen deterministic replay with exact dataset identity check",
            "neighbor_ordering": "minimum member-point XY distance, tie-break fragment_runtime_id",
            "NEIGHBOR_COUNT": 2,
            "spatial_reference": "target frozen best-seed PCA frame",
            "primary_quantity": "max(L1, L2)",
            "expected_direction": "POSITIVE_LOWER",
            "K_search_performed": False,
            "neighbor_radius": None,
        },
        "counts": {
            "POSITIVE": sum(row["label"] == "POSITIVE" for row in records),
            "N1": sum(row["label"] == "N1" for row in records),
        },
        "overall": overall,
        "folds": folds,
        "fold_direction_match_count": direction_matches,
        "Fold1_matches_overall": fold1_matches,
        "Fold5_matches_overall": fold5_matches,
        "singleton": singleton,
        "predecessor_comparison": {
            "phase1_3_quantity": "nearest_neighbor_absolute_lateral_projection",
            "absolute_SMD": 0.431271,
            "direction": "POSITIVE_LOWER",
            "phase1_4_absolute_SMD": absolute_smd,
            "absolute_SMD_change": (
                None if absolute_smd is None else float(absolute_smd - 0.431271)
            ),
        },
        "decision_rule": {
            "source": "reuse Phase-1.3 frozen effect/cross-fold consistency rule",
            "expected_direction": "POSITIVE_LOWER",
            "overall_absolute_SMD_min": MODERATE_SMD,
            "matching_fold_direction_min": MIN_FOLD_DIRECTION_MATCHES,
            "Fold1_and_Fold5_must_match": True,
            "NOT_SUPPORTED_if_wrong_direction_or_absolute_SMD_below": SMALL_SMD,
        },
        "LOCAL_MULTI_FRAGMENT_SUPPORT_COHERENCE": status,
        "NEIGHBORING_FRAGMENT_CONTEXT_STATUS": status,
        "MODEL_RETRAINED": False,
        "FORMAL_FEATURE_ADDED": False,
        "FRAGMENT_GRAPH_CHANGED": False,
        "FORMAL_100_EXECUTED": False,
    }
    output_dir = Path(output_dir)
    result_path = output_dir / "phase1_4_local_multi_fragment_analysis.json"
    record_path = output_dir / "phase1_4_local_multi_fragment_records.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = (
        "frame_id", "fragment_identity", "label", "group", "fold",
        "fragment_type", "target_point_count", "best_seed_component_runtime_id",
        "best_continuation_axis", "seed_outward_side", "neighbor_count_available",
        "multi_neighbor_valid", "N1_neighbor_fragment_identity",
        "N1_neighbor_fragment_type", "N1_neighbor_point_count",
        "N1_target_contact_source_index", "N1_neighbor_contact_source_index",
        "N1_neighbor_distance", "N1_neighbor_outward_projection",
        "N1_neighbor_lateral_projection",
        "N1_neighbor_absolute_lateral_projection",
        "N1_neighbor_outward_alignment_cosine", "N2_neighbor_fragment_identity",
        "N2_neighbor_fragment_type", "N2_neighbor_point_count",
        "N2_target_contact_source_index", "N2_neighbor_contact_source_index",
        "N2_neighbor_distance", "N2_neighbor_outward_projection",
        "N2_neighbor_lateral_projection",
        "N2_neighbor_absolute_lateral_projection",
        "N2_neighbor_outward_alignment_cosine", "L1", "L2",
        "multi_neighbor_lateral_envelope",
    )
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row[field] for field in fields})
    return result, result_path, record_path
