"""Read-only Phase-1.2 multi-seed competitive-context mechanism analysis."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from bev_tracking.fragment_learning_failure_analysis import _rebuild_frame_fragments
from bev_tracking.fragment_learning_training import _git_identity
from bev_tracking.fragment_phase0 import AXIS_ORDER, SIDE_ORDER, best_endpoint_relation


SCHEMA_VERSION = "learned-fragment-relation-phase1-2-v1"
SCORE_THRESHOLD = 0.50
GOOD_FOLDS = {1, 5}
BAD_FOLDS = {2, 3}
NEUTRAL_FOLDS = {4}
MODERATE_SMD = 0.50
MIN_DIRECTION_FRAMES = 3

GROUP_ORDER = ("P_HIGH", "P_MISS", "N0_FP", "N0_TN")
MULTIPLICITY_FIELDS = (
    "frame_valid_seed_component_count",
    "valid_seed_relation_count",
    "valid_seed_relation_fraction",
    "has_second_best_relation",
)
RELATION_FIELDS = (
    "best_endpoint_gap", "best_lateral_offset", "best_orientation_difference",
    "best_forward_projection", "second_endpoint_gap", "second_lateral_offset",
    "second_orientation_difference", "second_forward_projection",
)
CONTRAST_FIELDS = (
    "second_minus_best_endpoint_gap",
    "second_minus_best_lateral_offset",
    "second_minus_best_orientation_difference",
)
GEOMETRY_FIELDS = (
    "best_seed_point_count", "best_seed_major_span", "best_seed_minor_span",
    "best_seed_linearity", "second_seed_point_count", "second_seed_major_span",
    "second_seed_minor_span", "second_seed_linearity",
    "second_minus_best_seed_point_count", "second_minus_best_seed_major_span",
    "second_minus_best_seed_minor_span", "second_minus_best_seed_linearity",
)
DIAGNOSTIC_FIELDS = MULTIPLICITY_FIELDS + RELATION_FIELDS + CONTRAST_FIELDS + GEOMETRY_FIELDS
FAMILIES = {
    "seed_relation_multiplicity": (
        "valid_seed_relation_count", "valid_seed_relation_fraction",
        "has_second_best_relation",
    ),
    "best_vs_second_best_competition": CONTRAST_FIELDS,
    "seed_geometry_context": GEOMETRY_FIELDS,
}


class MultiSeedContextAnalysisError(ValueError):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relation_key(relation):
    orientation_valid = bool(relation["orientation_valid"])
    orientation = relation["orientation_difference"]
    return (
        float(relation["endpoint_gap"]),
        float(relation["lateral_offset"]),
        0 if orientation_valid else 1,
        float(orientation) if orientation is not None else float("inf"),
        int(relation["seed_component_runtime_id"]),
        AXIS_ORDER.index(relation["best_continuation_axis"]),
        SIDE_ORDER.index(relation["seed_outward_side"]),
        SIDE_ORDER.index(relation["fragment_endpoint_side"]),
    )


def _seed_geometry(component):
    geometry = component.geometry
    return {
        "seed_component_runtime_id": int(component.runtime_id),
        "seed_point_count": int(len(component.source_indices)),
        "seed_major_span": float(geometry.u_max - geometry.u_min),
        "seed_minor_span": float(geometry.v_max - geometry.v_min),
        "seed_linearity": (
            None
            if geometry.lambda1 == 0.0
            else float(1.0 - geometry.lambda2 / geometry.lambda1)
        ),
    }


def enumerate_distinct_seed_relations(fragment, valid_components):
    """Return one frozen best relation per distinct valid seed component."""
    per_component = []
    for component in valid_components:
        relation = best_endpoint_relation(fragment, [component])
        if not relation.get("has_computable_seed_relation"):
            continue
        relation = {
            key: value for key, value in relation.items()
            if key != "has_computable_seed_relation"
        }
        if int(relation["seed_component_runtime_id"]) != int(component.runtime_id):
            raise MultiSeedContextAnalysisError("relation/component identity mismatch")
        per_component.append({**relation, **_seed_geometry(component)})
    return sorted(per_component, key=_relation_key)


def _difference(second, best, field):
    left = second.get(field)
    right = best.get(field)
    if left is None or right is None:
        return None
    return float(left - right)


def build_multiseed_record(fragment, valid_components):
    relations = enumerate_distinct_seed_relations(fragment, valid_components)
    best = relations[0] if relations else None
    second = relations[1] if len(relations) >= 2 else None
    total = len(valid_components)
    record = {
        "fragment_type": fragment["fragment_type"],
        "frame_valid_seed_component_count": int(total),
        "valid_seed_relation_count": int(len(relations)),
        "valid_seed_relation_fraction": 0.0 if total == 0 else float(len(relations) / total),
        "has_best_relation": int(best is not None),
        "has_second_best_relation": int(second is not None),
        "best_relation": best,
        "second_best_relation": second,
    }
    relation_map = (
        ("best", best), ("second", second),
    )
    for prefix, relation in relation_map:
        record[f"{prefix}_endpoint_gap"] = None if relation is None else float(relation["endpoint_gap"])
        record[f"{prefix}_lateral_offset"] = None if relation is None else float(relation["lateral_offset"])
        record[f"{prefix}_orientation_difference"] = None if relation is None else relation["orientation_difference"]
        record[f"{prefix}_forward_projection"] = None if relation is None else float(relation["forward_projection"])
        for field in ("seed_point_count", "seed_major_span", "seed_minor_span", "seed_linearity"):
            record[f"{prefix}_{field}"] = None if relation is None else relation[field]
    record["second_minus_best_endpoint_gap"] = None if second is None else _difference(second, best, "endpoint_gap")
    record["second_minus_best_lateral_offset"] = None if second is None else _difference(second, best, "lateral_offset")
    record["second_minus_best_orientation_difference"] = None if second is None else _difference(second, best, "orientation_difference")
    for field in ("seed_point_count", "seed_major_span", "seed_minor_span", "seed_linearity"):
        record[f"second_minus_best_{field}"] = None if second is None else _difference(second, best, field)
    return record


def _distribution(values):
    valid = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if len(valid) == 0:
        return {"valid_N": 0, "min": None, "P25": None, "P50": None, "P75": None, "max": None}
    p25, p50, p75 = np.percentile(valid, [25, 50, 75])
    return {
        "valid_N": int(len(valid)), "min": float(valid.min()), "P25": float(p25),
        "P50": float(p50), "P75": float(p75), "max": float(valid.max()),
    }


def _group_distributions(records, *, singleton_only=False):
    output = {}
    for group in GROUP_ORDER:
        selected = [
            row for row in records
            if row["group"] == group and (not singleton_only or row["fragment_type"] == "SINGLETON")
        ]
        output[group] = {
            "sample_count": len(selected),
            "support_frame_count": len({row["frame_id"] for row in selected}),
            "distributions": {
                field: _distribution([row[field] for row in selected])
                for field in DIAGNOSTIC_FIELDS
            },
        }
    return output


def _smd_and_direction(left, right):
    left = np.asarray([value for value in left if value is not None and np.isfinite(value)], dtype=np.float64)
    right = np.asarray([value for value in right if value is not None and np.isfinite(value)], dtype=np.float64)
    if len(left) < 2 or len(right) < 2:
        return {"left_N": len(left), "right_N": len(right), "signed_SMD": None, "absolute_SMD": None, "direction": None}
    pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
    difference = float(left.mean() - right.mean())
    if pooled == 0.0:
        smd = 0.0 if difference == 0.0 else None
    else:
        smd = difference / pooled
    return {
        "left_N": len(left), "right_N": len(right),
        "signed_SMD": None if smd is None else float(smd),
        "absolute_SMD": None if smd is None else float(abs(smd)),
        "direction": None if smd is None or smd == 0.0 else ("P_HIGH_GREATER" if smd > 0 else "P_HIGH_LOWER"),
    }


def _comparison(records, field, folds=None):
    selected = records if folds is None else [row for row in records if row["fold"] in folds]
    return _smd_and_direction(
        [row[field] for row in selected if row["group"] == "P_HIGH"],
        [row[field] for row in selected if row["group"] == "N0_FP"],
    )


def _frame_direction_consistency(records, field, overall_direction):
    frames = sorted({row["frame_id"] for row in records if row["group"] == "P_HIGH"})
    rows = []
    for frame_id in frames:
        positive = [row[field] for row in records if row["frame_id"] == frame_id and row["group"] == "P_HIGH" and row[field] is not None]
        negative = [row[field] for row in records if row["frame_id"] == frame_id and row["group"] == "N0_FP" and row[field] is not None]
        if not positive or not negative:
            continue
        difference = float(np.median(positive) - np.median(negative))
        direction = None if difference == 0.0 else ("P_HIGH_GREATER" if difference > 0 else "P_HIGH_LOWER")
        rows.append({"frame_id": frame_id, "P_HIGH_N": len(positive), "N0_FP_N": len(negative), "median_difference": difference, "direction": direction, "matches_overall": direction == overall_direction})
    return {
        "co_support_frame_count": len(rows),
        "matching_direction_frame_count": sum(row["matches_overall"] for row in rows),
        "rows": rows,
    }


def evaluate_context_signal(records):
    """Apply the predeclared descriptive cross-frame support rule."""
    fields = {}
    for field in DIAGNOSTIC_FIELDS:
        overall = _comparison(records, field)
        good = _comparison(records, field, GOOD_FOLDS)
        bad = _comparison(records, field, BAD_FOLDS)
        frames = _frame_direction_consistency(records, field, overall["direction"])
        same_superfold_direction = (
            overall["direction"] is not None
            and good["direction"] == overall["direction"]
            and bad["direction"] == overall["direction"]
        )
        supported = (
            overall["absolute_SMD"] is not None
            and overall["absolute_SMD"] >= MODERATE_SMD
            and same_superfold_direction
            and frames["matching_direction_frame_count"] >= MIN_DIRECTION_FRAMES
        )
        fields[field] = {
            "overall": overall, "Good_Fold1_5": good, "Bad_Fold2_3": bad,
            "Neutral_Fold4": _comparison(records, field, NEUTRAL_FOLDS),
            "cross_frame": frames, "same_direction_in_Good_and_Bad": same_superfold_direction,
            "supported": bool(supported),
        }
    family_results = {}
    for family, family_fields in FAMILIES.items():
        supported_fields = [field for field in family_fields if fields[field]["supported"]]
        family_results[family] = {
            "supported": bool(supported_fields),
            "supported_fields": supported_fields,
            "strongest_fields": sorted(
                (
                    {"field": field, "absolute_SMD": fields[field]["overall"]["absolute_SMD"]}
                    for field in family_fields
                    if fields[field]["overall"]["absolute_SMD"] is not None
                ),
                key=lambda item: (-item["absolute_SMD"], item["field"]),
            )[:5],
        }
    if any(item["supported"] for item in family_results.values()):
        conclusion = "SUPPORTED"
    else:
        has_moderate_overall = any(
            item["overall"]["absolute_SMD"] is not None
            and item["overall"]["absolute_SMD"] >= MODERATE_SMD
            for item in fields.values()
        )
        conclusion = "INCONCLUSIVE" if has_moderate_overall else "NOT_SUPPORTED"
    return {
        "rule": {
            "overall_absolute_SMD_min": MODERATE_SMD,
            "same_direction_in_Good_Fold1_5_and_Bad_Fold2_3": True,
            "matching_direction_frame_count_min": MIN_DIRECTION_FRAMES,
            "singleton_evidence_is_supplementary_only": True,
        },
        "fields": fields,
        "families": family_results,
        "MULTI_SEED_CONTEXT_SIGNAL": conclusion,
    }


def _per_frame_summary(records):
    output = []
    frames = sorted({row["frame_id"] for row in records})
    key_fields = (
        "valid_seed_relation_count", "valid_seed_relation_fraction",
        "best_endpoint_gap", "second_endpoint_gap",
        "second_minus_best_endpoint_gap", "best_seed_point_count",
        "second_seed_point_count",
    )
    for frame_id in frames:
        item = {"frame_id": frame_id, "groups": {}}
        for group in GROUP_ORDER:
            selected = [row for row in records if row["frame_id"] == frame_id and row["group"] == group]
            item["groups"][group] = {
                "sample_count": len(selected),
                "distributions": {field: _distribution([row[field] for row in selected]) for field in key_fields},
            }
        output.append(item)
    return output


def _representative_cases(records, signal):
    candidates = []
    for field, item in signal["fields"].items():
        value = item["overall"]["absolute_SMD"]
        if value is not None:
            candidates.append((value, field))
    top_field = max(candidates)[1] if candidates else "valid_seed_relation_count"
    output = {"selection_field": top_field}
    for group in ("P_HIGH", "N0_FP", "P_MISS"):
        selected = [row for row in records if row["group"] == group and row[top_field] is not None]
        selected.sort(key=lambda row: (row[top_field], row["frame_id"], row["fragment_identity"]))
        if not selected:
            output[group] = []
            continue
        positions = sorted({0, len(selected) // 2, len(selected) - 1})
        output[group] = [
            {
                "frame_id": selected[index]["frame_id"],
                "fragment_identity": selected[index]["fragment_identity"],
                "fold": selected[index]["fold"],
                "M1_score": selected[index]["M1_score"],
                "fragment_type": selected[index]["fragment_type"],
                top_field: selected[index][top_field],
                "best_relation": selected[index]["best_relation"],
                "second_best_relation": selected[index]["second_best_relation"],
            }
            for index in positions
        ]
    return output


def _load_and_validate_oof(output_dir):
    oof = sorted(_read_csv(output_dir / "phase1_oof_predictions.csv"), key=lambda row: int(row["sample_row"]))
    split = json.loads((output_dir / "fragment_learning_splits.json").read_text(encoding="utf-8"))
    assignments = sorted(split.get("sample_assignments", []), key=lambda row: int(row["sample_row"]))
    if len(oof) != 3252 or len(assignments) != len(oof):
        raise MultiSeedContextAnalysisError("frozen OOF/split count mismatch")
    records = []
    for prediction, assignment in zip(oof, assignments):
        identity = (
            int(prediction["sample_row"]), str(prediction["frame_id"]).zfill(6),
            int(prediction["canonical_fragment_identity"]), prediction["label"],
            int(prediction["validation_fold"]),
        )
        expected = (
            int(assignment["sample_row"]), str(assignment["frame_id"]).zfill(6),
            int(assignment["canonical_fragment_identity"]), assignment["label"],
            int(assignment["validation_fold"]),
        )
        if identity != expected:
            raise MultiSeedContextAnalysisError(f"OOF/split identity mismatch at {identity[0]}")
        score = float(prediction["M1_score"])
        label = prediction["label"]
        group = None
        if label == "POSITIVE":
            group = "P_HIGH" if score >= SCORE_THRESHOLD else "P_MISS"
        elif label == "N0":
            group = "N0_FP" if score >= SCORE_THRESHOLD else "N0_TN"
        if group is not None:
            records.append({
                "sample_row": identity[0], "frame_id": identity[1],
                "fragment_identity": identity[2], "label": label, "fold": identity[4],
                "M1_score": score, "group": group,
            })
    counts = {group: sum(row["group"] == group for row in records) for group in GROUP_ORDER}
    if counts != {"P_HIGH": 29, "P_MISS": 27, "N0_FP": 268, "N0_TN": 2807}:
        raise MultiSeedContextAnalysisError(f"frozen analysis group mismatch: {counts}")
    return records


def run_multiseed_context_analysis(output_dir, data_root, *, progress_callback=None):
    output_dir = Path(output_dir)
    records = _load_and_validate_oof(output_dir)
    by_frame = defaultdict(list)
    for record in records:
        by_frame[record["frame_id"]].append(record)
    enriched = []
    for index, frame_id in enumerate(sorted(by_frame), start=1):
        if progress_callback:
            progress_callback(index, len(by_frame), frame_id)
        _, fragments, _, valid_components = _rebuild_frame_fragments(data_root, frame_id)
        for record in by_frame[frame_id]:
            fragment = fragments.get(record["fragment_identity"])
            if fragment is None:
                raise MultiSeedContextAnalysisError(
                    f"fragment identity replay failed: {frame_id}/{record['fragment_identity']}"
                )
            enriched.append({**record, **build_multiseed_record(fragment, valid_components)})
    enriched.sort(key=lambda row: row["sample_row"])
    distributions = _group_distributions(enriched)
    singleton = _group_distributions(enriched, singleton_only=True)
    signal = evaluate_context_signal(enriched)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "definitions": {
            "valid_seed_relation_count": "number of distinct valid seed components with at least one frozen computable outward relation; no distance cutoff",
            "best_second_ranking": "one best frozen relation per distinct seed component, then frozen lexicographic relation ordering",
            "contrast": "raw second-minus-best fields; no weighted score",
            "Good_folds": [1, 5], "Bad_folds": [2, 3], "Neutral_fold": [4],
        },
        "group_counts": {group: sum(row["group"] == group for row in enriched) for group in GROUP_ORDER},
        "group_distributions": distributions,
        "singleton_results": singleton,
        "cross_frame_signal": signal,
        "per_frame_results": _per_frame_summary(enriched),
        "representative_cases": _representative_cases(enriched, signal),
        "MULTI_SEED_CONTEXT_SIGNAL": signal["MULTI_SEED_CONTEXT_SIGNAL"],
        "supported_signal_families": [name for name, item in signal["families"].items() if item["supported"]],
        "MODEL_RETRAINED": False,
        "DATASET_REGENERATED": False,
        "SPLIT_REGENERATED": False,
        "PREDICTIONS_REGENERATED": False,
        "FORMAL_100_EXECUTED": False,
    }
    result_path = output_dir / "phase1_2_multiseed_context_analysis.json"
    record_path = output_dir / "phase1_2_multiseed_records.csv"
    evidence_path = output_dir / "phase1_2_relation_evidence.jsonl"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    flat_fields = (
        "sample_row", "frame_id", "fragment_identity", "label", "fold", "M1_score",
        "group", "fragment_type", *DIAGNOSTIC_FIELDS,
    )
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in enriched:
            writer.writerow({field: row.get(field) for field in flat_fields})
    with evidence_path.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps({
                "sample_row": row["sample_row"], "frame_id": row["frame_id"],
                "fragment_identity": row["fragment_identity"], "group": row["group"],
                "fragment_type": row["fragment_type"],
                "best_relation": row["best_relation"],
                "second_best_relation": row["second_best_relation"],
            }, separators=(",", ":")) + "\n")
    return result, result_path, record_path, evidence_path
