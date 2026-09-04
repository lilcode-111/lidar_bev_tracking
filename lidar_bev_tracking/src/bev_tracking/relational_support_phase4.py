"""Phase-4 read-only relational-learning support coverage statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from bev_tracking.fragment_learning_training import _git_identity
from bev_tracking.lfrr_v1 import load_compact_representation


SCHEMA_VERSION = "relational-learning-support-coverage-phase4-v1"
SUPERVISED_LABELS = ("POSITIVE", "N1", "N0")


class RelationalSupportError(ValueError):
    pass


def _key(frame_id, fragment_identity):
    return str(frame_id).zfill(6), int(fragment_identity)


def _percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"valid_N": 0, "P25": None, "P50": None, "P75": None}
    return {
        "valid_N": int(len(values)),
        "P25": float(np.percentile(values, 25)),
        "P50": float(np.median(values)),
        "P75": float(np.percentile(values, 75)),
    }


def _top_shares(counts):
    """Return concentration shares; the denominator is stated explicitly."""
    values = sorted((int(value) for value in counts if int(value) > 0), reverse=True)
    denominator = int(sum(values))
    output = {"support_count": denominator, "nonzero_unit_count": len(values)}
    for size in (1, 3, 5):
        output[f"Top{size}_share"] = (
            None if denominator == 0 else float(sum(values[:size]) / denominator)
        )
    return output


def _local_set_summary(targets, label):
    sizes = [len(row["neighbor_relations"]) for row in targets if row["label"] == label]
    return {
        "sample_count": len(sizes),
        "EMPTY": int(sum(value == 0 for value in sizes)),
        "ONE_MEMBER": int(sum(value == 1 for value in sizes)),
        "MULTI_MEMBER": int(sum(value >= 2 for value in sizes)),
        "local_set_size": _percentiles(sizes),
    }


def _composition_record(counter, target_count):
    total = int(sum(counter.values()))
    return {
        "target_count": int(target_count),
        "context_member_count": total,
        "labeled_P_count": int(counter["POSITIVE"]),
        "labeled_N1_count": int(counter["N1"]),
        "labeled_N0_count": int(counter["N0"]),
        "unlabeled_non_supervised_count": int(counter["UNLABELED_OTHER"]),
        "mean_context_members_per_target": (
            0.0 if target_count == 0 else float(total / target_count)
        ),
    }


def analyze_relational_support(output_dir):
    output_dir = Path(output_dir)
    catalog, targets = load_compact_representation(output_dir)
    targets = sorted(targets, key=lambda row: int(row["sample_row"]))
    if len(targets) != 3252:
        raise RelationalSupportError("frozen supervised target count changed")

    supervised = {}
    frame_fold = {}
    no_seed_targets = Counter()
    for expected_row, target in enumerate(targets):
        if int(target["sample_row"]) != expected_row:
            raise RelationalSupportError("sample_row identity is not frozen contiguous order")
        identity = _key(target["frame_id"], target["canonical_fragment_identity"])
        if identity in supervised:
            raise RelationalSupportError(f"duplicate supervised identity: {identity}")
        label = str(target["label"])
        if label not in SUPERVISED_LABELS:
            raise RelationalSupportError(f"unexpected supervised label: {label}")
        supervised[identity] = target
        frame, _ = identity
        fold = int(target["validation_fold"])
        previous = frame_fold.setdefault(frame, fold)
        if previous != fold:
            raise RelationalSupportError(f"frame crosses folds: {frame}")
        if target["best_seed_component_runtime_id"] is None:
            no_seed_targets[label] += 1
            if target["neighbor_relations"]:
                raise RelationalSupportError("no-seed target must have EMPTY local set")

    groups = defaultdict(lambda: {
        "member_identities": set(), "label_counts": Counter(),
        "unlabeled_identities": set(),
    })
    no_seed_catalog = 0
    for identity, row in catalog.items():
        normalized_identity = _key(*identity)
        if normalized_identity != identity:
            raise RelationalSupportError("catalog identity is not canonical")
        seed_id = row.get("best_seed_component_runtime_id")
        if seed_id is None:
            no_seed_catalog += 1
            continue
        group_key = (identity[0], int(seed_id))
        group = groups[group_key]
        group["member_identities"].add(identity)
        if identity in supervised:
            group["label_counts"][supervised[identity]["label"]] += 1
        else:
            group["unlabeled_identities"].add(identity)

    # Every seeded target must appear exactly once in its catalog-derived group.
    for identity, target in supervised.items():
        seed_id = target["best_seed_component_runtime_id"]
        if seed_id is None:
            continue
        group_key = (identity[0], int(seed_id))
        if identity not in groups[group_key]["member_identities"]:
            raise RelationalSupportError(f"target missing from relational group: {identity}")

    group_rows = []
    mixed_groups = set()
    for (frame, seed_id), group in sorted(groups.items()):
        counts = group["label_counts"]
        mixed = counts["POSITIVE"] > 0 and counts["N1"] > 0
        if mixed:
            mixed_groups.add((frame, seed_id))
        supervised_p_n1 = counts["POSITIVE"] + counts["N1"]
        group_rows.append({
            "frame_id": frame,
            "fold": int(frame_fold[frame]),
            "best_seed_component_id": int(seed_id),
            "member_count": len(group["member_identities"]),
            "P_count": int(counts["POSITIVE"]),
            "N1_count": int(counts["N1"]),
            "N0_count": int(counts["N0"]),
            "unlabeled_context_count": len(group["unlabeled_identities"]),
            "supervised_P_N1_target_count": int(supervised_p_n1),
            "MIXED_UTILITY_GROUP": bool(mixed),
        })

    group_by_key = {
        (row["frame_id"], row["best_seed_component_id"]): row for row in group_rows
    }
    sample_support = []
    by_frame = defaultdict(Counter)
    by_fold = defaultdict(Counter)
    eligible_counts = Counter()
    for target in targets:
        label = target["label"]
        if label not in {"POSITIVE", "N1"}:
            continue
        frame = str(target["frame_id"]).zfill(6)
        fold = int(target["validation_fold"])
        seed_id = target["best_seed_component_runtime_id"]
        if seed_id is None:
            category = f"{label}_NO_VALID_BEST_SEED"
        else:
            eligible_counts[label] += 1
            group = group_by_key[(frame, int(seed_id))]
            category = f"{label}_MIXED" if group["MIXED_UTILITY_GROUP"] else f"{label}_ONLY"
        by_frame[frame][category] += 1
        by_fold[fold][category] += 1
        sample_support.append({
            "sample_row": int(target["sample_row"]), "frame_id": frame,
            "fold": fold, "canonical_fragment_identity": int(target["canonical_fragment_identity"]),
            "label": label, "best_seed_component_id": seed_id,
            "support_category": category,
        })

    support_overall = Counter(row["support_category"] for row in sample_support)
    support_keys = (
        "POSITIVE_MIXED", "POSITIVE_ONLY", "POSITIVE_NO_VALID_BEST_SEED",
        "N1_MIXED", "N1_ONLY", "N1_NO_VALID_BEST_SEED",
    )
    support_overall = {key: int(support_overall[key]) for key in support_keys}
    frame_support_rows = [
        {"frame_id": frame, "fold": int(frame_fold[frame]), **{
            key: int(by_frame[frame][key]) for key in support_keys
        }}
        for frame in sorted(frame_fold)
    ]
    fold_support = {
        str(fold): {key: int(by_fold[fold][key]) for key in support_keys}
        for fold in sorted(set(frame_fold.values()))
    }

    supervised_bucket = Counter()
    for row in group_rows:
        value = row["supervised_P_N1_target_count"]
        bucket = "0" if value == 0 else ("1" if value == 1 else ("2" if value == 2 else ">=3"))
        supervised_bucket[bucket] += 1

    frames_p = {row["frame_id"] for row in group_rows if row["P_count"] > 0}
    frames_n1 = {row["frame_id"] for row in group_rows if row["N1_count"] > 0}
    frames_mixed = {row["frame_id"] for row in group_rows if row["MIXED_UTILITY_GROUP"]}
    fold_mixed = {}
    for fold in sorted(set(frame_fold.values())):
        selected = [row for row in group_rows if row["fold"] == fold and row["MIXED_UTILITY_GROUP"]]
        fold_mixed[str(fold)] = {
            "mixed_group_count": len(selected),
            "mixed_supported_P_count": int(sum(row["P_count"] for row in selected)),
            "mixed_supported_N1_count": int(sum(row["N1_count"] for row in selected)),
        }

    # Concentration uses only group-eligible (valid-best-seed) samples. Exclusions are explicit.
    p_group_counts = [row["P_count"] for row in group_rows]
    n1_group_counts = [row["N1_count"] for row in group_rows]
    mixed_sample_group_counts = [
        row["P_count"] + row["N1_count"] for row in group_rows if row["MIXED_UTILITY_GROUP"]
    ]
    group_concentration = {
        "P": {**_top_shares(p_group_counts), "excluded_no_valid_best_seed": int(no_seed_targets["POSITIVE"])},
        "N1": {**_top_shares(n1_group_counts), "excluded_no_valid_best_seed": int(no_seed_targets["N1"])},
        "mixed_group_supervised_samples": _top_shares(mixed_sample_group_counts),
    }
    p_frame = Counter()
    n1_frame = Counter()
    mixed_frame = Counter()
    for row in group_rows:
        p_frame[row["frame_id"]] += row["P_count"]
        n1_frame[row["frame_id"]] += row["N1_count"]
        if row["MIXED_UTILITY_GROUP"]:
            mixed_frame[row["frame_id"]] += row["P_count"] + row["N1_count"]
    frame_concentration = {
        "P": {**_top_shares(p_frame.values()), "excluded_no_valid_best_seed": int(no_seed_targets["POSITIVE"])},
        "N1": {**_top_shares(n1_frame.values()), "excluded_no_valid_best_seed": int(no_seed_targets["N1"])},
        "mixed_group_supervised_samples": _top_shares(mixed_frame.values()),
    }

    # Offline-only label composition of each P/N1 target's already-frozen local set.
    composition = {"ALL_P_N1_TARGETS": Counter(), "TARGET_POSITIVE": Counter(), "TARGET_N1": Counter()}
    composition_target_counts = Counter()
    for target in targets:
        if target["label"] not in {"POSITIVE", "N1"}:
            continue
        bucket = "TARGET_POSITIVE" if target["label"] == "POSITIVE" else "TARGET_N1"
        composition_target_counts["ALL_P_N1_TARGETS"] += 1
        composition_target_counts[bucket] += 1
        frame = str(target["frame_id"]).zfill(6)
        for relation in target["neighbor_relations"]:
            neighbor = (frame, int(relation["fragment_identity"]))
            label = supervised[neighbor]["label"] if neighbor in supervised else "UNLABELED_OTHER"
            composition["ALL_P_N1_TARGETS"][label] += 1
            composition[bucket][label] += 1

    label_group_counts = {
        "total_relational_groups": len(group_rows),
        "groups_containing_P": int(sum(row["P_count"] > 0 for row in group_rows)),
        "groups_containing_N1": int(sum(row["N1_count"] > 0 for row in group_rows)),
        "groups_containing_N0": int(sum(row["N0_count"] > 0 for row in group_rows)),
        "groups_containing_P_plus_N1": len(mixed_groups),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "analysis_scope": {
            "frame_count": len(frame_fold), "supervised_target_count": len(targets),
            "catalog_fragment_count": len(catalog),
            "relational_group_identity": "(frame_id, best_seed_component_runtime_id)",
            "no_valid_best_seed_semantics": "EXCLUDED_FROM_RELATIONAL_GROUPS_AND_REPORTED_SEPARATELY",
            "group_member_counting": "UNIQUE_(frame_id,canonical_fragment_identity)",
        },
        "relational_group_counts": label_group_counts,
        "no_valid_best_seed": {
            "catalog_fragment_count": int(no_seed_catalog),
            "target_counts": {label: int(no_seed_targets[label]) for label in SUPERVISED_LABELS},
        },
        "P_N1_mixed_group_support": {
            "overall": support_overall,
            "by_frame": frame_support_rows,
            "by_fold": fold_support,
            "ONLY_definition": "label has valid-best-seed group that does not contain the other P/N1 label; N0/unlabeled may be present",
        },
        "supervised_targets_per_group": {
            "zero_supervised_P_N1_targets": int(supervised_bucket["0"]),
            "one_supervised_P_N1_target": int(supervised_bucket["1"]),
            "two_supervised_P_N1_targets": int(supervised_bucket["2"]),
            "at_least_three_supervised_P_N1_targets": int(supervised_bucket[">=3"]),
        },
        "cross_frame_fold_support": {
            "frames_containing_P_relational_groups": len(frames_p),
            "frames_containing_N1_relational_groups": len(frames_n1),
            "frames_containing_mixed_P_N1_groups": len(frames_mixed),
            "by_fold": fold_mixed,
        },
        "support_concentration": {
            "denominator_semantics": "valid-best-seed relational-group-eligible samples only; no-seed exclusions reported",
            "by_group": group_concentration,
            "by_frame": frame_concentration,
        },
        "local_set_cardinality": {
            label: _local_set_summary(targets, label) for label in SUPERVISED_LABELS
        },
        "context_label_composition_offline_diagnostic": {
            key: _composition_record(composition[key], composition_target_counts[key])
            for key in ("ALL_P_N1_TARGETS", "TARGET_POSITIVE", "TARGET_N1")
        },
        "RELATIONAL_SUPPORT_OBSERVATION": "inconclusive",
        "observation_note": "No numerical interpretation gate was frozen; mechanical statistics are returned for reviewer judgment.",
        "RELATIONAL_LEARNING_SUPPORT": "NOT_FROZEN_BY_DEVELOPMENT",
        "MODEL_TRAINING_PERFORMED": False,
        "INDEPENDENT_MANIFEST_STATUS": "SEALED",
        "FIXED_100_EXECUTED": False,
        "FINAL_FIT_PERFORMED": False,
    }

    result_path = output_dir / "phase4_relational_support_coverage.json"
    group_path = output_dir / "phase4_relational_groups.csv"
    sample_path = output_dir / "phase4_P_N1_sample_support.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with group_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_support[0]))
        writer.writeheader(); writer.writerows(sample_support)
    return result, result_path, group_path, sample_path
