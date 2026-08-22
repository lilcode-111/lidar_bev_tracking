"""Frozen frame-grouped 5-fold split for fragment_learning_dev_v1."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from bev_tracking.fragment_learning_construction import TRAINING_LABELS
from bev_tracking.fragment_learning_dataset import DatasetConstructionError


SPLIT_SCHEMA_VERSION = "fragment-learning-splits-v1"
N_SPLITS = 5
SPLIT_RANDOM_STATE = 15531
FOLD_REQUIREMENTS = {
    "positive_min": 3,
    "positive_support_frames_min": 2,
    "N1_min": 2,
    "N0_min": 1,
}


class FragmentLearningSplitError(DatasetConstructionError):
    pass


def _read_manifest(path):
    values = [
        line.split("#", 1)[0].strip().zfill(6)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if len(values) != 64 or len(set(values)) != 64 or values != sorted(values):
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: learning manifest is not 64 unique sorted frames"
        )
    return values


def _read_fixed_ids(path):
    return {
        line.split("#", 1)[0].strip().zfill(6)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def _load_training_samples(dataset_path):
    samples = []
    identities = set()
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row["label_field"]["label"])
            if label not in TRAINING_LABELS:
                continue
            metadata = row["metadata_fields"]
            identity = (
                str(metadata["frame_id"]).zfill(6),
                int(metadata["canonical_fragment_identity"]),
            )
            if identity in identities:
                raise FragmentLearningSplitError(
                    f"SPLIT_CONSTRUCTION_ERROR: duplicate sample identity {identity}"
                )
            identities.add(identity)
            samples.append(
                {
                    "sample_row": len(samples),
                    "frame_id": identity[0],
                    "canonical_fragment_identity": identity[1],
                    "label": label,
                    "dataset_line": line_number,
                }
            )
    if not samples:
        raise FragmentLearningSplitError("SPLIT_CONSTRUCTION_ERROR: no training samples")
    return samples


def _validate_export_alignment(samples, x_model_path, y_labels_path):
    with Path(x_model_path).open(newline="", encoding="utf-8") as handle:
        x_count = sum(1 for _ in csv.DictReader(handle))
    with Path(y_labels_path).open(newline="", encoding="utf-8") as handle:
        y_rows = list(csv.DictReader(handle))
    if x_count != len(samples) or len(y_rows) != len(samples):
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: dataset/X_model/y_labels row counts differ"
        )
    for index, (sample, label_row) in enumerate(zip(samples, y_rows)):
        if int(label_row["sample_row"]) != index or label_row["label"] != sample["label"]:
            raise FragmentLearningSplitError(
                "SPLIT_CONSTRUCTION_ERROR: y_labels order differs from dataset"
            )


def _fold_record(fold_id, validation_indices, samples):
    selected = [samples[int(index)] for index in validation_indices]
    frames = sorted({item["frame_id"] for item in selected})
    counts = Counter(item["label"] for item in selected)
    support = {
        label: len({item["frame_id"] for item in selected if item["label"] == label})
        for label in TRAINING_LABELS
    }
    checks = {
        "positive_fragments": counts["POSITIVE"] >= FOLD_REQUIREMENTS["positive_min"],
        "positive_support_frames": support["POSITIVE"]
        >= FOLD_REQUIREMENTS["positive_support_frames_min"],
        "N1_fragments": counts["N1"] >= FOLD_REQUIREMENTS["N1_min"],
        "N0_fragments": counts["N0"] >= FOLD_REQUIREMENTS["N0_min"],
    }
    return {
        "fold_id": int(fold_id),
        "validation_frame_count": len(frames),
        "validation_frame_ids": frames,
        "validation_sample_count": len(selected),
        "label_counts": {label: int(counts[label]) for label in TRAINING_LABELS},
        "label_support_frame_counts": support,
        "checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }


def build_fragment_learning_splits(samples, manifest_frames):
    """Run the exact frozen StratifiedGroupKFold and validate its identities."""
    labels = np.asarray([item["label"] for item in samples], dtype=object)
    groups = np.asarray([item["frame_id"] for item in samples], dtype=object)
    if set(groups.tolist()) != set(manifest_frames):
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: training sample frames differ from manifest"
        )
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SPLIT_RANDOM_STATE,
    )
    assignment = np.full(len(samples), -1, dtype=np.int64)
    folds = []
    for fold_id, (_, validation_indices) in enumerate(
        splitter.split(np.zeros((len(samples), 1)), labels, groups), start=1
    ):
        if np.any(assignment[validation_indices] != -1):
            raise FragmentLearningSplitError(
                "SPLIT_CONSTRUCTION_ERROR: sample assigned to multiple folds"
            )
        assignment[validation_indices] = fold_id
        folds.append(_fold_record(fold_id, validation_indices, samples))
    if np.any(assignment == -1):
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: unassigned training sample"
        )
    frame_to_fold = {}
    for sample, fold_id in zip(samples, assignment.tolist()):
        previous = frame_to_fold.setdefault(sample["frame_id"], fold_id)
        if previous != fold_id:
            raise FragmentLearningSplitError(
                "SPLIT_CONSTRUCTION_ERROR: one frame spans multiple folds"
            )
    if set(frame_to_fold) != set(manifest_frames):
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: not all manifest frames assigned"
        )
    global_checks = {
        "exactly_5_folds": len(folds) == N_SPLITS,
        "all_training_samples_assigned_once": len(assignment) == len(samples),
        "all_64_frames_assigned_once": len(frame_to_fold) == 64,
        "frame_group_isolation": True,
        "all_fold_requirements_pass": all(item["result"] == "PASS" for item in folds),
    }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "parameters": {
            "group": "frame_id",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SPLIT_RANDOM_STATE,
        },
        "training_sample_count": len(samples),
        "manifest_frame_count": len(manifest_frames),
        "folds": folds,
        "frame_to_validation_fold": {
            frame_id: int(frame_to_fold[frame_id]) for frame_id in sorted(frame_to_fold)
        },
        "sample_assignments": [
            {
                "sample_row": item["sample_row"],
                "frame_id": item["frame_id"],
                "canonical_fragment_identity": item["canonical_fragment_identity"],
                "label": item["label"],
                "validation_fold": int(fold_id),
            }
            for item, fold_id in zip(samples, assignment.tolist())
        ],
        "global_checks": global_checks,
        "SPLIT_VALIDITY": "PASS" if all(global_checks.values()) else "FAIL",
        "MODEL_TRAINING_PERFORMED": False,
    }


def generate_fragment_learning_splits(output_dir, fixed_100_manifest):
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("DATASET_SUFFICIENCY", {}).get("result") != "PASS":
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: DATASET_SUFFICIENCY is not PASS"
        )
    if summary.get("FEATURE_LEAKAGE_GATE", {}).get("result") != "PASS":
        raise FragmentLearningSplitError(
            "SPLIT_CONSTRUCTION_ERROR: FEATURE_LEAKAGE_GATE is not PASS"
        )
    manifest = _read_manifest(output_dir / "fragment_learning_dev_manifest.txt")
    overlap = set(manifest) & _read_fixed_ids(fixed_100_manifest)
    if overlap:
        raise FragmentLearningSplitError(
            f"SPLIT_CONSTRUCTION_ERROR: learning/fixed-100 overlap: {sorted(overlap)}"
        )
    samples = _load_training_samples(output_dir / "fragment_dataset.jsonl")
    _validate_export_alignment(
        samples, output_dir / "X_model.csv", output_dir / "y_labels.csv"
    )
    result = build_fragment_learning_splits(samples, manifest)
    split_path = output_dir / "fragment_learning_splits.json"
    split_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary["split_generated"] = True
    summary["split_path"] = str(split_path)
    summary["SPLIT_VALIDITY"] = result["SPLIT_VALIDITY"]
    summary["MODEL_TRAINING_PERFORMED"] = False
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return result, split_path
