"""Streaming construction for fragment_learning_dev_v1."""

from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path

import numpy as np

from bev_tracking.clustering_detector import split_obstacle_filter_stages_with_indices
from bev_tracking.eval_policy import classify_gt_box, normalize_class_name
from bev_tracking.fragment_learning_dataset import (
    DATASET_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    MANIFEST_FRAME_COUNT,
    MODEL_FEATURE_FIELDS,
    DatasetConstructionError,
    build_dataset_row,
    build_offline_fragment_label,
    extract_runtime_fragment_features,
    feature_schema_payload,
    run_feature_leakage_gate,
    select_fragment_learning_manifest,
    write_feature_schema,
    write_fragment_learning_manifest,
)
from bev_tracking.fragment_phase0 import (
    build_candidate_fragments,
    union_box_mask,
)
from bev_tracking.geometry_sanity import points_in_oriented_3d_box
from bev_tracking.gesr_v1 import (
    CANDIDATE_INTENSITY_MIN,
    NUMERICAL_DTYPE,
    SEED_INTENSITY_MIN,
    build_seed_components_optimized,
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


TRAINING_LABELS = ("POSITIVE", "N1", "N0")
ALL_LABELS = (*TRAINING_LABELS, "UNLABELED_OTHER")
SUFFICIENCY_REQUIREMENTS = {
    "positive_fragments_min": 30,
    "positive_support_frames_min": 10,
    "N1_fragments_min": 20,
    "N1_support_frames_min": 5,
    "N0_fragments_min": 1,
}


def _read_frozen_manifest(path):
    values = [
        line.split("#", 1)[0].strip().zfill(6)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if len(values) != MANIFEST_FRAME_COUNT or len(set(values)) != len(values):
        raise DatasetConstructionError(
            "DATASET_CONSTRUCTION_ERROR: learning manifest must contain 64 unique frames"
        )
    if values != sorted(values):
        raise DatasetConstructionError(
            "DATASET_CONSTRUCTION_ERROR: learning manifest must be frame-id sorted"
        )
    return values


def prepare_or_validate_manifest(data_root, fixed_100_manifest, manifest_path):
    """Create the select-once manifest, or verify an existing frozen copy."""
    expected = select_fragment_learning_manifest(data_root, fixed_100_manifest)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        actual = _read_frozen_manifest(manifest_path)
        if actual != expected:
            raise DatasetConstructionError(
                "DATASET_CONSTRUCTION_ERROR: existing learning manifest differs "
                "from the frozen selection procedure"
            )
        return actual
    write_fragment_learning_manifest(expected, manifest_path)
    return expected


def _oracle_association(raw, fragment, gt_box, t0_gt_indices):
    fragment_indices = np.asarray(fragment["source_indices"], dtype=np.int64)
    inside = points_in_oriented_3d_box(raw[fragment_indices], gt_box)
    recovery_point_count = int(inside.sum())
    if recovery_point_count == 0:
        return None
    baseline = build_pca_oracle(raw[t0_gt_indices], gt_box)
    counterfactual = build_pca_oracle(
        raw[np.union1d(t0_gt_indices, fragment_indices)], gt_box
    )
    delta = (
        None
        if baseline["iou"] is None or counterfactual["iou"] is None
        else float(counterfactual["iou"] - baseline["iou"])
    )
    return {
        "gt_id": str(gt_box["id"]),
        "delta_iou": delta,
        "T2_minus_T0_point_count_inside_GT": recovery_point_count,
        "T0_pca_status": baseline["status"],
        "T0_iou": baseline["iou"],
        "counterfactual_pca_status": counterfactual["status"],
        "counterfactual_iou": counterfactual["iou"],
    }


def build_frame_dataset_rows(raw, boxes, frame_id):
    """Build one unique sample per runtime fragment from a single frame."""
    frame_id = str(frame_id).zfill(6)
    raw = np.asarray(raw)
    stages, stage_indices = split_obstacle_filter_stages_with_indices(
        raw, z_min=-0.9, intensity_min=SEED_INTENSITY_MIN
    )
    _, t2_indices = split_obstacle_filter_stages_with_indices(
        raw, z_min=-0.9, intensity_min=CANDIDATE_INTENSITY_MIN
    )
    z_points = stages["z_filter"]
    z_source_indices = stage_indices["z_filter"]
    intensities = np.asarray(z_points[:, 3], dtype=NUMERICAL_DTYPE)
    candidate_mask = (
        (intensities >= CANDIDATE_INTENSITY_MIN)
        & (intensities < SEED_INTENSITY_MIN)
    )
    seed_mask = intensities >= SEED_INTENSITY_MIN
    candidate_indices = np.asarray(z_source_indices[candidate_mask], dtype=np.int64)
    expected_candidate_indices = np.setdiff1d(
        t2_indices["intensity_filter"],
        stage_indices["intensity_filter"],
        assume_unique=True,
    )
    if not np.array_equal(np.sort(candidate_indices), expected_candidate_indices):
        raise DatasetConstructionError(
            f"DATASET_CONSTRUCTION_ERROR: T2-T0 candidate identity mismatch in {frame_id}"
        )

    fragments = build_candidate_fragments(
        z_points[candidate_mask], candidate_indices
    )
    seed_components = build_seed_components_optimized(
        z_points[seed_mask], z_source_indices[seed_mask]
    )
    valid_seed_components = [
        component for component in seed_components if component.geometry.valid
    ]
    positive_boxes = [box for box in boxes if classify_gt_box(box) == "positive"]
    positive_or_neutral_boxes = [
        box for box in boxes if classify_gt_box(box) in {"positive", "neutral"}
    ]
    annotation_boxes = [
        box for box in boxes
        if normalize_class_name(box.get("class_name")) in ANNOTATION_EXCLUSION_CLASSES
    ]
    t0_by_gt = {
        str(box["id"]): np.asarray(
            stage_indices["intensity_filter"]
        )[
            points_in_oriented_3d_box(
                raw[stage_indices["intensity_filter"]], box
            )
        ]
        for box in positive_boxes
    }

    rows = []
    identities = set()
    for fragment in fragments:
        identity = int(fragment["runtime_id"])
        if identity != int(np.min(fragment["source_indices"])) or identity in identities:
            raise DatasetConstructionError(
                f"DATASET_CONSTRUCTION_ERROR: non-canonical fragment identity in {frame_id}"
            )
        identities.add(identity)
        fragment_points = raw[fragment["source_indices"]]
        strict_background = not bool(
            union_box_mask(fragment_points, annotation_boxes).any()
        )
        positive_or_neutral_association = bool(
            union_box_mask(fragment_points, positive_or_neutral_boxes).any()
        )
        associations = []
        for box in positive_boxes:
            association = _oracle_association(
                raw, fragment, box, t0_by_gt[str(box["id"])]
            )
            if association is not None:
                associations.append(association)
        label = build_offline_fragment_label(
            associations,
            strict_background=strict_background,
            positive_or_neutral_annotation_association=(
                positive_or_neutral_association
            ),
        )
        features = extract_runtime_fragment_features(
            fragment, valid_seed_components
        )
        rows.append(
            build_dataset_row(
                frame_id=frame_id,
                canonical_fragment_identity=identity,
                model_features=features,
                label_record=label,
                diagnostics={
                    "associated_positive_car_GT": associations,
                    "strict_background": strict_background,
                    "positive_or_neutral_annotation_association": (
                        positive_or_neutral_association
                    ),
                },
            )
        )
    return {
        "frame_id": frame_id,
        "candidate_point_count": int(len(candidate_indices)),
        "fragment_count": int(len(fragments)),
        "rows": rows,
    }


def load_and_build_frame(data_root, frame_id):
    try:
        velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
        raw = load_kitti_point_cloud(velodyne_path)
        labels = load_kitti_labels(label_path)
        calib = load_kitti_calib(resolve_kitti_calib_path(data_root, frame_id))
        boxes = kitti_labels_to_lidar_boxes(labels, calib)
        return build_frame_dataset_rows(raw, boxes, frame_id)
    except DatasetConstructionError:
        raise
    except Exception as exc:
        raise DatasetConstructionError(
            f"DATASET_CONSTRUCTION_ERROR: frame {frame_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def dataset_sufficiency(counts, support_frames):
    checks = {
        "positive_fragments": counts["POSITIVE"]
        >= SUFFICIENCY_REQUIREMENTS["positive_fragments_min"],
        "positive_support_frames": len(support_frames["POSITIVE"])
        >= SUFFICIENCY_REQUIREMENTS["positive_support_frames_min"],
        "N1_fragments": counts["N1"]
        >= SUFFICIENCY_REQUIREMENTS["N1_fragments_min"],
        "N1_support_frames": len(support_frames["N1"])
        >= SUFFICIENCY_REQUIREMENTS["N1_support_frames_min"],
        "N0_fragments": counts["N0"]
        >= SUFFICIENCY_REQUIREMENTS["N0_fragments_min"],
    }
    return {"result": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def _merge_gate_checks(target, source):
    for key, value in source["checks"].items():
        target[key] = target.get(key, True) and bool(value)


def _temporary_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "dataset": output_dir / "fragment_dataset.jsonl.tmp",
        "X_model": output_dir / "X_model.csv.tmp",
        "y_labels": output_dir / "y_labels.csv.tmp",
    }


def construct_fragment_learning_dataset(
    data_root,
    frame_ids,
    output_dir,
    *,
    progress_callback=None,
    frame_builder=load_and_build_frame,
):
    """Stream all selected frames; publish tables only after complete success."""
    frame_ids = [str(value).zfill(6) for value in frame_ids]
    if len(frame_ids) != MANIFEST_FRAME_COUNT:
        raise DatasetConstructionError(
            "DATASET_CONSTRUCTION_ERROR: exactly 64 frozen frames are required"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_paths(output_dir)
    for path in temporary.values():
        if path.exists():
            path.unlink()

    counts = Counter({label: 0 for label in ALL_LABELS})
    support_frames = {label: set() for label in ALL_LABELS}
    candidate_point_count = 0
    fragment_count = 0
    gate_checks = {}
    try:
        with (
            temporary["dataset"].open("w", encoding="utf-8") as dataset_handle,
            temporary["X_model"].open("w", newline="", encoding="utf-8") as x_handle,
            temporary["y_labels"].open("w", newline="", encoding="utf-8") as y_handle,
        ):
            x_writer = csv.DictWriter(x_handle, fieldnames=MODEL_FEATURE_FIELDS)
            y_writer = csv.DictWriter(y_handle, fieldnames=("sample_row", "label"))
            x_writer.writeheader()
            y_writer.writeheader()
            training_row = 0
            for index, frame_id in enumerate(frame_ids, start=1):
                if progress_callback:
                    progress_callback(index, len(frame_ids), frame_id)
                frame = frame_builder(data_root, frame_id)
                if frame["frame_id"] != frame_id:
                    raise DatasetConstructionError(
                        "DATASET_CONSTRUCTION_ERROR: frame builder identity mismatch"
                    )
                candidate_point_count += int(frame["candidate_point_count"])
                fragment_count += int(frame["fragment_count"])
                if len(frame["rows"]) != int(frame["fragment_count"]):
                    raise DatasetConstructionError(
                        f"DATASET_CONSTRUCTION_ERROR: row/fragment mismatch in {frame_id}"
                    )
                for row in frame["rows"]:
                    leakage = run_feature_leakage_gate([row], MODEL_FEATURE_FIELDS)
                    _merge_gate_checks(gate_checks, leakage)
                    label = row["label_field"]["label"]
                    if label not in ALL_LABELS:
                        raise DatasetConstructionError(
                            f"DATASET_CONSTRUCTION_ERROR: unknown label {label}"
                        )
                    counts[label] += 1
                    support_frames[label].add(frame_id)
                    dataset_handle.write(
                        json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                    if label in TRAINING_LABELS:
                        x_writer.writerow(row["model_feature_fields"])
                        y_writer.writerow({"sample_row": training_row, "label": label})
                        training_row += 1
    except Exception:
        for path in temporary.values():
            if path.exists():
                path.unlink()
        raise

    leakage_result = {
        "result": "PASS" if gate_checks and all(gate_checks.values()) else "FAIL",
        "checks": gate_checks,
    }
    sufficiency = dataset_sufficiency(counts, support_frames)
    final_paths = {
        "dataset": output_dir / "fragment_dataset.jsonl",
        "X_model": output_dir / "X_model.csv",
        "y_labels": output_dir / "y_labels.csv",
    }
    for key, temporary_path in temporary.items():
        os.replace(temporary_path, final_paths[key])
    with final_paths["X_model"].open(newline="", encoding="utf-8") as handle:
        actual_x_model_columns = tuple(next(csv.reader(handle)))
    exported_header_gate = run_feature_leakage_gate([], actual_x_model_columns)
    _merge_gate_checks(leakage_result["checks"], exported_header_gate)
    leakage_result["result"] = (
        "PASS" if all(leakage_result["checks"].values()) else "FAIL"
    )
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "manifest_frame_count": len(frame_ids),
        "candidate_point_count": int(candidate_point_count),
        "unique_runtime_fragment_count": int(fragment_count),
        "label_counts": {label: int(counts[label]) for label in ALL_LABELS},
        "label_support_frame_counts": {
            label: len(support_frames[label]) for label in ALL_LABELS
        },
        "training_row_count": int(sum(counts[label] for label in TRAINING_LABELS)),
        "FEATURE_LEAKAGE_GATE": leakage_result,
        "DATASET_SUFFICIENCY": sufficiency,
        "MODEL_TRAINING_PERFORMED": False,
        "split_generated": False,
        "output_paths": {key: str(path) for key, path in final_paths.items()},
    }


def write_construction_summary(summary, output_dir):
    path = Path(output_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_and_construct_dataset(
    data_root,
    fixed_100_manifest,
    output_dir,
    *,
    progress_callback=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fragment_learning_dev_manifest.txt"
    frame_ids = prepare_or_validate_manifest(
        data_root, fixed_100_manifest, manifest_path
    )
    write_feature_schema(output_dir / "fragment_feature_schema_v1.json")
    summary = construct_fragment_learning_dataset(
        data_root,
        frame_ids,
        output_dir,
        progress_callback=progress_callback,
    )
    summary["manifest"] = str(manifest_path)
    summary_path = write_construction_summary(summary, output_dir)
    return summary, summary_path
