"""Streaming construction for fragment_learning_dev_v1."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import shutil
import zipfile

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
    select_fragment_learning_manifest_from_ids,
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

KITTI_ARCHIVE_SPECS = {
    "velodyne": {
        "archive": "data_object_velodyne.zip",
        "prefix": "training/velodyne/",
        "suffix": ".bin",
    },
    "label_2": {
        "archive": "data_object_label_2.zip",
        "prefix": "training/label_2/",
        "suffix": ".txt",
    },
    "calib": {
        "archive": "data_object_calib.zip",
        "prefix": "training/calib/",
        "suffix": ".txt",
    },
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


def _archive_frame_entries(archive, prefix, suffix):
    entries = {}
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        relative = name[len(prefix):]
        if "/" in relative or relative == suffix:
            continue
        frame_id = relative[: -len(suffix)].zfill(6)
        if frame_id in entries:
            raise DatasetConstructionError(
                f"DATASET_CONSTRUCTION_ERROR: duplicate archive frame {frame_id}"
            )
        entries[frame_id] = info
    return entries


def collect_complete_archive_frame_ids(archive_dir):
    """Read ZIP central directories only and intersect the three training inputs."""
    archive_dir = Path(archive_dir)
    frame_sets = []
    try:
        for spec in KITTI_ARCHIVE_SPECS.values():
            path = archive_dir / spec["archive"]
            if not path.is_file():
                raise DatasetConstructionError(
                    f"DATASET_CONSTRUCTION_ERROR: missing KITTI archive: {path}"
                )
            with zipfile.ZipFile(path) as archive:
                entries = _archive_frame_entries(
                    archive, spec["prefix"], spec["suffix"]
                )
            if not entries:
                raise DatasetConstructionError(
                    f"DATASET_CONSTRUCTION_ERROR: no training entries in {path}"
                )
            frame_sets.append(set(entries))
    except zipfile.BadZipFile as exc:
        raise DatasetConstructionError(
            f"DATASET_CONSTRUCTION_ERROR: invalid KITTI ZIP: {exc}"
        ) from exc
    return sorted(set.intersection(*frame_sets))


def _validate_or_write_expected_manifest(expected, manifest_path):
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        actual = _read_frozen_manifest(manifest_path)
        if actual != expected:
            raise DatasetConstructionError(
                "DATASET_CONSTRUCTION_ERROR: existing learning manifest differs "
                "from the frozen complete-archive selection"
            )
        return actual
    write_fragment_learning_manifest(expected, manifest_path)
    return expected


def _copy_archive_entry(archive, info, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == info.file_size:
        return False
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise DatasetConstructionError(
                f"DATASET_CONSTRUCTION_ERROR: extracted size mismatch: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return True


def prepare_selected_archive_cache(
    archive_dir,
    fixed_100_manifest,
    manifest_path,
    cache_root,
    *,
    progress_callback=None,
):
    """Select against all complete ZIP entries, then stage only the frozen 64."""
    complete_ids = collect_complete_archive_frame_ids(archive_dir)
    selected = select_fragment_learning_manifest_from_ids(
        complete_ids, fixed_100_manifest
    )
    selected = _validate_or_write_expected_manifest(selected, manifest_path)
    archive_dir = Path(archive_dir)
    cache_root = Path(cache_root)
    extracted_file_count = 0
    try:
        with ExitStack() as stack:
            archives = {}
            entries_by_kind = {}
            for kind, spec in KITTI_ARCHIVE_SPECS.items():
                archive = stack.enter_context(
                    zipfile.ZipFile(archive_dir / spec["archive"])
                )
                archives[kind] = archive
                entries_by_kind[kind] = _archive_frame_entries(
                    archive, spec["prefix"], spec["suffix"]
                )
            for index, frame_id in enumerate(selected, start=1):
                if progress_callback:
                    progress_callback(index, len(selected), frame_id)
                for kind, spec in KITTI_ARCHIVE_SPECS.items():
                    info = entries_by_kind[kind].get(frame_id)
                    if info is None:
                        raise DatasetConstructionError(
                            "DATASET_CONSTRUCTION_ERROR: selected frame missing from "
                            f"{kind} archive: {frame_id}"
                        )
                    destination = (
                        cache_root / "training" / kind / f"{frame_id}{spec['suffix']}"
                    )
                    extracted_file_count += int(
                        _copy_archive_entry(archives[kind], info, destination)
                    )
    except DatasetConstructionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise DatasetConstructionError(
            f"DATASET_CONSTRUCTION_ERROR: archive staging failed: {exc}"
        ) from exc
    return {
        "frame_ids": selected,
        "cache_root": cache_root,
        "complete_archive_frame_count": len(complete_ids),
        "selected_frame_count": len(selected),
        "newly_extracted_file_count": extracted_file_count,
    }


def _oracle_association(
    raw, fragment, gt_box, t0_gt_indices, t2_minus_t0_candidate_indices
):
    fragment_indices = np.asarray(fragment["source_indices"], dtype=np.int64)
    recovery_indices = np.intersect1d(
        fragment_indices,
        t2_minus_t0_candidate_indices,
        assume_unique=True,
    )
    inside = points_in_oriented_3d_box(raw[recovery_indices], gt_box)
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
    t2_minus_t0_candidate_indices = np.intersect1d(
        candidate_indices,
        expected_candidate_indices,
        assume_unique=True,
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
                raw,
                fragment,
                box,
                t0_by_gt[str(box["id"])],
                t2_minus_t0_candidate_indices,
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


def prepare_and_construct_archive_dataset(
    archive_dir,
    fixed_100_manifest,
    output_dir,
    cache_root,
    *,
    staging_progress_callback=None,
    construction_progress_callback=None,
):
    """Construct from official ZIPs while staging only the frozen 64 frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fragment_learning_dev_manifest.txt"
    staging = prepare_selected_archive_cache(
        archive_dir,
        fixed_100_manifest,
        manifest_path,
        cache_root,
        progress_callback=staging_progress_callback,
    )
    write_feature_schema(output_dir / "fragment_feature_schema_v1.json")
    summary = construct_fragment_learning_dataset(
        staging["cache_root"],
        staging["frame_ids"],
        output_dir,
        progress_callback=construction_progress_callback,
    )
    summary.update(
        {
            "input_mode": "official_KITTI_archives_selected_64_cache",
            "archive_dir": str(Path(archive_dir)),
            "selected_input_cache": str(staging["cache_root"]),
            "complete_archive_frame_count": staging[
                "complete_archive_frame_count"
            ],
            "newly_extracted_file_count": staging["newly_extracted_file_count"],
            "manifest": str(manifest_path),
        }
    )
    summary_path = write_construction_summary(summary, output_dir)
    return summary, summary_path
