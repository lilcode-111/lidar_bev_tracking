"""Frozen semantics for the fragment_learning_dev_v1 dataset.

Runtime feature extraction is deliberately GT-free.  GT and counterfactual
information is accepted only by the separate offline label builder.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import random

import numpy as np

from bev_tracking.fragment_phase0 import best_endpoint_relation


MANIFEST_SELECTION_SEED = 15501
MANIFEST_FRAME_COUNT = 64
MATERIAL_GAIN = 0.10
DATASET_SCHEMA_VERSION = "fragment-learning-dev-v1"
FEATURE_SCHEMA_VERSION = "fragment-feature-schema-v1"

MODEL_FEATURE_FIELDS = (
    "fragment_type",
    "fragment_point_count",
    "major_span",
    "minor_span",
    "lambda1",
    "lambda2",
    "linearity",
    "range_xy",
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    "fragment_axis_valid",
    "seed_relation_valid",
    "endpoint_gap",
    "lateral_offset",
    "orientation_difference",
    "seed_component_point_count",
    "seed_major_span",
    "seed_minor_span",
    "seed_linearity",
    "fragment_seed_center_distance",
    "fragment_relative_major_axis_coordinate",
    "fragment_relative_minor_axis_coordinate",
)

NUMERIC_MODEL_FEATURE_FIELDS = tuple(
    field for field in MODEL_FEATURE_FIELDS if field != "fragment_type"
)

RELATION_SENTINEL_FIELDS = (
    "endpoint_gap",
    "lateral_offset",
    "orientation_difference",
    "seed_component_point_count",
    "seed_major_span",
    "seed_minor_span",
    "seed_linearity",
    "fragment_seed_center_distance",
    "fragment_relative_major_axis_coordinate",
    "fragment_relative_minor_axis_coordinate",
)

FEATURE_DENYLIST = {
    "frame_id",
    "canonical_fragment_identity",
    "label",
    "label_reason",
    "gt_id",
    "associated_gt_identity",
    "counterfactual_iou",
    "t0_iou",
    "material_gain",
    "fragment_delta_iou",
    "oracle_fragment_purity",
    "oracle_status",
    "fold_id",
}


class FragmentLearningDatasetError(ValueError):
    pass


class DatasetConstructionError(FragmentLearningDatasetError):
    pass


class DatasetLabelConflict(FragmentLearningDatasetError):
    pass


def _manifest_ids(path):
    values = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw_line.split("#", 1)[0].strip()
        if value:
            values.append(value.zfill(6))
    return values


def collect_eligible_frame_ids(data_root):
    """Return sorted KITTI training frames with all three required inputs."""
    training = Path(data_root) / "training"
    roots = {
        "velodyne": training / "velodyne",
        "label_2": training / "label_2",
        "calib": training / "calib",
    }
    missing_roots = [str(path) for path in roots.values() if not path.is_dir()]
    if missing_roots:
        raise DatasetConstructionError(
            "DATASET_CONSTRUCTION_ERROR: missing KITTI input directories: "
            + ", ".join(missing_roots)
        )
    sets = {
        "velodyne": {path.stem for path in roots["velodyne"].glob("*.bin")},
        "label_2": {path.stem for path in roots["label_2"].glob("*.txt")},
        "calib": {path.stem for path in roots["calib"].glob("*.txt")},
    }
    return sorted(set.intersection(*sets.values()))


def select_fragment_learning_manifest(
    data_root,
    fixed_100_manifest,
    *,
    frame_count=MANIFEST_FRAME_COUNT,
    selection_seed=MANIFEST_SELECTION_SEED,
):
    """Apply the frozen select-once procedure without reading any labels."""
    return select_fragment_learning_manifest_from_ids(
        collect_eligible_frame_ids(data_root),
        fixed_100_manifest,
        frame_count=frame_count,
        selection_seed=selection_seed,
    )


def select_fragment_learning_manifest_from_ids(
    complete_frame_ids,
    fixed_100_manifest,
    *,
    frame_count=MANIFEST_FRAME_COUNT,
    selection_seed=MANIFEST_SELECTION_SEED,
):
    """Apply the same frozen selection to a complete, prevalidated frame catalog."""
    fixed = set(_manifest_ids(fixed_100_manifest))
    complete = sorted({str(frame_id).zfill(6) for frame_id in complete_frame_ids})
    eligible = [frame_id for frame_id in complete if frame_id not in fixed]
    if len(eligible) < frame_count:
        raise DatasetConstructionError(
            "DATASET_CONSTRUCTION_ERROR: "
            f"need {frame_count} eligible frames after fixed-100 exclusion, "
            f"found {len(eligible)}"
        )
    selected = list(eligible)
    random.Random(selection_seed).shuffle(selected)
    return sorted(selected[:frame_count])


def write_fragment_learning_manifest(frame_ids, output_path):
    frame_ids = [str(value).zfill(6) for value in frame_ids]
    if len(frame_ids) != MANIFEST_FRAME_COUNT or len(set(frame_ids)) != len(frame_ids):
        raise DatasetConstructionError(
            "fragment_learning_dev_manifest must contain 64 unique frames"
        )
    if frame_ids != sorted(frame_ids):
        raise DatasetConstructionError("final learning manifest must be frame-id sorted")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(frame_ids) + "\n", encoding="utf-8")
    return path


def _finite_float(value, field):
    value = float(value)
    if not math.isfinite(value):
        raise FragmentLearningDatasetError(f"non-finite runtime feature: {field}")
    return value


def _component_geometry_features(component, fragment):
    geometry = component.geometry
    center = np.asarray(geometry.center, dtype=np.float64)
    fragment_center = np.asarray(fragment["center"], dtype=np.float64)
    displacement = fragment_center - center
    major = np.asarray(geometry.major, dtype=np.float64)
    minor = np.asarray(geometry.minor, dtype=np.float64)
    seed_linearity = (
        0.0
        if geometry.lambda1 == 0.0
        else float(1.0 - geometry.lambda2 / geometry.lambda1)
    )
    return {
        "seed_component_point_count": float(len(component.source_indices)),
        "seed_major_span": float(geometry.u_max - geometry.u_min),
        "seed_minor_span": float(geometry.v_max - geometry.v_min),
        "seed_linearity": seed_linearity,
        "fragment_seed_center_distance": float(np.linalg.norm(displacement)),
        "fragment_relative_major_axis_coordinate": float(displacement @ major),
        "fragment_relative_minor_axis_coordinate": float(displacement @ minor),
    }


def extract_runtime_fragment_features(fragment, valid_seed_components):
    """Extract only inference-time data; this API cannot receive GT or labels."""
    relation = best_endpoint_relation(fragment, valid_seed_components)
    fragment_axis_valid = int(bool(fragment["major_axis_valid"]))
    features = {
        "fragment_type": str(fragment["fragment_type"]),
        "fragment_point_count": float(fragment["point_count"]),
        "major_span": float(fragment["major_span"]),
        "minor_span": float(fragment["minor_span"]),
        "lambda1": float(fragment["lambda1"] or 0.0),
        "lambda2": float(fragment["lambda2"] or 0.0),
        "linearity": float(fragment["linearity"] or 0.0),
        "range_xy": float(fragment["range"]),
        "intensity_mean": float(fragment["intensity_mean"]),
        "intensity_std": float(fragment["intensity_std"]),
        "intensity_min": float(fragment["intensity_min"]),
        "intensity_max": float(fragment["intensity_max"]),
        "fragment_axis_valid": float(fragment_axis_valid),
        "seed_relation_valid": float(bool(relation["has_computable_seed_relation"])),
    }
    features.update({field: 0.0 for field in RELATION_SENTINEL_FIELDS})
    if relation["has_computable_seed_relation"]:
        by_id = {component.runtime_id: component for component in valid_seed_components}
        component_id = relation["seed_component_runtime_id"]
        if component_id not in by_id:
            raise FragmentLearningDatasetError("selected seed component is unavailable")
        features.update(_component_geometry_features(by_id[component_id], fragment))
        features["endpoint_gap"] = float(relation["endpoint_gap"])
        features["lateral_offset"] = float(relation["lateral_offset"])
        features["orientation_difference"] = (
            float(relation["orientation_difference"])
            if fragment_axis_valid
            else 0.0
        )
    validate_model_feature_vector(features)
    return features


def validate_model_feature_vector(features):
    if tuple(features) != MODEL_FEATURE_FIELDS:
        missing = sorted(set(MODEL_FEATURE_FIELDS) - set(features))
        extra = sorted(set(features) - set(MODEL_FEATURE_FIELDS))
        raise FragmentLearningDatasetError(
            f"model feature schema mismatch; missing={missing}, extra={extra}"
        )
    if features["fragment_type"] not in {"SINGLETON", "STRUCTURED"}:
        raise FragmentLearningDatasetError("invalid fragment_type")
    for field in NUMERIC_MODEL_FEATURE_FIELDS:
        _finite_float(features[field], field)
    for flag in ("fragment_axis_valid", "seed_relation_valid"):
        if float(features[flag]) not in (0.0, 1.0):
            raise FragmentLearningDatasetError(f"{flag} must be binary")
    if float(features["seed_relation_valid"]) == 0.0:
        if any(float(features[field]) != 0.0 for field in RELATION_SENTINEL_FIELDS):
            raise FragmentLearningDatasetError(
                "missing seed relation must use zero sentinels"
            )
    if float(features["fragment_axis_valid"]) == 0.0:
        if float(features["orientation_difference"]) != 0.0:
            raise FragmentLearningDatasetError(
                "invalid fragment axis must use zero orientation sentinel"
            )
    return True


def build_offline_fragment_label(
    associations,
    *,
    strict_background,
    positive_or_neutral_annotation_association=False,
):
    """Build one oracle label from complete per-positive-Car associations."""
    associations = list(associations)
    if strict_background and positive_or_neutral_annotation_association:
        raise DatasetLabelConflict(
            "DATASET_LABEL_CONFLICT: strict background overlaps positive/neutral GT"
        )
    evaluable = [item for item in associations if item.get("delta_iou") is not None]
    if any(float(item["delta_iou"]) >= MATERIAL_GAIN for item in evaluable):
        return {"label": "POSITIVE", "label_reason": "MATERIAL_POSITIVE_FRAGMENT"}
    if associations:
        if len(evaluable) != len(associations):
            return {
                "label": "UNLABELED_OTHER",
                "label_reason": "LABEL_NOT_FULLY_EVALUABLE",
            }
        return {
            "label": "N1",
            "label_reason": "RECOVERY_ASSOCIATED_NON_MATERIAL",
        }
    if strict_background:
        return {"label": "N0", "label_reason": "STRICT_BACKGROUND_FRAGMENT"}
    return {"label": "UNLABELED_OTHER", "label_reason": "NO_FROZEN_LABEL"}


def build_dataset_row(
    *, frame_id, canonical_fragment_identity, model_features, label_record, diagnostics
):
    validate_model_feature_vector(model_features)
    return {
        "metadata_fields": {
            "frame_id": str(frame_id).zfill(6),
            "canonical_fragment_identity": int(canonical_fragment_identity),
        },
        "label_field": {"label": str(label_record["label"])},
        "model_feature_fields": dict(model_features),
        "diagnostic_fields": {
            **dict(diagnostics),
            "label_reason": str(label_record["label_reason"]),
        },
    }


def run_feature_leakage_gate(rows, x_model_columns):
    """Validate declared and exported model columns plus finite row values."""
    declared = set(MODEL_FEATURE_FIELDS)
    exported = set(x_model_columns)
    checks = {
        "declared_schema_exact": exported == declared,
        "denylist_absent_from_declared_schema": not bool(declared & FEATURE_DENYLIST),
        "denylist_absent_from_X_model": not bool(exported & FEATURE_DENYLIST),
        "feature_extractor_GT_free_signature": not bool(
            set(inspect.signature(extract_runtime_fragment_features).parameters)
            & {"gt", "gt_box", "gt_boxes", "label", "labels"}
        ),
        "row_field_isolation": True,
        "all_model_features_valid_and_finite": True,
    }
    expected_sections = {
        "metadata_fields", "label_field", "model_feature_fields", "diagnostic_fields"
    }
    for row in rows:
        if set(row) != expected_sections:
            checks["row_field_isolation"] = False
        try:
            validate_model_feature_vector(row["model_feature_fields"])
        except (KeyError, FragmentLearningDatasetError, TypeError, ValueError):
            checks["all_model_features_valid_and_finite"] = False
    return {"result": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def feature_schema_payload():
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "model_feature_fields": list(MODEL_FEATURE_FIELDS),
        "categorical_fields": ["fragment_type"],
        "validity_flags": ["fragment_axis_valid", "seed_relation_valid"],
        "missing_value_encoding": {
            "continuous_sentinel": 0.0,
            "requires_corresponding_valid_flag": True,
            "relation_fields": list(RELATION_SENTINEL_FIELDS),
        },
        "forbidden_field_classes": [
            "GT geometry", "counterfactual result", "label", "metadata", "fold identity"
        ],
    }


def write_feature_schema(output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_schema_payload(), indent=2) + "\n", encoding="utf-8")
    return path
