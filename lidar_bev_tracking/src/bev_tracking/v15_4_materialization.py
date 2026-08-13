import hashlib
import json
from pathlib import Path

from bev_tracking.experiment_gate import sha256_file


SCHEDULE_SCHEMA_VERSION = "15.4-threshold-schedule-v1"
IDENTITY_SCHEMA_VERSION = "15.4-pre-run-identity-v1"
EXPECTED_VARIANTS = {
    "T0": (0.38, False),
    "T1": (0.30, True),
    "T2": (0.15, True),
    "T_off": (0.00, False),
}
ALLOWED_CONFIG_DIFF_PATHS = ["detector.intensity_min"]


class V154MaterializationError(ValueError):
    pass


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def raw_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identity_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_ordered_manifest(path):
    frame_ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frame_ids:
        raise V154MaterializationError("manifest must not be empty")
    if any(len(frame_id) != 6 or not frame_id.isdigit() for frame_id in frame_ids):
        raise V154MaterializationError("manifest frame ids must be six digits")
    if len(set(frame_ids)) != len(frame_ids):
        raise V154MaterializationError("manifest frame ids must be unique")
    return frame_ids


def validate_threshold_schedule(schedule):
    if schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise V154MaterializationError("threshold schedule schema mismatch")
    if schedule.get("registration_status") != "PRE_REGISTERED":
        raise V154MaterializationError("threshold schedule is not pre-registered")
    expected_booleans = {
        "schedule_locked_before_results": True,
        "posthoc_threshold_addition_allowed": False,
        "posthoc_threshold_removal_allowed": False,
        "posthoc_threshold_change_allowed": False,
        "results_observed_at_registration": False,
    }
    for field, expected in expected_booleans.items():
        if schedule.get(field) is not expected:
            raise V154MaterializationError(f"invalid schedule lock field: {field}")
    if schedule.get("allowed_config_diff_paths") != ALLOWED_CONFIG_DIFF_PATHS:
        raise V154MaterializationError("only detector.intensity_min may differ")

    variants = schedule.get("variants", {})
    if list(variants) != list(EXPECTED_VARIANTS):
        raise V154MaterializationError("variant names or order changed")
    for name, (expected_threshold, expected_release_eligible) in EXPECTED_VARIANTS.items():
        variant = variants[name]
        if float(variant.get("intensity_min")) != expected_threshold:
            raise V154MaterializationError(f"{name} threshold changed")
        if variant.get("release_candidate_eligible") is not expected_release_eligible:
            raise V154MaterializationError(f"{name} release eligibility changed")

    frozen = schedule.get("frozen_algorithm", {})
    if (
        frozen.get("clustering_mode") != "fixed"
        or float(frozen.get("eps")) != 0.60
        or int(frozen.get("min_points")) != 20
        or frozen.get("min_points_includes_self") is not True
        or float(frozen.get("z_min")) != -0.9
    ):
        raise V154MaterializationError("frozen C0 clustering or z policy changed")
    if schedule.get("phase_1_state", {}).get("formal_run_authorized") is not False:
        raise V154MaterializationError("Day1 must not authorize a formal run")
    return {
        "status": "PASS",
        "variant_count": len(variants),
        "allowed_config_diff_paths": list(ALLOWED_CONFIG_DIFF_PATHS),
        "formal_run_authorized": False,
    }


def validate_pre_run_identity(identity, repo_root="."):
    root = Path(repo_root)
    if identity.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise V154MaterializationError("pre-run identity schema mismatch")

    diagnostic = identity["diagnostic_25"]
    formal = identity["formal_100"]
    diagnostic_ids = load_ordered_manifest(root / diagnostic["manifest_path"])
    formal_ids = load_ordered_manifest(root / formal["manifest_path"])
    _validate_manifest_identity(diagnostic, diagnostic_ids, root)
    _validate_manifest_identity(formal, formal_ids, root)
    if len(diagnostic_ids) != 25 or len(formal_ids) != 100:
        raise V154MaterializationError("frozen manifest frame count mismatch")
    if formal.get("dataset_role") != "development_validation_set":
        raise V154MaterializationError("100-frame dataset role changed")
    if formal.get("is_independent_holdout") is not False:
        raise V154MaterializationError("100-frame set must not be called an independent holdout")

    delta = identity["delta_22"]
    delta_ids = delta["ordered_identity_list"]
    if int(delta.get("count", -1)) != 22 or len(delta_ids) != 22:
        raise V154MaterializationError("delta-22 cohort size changed")
    normalized_delta = [[str(frame_id).zfill(6), str(gt_id)] for frame_id, gt_id in delta_ids]
    if len({tuple(item) for item in normalized_delta}) != 22:
        raise V154MaterializationError("delta-22 contains duplicate identities")
    if canonical_identity_sha256(normalized_delta) != delta["ordered_identity_sha256"]:
        raise V154MaterializationError("delta-22 ordered identity hash mismatch")
    if delta.get("runtime_rederivation_allowed") is not False:
        raise V154MaterializationError("delta-22 must not be re-derived at runtime")

    _validate_raw_reference_hash(root, identity["t0_100_reference"], "declaration")
    t0_25 = identity["t0_25_reference"]
    t0_100 = identity["t0_100_reference"]
    if t0_25["historical_fields_available"]["post_intensity_source_point_index_set"] is not False:
        raise V154MaterializationError("Day1 must preserve the known T0 25-frame point-set gap")
    if t0_100["historical_fields_available"]["candidate_identity_set"] is not False:
        raise V154MaterializationError("Day1 must preserve the known T0 100-frame identity gap")
    phase = identity["phase_1_state"]
    if phase.get("t0_exact_reference_complete") is not False:
        raise V154MaterializationError("T0 exact reference cannot be complete on Day1")
    if phase.get("formal_run_authorized") is not False:
        raise V154MaterializationError("Day1 must not authorize formal comparison")
    if phase.get("non_t0_variant_run_allowed") is not False:
        raise V154MaterializationError("non-T0 variants must remain blocked")
    return {
        "status": "PASS",
        "diagnostic_num_frames": len(diagnostic_ids),
        "formal_num_frames": len(formal_ids),
        "delta_22_count": len(normalized_delta),
        "t0_25_reference_status": t0_25["reference_status"],
        "t0_100_reference_status": t0_100["reference_status"],
        "formal_run_authorized": False,
    }


def _validate_manifest_identity(record, frame_ids, root):
    if int(record.get("num_frames", -1)) != len(frame_ids):
        raise V154MaterializationError("manifest declared frame count mismatch")
    if sha256_file(root / record["manifest_path"]) != record["manifest_sha256"]:
        raise V154MaterializationError("canonical manifest SHA-256 mismatch")
    if canonical_identity_sha256(frame_ids) != record["ordered_frame_ids_sha256"]:
        raise V154MaterializationError("ordered frame identity SHA-256 mismatch")
    if "ordered_frame_ids" in record and record["ordered_frame_ids"] != frame_ids:
        raise V154MaterializationError("embedded diagnostic frame order mismatch")


def _validate_raw_reference_hash(root, reference, prefix):
    path_key = f"{prefix}_path"
    hash_key = f"{prefix}_sha256"
    if reference.get(f"{prefix}_hash_mode") != "raw_file_bytes":
        raise V154MaterializationError(f"{prefix} hash mode must be raw_file_bytes")
    if raw_file_sha256(root / reference[path_key]) != reference[hash_key]:
        raise V154MaterializationError(f"{prefix} raw SHA-256 mismatch")


def validate_day1_materialization(schedule_path, identity_path, repo_root="."):
    return {
        "schema_version": "15.4-materialization-day1-validation-v1",
        "schedule": validate_threshold_schedule(load_json(schedule_path)),
        "identity": validate_pre_run_identity(load_json(identity_path), repo_root=repo_root),
        "pre_run_materialization_day1": "COMPLETE",
        "formal_run_authorized": False,
    }
