import hashlib
import json
from copy import deepcopy
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


def config_differences(config_a, config_b, prefix=""):
    if isinstance(config_a, dict) and isinstance(config_b, dict):
        output = []
        for key in sorted(set(config_a) | set(config_b)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in config_a or key not in config_b:
                output.append(path)
            else:
                output.extend(config_differences(config_a[key], config_b[key], path))
        return output
    return [] if config_a == config_b else [prefix]


def build_effective_config_matrix(base_config, schedule):
    validate_threshold_schedule(schedule)
    matrix = {}
    for variant_name, variant in schedule["variants"].items():
        effective = deepcopy(base_config)
        if not isinstance(effective.get("detector"), dict):
            raise V154MaterializationError("base config must contain detector mapping")
        effective["detector"]["intensity_min"] = float(variant["intensity_min"])
        matrix[variant_name] = effective
    validate_effective_config_matrix(matrix, schedule)
    return matrix


def validate_effective_config_matrix(matrix, schedule):
    validate_threshold_schedule(schedule)
    if list(matrix) != list(EXPECTED_VARIANTS):
        raise V154MaterializationError("effective config variant set changed")
    baseline = matrix["T0"]
    results = {}
    for variant_name, effective in matrix.items():
        expected_intensity = EXPECTED_VARIANTS[variant_name][0]
        if float(effective.get("detector", {}).get("intensity_min")) != expected_intensity:
            raise V154MaterializationError(f"{variant_name} effective intensity mismatch")
        differences = config_differences(baseline, effective)
        expected = [] if variant_name == "T0" else ALLOWED_CONFIG_DIFF_PATHS
        if differences != expected:
            raise V154MaterializationError(
                f"{variant_name} effective config differs outside whitelist: {differences}"
            )
        results[variant_name] = {"status": "PASS", "differences_from_T0": differences}
    return {"status": "PASS", "variants": results}


def validate_source_point_monotonicity(point_sets_by_variant):
    """Validate source identity inclusion; coordinate equality is never consulted."""
    if list(point_sets_by_variant) != list(EXPECTED_VARIANTS):
        raise V154MaterializationError("point-set variants must be T0/T1/T2/T_off in order")
    normalized = {
        variant: _normalize_point_universes(universes)
        for variant, universes in point_sets_by_variant.items()
    }
    universe_names = set(normalized["T0"])
    if any(set(universes) != universe_names for universes in normalized.values()):
        raise V154MaterializationError("point-set universe keys differ across variants")
    comparisons = []
    for left_name, right_name in zip(EXPECTED_VARIANTS, list(EXPECTED_VARIANTS)[1:]):
        for universe_name in sorted(universe_names):
            left = normalized[left_name][universe_name]
            right = normalized[right_name][universe_name]
            missing = sorted(left - right)
            if missing:
                raise V154MaterializationError(
                    f"source-point monotonicity failed: {left_name} not subset of "
                    f"{right_name} for {universe_name}; first missing={missing[0]}"
                )
            comparisons.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "universe": universe_name,
                    "left_count": len(left),
                    "right_count": len(right),
                    "passed": True,
                }
            )
    return {
        "status": "PASS",
        "point_identity": "(frame_id, raw_lidar_point_index)",
        "coordinate_row_dedup_used": False,
        "comparisons": comparisons,
    }


def _normalize_point_universes(universes):
    output = {}
    for name, values in universes.items():
        identities = {(str(frame_id).zfill(6), int(index)) for frame_id, index in values}
        if len(identities) != len(values):
            raise V154MaterializationError(f"duplicate source point identity in {name}")
        output[str(name)] = identities
    return output


def build_t0_reference_artifacts(report, historical_15_3_2, identity, declaration):
    """Build auditable T0-only references from one frozen 100-frame diagnostic replay."""
    expected_frames = identity["formal_100"]["num_frames"]
    if report.get("summary", {}).get("num_frames") != expected_frames:
        raise V154MaterializationError("T0 report does not contain the frozen 100 frames")
    t0_25 = _build_t0_25_reference(report, historical_15_3_2, identity)
    t0_100 = _build_t0_100_reference(report, declaration, identity)
    return t0_25, t0_100


def _build_t0_25_reference(report, historical, identity):
    requested = [tuple(item) for item in identity["delta_22"]["ordered_identity_list"]]
    current = {
        (str(item["frame_id"]).zfill(6), str(item["gt_id"])): item
        for item in report.get("source_point_identity_records", [])
    }
    historical_records = historical["corrected_point_retention_day3"]["delta_records"]
    old = {(str(item["frame_id"]).zfill(6), str(item["gt_id"])): item for item in historical_records}
    if set(current) != set(requested) or not set(requested).issubset(old):
        raise V154MaterializationError("T0 25-frame delta identity mismatch")
    records = []
    for key in requested:
        now = current[key]
        previous = old[key]
        stage = now["stages"]["intensity_filter"]
        oracle = now["post_intensity_diagnostic_pca"]
        if int(stage["count"]) != int(previous["stage_point_counts"]["intensity_filter"]):
            raise V154MaterializationError(f"T0 O3 point count mismatch: {key}")
        old_iou = previous["O3"]["iou"]
        new_iou = oracle["iou"]
        if old_iou is None or new_iou is None or abs(float(old_iou) - float(new_iou)) > 1e-8:
            raise V154MaterializationError(f"T0 O3 IoU mismatch: {key}")
        indices = [int(value) for value in stage["source_point_indices"]]
        records.append(
            {
                "frame_id": key[0],
                "gt_id": key[1],
                "post_intensity_point_count": len(indices),
                "post_intensity_source_point_indices": indices,
                "post_intensity_source_point_indices_sha256": canonical_identity_sha256(indices),
                "O3_iou": float(new_iou),
                "historical_O3_iou": float(old_iou),
                "historical_exact_match": True,
            }
        )
    return {
        "schema_version": "15.4-t0-25-reference-v1",
        "variant": "T0",
        "intensity_min": 0.38,
        "point_identity": "(frame_id, raw_lidar_point_index)",
        "delta_22_identity_sha256": identity["delta_22"]["ordered_identity_sha256"],
        "record_count": len(records),
        "replay_absolute_tolerance": 1e-8,
        "historical_match_status": "PASS",
        "records": records,
    }


def _build_t0_100_reference(report, declaration, identity):
    gt_records = report.get("gt_candidate_records", [])
    candidate_records = report.get("candidate_identity_records", [])
    positive_gt = sorted(
        [[str(item["frame_id"]).zfill(6), str(item["gt_id"])] for item in gt_records]
    )
    tp_by_iou = {
        iou_key: sorted(
            [[str(item["frame_id"]).zfill(6), str(item["gt_id"])] for item in gt_records if item["matched_by_iou"][iou_key]]
        )
        for iou_key in ("0.50", "0.25")
    }
    final_candidates = sorted(
        item["detection_identity"]
        for item in candidate_records
        if item["is_car_candidate_before_nms"] and item["after_nms"]
    )
    metrics = _aggregate_report_metrics(report)
    expected_metrics = declaration["metrics_by_iou"]
    for iou_key in ("0.50", "0.25"):
        for field in ("tp", "fp", "fn", "neutralized_detections"):
            if int(metrics[iou_key][field]) != int(expected_metrics[iou_key][field]):
                raise V154MaterializationError(f"T0 100-frame historical metric mismatch: {iou_key} {field}")
        for field in ("precision", "recall", "f1"):
            if abs(float(metrics[iou_key][field]) - float(expected_metrics[iou_key][field])) > 1e-8:
                raise V154MaterializationError(f"T0 100-frame historical metric mismatch: {iou_key} {field}")
    requested = report["source"]["requested_frame_ids"]
    geometry_failed_frames = int(report.get("summary", {}).get("geometry_failed_frames", 0))
    return {
        "schema_version": "15.4-t0-100-reference-v1",
        "variant": "T0",
        "intensity_min": 0.38,
        "manifest_sha256": identity["formal_100"]["manifest_sha256"],
        "num_frames": len(requested),
        "frame_status": [{"frame_id": str(frame_id).zfill(6), "status": "success"} for frame_id in requested],
        "positive_gt_count": len(positive_gt),
        "positive_gt_identity_sha256": canonical_identity_sha256(positive_gt),
        "tp_identity_by_iou": {
            key: {"count": len(values), "identity_sha256": canonical_identity_sha256(values), "identity_list": values}
            for key, values in tp_by_iou.items()
        },
        "effective_car_candidate_count": len(final_candidates),
        "candidate_identity_sha256": canonical_identity_sha256(final_candidates),
        "candidate_identity_set": final_candidates,
        "metrics_by_iou": metrics,
        "historical_aggregate_match_status": "PASS",
        "geometry_sanity_audit": {
            "role": "non_blocking_diagnostic",
            "failed_frames": geometry_failed_frames,
            "passed_frames": int(len(requested) - geometry_failed_frames),
            "blocks_t0_reference": False,
            "status": "PASS" if geometry_failed_frames == 0 else "OBSERVED_WITH_FAILURES",
        },
    }


def _aggregate_report_metrics(report):
    counts = {
        key: {"tp": 0, "fp": 0, "fn": 0, "neutralized_detections": 0}
        for key in ("0.50", "0.25")
    }
    for frame in report["frames"]:
        metrics_by_iou = frame["failure_evidence"]["summary"]["metrics_by_iou"]
        for key in counts:
            for field in counts[key]:
                counts[key][field] += int(metrics_by_iou[key][field])
    output = {}
    for key, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[key] = {**values, "precision": precision, "recall": recall, "f1": f1}
    return output


def validate_t0_replay(reference, replay, absolute_tolerance=1e-8):
    """Compare a future T0 replay to its materialized reference."""
    mismatches = []
    _compare_values(reference, replay, "", absolute_tolerance, mismatches)
    if mismatches:
        raise V154MaterializationError("T0 replay mismatch: " + ", ".join(mismatches[:10]))
    return {"status": "PASS", "absolute_tolerance": float(absolute_tolerance), "mismatch_count": 0}


def _compare_values(reference, replay, path, tolerance, mismatches):
    if isinstance(reference, dict):
        if not isinstance(replay, dict) or set(reference) != set(replay):
            mismatches.append(path or "root")
            return
        for key in sorted(reference):
            _compare_values(reference[key], replay[key], f"{path}.{key}" if path else key, tolerance, mismatches)
    elif isinstance(reference, list):
        if not isinstance(replay, list) or len(reference) != len(replay):
            mismatches.append(path)
            return
        for index, (left, right) in enumerate(zip(reference, replay)):
            _compare_values(left, right, f"{path}[{index}]", tolerance, mismatches)
    elif isinstance(reference, float):
        if not isinstance(replay, (int, float)) or abs(reference - float(replay)) > tolerance:
            mismatches.append(path)
    elif reference != replay:
        mismatches.append(path)
