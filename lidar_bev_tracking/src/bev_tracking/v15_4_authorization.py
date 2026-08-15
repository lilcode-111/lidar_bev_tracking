from pathlib import Path

from bev_tracking.v15_4_materialization import load_json, raw_file_sha256, validate_pre_run_identity, validate_threshold_schedule
from bev_tracking.v15_4_release import validate_release_gate_config


AUTHORIZATION_SCHEMA_VERSION = "15.4-formal-run-authorization-v1"


class V154AuthorizationError(ValueError):
    pass


def build_formal_run_authorization(*, repo_root, commit, working_tree_clean, schedule_path, gate_path, identity_path, reference_registry_path):
    root = Path(repo_root)
    if not working_tree_clean:
        raise V154AuthorizationError("working tree must be clean before authorization")
    if not isinstance(commit, str) or len(commit) != 40:
        raise V154AuthorizationError("formal comparison commit must be a full 40-character SHA")
    schedule = load_json(root / schedule_path)
    gate = load_json(root / gate_path)
    identity = load_json(root / identity_path)
    registry = load_json(root / reference_registry_path)
    validate_threshold_schedule(schedule)
    validate_release_gate_config(gate)
    validate_pre_run_identity(identity, repo_root=root)
    validate_t0_reference_registry(root, registry)
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED",
        "formal_run_authorized": True,
        "formal_comparison_commit": commit,
        "working_tree_clean_at_authorization": True,
        "formal_results_observed_before_authorization": False,
        "source": {
            "threshold_schedule": {"path": schedule_path, "sha256": raw_file_sha256(root / schedule_path)},
            "release_gate": {"path": gate_path, "sha256": raw_file_sha256(root / gate_path)},
            "pre_run_identity": {"path": identity_path, "sha256": raw_file_sha256(root / identity_path)},
            "t0_reference_registry": {"path": reference_registry_path, "sha256": raw_file_sha256(root / reference_registry_path)},
        },
        "data_identity": {
            "diagnostic_manifest_sha256": identity["diagnostic_25"]["manifest_sha256"],
            "formal_manifest_sha256": identity["formal_100"]["manifest_sha256"],
            "delta_22_identity_sha256": identity["delta_22"]["ordered_identity_sha256"],
        },
        "t0_replay_required_before_non_t0_interpretation": True,
        "authorized_variants": ["T0", "T1", "T2", "T_off"],
    }


def validate_formal_run_authorization(authorization, *, current_commit, working_tree_clean, repo_root="."):
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION or authorization.get("status") != "AUTHORIZED":
        raise V154AuthorizationError("formal authorization schema or status mismatch")
    if authorization.get("formal_run_authorized") is not True:
        raise V154AuthorizationError("formal run is not authorized")
    if not working_tree_clean:
        raise V154AuthorizationError("working tree became dirty after authorization")
    if authorization.get("formal_comparison_commit") != current_commit:
        raise V154AuthorizationError("current HEAD differs from formal comparison commit")
    root = Path(repo_root)
    for record in authorization["source"].values():
        if raw_file_sha256(root / record["path"]) != record["sha256"]:
            raise V154AuthorizationError(f"authorized artifact changed: {record['path']}")
    return {"status": "PASS", "formal_comparison_commit": current_commit}


def validate_t0_reference_registry(root, registry):
    root = Path(root)
    if registry.get("schema_version") != "15.4-t0-reference-index-v1" or registry.get("status") != "MATERIALIZED":
        raise V154AuthorizationError("T0 reference registry is not materialized")
    if registry.get("formal_variants_run") != ["T0"] or registry.get("non_t0_results_observed") is not False:
        raise V154AuthorizationError("non-T0 results were observed before authorization")
    required = ("t0_25_reference", "t0_100_reference")
    for name in required:
        record = registry.get("artifacts", {}).get(name)
        if not record or raw_file_sha256(root / record["path"]) != record["sha256"]:
            raise V154AuthorizationError(f"T0 reference hash mismatch: {name}")
