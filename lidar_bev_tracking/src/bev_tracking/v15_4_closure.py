from pathlib import Path

from bev_tracking.report_writer import atomic_write_json
from bev_tracking.v15_4_materialization import load_json, raw_file_sha256


CLOSURE_EVIDENCE_SCHEMA_VERSION = "15.4-closure-evidence-v1"
CLOSURE_SCHEMA_VERSION = "15.4-closure-v1"


DEFAULT_ARTIFACT_PATHS = {
    "pre_run_identity": "configs/experiments/v15_4/pre_run_identity.json",
    "threshold_schedule": "configs/experiments/v15_4/threshold_schedule.json",
    "release_gate": "configs/experiments/v15_4/v15_4_release_gate.json",
    "formal_authorization": "outputs/intensity_filter_ablation/pre_run/formal_run_authorization.json",
    "t0_replay_gate": "outputs/intensity_filter_ablation/formal/T0/t0_replay_gate.json",
    "t0_25_replay": "outputs/intensity_filter_ablation/formal/T0/t0_25_replay.json",
    "t0_100_replay": "outputs/intensity_filter_ablation/formal/T0/t0_100_replay.json",
    "t0_formal_report": "outputs/intensity_filter_ablation/formal/T0/t0_replay_report.json",
    "t1_formal_report": "outputs/intensity_filter_ablation/formal/T1/formal_report.json",
    "t2_formal_report": "outputs/intensity_filter_ablation/formal/T2/formal_report.json",
    "t_off_formal_report": "outputs/intensity_filter_ablation/formal/T_off/formal_report.json",
    "matrix_identity_audit": "outputs/intensity_filter_ablation/formal/matrix_identity_audit.json",
    "formal_experiment": "outputs/intensity_filter_ablation/formal/v15_4_formal_experiment.json",
}


class V154ClosureError(ValueError):
    pass


def build_v15_4_closure(
    *,
    repo_root=".",
    artifact_paths=None,
    evidence_output="docs/v15_4_closure_evidence.json",
    closure_output="docs/v15_4_closure.json",
):
    root = Path(repo_root)
    paths = dict(DEFAULT_ARTIFACT_PATHS if artifact_paths is None else artifact_paths)
    _require_artifact_ids(paths)
    registry = _build_registry(root, paths)
    identity = load_json(root / paths["pre_run_identity"])
    authorization = load_json(root / paths["formal_authorization"])
    formal = load_json(root / paths["formal_experiment"])

    evidence = {
        "schema_version": CLOSURE_EVIDENCE_SCHEMA_VERSION,
        "status": "CLOSED",
        "evidence_registry": registry,
        "gate0_evidence": _build_gate0_evidence(identity, authorization),
        "formal_results_modified": False,
        "formal_100_rerun": False,
    }
    evidence_path = root / evidence_output
    atomic_write_json(evidence_path, evidence)

    qualified, selected, safe = recompute_selection(formal)
    closure = {
        "schema_version": CLOSURE_SCHEMA_VERSION,
        "status": "CLOSED",
        "canonical_artifacts": {
            "experiment_result": {
                "path": paths["formal_experiment"],
                "sha256": registry["formal_experiment"]["sha256"],
                "formal_comparison_commit": formal["formal_comparison_commit"],
                "role": "canonical experiment result",
            },
            "closure_evidence": {
                "path": Path(evidence_output).as_posix(),
                "sha256": raw_file_sha256(evidence_path),
                "role": "canonical evidence archive",
            },
        },
        "frozen_conclusion": {
            "experiment_valid": _all_experiments_valid(formal),
            "root_cause_intervention_validated": _root_cause_intervention_validated(formal),
            "release_goal_achieved": bool(formal["release_goal_achieved"]),
            "qualified_release_candidates": qualified,
            "selected_release_candidate": selected,
            "safe_configuration": safe,
            "safe_intensity_min": float(identity["t0_100_reference"]["intensity_min"]),
        },
        "product_review": {"status": "PENDING"},
        "formal_results_modified": False,
        "formal_100_rerun": False,
    }
    atomic_write_json(root / closure_output, closure)
    validation = validate_v15_4_closure(
        repo_root=root,
        evidence_path=evidence_output,
        closure_path=closure_output,
    )
    return evidence, closure, validation


def validate_v15_4_closure(
    *,
    repo_root=".",
    evidence_path="docs/v15_4_closure_evidence.json",
    closure_path="docs/v15_4_closure.json",
):
    root = Path(repo_root)
    evidence_file = root / evidence_path
    closure_file = root / closure_path
    evidence = load_json(evidence_file)
    closure = load_json(closure_file)
    if evidence.get("schema_version") != CLOSURE_EVIDENCE_SCHEMA_VERSION:
        raise V154ClosureError("closure evidence schema mismatch")
    if closure.get("schema_version") != CLOSURE_SCHEMA_VERSION:
        raise V154ClosureError("closure schema mismatch")

    registry = evidence.get("evidence_registry", {})
    all_registry_hashes_verified = _verify_registry_hashes(root, registry)
    if not all_registry_hashes_verified:
        raise V154ClosureError(
            "closure validation failed: all_registry_hashes_verified"
        )
    ref_status = _resolve_gate0_evidence(root, registry, evidence.get("gate0_evidence", {}))

    formal_record = registry["formal_experiment"]
    formal = load_json(root / formal_record["path"])
    authorization = load_json(root / registry["formal_authorization"]["path"])
    matrix = load_json(root / registry["matrix_identity_audit"]["path"])
    qualified, selected, safe = recompute_selection(formal)
    conclusion = closure["frozen_conclusion"]
    result_selection_recomputed = (
        qualified == conclusion["qualified_release_candidates"]
        and selected == conclusion["selected_release_candidate"]
        and safe == conclusion["safe_configuration"]
        and qualified == formal["qualified_release_candidates"]
        and selected == formal["selected_release_candidate"]
        and safe == formal["safe_configuration"]
    )
    formal_commit_identity_verified = (
        formal["formal_comparison_commit"]
        == authorization["formal_comparison_commit"]
        == closure["canonical_artifacts"]["experiment_result"]["formal_comparison_commit"]
    )
    source_point_monotonicity_verified = (
        matrix["source_point_monotonicity"]["status"] == "PASS"
        and formal["source_point_monotonicity"] == "PASS"
    )
    evidence_link_verified = (
        closure["canonical_artifacts"]["closure_evidence"]["sha256"]
        == raw_file_sha256(evidence_file)
    )
    result_link_verified = (
        closure["canonical_artifacts"]["experiment_result"]["sha256"]
        == formal_record["sha256"]
    )
    if closure.get("formal_results_modified") is not False:
        raise V154ClosureError("closure must declare formal_results_modified=false")
    if closure.get("formal_100_rerun") is not False:
        raise V154ClosureError("closure must declare formal_100_rerun=false")
    checks = {
        "all_registry_hashes_verified": all_registry_hashes_verified,
        "all_gate0_refs_resolved": ref_status["all_refs_resolved"],
        "all_gate0_fields_resolved": ref_status["all_fields_resolved"],
        "all_gate0_assertions_passed": ref_status["all_assertions_passed"],
        "formal_commit_identity_verified": formal_commit_identity_verified,
        "source_point_monotonicity_verified": source_point_monotonicity_verified,
        "result_selection_recomputed": result_selection_recomputed,
        "evidence_link_verified": evidence_link_verified,
        "result_link_verified": result_link_verified,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V154ClosureError("closure validation failed: " + ", ".join(failed))
    return {
        "schema_version": "15.4-closure-validation-v1",
        "status": "PASS",
        **checks,
        "formal_results_modified": False,
        "formal_100_rerun": False,
    }


def recompute_selection(formal):
    gate_results = formal.get("release_gate_results", {})
    qualified = [
        name
        for name in ("T1", "T2")
        if gate_results.get(name, {}).get("release_candidate_qualified") is True
    ]
    if qualified:
        if qualified != formal.get("qualified_release_candidates"):
            raise V154ClosureError("qualified candidate list cannot be recomputed")
        selected = formal.get("selected_release_candidate")
        if selected not in qualified:
            raise V154ClosureError("selected candidate is not qualified")
    else:
        selected = None
    return qualified, selected, selected or "T0"


def _require_artifact_ids(paths):
    missing = sorted(set(DEFAULT_ARTIFACT_PATHS) - set(paths))
    if missing:
        raise V154ClosureError("artifact path mapping is incomplete: " + ", ".join(missing))


def _build_registry(root, paths):
    registry = {}
    for artifact_id, relative in paths.items():
        path = root / relative
        if not path.is_file():
            raise V154ClosureError(f"required artifact is missing: {relative}")
        registry[artifact_id] = {
            "path": Path(relative).as_posix(),
            "sha256": raw_file_sha256(path),
            "hash_mode": "raw_file_bytes",
        }
    return registry


def _build_gate0_evidence(identity, authorization):
    diagnostic_hash = identity["diagnostic_25"]["manifest_sha256"]
    formal_hash = identity["formal_100"]["manifest_sha256"]
    delta_hash = identity["delta_22"]["ordered_identity_sha256"]
    commit = authorization["formal_comparison_commit"]
    return {
        "diagnostic_manifest_consistency": _all_equal(
            diagnostic_hash,
            ("pre_run_identity", "/diagnostic_25/manifest_sha256"),
            ("formal_authorization", "/data_identity/diagnostic_manifest_sha256"),
        ),
        "formal_manifest_consistency": _all_equal(
            formal_hash,
            ("pre_run_identity", "/formal_100/manifest_sha256"),
            ("formal_authorization", "/data_identity/formal_manifest_sha256"),
        ),
        "delta_22_identity_consistency": _all_equal(
            delta_hash,
            ("pre_run_identity", "/delta_22/ordered_identity_sha256"),
            ("formal_authorization", "/data_identity/delta_22_identity_sha256"),
        ),
        "comparison_commit_consistency": _all_equal(
            commit,
            ("formal_authorization", "/formal_comparison_commit"),
            ("formal_experiment", "/formal_comparison_commit"),
            ("matrix_identity_audit", "/formal_comparison_commit"),
        ),
        "effective_config_diff": {
            "operator": "equals",
            "expected": ["detector.intensity_min"],
            "evidence": [_ref("threshold_schedule", "/allowed_config_diff_paths")],
        },
        "t0_replay_25": {
            "operator": "equals",
            "expected": "PASS",
            "evidence": [_ref("t0_replay_gate", "/replay_25/status")],
        },
        "t0_replay_100": {
            "operator": "equals",
            "expected": "PASS",
            "evidence": [_ref("t0_replay_gate", "/replay_100/status")],
        },
        "dual_iou_complete": {
            "operator": "contains_keys",
            "expected": ["0.50", "0.25"],
            "evidence": [
                _ref("formal_experiment", f"/variant_scalar_summaries/{name}/metrics_by_iou")
                for name in ("T0", "T1", "T2", "T_off")
            ],
        },
        "required_artifacts_written": {
            "operator": "field_exists",
            "expected": True,
            "evidence": [
                _ref("t0_25_replay", "/schema_version"),
                _ref("t0_100_replay", "/schema_version"),
                _ref("t0_formal_report", "/schema_version"),
                _ref("t1_formal_report", "/schema_version"),
                _ref("t2_formal_report", "/schema_version"),
                _ref("t_off_formal_report", "/schema_version"),
                _ref("matrix_identity_audit", "/schema_version"),
                _ref("formal_experiment", "/schema_version"),
            ],
        },
        "source_point_monotonicity": _all_equal(
            "PASS",
            ("matrix_identity_audit", "/source_point_monotonicity/status"),
            ("formal_experiment", "/source_point_monotonicity"),
        ),
    }


def _all_equal(expected, *references):
    return {
        "operator": "all_equal",
        "expected": expected,
        "evidence": [_ref(artifact, field) for artifact, field in references],
    }


def _ref(artifact, field):
    return {"evidence_ref": artifact, "field": field}


def _verify_registry_hashes(root, registry):
    if set(registry) != set(DEFAULT_ARTIFACT_PATHS):
        return False
    return all(
        record.get("hash_mode") == "raw_file_bytes"
        and (root / record["path"]).is_file()
        and raw_file_sha256(root / record["path"]) == record["sha256"]
        for record in registry.values()
    )


def _resolve_gate0_evidence(root, registry, gate0):
    refs_resolved = True
    fields_resolved = True
    assertions_passed = True
    required = {
        "diagnostic_manifest_consistency",
        "formal_manifest_consistency",
        "delta_22_identity_consistency",
        "comparison_commit_consistency",
        "effective_config_diff",
        "t0_replay_25",
        "t0_replay_100",
        "dual_iou_complete",
        "required_artifacts_written",
        "source_point_monotonicity",
    }
    if set(gate0) != required:
        raise V154ClosureError("Gate 0 evidence set is incomplete")
    cache = {}
    for assertion in gate0.values():
        values = []
        for reference in assertion.get("evidence", []):
            artifact_id = reference.get("evidence_ref")
            if artifact_id not in registry:
                refs_resolved = False
                continue
            if artifact_id not in cache:
                cache[artifact_id] = load_json(root / registry[artifact_id]["path"])
            try:
                values.append(_resolve_json_pointer(cache[artifact_id], reference.get("field")))
            except (KeyError, IndexError, TypeError, ValueError):
                fields_resolved = False
        if len(values) != len(assertion.get("evidence", [])):
            assertions_passed = False
            continue
        operator = assertion.get("operator")
        expected = assertion.get("expected")
        if operator in ("equals", "all_equal"):
            passed = all(value == expected for value in values)
        elif operator == "contains_keys":
            passed = all(isinstance(value, dict) and set(expected).issubset(value) for value in values)
        elif operator == "field_exists":
            passed = all(value is not None for value in values)
        else:
            raise V154ClosureError(f"unsupported evidence operator: {operator}")
        assertions_passed = assertions_passed and passed
    return {
        "all_refs_resolved": refs_resolved,
        "all_fields_resolved": fields_resolved,
        "all_assertions_passed": assertions_passed,
    }


def _resolve_json_pointer(document, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("field must be a JSON pointer")
    value = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _all_experiments_valid(formal):
    return all(
        formal["release_gate_results"][name]["experiment_valid"] is True
        for name in ("T1", "T2", "T_off")
    )


def _root_cause_intervention_validated(formal):
    return any(
        formal["release_gate_results"][name]["gate_results"].get("Gate A", {}).get("passed") is True
        for name in ("T1", "T2")
    )
