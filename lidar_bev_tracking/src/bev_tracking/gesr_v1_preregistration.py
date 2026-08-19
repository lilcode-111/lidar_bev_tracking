"""Frozen GESR-v1 preregistration artifacts; no runtime algorithm lives here."""

import hashlib
import json
from pathlib import Path

from bev_tracking.experiment_gate import sha256_file
from bev_tracking.v15_4_materialization import (
    canonical_identity_sha256,
    load_json,
    raw_file_sha256,
)


PREREGISTRATION_DIR = Path("configs/experiments/v15_5/gesr_v1")
ARTIFACT_FILENAMES = (
    "gesr_v1_evaluation_spec.json",
    "gesr_v1_release_gate.json",
    "T0_T2_replay_contract.json",
    "no_gt_leakage_test_spec.json",
    "comparison_identity.json",
)
TERMINAL_CODES = (
    "ACCEPTED",
    "GEOMETRY_EXTENSION_INVALID",
    "INSIDE_CURRENT_EXTENT",
    "INSUFFICIENT_DIRECT_ANCHORS",
    "NO_VALID_COMPONENT",
)
ASSOCIATION_OUTCOMES = ("SELECTED", "MULTI_COMPONENT_LOST")


class GESRPreregistrationError(ValueError):
    pass


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def artifact_bytes(value):
    return (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def frozen_algorithm_spec():
    return {
        "schema_version": "15.5-gesr-v1-algorithm-spec-snapshot-v1",
        "status": "FROZEN",
        "algorithm_name": "Geometry-Extension Selective Retention v1",
        "short_name": "GESR-v1",
        "new_tunable_parameter_count": 0,
        "input_stage": "post-ROI post-z pre-final-intensity-filter point set",
        "point_identity": "(frame_id, raw_lidar_point_index)",
        "intensity_split": {
            "P_seed": "intensity >= 0.38",
            "P_candidate": "0.15 <= intensity < 0.38",
            "P_discard": "intensity < 0.15",
        },
        "seed_components": {
            "space": "XY",
            "connectivity_radius_m": 0.60,
            "MIN_SEED_COMPONENT_POINTS": 4,
            "component_signature": "SHA-256(sorted source raw_lidar_point_index list)",
            "display_component_id_is_cross_run_identity": False,
        },
        "runtime_geometry": {
            "method": "seed-only 2D PCA",
            "center": "mean(seed xy)",
            "axes": ["major PCA eigenvector", "minor PCA eigenvector"],
            "extent": ["u_min", "u_max", "v_min", "v_max"],
            "validity": ["|S| >= 4", "lambda1 > numerical_epsilon"],
        },
        "candidate_direct_association": {
            "space": "XY",
            "distance_m_max": 0.60,
            "MIN_DIRECT_SEED_ANCHORS": 2,
        },
        "extension": {
            "d_major": "max(u_min-u_p,0,u_p-u_max)",
            "d_minor": "max(v_min-v_p,0,v_p-v_max)",
            "extension_score": "sqrt(d_major^2+d_minor^2)",
            "decision": "extension_score > numerical_epsilon",
        },
        "candidate_eligibility": [
            "VALID_COMPONENT",
            "direct anchors >= 2",
            "outside original PCA extent",
        ],
        "multi_component_arbitration": {
            "evaluate": "all eligible components",
            "association_distance": "mean(nearest direct anchor distance, second-nearest direct anchor distance)",
            "winner": "minimum association_distance",
            "tie_break": "minimum deterministic component identity",
        },
        "expansion": {
            "mode": "SINGLE_PASS",
            "geometry_update": False,
            "candidate_reanchoring": False,
            "iterative_expansion": False,
        },
        "final_point_set": "P_seed UNION P_accept",
        "frozen_downstream": [
            "C0 clustering",
            "classifier",
            "PCA detection box",
            "score",
            "NMS",
            "Evaluation Policy",
        ],
        "candidate_universe": {
            "stage": "post-ROI post-z",
            "intensity": "0.15 <= intensity < 0.38",
            "coordinate_row_identity": False,
            "coordinate_dedup": False,
            "invariants": [
                "GESR accepted source points subset of candidate universe",
                "T2 recovered candidate set equals complete candidate universe",
                "GESR representation equals T0 retained points UNION GESR accepted candidate points",
            ],
        },
        "runtime_failure_handling": "frozen terminal policy is attribution-only after accepted/rejected result",
    }


def terminal_reason_policy():
    return {
        "level": "point",
        "point_terminal_decision_codes": list(TERMINAL_CODES),
        "terminal_reason_precedence": list(TERMINAL_CODES),
        "precedence_semantics": "furthest-valid-progress",
        "exactly_one_terminal_decision_per_candidate": True,
        "attribution_timing": "after runtime accepted/rejected and selected component are fixed",
        "attribution_may_change_runtime_result": False,
        "invariants": [
            "runtime accepted=true iff terminal=ACCEPTED",
            "runtime accepted=false iff terminal!=ACCEPTED",
            "reason attribution ON/OFF preserves accepted point identity set exactly",
        ],
    }


def frozen_evaluation_spec(algorithm_spec_sha256):
    terminal = terminal_reason_policy()
    return {
        "schema_version": "15.5-gesr-v1-evaluation-spec-v1",
        "status": "PRE_REGISTERED",
        "source_precedence": {
            "runtime_algorithm_conflict": "SOT-1 wins",
            "evaluation_metric_gate_conflict": "SOT-2 wins",
            "Gate0_terminal_association_conflict": "SOT-3 supersedes SOT-2",
        },
        "algorithm_spec_binding": {
            "hash_mode": "canonical_json_subobject_sha256",
            "sha256": algorithm_spec_sha256,
        },
        "experiment_structure": {
            "Phase2": "diagnostic 25-frame comparison",
            "Phase3": "formal 100-frame comparison only after Phase2 PASS and GateA PASS",
            "comparison_variants": ["T0", "GESR-v1", "T2"],
        },
        "candidate_evaluation_class": {
            "identity": "(frame_id, raw_lidar_point_index)",
            "exactly_one_primary_class": True,
            "precedence": ["POSITIVE_CAR", "OTHER_ANNOTATED", "ANNOTATION_EXCLUDED_BACKGROUND"],
            "definitions": {
                "POSITIVE_CAR": "inside >=1 positive Car GT oriented 3D box",
                "OTHER_ANNOTATED": "not POSITIVE_CAR but inside another usable annotation box",
                "ANNOTATION_EXCLUDED_BACKGROUND": "inside no usable annotation box",
            },
            "OTHER_ANNOTATED_in_primary_VRR_BRR": False,
        },
        "point_selectivity": {
            "aggregation": "batch_micro",
            "VRR": "accepted POSITIVE_CAR candidate points / all POSITIVE_CAR candidate points",
            "BRR": "accepted ANNOTATION_EXCLUDED_BACKGROUND points / all ANNOTATION_EXCLUDED_BACKGROUND candidate points",
            "primary": {"selectivity_gain": "VRR-BRR"},
            "diagnostic": {"selectivity_ratio": "VRR/BRR"},
            "per_frame_diagnostic_required": True,
            "zero_denominator": {
                "V_candidate=0": {"VRR": None, "GateB": "not_evaluable"},
                "B_candidate=0": {"BRR": None, "background": "empty_universe", "selectivity_gain": None},
            },
            "fabricated_BRR_zero_forbidden": True,
        },
        "Gate0": {
            "formula": "Gate0_Phase2 AND Gate0_Phase3",
            "Gate0_Phase2": {
                "non_compensatory": True,
                "requirements": [
                    "diagnostic_25_manifest_identity PASS",
                    "delta_22_identity PASS",
                    "comparison_commit_identity PASS",
                    "T0_replay_25 PASS",
                    "T2_replay_25 PASS",
                    "candidate_universe_identity PASS",
                    "source_point_identity PASS",
                    "GESR_accepted_subset_of_T2_candidate_universe PASS",
                    "decision_determinism PASS",
                    "component_determinism PASS",
                    "no_GT_runtime_leakage PASS",
                    "algorithm_spec_SHA_binding PASS",
                    "evaluation_spec_SHA_binding PASS",
                    "release_gate_SHA_binding PASS",
                    "new_tunable_parameter_count=0",
                    "no_post_hoc_algorithm_change PASS",
                    "required_Phase2_artifacts_written PASS",
                ],
                "initial_status": "not_evaluated",
                "failure_decision": "EXPERIMENT_INVALID",
            },
            "Phase3_authorization": "Gate0_Phase2 PASS AND GateA PASS",
            "GateA_failure_before_Phase3": "GESR_V1_REJECTED",
            "Gate0_Phase3": {
                "requirements": [
                    "formal_100_manifest_identity PASS",
                    "same_comparison_commit_as_Phase2 PASS",
                    "T0_replay_100 PASS",
                    "T2_replay_100 PASS",
                    "GESR_algorithm_spec_SHA_unchanged PASS",
                    "evaluation_spec_SHA_unchanged PASS",
                    "release_gate_SHA_unchanged PASS",
                    "requested=100",
                    "processed=100",
                    "success=100",
                    "metric_valid=100",
                    "partial_success=0",
                    "skipped=0",
                    "failed=0",
                    "dual_IoU_metrics_complete PASS",
                    "formal_result_artifacts_complete PASS",
                    "formal_result_SHA_complete PASS",
                    "formal_provenance_complete PASS",
                    "no_post_hoc_algorithm_change PASS",
                ],
                "initial_status": "not_evaluated",
                "failure_decision": "formal release experiment invalid",
            },
            "initial_Final_Gate0": "not_evaluated",
            "premature_PASS_or_FAIL_for_Phase3_forbidden": True,
        },
        "material_recovery": "IoU_GESR-IoU_T0 >= +0.10",
        "material_regression": "IoU_GESR-IoU_T0 <= -0.10",
        "point_terminal_decision_policy": terminal,
        "association_arbitration": {
            "level": "candidate x component association",
            "outcomes": list(ASSOCIATION_OUTCOMES),
            "MULTI_COMPONENT_LOST_is_point_reject_reason": False,
            "required_fields": [
                "frame_id",
                "raw_lidar_point_index",
                "component_signature",
                "arbitration_eligible",
                "arbitration_outcome",
            ],
            "eligible": {"outcome": list(ASSOCIATION_OUTCOMES)},
            "not_eligible": {"arbitration_outcome": None, "status": "not_applicable"},
            "failed_anchor_or_geometry_may_be_MULTI_COMPONENT_LOST": False,
            "accepted_candidate_invariant": {
                "eligible_component_count": "k",
                "SELECTED_count": 1,
                "MULTI_COMPONENT_LOST_count": "k-1",
            },
            "rejected_candidate_SELECTED_count": 0,
        },
        "arbitration_metrics": {
            "multi_component_candidate_count": "unique candidate point with candidate_component_count>=2",
            "conflict_arbitration_count": "unique candidate point with arbitration_eligible_component_count>=2",
            "lost_association_count": "candidate-component association with MULTI_COMPONENT_LOST",
        },
        "stable_component_identity": {
            "component_signature": "SHA-256(sorted source raw_lidar_point_index list)",
            "component_id_cross_run_identity": False,
        },
        "registration_state": {
            "results_observed_at_registration": False,
            "Gate0_Phase2": "not_evaluated",
            "Gate0_Phase3": "not_evaluated",
            "Final_Gate0": "not_evaluated",
            "GESR_treatment_metrics_present": False,
        },
    }


def frozen_release_gate(algorithm_sha, evaluation_sha):
    return {
        "schema_version": "15.5-gesr-v1-release-gate-v1",
        "status": "PRE_REGISTERED",
        "algorithm_spec_binding": {"hash_mode": "canonical_json_subobject_sha256", "sha256": algorithm_sha},
        "evaluation_spec_binding": {"hash_mode": "raw_file_sha256", "sha256": evaluation_sha},
        "non_compensatory": True,
        "gates": {
            "GateA": {
                "role": "mechanism_geometry_recovery",
                "delta_22_count": 22,
                "material_recovery_count_min": 6,
                "median_iou_gain_vs_T0_min": 0.05,
                "iou_ge_0_25_count_gain_vs_T0_min": 4,
                "material_regression_count_max": 2,
                "material_recovery_frame_count_min": 3,
                "failure_decision": "GESR_V1_REJECTED",
            },
            "GateB": {
                "role": "selectivity",
                "geometry_recovery_retention_vs_T2_min": 0.60,
                "BRR_max": 0.50,
                "VRR_minus_BRR_min": 0.10,
                "zero_denominator_semantics": "not_evaluable; never fabricate BRR=0",
            },
            "GateC": {
                "role": "background_FP_cost",
                "vs_T0": {
                    "FP@0.50": {"absolute_growth_max": 50, "relative_growth_max": 0.10},
                    "FP@0.25": {"absolute_growth_max": 50, "relative_growth_max": 0.10},
                    "effective_Car_detection": {"absolute_growth_max": 75, "relative_growth_max": 0.15},
                    "raw_Car_candidate_before_NMS": {"absolute_growth_max": 100, "relative_growth_max": 0.20},
                    "strict_background_Car_candidate_after_NMS_growth_max": "max(2,ceil(0.10*B0))",
                },
                "vs_T2": {
                    "FP@0.50_cost_avoidance_min": 0.50,
                    "strict_background_Car_candidate_cost_avoidance_min": 0.50,
                    "cost_retention": "max(0,X_GESR-X_T0)/(X_T2-X_T0)",
                    "cost_avoidance": "1-min(1,cost_retention)",
                    "when_X_T2_le_X_T0": "not_applicable; PASS fabrication forbidden",
                },
            },
            "GateD": {
                "role": "downstream_benefit",
                "TP@0.50_gain_min": 1,
                "TP@0.25_gain_min": 3,
                "F1@0.50_regression_allowed": 0.0,
                "F1@0.25_regression_allowed": 0.0,
                "gain_spread": "if new TP at an IoU >=2, new TP frame count at that IoU >=2",
            },
            "GateE": {
                "role": "regression_safety",
                "T0_TP@0.50_to_GESR_FN_max": 0,
                "T0_TP@0.25_to_GESR_FN_max": 1,
                "candidate_regression_count_max": "max(1,floor(0.05*T0_candidate_positive_GT_count))",
                "material_regression_count_max": 2,
                "unexplained_material_regression_count_max": 0,
                "unexplained_tp_regression_count_max": 0,
                "unexplained_candidate_regression_count_max": 0,
            },
            "GateF": {
                "role": "decision_mapping",
                "validity_failure": "EXPERIMENT_INVALID",
                "RELEASE_CANDIDATE": "A PASS AND B PASS AND C PASS AND D PASS AND E PASS",
                "MECHANISM_VALIDATED_BUT_NOT_RELEASE_READY": "A PASS AND B PASS AND C PASS AND E PASS AND D FAIL AND downstream_attribution_coverage=100% AND unexplained_downstream_count=0",
                "GESR_V1_REFINEMENT_REQUIRED": [
                    "A PASS AND B FAIL",
                    "C FAIL",
                    "E FAIL",
                    "D FAIL primarily attributable to GESR structural issue",
                ],
                "GESR_V1_REJECTED": "GateA FAIL",
            },
        },
        "success_definitions": {
            "mechanism_success": "Gate0 PASS AND GateA PASS AND GateB PASS AND unexplained_material_regression_count=0",
            "GateD_required_for_mechanism_success": False,
            "release_success": "Gate0 PASS AND GateA PASS AND GateB PASS AND GateC PASS AND GateD PASS AND GateE PASS",
        },
        "selection_policy": {
            "allowed_output": ["GESR-v1", None],
            "T2_release_candidate_eligible": False,
            "selected_release_candidate_when_release_success": "GESR-v1",
            "selected_release_candidate_otherwise": None,
        },
        "registration_state": {
            "results_observed_at_registration": False,
            "all_gate_results": "not_evaluated",
            "selected_release_candidate": None,
        },
    }


def frozen_replay_contract(identity, algorithm_sha):
    return {
        "schema_version": "15.5-gesr-v1-T0-T2-replay-contract-v1",
        "status": "PRE_REGISTERED",
        "algorithm_spec_binding": {"hash_mode": "canonical_json_subobject_sha256", "sha256": algorithm_sha},
        "variants": ["T0", "T2"],
        "phases": {"Phase2": 25, "Phase3": 100},
        "reference_source": "frozen 15.4 T0/T2 references",
        "manifest_identity": {
            "25_frame_manifest_sha256": identity["diagnostic_25"]["manifest_sha256"],
            "100_frame_manifest_sha256": identity["formal_100"]["manifest_sha256"],
        },
        "exact_match_fields": [
            "source point identity",
            "filtered point sets",
            "cluster membership/hash",
            "raw detections",
            "Car candidates",
            "NMS outputs",
            "positive GT identity",
            "TP/FP/FN",
            "effective detection",
            "regression identity",
        ],
        "floating_comparison_fields": ["IoU", "PCA box", "Precision", "Recall", "F1"],
        "absolute_tolerance": 1e-8,
        "allowed_differences": ["run_id", "timestamp", "runtime", "artifact path"],
        "failure_semantics": {
            "any_T0_or_T2_replay_failure": "relevant experiment validity FAIL",
            "Phase2": "Gate0_Phase2 FAIL -> EXPERIMENT_INVALID",
            "Phase3": "Gate0_Phase3 FAIL -> formal release experiment invalid",
            "interpret_as_GESR_algorithm_failure": False,
        },
        "results_observed_at_registration": False,
    }


def frozen_no_gt_leakage_spec(algorithm_sha):
    return {
        "schema_version": "15.5-gesr-v1-no-gt-leakage-test-spec-v1",
        "status": "PRE_REGISTERED",
        "algorithm_spec_binding": {"hash_mode": "canonical_json_subobject_sha256", "sha256": algorithm_sha},
        "runtime_core_forbidden_inputs": ["GT", "label", "Evaluation matching", "FailureEvidence", "Oracle"],
        "runtime_core_signature_accepts_GT_or_evaluation": False,
        "metamorphic_cases": [
            {"id": "A", "input": "same point cloud", "GT": "normal GT"},
            {"id": "B", "input": "same point cloud", "GT": "no GT"},
            {"id": "C", "input": "same point cloud", "GT": "GT order changed"},
            {"id": "D", "input": "same point cloud", "GT": "GT content perturbed"},
        ],
        "required_exact_outputs": [
            "accepted source point identity set",
            "seed component signatures",
            "candidate decisions",
            "selected component signatures",
            "runtime accepted/rejected result",
        ],
        "any_difference": {
            "no_GT_runtime_leakage": "FAIL",
            "Gate0_relevant_phase": "FAIL",
            "experiment_validity": False,
        },
        "results_observed_at_registration": False,
        "runtime_implementation_present": False,
    }


def build_preregistration_payloads(repo_root="."):
    root = Path(repo_root)
    identity_path = root / "configs/experiments/v15_4/pre_run_identity.json"
    identity = load_json(identity_path)
    algorithm = frozen_algorithm_spec()
    algorithm_sha = canonical_json_sha256(algorithm)
    evaluation = frozen_evaluation_spec(algorithm_sha)
    evaluation_sha = bytes_sha256(artifact_bytes(evaluation))
    release = frozen_release_gate(algorithm_sha, evaluation_sha)
    release_sha = bytes_sha256(artifact_bytes(release))
    replay = frozen_replay_contract(identity, algorithm_sha)
    replay_sha = bytes_sha256(artifact_bytes(replay))
    no_gt = frozen_no_gt_leakage_spec(algorithm_sha)
    no_gt_sha = bytes_sha256(artifact_bytes(no_gt))
    terminal_sha = canonical_json_sha256(evaluation["point_terminal_decision_policy"])
    base = PREREGISTRATION_DIR.as_posix()
    comparison = {
        "schema_version": "15.5-gesr-v1-comparison-identity-v1",
        "status": "PRE_REGISTERED",
        "registration_role": "final SHA aggregation record",
        "algorithm_spec_snapshot": algorithm,
        "identity_bindings": {
            "25_frame_manifest": {
                "path": identity["diagnostic_25"]["manifest_path"],
                "hash_mode": "canonical_utf8_lf_sha256",
                "sha256": identity["diagnostic_25"]["manifest_sha256"],
                "count": 25,
            },
            "100_frame_manifest": {
                "path": identity["formal_100"]["manifest_path"],
                "hash_mode": "canonical_utf8_lf_sha256",
                "sha256": identity["formal_100"]["manifest_sha256"],
                "count": 100,
            },
            "delta_22": {
                "source_path": "configs/experiments/v15_4/pre_run_identity.json",
                "json_pointer": "/delta_22/ordered_identity_list",
                "hash_mode": "canonical_json_subobject_sha256",
                "sha256": identity["delta_22"]["ordered_identity_sha256"],
                "count": 22,
            },
            "algorithm_spec": {
                "source_path": f"{base}/comparison_identity.json",
                "json_pointer": "/algorithm_spec_snapshot",
                "hash_mode": "canonical_json_subobject_sha256",
                "sha256": algorithm_sha,
            },
            "terminal_reason_policy": {
                "source_path": f"{base}/gesr_v1_evaluation_spec.json",
                "json_pointer": "/point_terminal_decision_policy",
                "hash_mode": "canonical_json_subobject_sha256",
                "sha256": terminal_sha,
            },
        },
        "comparisons": {
            "T0": {"intensity_min": 0.38, "GESR_enabled": False, "role": "safe baseline", "release_candidate_eligible": False},
            "GESR-v1": {"base_intensity_min": 0.38, "candidate_intensity": "0.15 <= intensity < 0.38", "GESR_enabled": True, "role": "treatment", "release_candidate_eligible": True},
            "T2": {"intensity_min": 0.15, "GESR_enabled": False, "role": "uniform recovery reference", "release_candidate_eligible": False},
        },
        "release_selector_allowed_output": ["GESR-v1", None],
        "artifact_registry": {
            "evaluation_spec": {"path": f"{base}/gesr_v1_evaluation_spec.json", "hash_mode": "raw_file_sha256", "sha256": evaluation_sha},
            "release_gate": {"path": f"{base}/gesr_v1_release_gate.json", "hash_mode": "raw_file_sha256", "sha256": release_sha},
            "T0_T2_replay_contract": {"path": f"{base}/T0_T2_replay_contract.json", "hash_mode": "raw_file_sha256", "sha256": replay_sha},
            "no_gt_leakage_test_spec": {"path": f"{base}/no_gt_leakage_test_spec.json", "hash_mode": "raw_file_sha256", "sha256": no_gt_sha},
        },
        "content_sha256": {
            "algorithm_spec_sha256": algorithm_sha,
            "evaluation_spec_sha256": evaluation_sha,
            "release_gate_sha256": release_sha,
            "terminal_reason_policy_sha256": terminal_sha,
        },
        "registration_state": {
            "results_observed_at_registration": False,
            "GESR_runtime_implemented": False,
            "GESR_25_frame_treatment_run": False,
            "GESR_100_frame_treatment_run": False,
            "GESR_formal_result_generated": False,
            "formal_results_modified": False,
            "registered_treatment_result_artifacts": [],
            "algorithm_implementation_authorized": False,
        },
    }
    return {
        "gesr_v1_evaluation_spec.json": evaluation,
        "gesr_v1_release_gate.json": release,
        "T0_T2_replay_contract.json": replay,
        "no_gt_leakage_test_spec.json": no_gt,
        "comparison_identity.json": comparison,
    }


def write_preregistration_artifacts(repo_root="."):
    root = Path(repo_root)
    output = root / PREREGISTRATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    payloads = build_preregistration_payloads(root)
    for filename in ARTIFACT_FILENAMES:
        (output / filename).write_bytes(artifact_bytes(payloads[filename]))
    return validate_preregistration_artifacts(root)


def validate_preregistration_artifacts(repo_root="."):
    root = Path(repo_root)
    directory = root / PREREGISTRATION_DIR
    expected = build_preregistration_payloads(root)
    observed = {}
    artifact_hashes = {}
    for filename in ARTIFACT_FILENAMES:
        path = directory / filename
        if not path.is_file():
            raise GESRPreregistrationError(f"missing preregistration artifact: {path}")
        observed[filename] = load_json(path)
        if observed[filename] != expected[filename]:
            raise GESRPreregistrationError(f"preregistration artifact differs from frozen SOT: {filename}")
        artifact_hashes[filename] = raw_file_sha256(path)

    comparison = observed["comparison_identity.json"]
    evaluation = observed["gesr_v1_evaluation_spec.json"]
    release = observed["gesr_v1_release_gate.json"]
    for name, record in comparison["artifact_registry"].items():
        if record["hash_mode"] != "raw_file_sha256":
            raise GESRPreregistrationError(f"artifact hash mode changed: {name}")
        if raw_file_sha256(root / record["path"]) != record["sha256"]:
            raise GESRPreregistrationError(f"artifact SHA mismatch: {name}")
    identities = comparison["identity_bindings"]
    if sha256_file(root / identities["25_frame_manifest"]["path"]) != identities["25_frame_manifest"]["sha256"]:
        raise GESRPreregistrationError("25-frame manifest SHA mismatch")
    if sha256_file(root / identities["100_frame_manifest"]["path"]) != identities["100_frame_manifest"]["sha256"]:
        raise GESRPreregistrationError("100-frame manifest SHA mismatch")
    pre_run = load_json(root / "configs/experiments/v15_4/pre_run_identity.json")
    if canonical_identity_sha256(pre_run["delta_22"]["ordered_identity_list"]) != identities["delta_22"]["sha256"]:
        raise GESRPreregistrationError("delta-22 identity SHA mismatch")
    algorithm_sha = canonical_json_sha256(comparison["algorithm_spec_snapshot"])
    terminal_sha = canonical_json_sha256(evaluation["point_terminal_decision_policy"])
    if algorithm_sha != comparison["content_sha256"]["algorithm_spec_sha256"]:
        raise GESRPreregistrationError("algorithm spec canonical SHA mismatch")
    if terminal_sha != comparison["content_sha256"]["terminal_reason_policy_sha256"]:
        raise GESRPreregistrationError("terminal reason policy canonical SHA mismatch")
    if release["evaluation_spec_binding"]["sha256"] != artifact_hashes["gesr_v1_evaluation_spec.json"]:
        raise GESRPreregistrationError("release gate evaluation SHA binding mismatch")
    terminal = evaluation["point_terminal_decision_policy"]
    if terminal["point_terminal_decision_codes"] != list(TERMINAL_CODES):
        raise GESRPreregistrationError("point terminal code order changed")
    if terminal["terminal_reason_precedence"] != list(TERMINAL_CODES):
        raise GESRPreregistrationError("terminal precedence changed")
    if evaluation["association_arbitration"]["outcomes"] != list(ASSOCIATION_OUTCOMES):
        raise GESRPreregistrationError("association arbitration outcomes changed")
    gate0 = evaluation["Gate0"]
    if gate0["formula"] != "Gate0_Phase2 AND Gate0_Phase3":
        raise GESRPreregistrationError("Final Gate0 formula changed")
    if any(gate0[field]["initial_status"] != "not_evaluated" for field in ("Gate0_Phase2", "Gate0_Phase3")):
        raise GESRPreregistrationError("Gate0 phase was evaluated at registration")
    state = comparison["registration_state"]
    required_false = (
        "results_observed_at_registration",
        "GESR_runtime_implemented",
        "GESR_25_frame_treatment_run",
        "GESR_100_frame_treatment_run",
        "GESR_formal_result_generated",
        "formal_results_modified",
        "algorithm_implementation_authorized",
    )
    if any(state[field] is not False for field in required_false):
        raise GESRPreregistrationError("registration state contains observed or implemented work")
    if state["registered_treatment_result_artifacts"] != []:
        raise GESRPreregistrationError("treatment result artifact registered before authorization")
    return {
        "schema_version": "15.5-gesr-v1-preregistration-validation-v1",
        "status": "PASS",
        "artifact_raw_sha256": artifact_hashes,
        "25_frame_manifest_sha256": identities["25_frame_manifest"]["sha256"],
        "100_frame_manifest_sha256": identities["100_frame_manifest"]["sha256"],
        "delta_22_identity_sha256": identities["delta_22"]["sha256"],
        "algorithm_spec_sha256": algorithm_sha,
        "evaluation_spec_sha256": artifact_hashes["gesr_v1_evaluation_spec.json"],
        "release_gate_sha256": artifact_hashes["gesr_v1_release_gate.json"],
        "terminal_reason_policy_sha256": terminal_sha,
        "all_schemas_valid": True,
        "all_references_resolved": True,
        "all_SHA_bindings_verified": True,
        "Gate0_structure_valid": True,
        "terminal_association_layers_separated": True,
        "results_observed_at_registration": False,
        "GESR_runtime_implemented": False,
        "GESR_formal_result_generated": False,
        "formal_results_modified": False,
    }
