from bev_tracking.v15_4_materialization import EXPECTED_VARIANTS, build_effective_config_matrix, validate_t0_replay, validate_threshold_schedule
from bev_tracking.v15_4_materialization import validate_source_point_monotonicity
from bev_tracking.v15_4_audit import build_candidate_regression_audit, build_tp_regression_audit, classify_regression_reasons, extract_monotonicity_universes


FORMAL_PLAN_SCHEMA_VERSION = "15.4-formal-run-plan-v1"


class V154FormalRunError(ValueError):
    pass


def build_formal_run_plan(schedule, base_config, authorization):
    validate_threshold_schedule(schedule)
    if authorization.get("formal_run_authorized") is not True:
        raise V154FormalRunError("formal authorization is required")
    matrix = build_effective_config_matrix(base_config, schedule)
    return {
        "schema_version": FORMAL_PLAN_SCHEMA_VERSION,
        "formal_comparison_commit": authorization["formal_comparison_commit"],
        "execution_order": list(EXPECTED_VARIANTS),
        "t0_replay_gate": {"required_25": True, "required_100": True, "status": "PENDING"},
        "non_t0_interpretation_allowed": False,
        "variants": {
            name: {
                "intensity_min": config["detector"]["intensity_min"],
                "diagnostic_25": "PENDING",
                "formal_100": "PENDING",
                "release_candidate_eligible": schedule["variants"][name]["release_candidate_eligible"],
            }
            for name, config in matrix.items()
        },
        "formal_results_observed": False,
    }


def apply_t0_replay_result(plan, *, replay_25_passed, replay_100_passed):
    result = dict(plan)
    passed = replay_25_passed is True and replay_100_passed is True
    result["t0_replay_gate"] = {"required_25": True, "required_100": True, "replay_25": bool(replay_25_passed), "replay_100": bool(replay_100_passed), "status": "PASS" if passed else "FAIL"}
    result["non_t0_interpretation_allowed"] = passed
    result["experiment_valid"] = passed
    return result


def validate_both_t0_replays(reference_25, replay_25, reference_100, replay_100):
    result_25 = validate_t0_replay(reference_25, replay_25, absolute_tolerance=1e-8)
    result_100 = validate_t0_replay(reference_100, replay_100, absolute_tolerance=1e-8)
    return {
        "schema_version": "15.4-t0-replay-gate-v1",
        "replay_25": result_25,
        "replay_100": result_100,
        "status": "PASS",
        "non_t0_interpretation_allowed": True,
    }


def require_passed_t0_gate(gate):
    if gate.get("schema_version") != "15.4-t0-replay-gate-v1" or gate.get("status") != "PASS":
        raise V154FormalRunError("T0 replay gate must pass before matrix execution")
    if gate.get("non_t0_interpretation_allowed") is not True:
        raise V154FormalRunError("T0 gate does not allow non-T0 interpretation")
    return {"status": "PASS"}


def build_matrix_identity_audit(reports):
    if list(reports) != list(EXPECTED_VARIANTS):
        raise V154FormalRunError("matrix reports must be ordered T0/T1/T2/T_off")
    point_sets = {name: extract_monotonicity_universes(report) for name, report in reports.items()}
    monotonicity = validate_source_point_monotonicity(point_sets)
    comparisons = {}
    for name in ("T1", "T2", "T_off"):
        comparisons[name] = {
            "tp_regression": build_tp_regression_audit(reports["T0"], reports[name]),
            "candidate_regression": build_candidate_regression_audit(reports["T0"], reports[name]),
            "regression_reasons": {
                key: classify_regression_reasons(reports["T0"], reports[name], key)
                for key in ("0.50", "0.25")
            },
        }
    return {
        "schema_version": "15.4-formal-matrix-identity-audit-v1",
        "source_point_monotonicity": monotonicity,
        "comparisons_vs_T0": comparisons,
    }
