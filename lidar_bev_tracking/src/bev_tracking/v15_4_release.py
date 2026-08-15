import math

from bev_tracking.v15_4_materialization import EXPECTED_VARIANTS


RELEASE_GATE_SCHEMA_VERSION = "15.4-release-gate-v1"
ARTIFACT_SCHEMA_VERSION = "15.4-intensity-filter-ablation-v1"


class V154ReleaseError(ValueError):
    pass


def validate_release_gate_config(config):
    if config.get("schema_version") != RELEASE_GATE_SCHEMA_VERSION:
        raise V154ReleaseError("release gate schema mismatch")
    if config.get("registration_status") != "PRE_REGISTERED":
        raise V154ReleaseError("release gate is not pre-registered")
    if config.get("baseline_variant") != "T0":
        raise V154ReleaseError("release baseline must be T0")
    if config.get("candidate_variants") != ["T1", "T2"]:
        raise V154ReleaseError("only T1 and T2 may enter candidate selection")
    if config.get("diagnostic_only_variants") != ["T_off"]:
        raise V154ReleaseError("T_off must remain diagnostic-only")
    if config.get("non_compensatory") is not True:
        raise V154ReleaseError("release gates must be non-compensatory")
    if config.get("cli_override_allowed") is not False or config.get("environment_override_allowed") is not False:
        raise V154ReleaseError("release gate overrides must remain disabled")
    expected_order = ["Gate 0", "Gate A", "Gate B", "Gate C", "Gate D", "Gate E", "Gate F", "Candidate Qualified"]
    if config.get("execution_order") != expected_order:
        raise V154ReleaseError("release gate execution order changed")
    gates = config.get("gates", {})
    if list(gates) != ["Gate 0", "Gate A", "Gate B", "Gate C", "Gate D", "Gate E", "Gate F", "Gate G"]:
        raise V154ReleaseError("Gate 0/A/B/C/D/E/F/G must all be present in order")
    frozen = {
        ("Gate A", "material_recovery_count_min"): 6,
        ("Gate A", "median_iou_gain_min"): 0.05,
        ("Gate A", "iou_ge_0_25_count_gain_min"): 4,
        ("Gate A", "material_regression_count_max"): 2,
        ("Gate A", "material_recovery_unique_frames_min"): 3,
        ("Gate B", "candidate_net_gain_min"): 2,
        ("Gate B", "tp_0_50_gain_min"): 1,
        ("Gate B", "tp_0_25_gain_min"): 3,
        ("Gate C", "tp_regressed_0_50_max"): 0,
        ("Gate C", "tp_regressed_0_25_max"): 1,
        ("Gate D", "fp_relative_growth_max"): 0.10,
        ("Gate D", "fp_absolute_growth_max"): 50,
        ("Gate D", "fp_when_baseline_zero_max"): 2,
    }
    for (gate, field), expected in frozen.items():
        if gates[gate].get(field) != expected:
            raise V154ReleaseError(f"frozen release threshold changed: {gate}.{field}")
    if gates["Gate G"].get("release_candidate_eligible") is not False:
        raise V154ReleaseError("T_off cannot become release eligible")
    return {"status": "PASS", "schema_version": RELEASE_GATE_SCHEMA_VERSION, "gate_count": 8}


def absolute_relative_delta(baseline, variant):
    baseline = int(baseline)
    variant = int(variant)
    absolute = variant - baseline
    return {
        "baseline": baseline,
        "variant": variant,
        "absolute_delta": absolute,
        "relative_delta": absolute / baseline if baseline else None,
        "relative_status": "applicable" if baseline else "not_applicable",
    }


def evaluate_variant_release_gates(baseline, variant, variant_name, config):
    validate_release_gate_config(config)
    if variant_name not in EXPECTED_VARIANTS:
        raise V154ReleaseError(f"unknown variant: {variant_name}")
    gate_results = {"Gate 0": _evaluate_gate_0(variant, config["gates"]["Gate 0"])}
    experiment_valid = gate_results["Gate 0"]["passed"]
    if not experiment_valid:
        return _evaluation_result(variant_name, gate_results, False, "gate_0_failed")
    if variant_name in ("T0", "T_off"):
        if variant_name == "T_off":
            gate_results["Gate G"] = {
                "passed": True,
                "release_candidate_eligible": False,
                "full_metrics": True,
                "checks": [_check("release_candidate_eligible", False, False, False, "==", False, True)],
            }
        return _evaluation_result(variant_name, gate_results, False, "baseline_or_diagnostic_only")

    gate_results["Gate A"] = _evaluate_gate_a(baseline, variant, config["gates"]["Gate A"])
    gate_results["Gate B"] = _evaluate_gate_b(baseline, variant, config["gates"]["Gate B"])
    gate_results["Gate C"] = _evaluate_gate_c(baseline, variant, config["gates"]["Gate C"])
    gate_results["Gate D"] = _evaluate_gate_d(baseline, variant, config["gates"]["Gate D"])
    gate_results["Gate E"] = _evaluate_gate_e(variant, config["gates"]["Gate E"])
    gate_results["Gate F"] = _evaluate_gate_f(variant, config["gates"]["Gate F"])
    qualified = all(item["passed"] for item in gate_results.values())
    return _evaluation_result(variant_name, gate_results, qualified, "all_hard_gates_passed" if qualified else "hard_gate_failed")


def _evaluate_gate_0(variant, policy):
    checks = []
    invariants = variant.get("validity", {}).get("invariants", {})
    for name in policy["required_invariants"]:
        value = invariants.get(name)
        checks.append(_check(name, True, value, True, "==", value, value is True))
    expected_summary = policy["formal_100"]
    summary = variant.get("validity", {}).get("formal_100", {})
    for name, expected in expected_summary.items():
        value = summary.get(name)
        checks.append(_check(f"formal_100.{name}", expected, value, expected, "==", value, value == expected))
    return _gate(checks)


def _evaluate_gate_a(baseline, variant, policy):
    base = baseline["delta_22"]
    value = variant["delta_22"]
    checks = [
        _ge("delta_22_material_recovery_count", base.get("material_recovery_count", 0), value["material_recovery_count"], policy["material_recovery_count_min"]),
        _ge("delta_22_median_iou_gain_vs_T0", 0.0, value["median_iou_gain_vs_T0"], policy["median_iou_gain_min"]),
        _ge("delta_22_iou_ge_0_25_count_gain", base["iou_ge_0_25_count"], value["iou_ge_0_25_count"] - base["iou_ge_0_25_count"], policy["iou_ge_0_25_count_gain_min"]),
        _le("delta_22_material_regression_count", base.get("material_regression_count", 0), value["material_regression_count"], policy["material_regression_count_max"]),
        _ge("delta_22_material_recovery_unique_frames", 0, value["material_recovery_unique_frames"], policy["material_recovery_unique_frames_min"]),
    ]
    return _gate(checks)


def _evaluate_gate_b(baseline, variant, policy):
    base_metrics = baseline["metrics_by_iou"]
    metrics = variant["metrics_by_iou"]
    candidate_net = variant["candidate"]["improved_gt_count"] - variant["candidate"]["regressed_gt_count"]
    checks = [
        _ge("candidate_improved_minus_regressed", 0, candidate_net, policy["candidate_net_gain_min"]),
        _ge("TP_gain@0.50", base_metrics["0.50"]["tp"], metrics["0.50"]["tp"] - base_metrics["0.50"]["tp"], policy["tp_0_50_gain_min"]),
        _ge("TP_gain@0.25", base_metrics["0.25"]["tp"], metrics["0.25"]["tp"] - base_metrics["0.25"]["tp"], policy["tp_0_25_gain_min"]),
        _ge("F1@0.50", base_metrics["0.50"]["f1"], metrics["0.50"]["f1"], base_metrics["0.50"]["f1"]),
        _ge("F1@0.25", base_metrics["0.25"]["f1"], metrics["0.25"]["f1"], base_metrics["0.25"]["f1"]),
    ]
    return _gate(checks)


def _evaluate_gate_c(baseline, variant, policy):
    positive = baseline["candidate"]["positive_gt_count"]
    candidate_limit = max(policy["candidate_regression_floor_max"], math.floor(policy["candidate_regression_fraction_max"] * positive))
    regressed = variant["regression"]["tp_regressed_gt_count"]
    checks = [
        _le("TP_regressed_GT@0.50", 0, regressed["0.50"], policy["tp_regressed_0_50_max"]),
        _le("TP_regressed_GT@0.25", 0, regressed["0.25"], policy["tp_regressed_0_25_max"]),
        _le("candidate_regressed_GT", positive, variant["candidate"]["regressed_gt_count"], candidate_limit),
    ]
    return _gate(checks)


def _evaluate_gate_d(baseline, variant, policy):
    checks = []
    checks.extend(_growth_checks("FP@0.50", baseline["metrics_by_iou"]["0.50"]["fp"], variant["metrics_by_iou"]["0.50"]["fp"], policy["fp_relative_growth_max"], policy["fp_absolute_growth_max"], policy["fp_when_baseline_zero_max"]))
    checks.extend(_growth_checks("effective_car_detection", baseline["counts"]["effective_car_detection"], variant["counts"]["effective_car_detection"], policy["effective_car_relative_growth_max"], policy["effective_car_absolute_growth_max"]))
    checks.extend(_growth_checks("raw_car_candidate_before_nms", baseline["counts"]["raw_car_candidate_before_nms"], variant["counts"]["raw_car_candidate_before_nms"], policy["raw_car_relative_growth_max"], policy["raw_car_absolute_growth_max"]))
    base_background = baseline["counts"]["strict_background_candidate_after_nms"]
    background_growth = variant["counts"]["strict_background_candidate_after_nms"] - base_background
    background_limit = max(policy["strict_background_growth_floor_max"], math.ceil(policy["strict_background_growth_fraction_max"] * base_background))
    checks.append(_le("strict_background_candidate_growth", base_background, background_growth, background_limit))
    return _gate(checks)


def _evaluate_gate_e(variant, policy):
    checks = []
    for iou in ("0.50", "0.25"):
        gt_count = variant["gain_distribution"][iou]["new_tp_gt_count"]
        frame_count = variant["gain_distribution"][iou]["new_tp_frame_count"]
        checks.append(_ge(f"new_tp_frame_count@{iou}", gt_count, frame_count, min(policy["new_tp_unique_frames_cap"], gt_count)))
    return _gate(checks)


def _evaluate_gate_f(variant, policy):
    value = variant["regression"]["unexplained_count"]
    return _gate([_le("unexplained_stable_path_regression", 0, value, policy["unexplained_stable_path_regression_max"])])


def _growth_checks(metric, baseline, variant, relative_max, absolute_max, zero_max=None):
    delta = absolute_relative_delta(baseline, variant)
    absolute_check = _le(f"{metric}_absolute_growth", baseline, delta["absolute_delta"], zero_max if baseline == 0 and zero_max is not None else absolute_max)
    if baseline == 0:
        relative_check = _check(f"{metric}_relative_growth", baseline, variant, relative_max, "<=", None, True, "not_applicable")
    else:
        relative_check = _le(f"{metric}_relative_growth", baseline, delta["relative_delta"], relative_max)
    return [relative_check, absolute_check]


def _gate(checks):
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def _ge(metric, baseline, value, threshold):
    return _check(metric, baseline, value, threshold, ">=", value, value >= threshold)


def _le(metric, baseline, value, threshold):
    return _check(metric, baseline, value, threshold, "<=", value, value <= threshold)


def _check(metric, baseline, variant, threshold, operator, computed_value, passed, status="applicable"):
    return {"metric": metric, "baseline": baseline, "variant": variant, "threshold": threshold, "operator": operator, "computed_value": computed_value, "status": status, "passed": bool(passed)}


def _evaluation_result(variant_name, gate_results, qualified, reason):
    return {
        "schema_version": RELEASE_GATE_SCHEMA_VERSION,
        "variant": variant_name,
        "experiment_valid": gate_results["Gate 0"]["passed"],
        "release_gate_status": "evaluated" if gate_results["Gate 0"]["passed"] else "not_evaluated",
        "gate_results": gate_results,
        "release_candidate_qualified": bool(qualified),
        "qualification_reason": reason,
    }


def select_release_candidate(candidate_results, selection_metrics, config):
    validate_release_gate_config(config)
    qualified = [name for name in ("T1", "T2") if candidate_results[name]["release_candidate_qualified"]]
    if not qualified:
        return _selection(None, qualified, "no_qualified_candidate", False, None)
    if len(qualified) == 1:
        return _selection(qualified[0], qualified, "only_qualified_candidate", False, None)
    left, right = qualified
    pareto = _pareto_result(selection_metrics[left], selection_metrics[right], config["selection_policy"]["pareto"])
    if pareto["dominant"]:
        return _selection(pareto["dominant"], qualified, "pareto_dominance", False, pareto)
    equivalent = _performance_equivalent(selection_metrics[left], selection_metrics[right], config["selection_policy"]["performance_equivalent"])
    if equivalent:
        selected = _choose_equivalent(left, right, selection_metrics)
        return _selection(selected, qualified, "performance_equivalent_tiebreak", True, pareto)
    selected = _choose_lexicographic(left, right, selection_metrics)
    return _selection(selected, qualified, "non_pareto_non_equivalent_lexicographic", False, pareto)


def _pareto_result(left, right, policy):
    def dominates(a, b):
        comparisons = [a[name] >= b[name] for name in policy["maximize"]] + [a[name] <= b[name] for name in policy["minimize"]]
        strict = [a[name] > b[name] for name in policy["maximize"]] + [a[name] < b[name] for name in policy["minimize"]]
        return all(comparisons) and any(strict)
    left_name, right_name = left["variant"], right["variant"]
    dominant = left_name if dominates(left, right) else right_name if dominates(right, left) else None
    return {"dominant": dominant, "mutually_non_dominating": dominant is None}


def _performance_equivalent(left, right, policy):
    return (
        abs(left["f1_0_50"] - right["f1_0_50"]) <= policy["f1_0_50_abs_max"]
        and abs(left["tp_0_50"] - right["tp_0_50"]) <= policy["tp_0_50_abs_max"]
        and abs(left["f1_0_25"] - right["f1_0_25"]) <= policy["f1_0_25_abs_max"]
        and abs(left["tp_0_25"] - right["tp_0_25"]) <= policy["tp_0_25_abs_max"]
    )


def _choose_equivalent(left, right, metrics):
    def key(name):
        item = metrics[name]
        return (item["fp_0_50"], item["effective_car_detection_count"], item["strict_background_candidate_after_nms"], item["candidate_regressed_gt_count"], item["background_car_candidate_growth"], -item["intensity_min"], 0 if name == "T1" else 1)
    return min((left, right), key=key)


def _choose_lexicographic(left, right, metrics):
    def key(name):
        item = metrics[name]
        return (-item["f1_0_50"], -item["tp_0_50"], -item["f1_0_25"], -item["tp_0_25"], item["fp_0_50"], item["effective_car_detection_growth"], item["strict_background_candidate_after_nms"], item["candidate_regressed_gt_count"], item["background_car_candidate_growth"], -item["intensity_min"], 0 if name == "T1" else 1)
    return min((left, right), key=key)


def _selection(selected, qualified, path, equivalent, pareto):
    return {
        "qualified_release_candidates": qualified,
        "pareto_result": pareto,
        "performance_equivalent": equivalent,
        "selection_path": path,
        "selected_release_candidate": selected,
        "release_goal_achieved": selected is not None,
        "safe_configuration": selected or "T0",
        "selection_reason": path,
    }


def build_intensity_filter_ablation_skeleton(schedule_path, schedule_sha256, release_gate_path, release_gate_sha256, identity):
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "analysis_version": "15.4-phase1-day4",
        "source": {"threshold_schedule_path": schedule_path, "threshold_schedule_sha256": schedule_sha256, "release_gate_path": release_gate_path, "release_gate_sha256": release_gate_sha256, "source_15_3_2_diagnostic_sha256": identity["delta_22"]["source_artifact"]["sha256"]},
        "data_identity": {"diagnostic_manifest_sha256": identity["diagnostic_25"]["manifest_sha256"], "formal_manifest_sha256": identity["formal_100"]["manifest_sha256"], "delta_22_identity_sha256": identity["delta_22"]["ordered_identity_sha256"]},
        "invariants": {name: {"status": "PENDING_FORMAL_RUN"} for name in ("T0_replay", "config_diff", "commit_consistency", "manifest_consistency", "source_point_monotonicity", "metric_valid")},
        "variants": {name: {"status": "NOT_RUN"} for name in EXPECTED_VARIANTS},
        "qualified_release_candidates": [],
        "pareto_result": None,
        "performance_equivalent": None,
        "selection_path": None,
        "selected_release_candidate": None,
        "release_goal_achieved": None,
        "selection_reason": None,
        "formal_results_observed": False,
    }
