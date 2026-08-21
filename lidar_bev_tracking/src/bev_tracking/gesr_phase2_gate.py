"""Minimal Gate0 / Gate A analysis for the frozen GESR-v1 Phase-2 run."""

import json
from pathlib import Path
from statistics import median


VARIANTS = ("T0", "T2", "GESR-v1")
EXPECTED_PARAMETERS = {
    "T0": {"intensity_min": 0.38, "gesr_enabled": False},
    "T2": {"intensity_min": 0.15, "gesr_enabled": False},
    "GESR-v1": {"intensity_min": 0.38, "gesr_enabled": True},
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen_phase2_contract(
    identity_path="configs/experiments/v15_4/pre_run_identity.json",
    gate_path="configs/experiments/v15_5/gesr_v1/gesr_v1_release_gate.json",
):
    identity = load_json(identity_path)
    gate = load_json(gate_path)["gates"]["GateA"]
    return {
        "frame_ids": [str(value).zfill(6) for value in identity["diagnostic_25"]["ordered_frame_ids"]],
        "delta22": [
            (str(frame_id).zfill(6), str(gt_id))
            for frame_id, gt_id in identity["delta_22"]["ordered_identity_list"]
        ],
        "gate": gate,
    }


def load_run(run_dir, expected_variant):
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "summary.json")
    config = load_json(run_dir / "config_effective.json")
    records = {}
    invariant_items = []
    for frame_path in sorted((run_dir / "frames").glob("*.json")):
        frame = load_json(frame_path)
        geometry = frame.get("artifacts", {}).get("phase2_delta22_geometry")
        if geometry is not None:
            if geometry.get("variant") != expected_variant:
                raise ValueError(f"variant mismatch in {frame_path}")
            for record in geometry.get("records", []):
                key = (str(record["frame_id"]).zfill(6), str(record["gt_id"]))
                if key in records:
                    raise ValueError(f"duplicate delta-22 record: {key}")
                records[key] = record
        if expected_variant == "GESR-v1":
            runtime = frame.get("artifacts", {}).get("gesr", {}).get("runtime_evidence")
            if runtime is not None:
                invariant_items.extend(runtime.get("invariants", {}).items())
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "config": config,
        "records": records,
        "gesr_invariants": invariant_items,
    }


def close_enough(left, right, tolerance=1e-12):
    return abs(float(left) - float(right)) <= tolerance


def build_gate0(runs, contract):
    expected_frames = contract["frame_ids"]
    expected_identities = set(contract["delta22"])
    commits = {
        run["summary"].get("reproducibility", {}).get("git_commit")
        for run in runs.values()
    }
    checks = {
        "fixed_25_frame_input": all(
            run["summary"].get("reproducibility", {}).get("requested_frame_ids")
            == expected_frames
            for run in runs.values()
        ),
        "same_commit": len(commits) == 1 and None not in commits,
        "working_tree_clean_at_run": all(
            run["summary"].get("reproducibility", {}).get("git_dirty") is False
            for run in runs.values()
        ),
        "all_variants_25_of_25_success": all(
            run["summary"].get("run", {}).get("batch_status") == "success"
            and run["summary"].get("frames", {}).get("requested") == 25
            and run["summary"].get("frames", {}).get("success") == 25
            and run["summary"].get("frames", {}).get("metric_valid") == 25
            and run["summary"].get("frames", {}).get("failed") == 0
            and run["summary"].get("frames", {}).get("skipped") == 0
            for run in runs.values()
        ),
        "variant_parameters_frozen": all(
            run["config"].get("phase2", {}).get("variant") == variant
            and close_enough(
                run["config"].get("detector", {}).get("intensity_min"),
                EXPECTED_PARAMETERS[variant]["intensity_min"],
            )
            and run["config"].get("detector", {}).get("gesr_enabled")
            is EXPECTED_PARAMETERS[variant]["gesr_enabled"]
            for variant, run in runs.items()
        ),
        "baseline_metrics_complete": all(
            set(run["summary"].get("metrics_by_iou", {})) >= {"0.50", "0.25"}
            for run in runs.values()
        ),
        "delta22_identity_complete": all(
            set(run["records"]) == expected_identities for run in runs.values()
        ),
        "source_point_identity_runtime_invariants": bool(runs["GESR-v1"]["gesr_invariants"])
        and all(
            value == 0 if name == "rejected_candidate_SELECTED_count" else value is True
            for name, value in runs["GESR-v1"]["gesr_invariants"]
        ),
    }
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "actual_commit": next(iter(commits)) if len(commits) == 1 else None,
    }


def numeric_gain(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_gate_a(runs, contract):
    rows = []
    for frame_id, gt_id in contract["delta22"]:
        key = (frame_id, gt_id)
        iou_t0 = runs["T0"]["records"].get(key, {}).get("pca_iou")
        iou_t2 = runs["T2"]["records"].get(key, {}).get("pca_iou")
        iou_gesr = runs["GESR-v1"]["records"].get(key, {}).get("pca_iou")
        rows.append(
            {
                "frame_id": frame_id,
                "gt_id": gt_id,
                "IoU_T0": iou_t0,
                "IoU_T2": iou_t2,
                "IoU_GESR": iou_gesr,
                "GESR_minus_T0": numeric_gain(iou_gesr, iou_t0),
                "T2_minus_T0": numeric_gain(iou_t2, iou_t0),
            }
        )

    gains = [row["GESR_minus_T0"] for row in rows if row["GESR_minus_T0"] is not None]
    material_recovery_count = sum(gain >= 0.10 for gain in gains)
    material_regression_count = sum(gain <= -0.10 for gain in gains)
    recovery_frames = {
        row["frame_id"]
        for row in rows
        if row["GESR_minus_T0"] is not None and row["GESR_minus_T0"] >= 0.10
    }
    median_gain = float(median(gains)) if gains else None
    t0_iou_ge_025 = sum(row["IoU_T0"] is not None and row["IoU_T0"] >= 0.25 for row in rows)
    gesr_iou_ge_025 = sum(
        row["IoU_GESR"] is not None and row["IoU_GESR"] >= 0.25 for row in rows
    )
    thresholds = contract["gate"]
    checks = {
        "material_recovery_count": material_recovery_count
        >= int(thresholds["material_recovery_count_min"]),
        "median_iou_gain": median_gain is not None
        and median_gain >= float(thresholds["median_iou_gain_vs_T0_min"]),
        "iou_ge_0_25_count_gain": gesr_iou_ge_025 - t0_iou_ge_025
        >= int(thresholds["iou_ge_0_25_count_gain_vs_T0_min"]),
        "material_regression_count": material_regression_count
        <= int(thresholds["material_regression_count_max"]),
        "material_recovery_frame_count": len(recovery_frames)
        >= int(thresholds["material_recovery_frame_count_min"]),
    }
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "material_recovery_count": material_recovery_count,
            "median_iou_gain": median_gain,
            "T0_iou_ge_0_25_count": t0_iou_ge_025,
            "GESR_iou_ge_0_25_count": gesr_iou_ge_025,
            "iou_ge_0_25_count_gain": gesr_iou_ge_025 - t0_iou_ge_025,
            "material_regression_count": material_regression_count,
            "material_recovery_frame_count": len(recovery_frames),
            "valid_gain_count": len(gains),
        },
        "delta22": rows,
    }


def analyze_phase2_runs(run_dirs, identity_path, gate_path):
    contract = load_frozen_phase2_contract(identity_path, gate_path)
    runs = {variant: load_run(run_dirs[variant], variant) for variant in VARIANTS}
    gate0 = build_gate0(runs, contract)
    gate_a = build_gate_a(runs, contract)
    if gate0["result"] != "PASS":
        gate_a["result"] = "NOT_EVALUATED"
    return {
        "schema_version": "15.5-gesr-v1-phase2-gate-result-v1",
        "actual_commit": gate0["actual_commit"],
        "implementation_change_declaration": {
            "algorithm_semantics_changed": False,
            "algorithm_parameters_changed": False,
            "Evaluation_or_Gate_changed": False,
            "25_frame_manifest_changed": False,
            "scope": "batch exposure, progress, bounded-memory streaming, compact evidence, and minimal delta-22 PCA logging",
        },
        "Gate0_Phase2": gate0,
        "Gate_A": gate_a,
    }
