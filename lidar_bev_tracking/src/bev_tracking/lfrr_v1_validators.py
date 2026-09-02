"""Mandatory LFRR-v1 pre-training hard validators.

This module has no optimizer and cannot execute a training step.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from bev_tracking.fragment_learning_training import M1_PARAMETERS
from bev_tracking.lfrr_v1 import (
    EXPECTED_PARAMETER_COUNT,
    FoldNormalizer,
    LFRRV1Error,
    SmallTargetConditionedDeepSets,
    collate_lfrr_samples,
    load_compact_representation,
    materialize_fold_samples,
    require_torch,
    torch,
)


SCHEMA_VERSION = "lfrr-v1-pretraining-validators-v1"
EXPECTED_M2_PARAMETERS = dict(M1_PARAMETERS)
ATOL = 1.0e-6
RTOL = 1.0e-6


class LFRRValidatorError(LFRRV1Error):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_assignments(output_dir):
    split = json.loads(
        (Path(output_dir) / "fragment_learning_splits.json").read_text(encoding="utf-8")
    )
    rows = sorted(
        split.get("sample_assignments", []), key=lambda row: int(row["sample_row"])
    )
    if len(rows) != 3252:
        raise LFRRValidatorError("frozen split must contain 3252 assignments")
    return rows


def validate_m2_baseline_alignment(output_dir):
    output_dir = Path(output_dir)
    assignments = _load_assignments(output_dir)
    rows = _read_csv(output_dir / "lightgbm_v2_oof_predictions.csv")
    result = json.loads(
        (output_dir / "lightgbm_v2_development_result.json").read_text(encoding="utf-8")
    )
    identities = set()
    checks = {
        "sample_count_3252": len(rows) == len(assignments) == 3252,
        "M2_config_frozen": result.get("M2_parameters") == EXPECTED_M2_PARAMETERS,
        "M2_N0_FP_frozen": int(result.get("N0_FP", {}).get("M2", -1)) == 225,
        "M2_overall_FP_frozen": int(result.get("M2", {}).get("FP", -1)) == 252,
        "M2_P_vs_N1_AP_frozen": np.isclose(
            float(result.get("P_vs_N1", {}).get("M2_AP", np.nan)),
            0.5402350133580981,
            atol=5e-12,
        ),
    }
    row_checks = []
    if checks["sample_count_3252"]:
        for assignment, row in zip(assignments, rows):
            expected = (
                int(assignment["sample_row"]),
                str(assignment["frame_id"]).zfill(6),
                int(assignment["canonical_fragment_identity"]),
                str(assignment["label"]),
                int(assignment["validation_fold"]),
            )
            actual = (
                int(row["sample_row"]), str(row["frame_id"]).zfill(6),
                int(row["canonical_fragment_identity"]), str(row["label"]),
                int(row["validation_fold"]),
            )
            score = float(row["M2_score"])
            prediction = int(row["M2_prediction_at_0_50"])
            row_checks.append(
                actual == expected
                and actual[:3] not in identities
                and np.isfinite(score)
                and prediction == int(score >= 0.50)
            )
            identities.add(actual[:3])
    checks.update({
        "one_to_one_identity": len(identities) == 3252 and all(row_checks),
        "fold_identity_identical": all(row_checks),
        "label_identity_identical": all(row_checks),
        "M2_scores_finite": all(row_checks),
        "M2_prediction_matches_threshold": all(row_checks),
    })
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def validate_normalization(output_dir):
    catalog, targets = load_compact_representation(output_dir)
    folds = {}
    for fold in range(1, 6):
        normalizer = FoldNormalizer.fit(catalog, targets, fold)
        validation_frames = {
            str(row["frame_id"]).zfill(6)
            for row in targets if int(row["validation_fold"]) == fold
        }
        perturbed_catalog = {
            key: ({**row, "base24": [value + 1.0e6 for value in row["base24"]]}
                  if key[0] in validation_frames else row)
            for key, row in catalog.items()
        }
        perturbed_targets = copy.deepcopy(targets)
        for row in perturbed_targets:
            if int(row["validation_fold"]) == fold:
                if row["target_margin"] is not None:
                    row["target_margin"] = float(row["target_margin"]) + 1.0e6
                for relation in row["neighbor_relations"]:
                    relation["delta_u"] = float(relation["delta_u"]) + 1.0e6
                    relation["delta_v"] = float(relation["delta_v"]) - 1.0e6
        other = FoldNormalizer.fit(perturbed_catalog, perturbed_targets, fold)
        checks = {
            "base24_shape_24": normalizer.base_mean.shape == (24,) and normalizer.base_scale.shape == (24,),
            "delta_shape_2": normalizer.delta_mean.shape == (2,) and normalizer.delta_scale.shape == (2,),
            "all_stats_finite": all(np.isfinite(value).all() for value in (
                normalizer.base_mean, normalizer.base_scale,
                normalizer.delta_mean, normalizer.delta_scale,
                np.asarray([normalizer.margin_mean, normalizer.margin_scale]),
            )),
            "all_scales_positive": bool(
                np.all(normalizer.base_scale > 0.0)
                and np.all(normalizer.delta_scale > 0.0)
                and normalizer.margin_scale > 0.0
            ),
            "validation_perturbation_does_not_change_base_stats": bool(
                np.array_equal(normalizer.base_mean, other.base_mean)
                and np.array_equal(normalizer.base_scale, other.base_scale)
            ),
            "validation_perturbation_does_not_change_delta_stats": bool(
                np.array_equal(normalizer.delta_mean, other.delta_mean)
                and np.array_equal(normalizer.delta_scale, other.delta_scale)
            ),
            "validation_perturbation_does_not_change_margin_stats": bool(
                normalizer.margin_mean == other.margin_mean
                and normalizer.margin_scale == other.margin_scale
            ),
            "unique_fragment_weighting": normalizer.evidence["unique_base24_fragment_count"] > 0,
            "training_relation_universe_nonempty": normalizer.evidence["relation_edge_count"] > 0,
        }
        folds[str(fold)] = {
            "result": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "normalization_evidence": normalizer.payload(),
        }
    return {
        "result": "PASS" if all(item["result"] == "PASS" for item in folds.values()) else "FAIL",
        "folds": folds,
    }


def validate_missing_values(output_dir):
    catalog, targets = load_compact_representation(output_dir)
    folds = {}
    for fold in range(1, 6):
        normalizer = FoldNormalizer.fit(catalog, targets, fold)
        training = materialize_fold_samples(
            catalog, targets, normalizer, validation_fold=fold, training=True
        )
        validation = materialize_fold_samples(
            catalog, targets, normalizer, validation_fold=fold, training=False
        )
        missing_targets = [row for row in targets if row["target_margin"] is None]
        missing_maps_to_zero = all(
            normalizer.transform_target(
                catalog[(str(row["frame_id"]).zfill(6), int(row["canonical_fragment_identity"]))],
                None,
            )[-1] == 0.0
            for row in missing_targets
        )
        values_finite = all(
            np.isfinite(item["target"]).all() and np.isfinite(item["neighbors"]).all()
            for item in training + validation
        )
        checks = {
            "valid_margin_count_nonzero": normalizer.evidence["valid_margin_count"] > 0,
            "missing_margin_normalizes_to_exact_zero": bool(missing_maps_to_zero),
            "all_training_inputs_finite": bool(values_finite),
            "all_validation_inputs_finite": bool(values_finite),
            "target_dimension_25": all(item["target"].shape == (25,) for item in training + validation),
            "neighbor_dimension_26": all(item["neighbors"].ndim == 2 and item["neighbors"].shape[1] == 26 for item in training + validation),
        }
        folds[str(fold)] = {
            "result": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "missing_margin_target_count": len(missing_targets),
        }
    return {
        "result": "PASS" if all(item["result"] == "PASS" for item in folds.values()) else "FAIL",
        "folds": folds,
    }


def _model_for_validation():
    require_torch()
    torch.manual_seed(15531)
    torch.use_deterministic_algorithms(True)
    model = SmallTargetConditionedDeepSets().eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise LFRRValidatorError(f"model parameter count changed: {count}")
    return model


def _forward(model, samples):
    batch = collate_lfrr_samples(samples)
    with torch.no_grad():
        logits, pooled = model(
            batch["target"], batch["neighbors"], batch["neighbor_mask"],
            return_pooled=True,
        )
        probabilities = torch.sigmoid(logits)
    return pooled, logits, probabilities


def _representative_samples(output_dir):
    catalog, targets = load_compact_representation(output_dir)
    normalizer = FoldNormalizer.fit(catalog, targets, validation_fold=1)
    samples = materialize_fold_samples(catalog, targets, normalizer)
    by_row = {item["sample_row"]: item for item in samples}
    target_by_row = {int(item["sample_row"]): item for item in targets}
    empty = next(item for item in samples if len(item["neighbors"]) == 0)
    one = next(item for item in samples if len(item["neighbors"]) == 1)
    multi = next(item for item in samples if len(item["neighbors"]) >= 2)
    singleton = next(
        by_row[int(row["sample_row"])]
        for row in targets
        if catalog[(str(row["frame_id"]).zfill(6), int(row["canonical_fragment_identity"]))]["base24"][0] == 0.0
        and row["best_seed_component_runtime_id"] is not None
        and len(row["neighbor_relations"]) == 0
    )
    return {"EMPTY": empty, "ONE_MEMBER": one, "MULTI_MEMBER": multi, "SINGLETON_TARGET": singleton}


def validate_empty_set_batching(output_dir):
    model = _model_for_validation()
    samples = _representative_samples(output_dir)
    ordered = [samples["EMPTY"], samples["ONE_MEMBER"], samples["MULTI_MEMBER"]]
    pooled_batch, logits_batch, probabilities_batch = _forward(model, ordered)
    checks = {}
    for index, sample in enumerate(ordered):
        pooled, logits, probabilities = _forward(model, [sample])
        checks[f"sample_{index}_pooled_equal"] = bool(torch.allclose(pooled_batch[index], pooled[0], atol=ATOL, rtol=RTOL))
        checks[f"sample_{index}_logit_equal"] = bool(torch.allclose(logits_batch[index], logits[0], atol=ATOL, rtol=RTOL))
        checks[f"sample_{index}_probability_equal"] = bool(torch.allclose(probabilities_batch[index], probabilities[0], atol=ATOL, rtol=RTOL))
    checks["empty_pooled_exact_zero"] = bool(torch.equal(pooled_batch[0], torch.zeros_like(pooled_batch[0])))
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "atol": ATOL,
        "rtol": RTOL,
    }


def validate_permutation(output_dir):
    model = _model_for_validation()
    samples = _representative_samples(output_dir)
    generator = np.random.default_rng(15531)
    cases = {}
    for name, sample in samples.items():
        baseline = _forward(model, [sample])
        checks = []
        if len(sample["neighbors"]) == 0:
            repeated = _forward(model, [sample])
            checks.append(
                torch.equal(repeated[0], torch.zeros_like(repeated[0]))
                and torch.allclose(baseline[1], repeated[1], atol=ATOL, rtol=RTOL)
                and torch.allclose(baseline[2], repeated[2], atol=ATOL, rtol=RTOL)
            )
        else:
            for _ in range(20):
                permuted = dict(sample)
                permuted["neighbors"] = sample["neighbors"][generator.permutation(len(sample["neighbors"]))]
                current = _forward(model, [permuted])
                checks.append(
                    torch.allclose(baseline[0], current[0], atol=ATOL, rtol=RTOL)
                    and torch.allclose(baseline[1], current[1], atol=ATOL, rtol=RTOL)
                    and torch.allclose(baseline[2], current[2], atol=ATOL, rtol=RTOL)
                )
        cases[name] = {
            "neighbor_count": len(sample["neighbors"]),
            "comparison_count": len(checks),
            "result": "PASS" if all(checks) else "FAIL",
        }
    return {
        "result": "PASS" if all(item["result"] == "PASS" for item in cases.values()) else "FAIL",
        "cases": cases,
        "atol": ATOL,
        "rtol": RTOL,
    }


def run_pretraining_validators(output_dir):
    """Run all five hard gates without constructing an optimizer."""
    results = {
        "NORMALIZATION_VALIDATOR": validate_normalization(output_dir),
        "MISSING_VALUE_VALIDATOR": validate_missing_values(output_dir),
        "EMPTY_SET_BATCHING_TEST": validate_empty_set_batching(output_dir),
        "PERMUTATION_TEST": validate_permutation(output_dir),
        "M2_BASELINE_ALIGNMENT": validate_m2_baseline_alignment(output_dir),
    }
    readiness = "PASS" if all(item["result"] == "PASS" for item in results.values()) else "FAIL"
    report = {
        "schema_version": SCHEMA_VERSION,
        "validators": results,
        "TRAINING_READINESS": readiness,
        "TRAINING_STARTED": False,
        "OPTIMIZER_STEP_EXECUTED": False,
        "INDEPENDENT_RESULTS_OBSERVED": False,
        "FIXED_100_EXECUTED": False,
    }
    path = Path(output_dir) / "lfrr_v1_pretraining_validators.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report, path
