"""Fixed LightGBM-v2 development confirmation with one multi-seed feature."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import warnings

import numpy as np

from bev_tracking.fragment_learning_failure_analysis import _rebuild_frame_fragments
from bev_tracking.fragment_learning_multiseed_analysis import build_multiseed_record
from bev_tracking.fragment_learning_training import (
    M1_PARAMETERS,
    _binary_metrics,
    _default_m1_factory,
    _git_identity,
    _load_inputs,
)


SCHEMA_VERSION = "learned-fragment-relation-lightgbm-v2-development-v1"
MARGIN_FEATURE = "seed_relation_endpoint_gap_margin"
FROZEN_M1 = {
    "average_precision": 0.1376157174,
    "roc_auc": 0.791156,
    "TP": 29,
    "FP": 294,
    "FN": 27,
    "N0_FP": 268,
    "P_vs_N1_AP": 0.590518,
}


class FragmentLearningV2Error(ValueError):
    pass


def _to_builtin(value):
    """Recursively normalize NumPy scalars before JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    return value


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_frozen_m1(output_dir, assignments):
    output_dir = Path(output_dir)
    result = json.loads(
        (output_dir / "phase1_feasibility_result.json").read_text(encoding="utf-8")
    )
    rows = sorted(
        _read_csv(output_dir / "phase1_oof_predictions.csv"),
        key=lambda row: int(row["sample_row"]),
    )
    if len(rows) != 3252 or len(assignments) != len(rows):
        raise FragmentLearningV2Error("frozen M1 OOF row count changed")
    scores = np.empty(len(rows), dtype=np.float64)
    for index, (row, assignment) in enumerate(zip(rows, assignments)):
        identity = (
            int(row["sample_row"]),
            str(row["frame_id"]).zfill(6),
            int(row["canonical_fragment_identity"]),
            row["label"],
            int(row["validation_fold"]),
        )
        expected = (
            int(assignment["sample_row"]),
            str(assignment["frame_id"]).zfill(6),
            int(assignment["canonical_fragment_identity"]),
            assignment["label"],
            int(assignment["validation_fold"]),
        )
        if identity != expected:
            raise FragmentLearningV2Error(f"M1/sample identity mismatch at row {index}")
        scores[index] = float(row["M1_score"])
    if not np.isfinite(scores).all():
        raise FragmentLearningV2Error("frozen M1 OOF contains non-finite scores")
    checks = {
        "AP": np.isclose(result["M1"]["average_precision"], FROZEN_M1["average_precision"], atol=5e-10),
        "ROC_AUC": np.isclose(result["M1"]["roc_auc"], FROZEN_M1["roc_auc"], atol=5e-7),
        "TP": int(result["M1"]["TP"]) == FROZEN_M1["TP"],
        "FP": int(result["M1"]["FP"]) == FROZEN_M1["FP"],
        "FN": int(result["M1"]["FN"]) == FROZEN_M1["FN"],
        "P_vs_N1_AP": np.isclose(
            result["subsets"]["P_vs_N1"]["M1"]["average_precision"],
            FROZEN_M1["P_vs_N1_AP"],
            atol=5e-7,
        ),
    }
    if not all(checks.values()):
        raise FragmentLearningV2Error(f"frozen M1 baseline mismatch: {checks}")
    return result, rows, scores


def build_margin_values(assignments, data_root, *, progress_callback=None):
    """Replay runtime fragments and compute exactly one GT-free M2 feature."""
    by_frame = defaultdict(list)
    for index, row in enumerate(assignments):
        by_frame[str(row["frame_id"]).zfill(6)].append(
            (index, int(row["canonical_fragment_identity"]))
        )
    margins = np.full(len(assignments), np.nan, dtype=np.float64)
    evidence = [None] * len(assignments)
    for position, frame_id in enumerate(sorted(by_frame), start=1):
        if progress_callback:
            progress_callback(position, len(by_frame), frame_id)
        _, fragments, _, valid_components = _rebuild_frame_fragments(data_root, frame_id)
        for sample_index, identity in by_frame[frame_id]:
            fragment = fragments.get(identity)
            if fragment is None:
                raise FragmentLearningV2Error(
                    f"fragment identity replay failed: {frame_id}/{identity}"
                )
            context = build_multiseed_record(fragment, valid_components)
            best = context["best_relation"]
            second = context["second_best_relation"]
            if second is not None:
                best_id = int(best["seed_component_runtime_id"])
                second_id = int(second["seed_component_runtime_id"])
                if best_id == second_id:
                    raise FragmentLearningV2Error("best and second-best seed must be distinct")
                margin = float(context["second_minus_best_endpoint_gap"])
                if not np.isfinite(margin) or margin < 0.0:
                    raise FragmentLearningV2Error("margin must be finite and non-negative")
                margins[sample_index] = margin
            evidence[sample_index] = {
                "sample_row": int(assignments[sample_index]["sample_row"]),
                "frame_id": frame_id,
                "canonical_fragment_identity": identity,
                MARGIN_FEATURE: None if second is None else float(margins[sample_index]),
                "margin_valid": bool(second is not None),
                "best_seed_component_runtime_id": (
                    None if best is None else int(best["seed_component_runtime_id"])
                ),
                "second_seed_component_runtime_id": (
                    None if second is None else int(second["seed_component_runtime_id"])
                ),
            }
    if any(item is None for item in evidence):
        raise FragmentLearningV2Error("margin feature coverage incomplete")
    return margins, evidence


def _margin_coverage(margins, raw_labels):
    output = {}
    for label in ("ALL", "POSITIVE", "N1", "N0"):
        mask = np.ones(len(margins), dtype=bool) if label == "ALL" else raw_labels == label
        valid = int(np.isfinite(margins[mask]).sum())
        total = int(mask.sum())
        output[label] = {
            "sample_count": total,
            "valid": valid,
            "missing": total - valid,
            "valid_rate": 0.0 if total == 0 else float(valid / total),
        }
    return output


def _per_frame_comparison(frames, raw_labels, y, m1_scores, m2_scores):
    rows = []
    for frame_id in sorted(set(frames.tolist())):
        frame = frames == frame_id
        n0 = frame & (raw_labels == "N0")
        positive = frame & (y == 1)
        m1_fp = int(np.sum(m1_scores[n0] >= 0.50))
        m2_fp = int(np.sum(m2_scores[n0] >= 0.50))
        rows.append({
            "frame_id": frame_id,
            "positive_count": int(positive.sum()),
            "M1_N0_FP": m1_fp,
            "M2_N0_FP": m2_fp,
            "Delta_N0_FP": m2_fp - m1_fp,
        })
    return rows


def _interpret(gates, delta_ap, n0_reduction, m2_tp):
    if all(item["passed"] for item in gates.values()):
        return "PASS"
    if m2_tp < 27:
        return "NOT_SUPPORTED"
    if delta_ap <= 0.0 and n0_reduction < 14:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def train_fragment_learning_v2(
    output_dir,
    data_root,
    *,
    m2_factory=None,
    progress_callback=None,
):
    """Run the single authorized M2 5-fold grouped-OOF confirmation."""
    (
        summary, split, prerequisites, assignments, X_base, y, raw_labels, folds, frames
    ) = _load_inputs(output_dir)
    m1_result, m1_rows, m1_scores = _load_frozen_m1(output_dir, assignments)
    margins, margin_evidence = build_margin_values(
        assignments, data_root, progress_callback=progress_callback
    )
    X_m2 = np.column_stack((X_base, margins))
    if X_m2.shape != (3252, X_base.shape[1] + 1):
        raise FragmentLearningV2Error("M2 feature matrix identity changed")

    m2_factory = m2_factory or _default_m1_factory
    m2_scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    for fold_id in range(1, 6):
        validation = folds == fold_id
        training = ~validation
        positive_train = int(y[training].sum())
        negative_train = int(training.sum() - positive_train)
        model = m2_factory(negative_train / positive_train)
        model.fit(X_m2[training], y[training])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            m2_scores[validation] = model.predict_proba(X_m2[validation])[:, 1]
        m1_metrics = _binary_metrics(y[validation], m1_scores[validation])
        m2_metrics = _binary_metrics(y[validation], m2_scores[validation])
        fold_rows.append({
            "fold_id": fold_id,
            "validation_frames": sorted(set(frames[validation].tolist())),
            "M1_AP": m1_metrics["average_precision"],
            "M2_AP": m2_metrics["average_precision"],
            "Delta_AP": m2_metrics["average_precision"] - m1_metrics["average_precision"],
        })
    if not np.isfinite(m2_scores).all():
        raise FragmentLearningV2Error("M2 OOF coverage incomplete or non-finite")

    m1 = _binary_metrics(y, m1_scores)
    m2 = _binary_metrics(y, m2_scores)
    p_n1 = (raw_labels == "POSITIVE") | (raw_labels == "N1")
    m2_p_n1 = _binary_metrics(y[p_n1], m2_scores[p_n1])
    n0 = raw_labels == "N0"
    m1_n0_fp = int(np.sum(m1_scores[n0] >= 0.50))
    m2_n0_fp = int(np.sum(m2_scores[n0] >= 0.50))
    if m1_n0_fp != FROZEN_M1["N0_FP"]:
        raise FragmentLearningV2Error("frozen M1 N0-FP changed")
    reduction = m1_n0_fp - m2_n0_fp
    delta_ap = m2["average_precision"] - m1["average_precision"]
    fold_deltas = [row["Delta_AP"] for row in fold_rows]
    fold_noninferior = sum(value >= 0.0 for value in fold_deltas)

    identity_checks = {
        "64_frames_unchanged": len(set(frames.tolist())) == 64,
        "3252_samples_unchanged": len(y) == 3252,
        "same_labels": Counter(raw_labels.tolist()) == Counter({"POSITIVE": 56, "N1": 121, "N0": 3075}),
        "same_folds": len(folds) == 3252 and set(folds.tolist()) == {1, 2, 3, 4, 5},
        "same_base24_features": X_base.shape[1] == 24,
        "exactly_one_new_feature": X_m2.shape[1] - X_base.shape[1] == 1,
        "same_LightGBM_config": M1_PARAMETERS == m1_result["M1_parameters"],
        "3252_OOF_predictions_exactly_once": len(m2_scores) == 3252 and np.isfinite(m2_scores).all(),
        "same_sample_identity_as_M1": len(m1_rows) == len(assignments),
    }
    gates = {
        "V2-0_Identity": {"passed": all(identity_checks.values()), "checks": identity_checks},
        "V2-A_Ranking": {"passed": delta_ap >= 0.010, "value": delta_ap, "threshold": 0.010},
        "V2-B_N0_FP_Suppression": {"passed": m2_n0_fp <= 254, "value": m2_n0_fp, "threshold": "<= 254"},
        "V2-C_Positive_Safety": {"passed": m2["TP"] >= 27, "value": m2["TP"], "threshold": ">= 27"},
        "V2-D_Operating_Point": {
            "passed": m2["precision_at_0_50"] > m1["precision_at_0_50"] and m2["f1_at_0_50"] >= m1["f1_at_0_50"],
            "precision_improved": m2["precision_at_0_50"] > m1["precision_at_0_50"],
            "F1_noninferior": m2["f1_at_0_50"] >= m1["f1_at_0_50"],
        },
        "V2-E_N1_Safety": {"passed": m2_p_n1["average_precision"] >= 0.580518, "value": m2_p_n1["average_precision"], "threshold": ">= 0.580518"},
        "V2-F_Fold_Stability": {
            "passed": fold_noninferior >= 3 and float(np.median(fold_deltas)) > 0.0,
            "M2_AP_ge_M1_AP_fold_count": fold_noninferior,
            "median_fold_Delta_AP": float(np.median(fold_deltas)),
        },
    }
    conclusion = _interpret(gates, delta_ap, reduction, m2["TP"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "execution_identity": _git_identity(Path.cwd()),
        "dataset": "fragment_learning_dev_v1",
        "training_sample_count": len(y),
        "base_feature_fields": m1_result["feature_fields"],
        "new_feature_fields": [MARGIN_FEATURE],
        "M2_parameters": M1_PARAMETERS,
        "M1": m1,
        "M2": m2,
        "Delta_AP": delta_ap,
        "N0_FP": {
            "M1": m1_n0_fp,
            "M2": m2_n0_fp,
            "absolute_reduction": reduction,
            "relative_reduction": float(reduction / m1_n0_fp),
        },
        "P_vs_N1": {
            "M1_AP": FROZEN_M1["P_vs_N1_AP"],
            "M2_AP": m2_p_n1["average_precision"],
            "Delta_AP": m2_p_n1["average_precision"] - FROZEN_M1["P_vs_N1_AP"],
        },
        "folds": fold_rows,
        "M2_AP_ge_M1_AP_fold_count": fold_noninferior,
        "median_fold_Delta_AP": float(np.median(fold_deltas)),
        "margin_coverage": _margin_coverage(margins, raw_labels),
        "per_frame": _per_frame_comparison(frames, raw_labels, y, m1_scores, m2_scores),
        "Development_Gates": gates,
        "M2_DEVELOPMENT_CONFIRMATION": conclusion,
        "MULTI_SEED_CONTEXT_INCREMENTAL_VALUE": "SUPPORTED" if conclusion == "PASS" else ("NOT_SUPPORTED" if conclusion == "NOT_SUPPORTED" else "INCONCLUSIVE"),
        "M1_RETRAINED": False,
        "DATASET_REGENERATED": False,
        "SPLIT_REGENERATED": False,
        "FORMAL_100_EXECUTED": False,
    }

    output_dir = Path(output_dir)
    result_path = output_dir / "lightgbm_v2_development_result.json"
    oof_path = output_dir / "lightgbm_v2_oof_predictions.csv"
    margin_path = output_dir / "lightgbm_v2_margin_feature.csv"
    result = _to_builtin(result)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "sample_row", "frame_id", "canonical_fragment_identity", "label",
            "validation_fold", "M1_score", "M2_score",
            "M1_prediction_at_0_50", "M2_prediction_at_0_50",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, m1_score, m2_score in zip(assignments, m1_scores, m2_scores):
            writer.writerow({
                **{field: row[field] for field in fields[:5]},
                "M1_score": float(m1_score), "M2_score": float(m2_score),
                "M1_prediction_at_0_50": int(m1_score >= 0.50),
                "M2_prediction_at_0_50": int(m2_score >= 0.50),
            })
    with margin_path.open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(margin_evidence[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(margin_evidence)
    return result, result_path, oof_path, margin_path
