"""Frozen Phase-1 grouped-OOF feasibility training for fragment learning."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from bev_tracking.fragment_learning_dataset import MODEL_FEATURE_FIELDS


TRAINING_SCHEMA_VERSION = "learned-fragment-relation-phase1-v1"
RANDOM_STATE = 15531
POSITIVE_LABEL = "POSITIVE"
NEGATIVE_LABELS = {"N1", "N0"}
FRAGMENT_TYPE_ENCODING = {"SINGLETON": 0.0, "STRUCTURED": 1.0}
EXPECTED_LABEL_COUNTS = {"POSITIVE": 56, "N1": 121, "N0": 3075}

M1_PARAMETERS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "n_estimators": 80,
    "learning_rate": 0.05,
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "max_bin": 63,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
L0_PARAMETERS = {
    "C": 1.0,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}


class FragmentLearningTrainingError(ValueError):
    pass


def _binary_metrics(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y_true) == 0 or len(np.unique(y_true)) != 2:
        raise FragmentLearningTrainingError("metric subset must contain both classes")
    predicted = (scores >= 0.50).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "sample_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "positive_prevalence": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision_at_0_50": float(precision),
        "recall_at_0_50": float(recall),
        "f1_at_0_50": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def _load_inputs(output_dir):
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    split = json.loads(
        (output_dir / "fragment_learning_splits.json").read_text(encoding="utf-8")
    )
    prerequisites = {
        "DATASET_SUFFICIENCY": summary.get("DATASET_SUFFICIENCY", {}).get("result") == "PASS",
        "FEATURE_LEAKAGE_GATE": summary.get("FEATURE_LEAKAGE_GATE", {}).get("result") == "PASS",
        "SPLIT_VALIDITY": split.get("SPLIT_VALIDITY") == "PASS",
    }
    if not all(prerequisites.values()):
        raise FragmentLearningTrainingError(f"training prerequisites failed: {prerequisites}")
    actual_counts = summary.get("label_counts", {})
    if any(int(actual_counts.get(label, -1)) != count for label, count in EXPECTED_LABEL_COUNTS.items()):
        raise FragmentLearningTrainingError("frozen training label counts changed")
    if int(summary.get("training_row_count", -1)) != 3252:
        raise FragmentLearningTrainingError("frozen training sample count changed")
    expected_split = {
        "group": "frame_id", "n_splits": 5, "shuffle": True,
        "random_state": RANDOM_STATE,
    }
    if split.get("parameters") != expected_split or int(split.get("training_sample_count", -1)) != 3252:
        raise FragmentLearningTrainingError("frozen split identity changed")

    with (output_dir / "X_model.csv").open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle))
    if tuple(feature_rows[0]) != MODEL_FEATURE_FIELDS:
        raise FragmentLearningTrainingError("frozen feature schema mismatch")
    assignments = sorted(split["sample_assignments"], key=lambda row: int(row["sample_row"]))
    if len(feature_rows) != len(assignments):
        raise FragmentLearningTrainingError("X_model/split row count mismatch")

    matrix = np.empty((len(feature_rows), len(MODEL_FEATURE_FIELDS)), dtype=np.float64)
    for row_index, row in enumerate(feature_rows):
        for column_index, field in enumerate(MODEL_FEATURE_FIELDS):
            if field == "fragment_type":
                try:
                    value = FRAGMENT_TYPE_ENCODING[row[field]]
                except KeyError as exc:
                    raise FragmentLearningTrainingError(
                        f"unknown fragment_type: {row[field]}"
                    ) from exc
            else:
                value = float(row[field])
            matrix[row_index, column_index] = value
    if not np.isfinite(matrix).all():
        raise FragmentLearningTrainingError("non-finite model input")

    labels = np.asarray(
        [1 if row["label"] == POSITIVE_LABEL else 0 for row in assignments],
        dtype=np.int64,
    )
    raw_labels = np.asarray([row["label"] for row in assignments], dtype=object)
    if any(label not in ({POSITIVE_LABEL} | NEGATIVE_LABELS) for label in raw_labels):
        raise FragmentLearningTrainingError("unexpected training label")
    folds = np.asarray([int(row["validation_fold"]) for row in assignments], dtype=np.int64)
    frames = np.asarray([str(row["frame_id"]).zfill(6) for row in assignments], dtype=object)
    return summary, split, prerequisites, assignments, matrix, labels, raw_labels, folds, frames


def _git_identity(repo_root):
    def run(*args):
        completed = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    try:
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "working_tree_clean": run("status", "--porcelain") == "",
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FragmentLearningTrainingError("unable to resolve Git execution identity") from exc


def _default_m1_factory(scale_pos_weight):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise FragmentLearningTrainingError(
            "LightGBM is required; install requirements.txt in the active environment"
        ) from exc
    return LGBMClassifier(scale_pos_weight=scale_pos_weight, verbosity=-1, **M1_PARAMETERS)


def _per_frame_records(frames, raw_labels, y_true, scores):
    records = []
    for frame_id in sorted(set(frames.tolist())):
        mask = frames == frame_id
        frame_y = y_true[mask]
        frame_scores = scores[mask]
        predicted = (frame_scores >= 0.50).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(frame_y, predicted, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            frame_y, predicted, average="binary", zero_division=0
        )
        counts = Counter(raw_labels[mask].tolist())
        records.append(
            {
                "frame_id": frame_id,
                "sample_count": int(mask.sum()),
                "positive_count": int(counts["POSITIVE"]),
                "N1_count": int(counts["N1"]),
                "N0_count": int(counts["N0"]),
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "precision_at_0_50": float(precision),
                "recall_at_0_50": float(recall),
                "f1_at_0_50": float(f1),
            }
        )
    return records


def _interpret_gates(gates):
    if all(item["passed"] for item in gates.values()):
        return "SUPPORTED_FOR_RUNTIME_PROTOTYPE"
    if gates["M1_OOF_AP"]["passed"] and gates["M1_AP_MINUS_PREVALENCE"]["passed"]:
        return "INCONCLUSIVE"
    return "NOT_SUPPORTED_BY_CURRENT_REPRESENTATION"


def train_fragment_learning_phase1(output_dir, *, m1_factory=None, l0_factory=None):
    """Execute exactly one frozen 5-fold OOF evaluation and write compact results."""
    (
        summary, split, prerequisites, assignments, X, y, raw_labels, folds, frames
    ) = _load_inputs(output_dir)
    m1_factory = m1_factory or _default_m1_factory
    l0_factory = l0_factory or (lambda: LogisticRegression(**L0_PARAMETERS))
    m1_scores = np.full(len(y), np.nan, dtype=np.float64)
    l0_scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_records = []

    for fold_id in range(1, 6):
        validation = folds == fold_id
        training = ~validation
        positive_train = int(y[training].sum())
        negative_train = int(training.sum() - positive_train)
        if positive_train == 0 or len(np.unique(y[validation])) != 2:
            raise FragmentLearningTrainingError(f"invalid class support in fold {fold_id}")
        scale_pos_weight = negative_train / positive_train

        m1 = m1_factory(scale_pos_weight)
        m1.fit(X[training], y[training])
        m1_scores[validation] = m1.predict_proba(X[validation])[:, 1]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[training])
        X_validation_scaled = scaler.transform(X[validation])
        l0 = l0_factory()
        l0.fit(X_train_scaled, y[training])
        l0_scores[validation] = l0.predict_proba(X_validation_scaled)[:, 1]

        m1_metrics = _binary_metrics(y[validation], m1_scores[validation])
        l0_metrics = _binary_metrics(y[validation], l0_scores[validation])
        counts = Counter(raw_labels[validation].tolist())
        fold_records.append(
            {
                "fold_id": fold_id,
                "validation_frames": sorted(set(frames[validation].tolist())),
                "validation_frame_count": len(set(frames[validation].tolist())),
                "sample_count": int(validation.sum()),
                "P": int(counts["POSITIVE"]), "N1": int(counts["N1"]), "N0": int(counts["N0"]),
                "positive_support_frames": len(set(frames[validation & (y == 1)].tolist())),
                "scale_pos_weight": float(scale_pos_weight),
                "M1": m1_metrics,
                "L0": l0_metrics,
                "M1_AP_minus_L0_AP": m1_metrics["average_precision"] - l0_metrics["average_precision"],
                "M1_AP_minus_prevalence": m1_metrics["average_precision"] - m1_metrics["positive_prevalence"],
            }
        )

    if not np.isfinite(m1_scores).all() or not np.isfinite(l0_scores).all():
        raise FragmentLearningTrainingError("OOF coverage incomplete or non-finite")
    m1_overall = _binary_metrics(y, m1_scores)
    l0_overall = _binary_metrics(y, l0_scores)
    subset_results = {}
    for name, negative_label in (("P_vs_N0", "N0"), ("P_vs_N1", "N1")):
        mask = (raw_labels == POSITIVE_LABEL) | (raw_labels == negative_label)
        subset_results[name] = {
            "M1": _binary_metrics(y[mask], m1_scores[mask]),
            "L0": _binary_metrics(y[mask], l0_scores[mask]),
        }
    p_n1 = subset_results["P_vs_N1"]["M1"]
    p_n1["AP_lift_over_prevalence"] = p_n1["average_precision"] - p_n1["positive_prevalence"]

    m1_fold_wins = sum(row["M1"]["average_precision"] > row["L0"]["average_precision"] for row in fold_records)
    m1_above_fold_prevalence = sum(row["M1"]["average_precision"] > row["M1"]["positive_prevalence"] for row in fold_records)
    prevalence = m1_overall["positive_prevalence"]
    gates = {
        "prerequisites": {"passed": all(prerequisites.values()), "details": prerequisites},
        "M1_OOF_AP": {"value": m1_overall["average_precision"], "threshold": 0.25, "passed": m1_overall["average_precision"] >= 0.25},
        "M1_AP_MINUS_PREVALENCE": {"value": m1_overall["average_precision"] - prevalence, "threshold": 0.15, "passed": m1_overall["average_precision"] - prevalence >= 0.15},
        "M1_AP_MINUS_L0_AP": {"value": m1_overall["average_precision"] - l0_overall["average_precision"], "threshold": 0.05, "passed": m1_overall["average_precision"] - l0_overall["average_precision"] >= 0.05},
        "cross_fold_stability": {"M1_over_L0_fold_count": m1_fold_wins, "M1_over_prevalence_fold_count": m1_above_fold_prevalence, "threshold": "both >= 4/5", "passed": m1_fold_wins >= 4 and m1_above_fold_prevalence >= 4},
        "P_vs_N1_AP_lift": {"value": p_n1["AP_lift_over_prevalence"], "threshold": 0.10, "passed": p_n1["AP_lift_over_prevalence"] >= 0.10},
    }
    per_frame = _per_frame_records(frames, raw_labels, y, m1_scores)
    positive_frames = [row for row in per_frame if row["positive_count"] > 0]
    frame_summary = {
        "positive_containing_validation_frames": len(positive_frames),
        "frames_with_at_least_one_correctly_predicted_positive": sum(row["TP"] >= 1 for row in positive_frames),
        "frames_with_zero_recovered_positive": sum(row["TP"] == 0 for row in positive_frames),
        "per_frame_FP_count": {row["frame_id"]: row["FP"] for row in per_frame},
    }
    result = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "execution_identity": _git_identity(Path.cwd()),
        "dataset": "fragment_learning_dev_v1",
        "training_sample_count": len(y),
        "feature_fields": list(MODEL_FEATURE_FIELDS),
        "fragment_type_encoding": FRAGMENT_TYPE_ENCODING,
        "split_parameters": split["parameters"],
        "M1_parameters": M1_PARAMETERS,
        "L0_parameters": L0_PARAMETERS,
        "positive_prevalence": prevalence,
        "M1": m1_overall,
        "L0": l0_overall,
        "M1_AP_minus_prevalence": m1_overall["average_precision"] - prevalence,
        "M1_AP_minus_L0_AP": m1_overall["average_precision"] - l0_overall["average_precision"],
        "folds": fold_records,
        "M1_vs_L0_fold_win_count": m1_fold_wins,
        "M1_above_fold_prevalence_count": m1_above_fold_prevalence,
        "subsets": subset_results,
        "per_frame": per_frame,
        "per_frame_summary": frame_summary,
        "Success_Gates": gates,
    }
    result["LEARNED_FRAGMENT_RELATION"] = _interpret_gates(gates)

    output_dir = Path(output_dir)
    result_path = output_dir / "phase1_feasibility_result.json"
    oof_path = output_dir / "phase1_oof_predictions.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("sample_row", "frame_id", "canonical_fragment_identity", "label", "validation_fold", "M1_score", "L0_score", "M1_prediction_at_0_50", "L0_prediction_at_0_50")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, m1_score, l0_score in zip(assignments, m1_scores, l0_scores):
            writer.writerow({**{field: row[field] for field in fields[:5]}, "M1_score": float(m1_score), "L0_score": float(l0_score), "M1_prediction_at_0_50": int(m1_score >= 0.50), "L0_prediction_at_0_50": int(l0_score >= 0.50)})
    summary["MODEL_TRAINING_PERFORMED"] = True
    summary["phase1_result_path"] = str(result_path)
    summary["phase1_oof_predictions_path"] = str(oof_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return result, result_path, oof_path
