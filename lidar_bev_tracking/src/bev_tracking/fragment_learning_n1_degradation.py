"""Read-only M2 N1 ranking-degradation diagnostic."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from bev_tracking.fragment_learning_training import _git_identity


SCHEMA_VERSION = "learned-fragment-relation-m2-n1-degradation-v1"
SIGNIFICANT_SCORE_INCREASE = 0.10
TOP_RANK_LIFT_COUNT = 20
SCENE_TOP_FRAME_COUNT = 3


class N1DegradationAnalysisError(ValueError):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _distribution(values):
    valid = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if len(valid) == 0:
        return {
            "valid_N": 0, "min": None, "P25": None, "P50": None,
            "P75": None, "max": None, "mean": None,
        }
    p25, p50, p75 = np.percentile(valid, [25, 50, 75])
    return {
        "valid_N": int(len(valid)), "min": float(valid.min()),
        "P25": float(p25), "P50": float(p50), "P75": float(p75),
        "max": float(valid.max()), "mean": float(valid.mean()),
    }


def _spearman(left, right):
    pairs = [
        (float(a), float(b)) for a, b in zip(left, right)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
    ]
    if len(pairs) < 3:
        return {"valid_N": len(pairs), "rho": None, "p_value": None}
    a, b = zip(*pairs)
    result = spearmanr(a, b)
    return {
        "valid_N": len(pairs), "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _load_dataset_evidence(path):
    evidence = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row["label_field"]["label"]
            if label not in {"POSITIVE", "N1", "N0"}:
                continue
            metadata = row["metadata_fields"]
            identity = (
                str(metadata["frame_id"]).zfill(6),
                int(metadata["canonical_fragment_identity"]),
            )
            if identity in evidence:
                raise N1DegradationAnalysisError(f"duplicate dataset identity: {identity}")
            associations = row["diagnostic_fields"].get(
                "associated_positive_car_GT", []
            )
            gains = [
                float(item["delta_iou"]) for item in associations
                if item.get("delta_iou") is not None
            ]
            evidence[identity] = {
                "label": label,
                "max_material_gain": max(gains) if gains else None,
                "label_reason": row["diagnostic_fields"].get("label_reason"),
            }
    return evidence


def _load_records(output_dir):
    output_dir = Path(output_dir)
    m2 = sorted(
        _read_csv(output_dir / "lightgbm_v2_oof_predictions.csv"),
        key=lambda row: int(row["sample_row"]),
    )
    margin = sorted(
        _read_csv(output_dir / "lightgbm_v2_margin_feature.csv"),
        key=lambda row: int(row["sample_row"]),
    )
    split = json.loads(
        (output_dir / "fragment_learning_splits.json").read_text(encoding="utf-8")
    )
    assignments = sorted(
        split["sample_assignments"], key=lambda row: int(row["sample_row"])
    )
    evidence = _load_dataset_evidence(output_dir / "fragment_dataset.jsonl")
    if not (len(m2) == len(margin) == len(assignments) == 3252):
        raise N1DegradationAnalysisError("frozen input row count mismatch")

    records = []
    for prediction, feature, assignment in zip(m2, margin, assignments):
        identity = (
            int(prediction["sample_row"]),
            str(prediction["frame_id"]).zfill(6),
            int(prediction["canonical_fragment_identity"]),
            prediction["label"],
            int(prediction["validation_fold"]),
        )
        expected = (
            int(assignment["sample_row"]),
            str(assignment["frame_id"]).zfill(6),
            int(assignment["canonical_fragment_identity"]),
            assignment["label"],
            int(assignment["validation_fold"]),
        )
        feature_identity = (
            int(feature["sample_row"]), str(feature["frame_id"]).zfill(6),
            int(feature["canonical_fragment_identity"]),
        )
        if identity != expected or feature_identity != identity[:3]:
            raise N1DegradationAnalysisError(
                f"prediction/margin/split identity mismatch at {identity[0]}"
            )
        oracle = evidence.get((identity[1], identity[2]))
        if oracle is None or oracle["label"] != identity[3]:
            raise N1DegradationAnalysisError(
                f"dataset evidence identity mismatch at {identity[0]}"
            )
        margin_text = feature["seed_relation_endpoint_gap_margin"].strip()
        margin_value = float(margin_text) if margin_text else None
        m1_score = float(prediction["M1_score"])
        m2_score = float(prediction["M2_score"])
        records.append({
            "sample_row": identity[0], "frame_id": identity[1],
            "fragment_identity": identity[2], "label": identity[3],
            "fold": identity[4], "M1_score": m1_score, "M2_score": m2_score,
            "Delta_score": m2_score - m1_score,
            "margin": margin_value,
            "max_material_gain": oracle["max_material_gain"],
            "M1_prediction_at_0_50": int(m1_score >= 0.50),
            "M2_prediction_at_0_50": int(m2_score >= 0.50),
        })
    counts = {label: sum(row["label"] == label for row in records) for label in ("POSITIVE", "N1", "N0")}
    if counts != {"POSITIVE": 56, "N1": 121, "N0": 3075}:
        raise N1DegradationAnalysisError(f"frozen labels changed: {counts}")
    if any(row["max_material_gain"] is None for row in records if row["label"] == "N1"):
        raise N1DegradationAnalysisError("N1 max_material_gain evidence missing")
    return records


def _score_movement(n1):
    transitions = defaultdict(int)
    for row in n1:
        transitions[
            (row["M1_prediction_at_0_50"], row["M2_prediction_at_0_50"])
        ] += 1
    return {
        "M1_score": _distribution([row["M1_score"] for row in n1]),
        "M2_score": _distribution([row["M2_score"] for row in n1]),
        "Delta_score": _distribution([row["Delta_score"] for row in n1]),
        "score_increased_count": sum(row["Delta_score"] > 0.0 for row in n1),
        "score_decreased_count": sum(row["Delta_score"] < 0.0 for row in n1),
        "score_unchanged_count": sum(row["Delta_score"] == 0.0 for row in n1),
        "M1_below_to_M2_at_or_above_0_50": transitions[(0, 1)],
        "M1_at_or_above_to_M2_below_0_50": transitions[(1, 0)],
        "both_at_or_above_0_50": transitions[(1, 1)],
        "both_below_0_50": transitions[(0, 0)],
    }


def _materiality_analysis(records):
    n1 = [row for row in records if row["label"] == "N1"]
    pn1 = [row for row in records if row["label"] in {"POSITIVE", "N1"}]
    rank_m1 = rankdata(-np.asarray([row["M1_score"] for row in pn1]), method="average")
    rank_m2 = rankdata(-np.asarray([row["M2_score"] for row in pn1]), method="average")
    rank_by_sample = {
        row["sample_row"]: (float(left), float(right), float(left - right))
        for row, left, right in zip(pn1, rank_m1, rank_m2)
    }
    for row in n1:
        row["P_vs_N1_M1_rank"], row["P_vs_N1_M2_rank"], row["P_vs_N1_rank_lift"] = rank_by_sample[row["sample_row"]]
    significant = [
        row for row in n1 if row["Delta_score"] >= SIGNIFICANT_SCORE_INCREASE
    ]
    m2_fp = [row for row in n1 if row["M2_score"] >= 0.50]
    top_rank = sorted(
        n1, key=lambda row: (-row["P_vs_N1_rank_lift"], row["sample_row"])
    )[:TOP_RANK_LIFT_COUNT]
    groups = {
        "ALL_N1": n1,
        "SIGNIFICANT_SCORE_UP_N1": significant,
        "M2_N1_FP_AT_0_50": m2_fp,
        "TOP_20_P_VS_N1_RANK_LIFT": top_rank,
    }
    return {
        "definitions": {
            "significant_score_up": f"M2-M1 >= {SIGNIFICANT_SCORE_INCREASE}",
            "top_rank_lift": f"largest {TOP_RANK_LIFT_COUNT} improvements in P-vs-N1 descending-score rank",
        },
        "groups": {
            name: {
                "sample_count": len(rows),
                "support_frame_count": len({row["frame_id"] for row in rows}),
                "max_material_gain": _distribution([row["max_material_gain"] for row in rows]),
                "Delta_score": _distribution([row["Delta_score"] for row in rows]),
            }
            for name, rows in groups.items()
        },
        "spearman": {
            "max_material_gain_vs_M1_score": _spearman(
                [row["max_material_gain"] for row in n1],
                [row["M1_score"] for row in n1],
            ),
            "max_material_gain_vs_M2_score": _spearman(
                [row["max_material_gain"] for row in n1],
                [row["M2_score"] for row in n1],
            ),
            "max_material_gain_vs_Delta_score": _spearman(
                [row["max_material_gain"] for row in n1],
                [row["Delta_score"] for row in n1],
            ),
        },
        "top_rank_lift_records": [
            {
                key: row[key] for key in (
                    "sample_row", "frame_id", "fragment_identity", "fold",
                    "max_material_gain", "margin", "M1_score", "M2_score",
                    "Delta_score", "P_vs_N1_M1_rank", "P_vs_N1_M2_rank",
                    "P_vs_N1_rank_lift",
                )
            }
            for row in top_rank
        ],
    }


def _margin_discrimination(records):
    output = {
        label: _distribution([row["margin"] for row in records if row["label"] == label])
        for label in ("POSITIVE", "N1", "N0")
    }
    comparisons = {}
    definitions = (
        ("P_vs_N1", {"POSITIVE"}, {"N1"}),
        ("P_vs_N0", {"POSITIVE"}, {"N0"}),
        ("P_plus_N1_vs_N0", {"POSITIVE", "N1"}, {"N0"}),
    )
    for name, positive_labels, negative_labels in definitions:
        selected = [
            row for row in records
            if row["label"] in positive_labels | negative_labels and row["margin"] is not None
        ]
        labels = [int(row["label"] in positive_labels) for row in selected]
        scores = [row["margin"] for row in selected]
        comparisons[name] = {
            "sample_count": len(selected),
            "positive_count": sum(labels),
            "ROC_AUC_higher_margin_is_positive": float(roc_auc_score(labels, scores)),
        }
    return {"distributions": output, "rank_discrimination": comparisons}


def _fold_analysis(records):
    rows = []
    for fold in range(1, 6):
        selected = [
            row for row in records
            if row["fold"] == fold and row["label"] in {"POSITIVE", "N1"}
        ]
        y = np.asarray([row["label"] == "POSITIVE" for row in selected], dtype=np.int64)
        m1 = float(average_precision_score(y, [row["M1_score"] for row in selected]))
        m2 = float(average_precision_score(y, [row["M2_score"] for row in selected]))
        rows.append({
            "fold_id": fold,
            "P": int(y.sum()), "N1": int(len(y) - y.sum()),
            "M1_P_vs_N1_AP": m1, "M2_P_vs_N1_AP": m2,
            "Delta_AP": m2 - m1,
        })
    return rows


def _concentration(records):
    n1 = [row for row in records if row["label"] == "N1"]
    by_frame = defaultdict(lambda: {
        "N1_count": 0, "M2_N1_FP": 0, "score_up_count": 0,
        "significant_score_up_count": 0, "positive_Delta_score_mass": 0.0,
    })
    for row in n1:
        item = by_frame[row["frame_id"]]
        item["N1_count"] += 1
        item["M2_N1_FP"] += int(row["M2_score"] >= 0.50)
        item["score_up_count"] += int(row["Delta_score"] > 0.0)
        item["significant_score_up_count"] += int(
            row["Delta_score"] >= SIGNIFICANT_SCORE_INCREASE
        )
        item["positive_Delta_score_mass"] += max(row["Delta_score"], 0.0)
    rows = [{"frame_id": frame, **values} for frame, values in sorted(by_frame.items())]
    concentration = {}
    for field in ("M2_N1_FP", "score_up_count", "significant_score_up_count", "positive_Delta_score_mass"):
        ranked = sorted(rows, key=lambda row: (-row[field], row["frame_id"]))
        total = sum(row[field] for row in rows)
        concentration[field] = {}
        for count in (1, 3, 5):
            value = sum(row[field] for row in ranked[:count])
            concentration[field][f"top_{count}_frames"] = {
                "value": value, "ratio": 0.0 if total == 0 else float(value / total),
                "frames": [row["frame_id"] for row in ranked[:count]],
            }
    return {"per_frame": rows, "concentration": concentration}


def _conclusion(materiality, folds, scene):
    all_median = materiality["groups"]["ALL_N1"]["max_material_gain"]["P50"]
    up_median = materiality["groups"]["SIGNIFICANT_SCORE_UP_N1"]["max_material_gain"]["P50"]
    rho = materiality["spearman"]["max_material_gain_vs_Delta_score"]["rho"]
    top3_fp = scene["concentration"]["M2_N1_FP"]["top_3_frames"]["ratio"]
    top3_up = scene["concentration"]["significant_score_up_count"]["top_3_frames"]["ratio"]
    degraded_folds = [row["fold_id"] for row in folds if row["Delta_AP"] < 0.0]
    nondegraded_folds = [row["fold_id"] for row in folds if row["Delta_AP"] >= 0.0]
    fold_concentrated = (
        1 <= len(degraded_folds) <= 2 and len(nondegraded_folds) >= 3
    )
    scene_supported = (
        fold_concentrated or top3_fp >= 0.50 or top3_up >= 0.50
    )
    boundary_supported = (
        rho is not None and rho >= 0.20
        and up_median is not None and all_median is not None and up_median > all_median
    )
    true_regression_supported = (
        rho is not None and rho <= 0.0
        and up_median is not None and all_median is not None and up_median <= all_median
    )
    supported = [
        name for name, value in (
            ("MATERIALITY_BOUNDARY_CONFLICT", boundary_supported),
            ("TRUE_N1_HARD_NEGATIVE_REGRESSION", true_regression_supported),
            ("FOLD / SCENE CONCENTRATED", scene_supported),
        ) if value
    ]
    result = supported[0] if len(supported) == 1 else "MIXED / INCONCLUSIVE"
    return {
        "descriptive_rule": {
            "boundary": "Spearman(max_material_gain, Delta_score) >= 0.20 and significant-up gain median > all-N1 median",
            "true_regression": "Spearman <= 0 and significant-up gain median <= all-N1 median",
            "fold_scene_concentrated": "degradation occurs in <=2 folds while >=3 folds are non-degrading, or top-3 frame share >= 0.50 for M2 N1-FP/significant score-up count",
            "mixed": "zero or multiple mechanisms meet the descriptive rules",
        },
        "evidence": {
            "all_N1_gain_median": all_median,
            "significant_up_gain_median": up_median,
            "gain_vs_Delta_score_spearman": rho,
            "top3_M2_N1_FP_ratio": top3_fp,
            "top3_significant_up_ratio": top3_up,
            "P_vs_N1_degraded_folds": degraded_folds,
            "P_vs_N1_nondegraded_folds": nondegraded_folds,
            "fold_concentrated": fold_concentrated,
        },
        "supported_mechanisms": supported,
        "N1_RANKING_DEGRADATION_DIAGNOSIS": result,
    }


def analyze_m2_n1_degradation(output_dir):
    """Read frozen artifacts, write diagnostics, and never fit a model."""
    output_dir = Path(output_dir)
    records = _load_records(output_dir)
    n1 = [row for row in records if row["label"] == "N1"]
    movement = _score_movement(n1)
    materiality = _materiality_analysis(records)
    margin = _margin_discrimination(records)
    folds = _fold_analysis(records)
    scene = _concentration(records)
    conclusion = _conclusion(materiality, folds, scene)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "read_only_inputs": [
            "fragment_dataset.jsonl", "fragment_learning_splits.json",
            "lightgbm_v2_oof_predictions.csv", "lightgbm_v2_margin_feature.csv",
        ],
        "frozen_counts": {"POSITIVE": 56, "N1": 121, "N0": 3075},
        "N1_score_movement": movement,
        "N1_materiality_continuum": materiality,
        "margin_label_relationship": margin,
        "P_vs_N1_per_fold": folds,
        "scene_concentration": scene,
        "final_diagnosis": conclusion,
        "MODEL_RETRAINED": False,
        "PREDICTIONS_REGENERATED": False,
        "DATASET_REGENERATED": False,
        "SPLIT_REGENERATED": False,
        "FORMAL_100_EXECUTED": False,
    }
    result_path = output_dir / "m2_n1_ranking_degradation_analysis.json"
    record_path = output_dir / "m2_n1_diagnostic_records.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = (
        "sample_row", "frame_id", "fragment_identity", "fold",
        "M1_score", "M2_score", "Delta_score", "margin",
        "max_material_gain", "M1_prediction_at_0_50",
        "M2_prediction_at_0_50", "P_vs_N1_M1_rank",
        "P_vs_N1_M2_rank", "P_vs_N1_rank_lift",
    )
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in n1:
            writer.writerow({field: row.get(field) for field in fields})
    return result, result_path, record_path
