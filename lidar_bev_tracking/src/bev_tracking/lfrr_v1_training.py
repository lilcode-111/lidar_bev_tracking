"""Frozen LFRR-v1 5-seed x 5-fold development training and evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from bev_tracking.fragment_learning_training import _binary_metrics, _git_identity
from bev_tracking.lfrr_v1 import (
    FoldNormalizer,
    LFRRV1Error,
    SmallTargetConditionedDeepSets,
    collate_lfrr_samples,
    load_compact_representation,
    materialize_fold_samples,
    require_torch,
    torch,
)
from bev_tracking.lfrr_v1_validators import run_pretraining_validators


SCHEMA_VERSION = "lfrr-v1-development-training-v1"
TRAINING_SEEDS = (15531, 15532, 15533, 15534, 15535)
FOLDS = (1, 2, 3, 4, 5)
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
M2_P_VS_N1_AP = 0.5402350133580981
M2_N0_FP = 225
M2_OVERALL_FP = 252


class LFRRTrainingError(LFRRV1Error):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_metrics(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y_true) == 0:
        return None
    predicted = scores >= 0.50
    tp = int(np.sum((y_true == 1) & predicted))
    fp = int(np.sum((y_true == 0) & predicted))
    fn = int(np.sum((y_true == 1) & ~predicted))
    tn = int(np.sum((y_true == 0) & ~predicted))
    precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
    f1 = 0.0 if precision + recall == 0.0 else float(2 * precision * recall / (precision + recall))
    output = {
        "sample_count": int(len(y_true)), "positive_count": int(y_true.sum()),
        "precision_at_0_50": precision, "recall_at_0_50": recall,
        "f1_at_0_50": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "average_precision": None, "roc_auc": None,
    }
    if len(np.unique(y_true)) == 2:
        complete = _binary_metrics(y_true, scores)
        output.update(complete)
    return output


def _load_frozen_m2(output_dir, targets):
    rows = _read_csv(Path(output_dir) / "lightgbm_v2_oof_predictions.csv")
    if len(rows) != len(targets) or len(targets) != 3252:
        raise LFRRTrainingError("M2/target sample count changed")
    scores = np.full(len(targets), np.nan, dtype=np.float64)
    for target, row in zip(targets, rows):
        expected = (
            int(target["sample_row"]), str(target["frame_id"]).zfill(6),
            int(target["canonical_fragment_identity"]), str(target["label"]),
            int(target["validation_fold"]),
        )
        actual = (
            int(row["sample_row"]), str(row["frame_id"]).zfill(6),
            int(row["canonical_fragment_identity"]), str(row["label"]),
            int(row["validation_fold"]),
        )
        if actual != expected:
            raise LFRRTrainingError(f"M2 identity mismatch: {actual} != {expected}")
        scores[expected[0]] = float(row["M2_score"])
    if not np.isfinite(scores).all():
        raise LFRRTrainingError("M2 OOF scores incomplete")
    return scores


def _tensorize(samples):
    batch = collate_lfrr_samples(samples)
    return {
        "target": batch["target"], "neighbors": batch["neighbors"],
        "neighbor_mask": batch["neighbor_mask"], "label": batch["label"],
        "sample_rows": np.asarray(batch["sample_rows"], dtype=np.int64),
    }


def _set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _train_one_model(fold_data, seed, fold, checkpoint_path, progress_callback=None):
    _set_determinism(seed)
    model = SmallTargetConditionedDeepSets().cpu()
    train = fold_data["training"]
    validation = fold_data["validation"]
    positive = int(train["label"].sum().item())
    negative = int(len(train["label"]) - positive)
    if positive <= 0 or negative <= 0:
        raise LFRRTrainingError("training fold must contain positive and negative samples")
    pos_weight = float(negative / positive)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=BETAS, eps=ADAM_EPS,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 100 + fold)
    losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train["label"]), generator=generator)
        weighted_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                train["target"][index], train["neighbors"][index],
                train["neighbor_mask"][index],
            )
            loss = criterion(logits, train["label"][index])
            if not torch.isfinite(loss):
                raise LFRRTrainingError(
                    f"non-finite loss: seed={seed}, fold={fold}, epoch={epoch}"
                )
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().item()) * len(index)
        losses.append(weighted_loss / len(order))
        if progress_callback and (epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS):
            progress_callback(seed, fold, epoch, EPOCHS, losses[-1])
    model.eval()
    logits_parts = []
    with torch.no_grad():
        for start in range(0, len(validation["label"]), BATCH_SIZE):
            stop = start + BATCH_SIZE
            logits_parts.append(model(
                validation["target"][start:stop],
                validation["neighbors"][start:stop],
                validation["neighbor_mask"][start:stop],
            ))
    logits_tensor = torch.cat(logits_parts).cpu()
    probabilities = torch.sigmoid(logits_tensor).numpy().astype(np.float64)
    logits = logits_tensor.numpy().astype(np.float64)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise LFRRTrainingError("validation prediction is non-finite")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION, "seed": int(seed), "fold": int(fold),
        "epoch": EPOCHS, "model_state_dict": model.state_dict(),
    }, checkpoint_path)
    return {
        "seed": int(seed), "fold": int(fold), "pos_weight": pos_weight,
        "training_loss_by_epoch": losses,
        "validation_sample_rows": validation["sample_rows"].tolist(),
        "validation_logits": logits,
        "validation_probabilities": probabilities,
        "checkpoint": str(checkpoint_path),
    }


def _layer_metrics(y, labels, scores):
    overall = _binary_metrics(y, scores)
    layer_a = (labels == "POSITIVE") | (labels == "N0")
    layer_b = (labels == "POSITIVE") | (labels == "N1")
    a = _binary_metrics(y[layer_a], scores[layer_a])
    b = _binary_metrics(y[layer_b], scores[layer_b])
    n0 = labels == "N0"
    a.update({
        "N0_FP_at_0_50": int(np.sum(scores[n0] >= 0.50)),
        "N0_FPR_at_0_50": float(np.mean(scores[n0] >= 0.50)),
        "P_recall_at_0_50": a["recall_at_0_50"],
        "overall_FP_at_0_50": overall["FP"],
    })
    return {"Overall": overall, "Layer_A_P_vs_N0": a, "Layer_B_P_vs_N1": b}


def _per_frame(seed, targets, y, labels, scores):
    frames = np.asarray([str(row["frame_id"]).zfill(6) for row in targets], dtype=object)
    rows = []
    for frame in sorted(set(frames.tolist())):
        mask = frames == frame
        frame_y = y[mask]
        frame_labels = labels[mask]
        frame_scores = scores[mask]
        overall = _safe_metrics(frame_y, frame_scores)
        layer_a = (frame_labels == "POSITIVE") | (frame_labels == "N0")
        layer_b = (frame_labels == "POSITIVE") | (frame_labels == "N1")
        a = _safe_metrics(frame_y[layer_a], frame_scores[layer_a])
        b = _safe_metrics(frame_y[layer_b], frame_scores[layer_b])
        rows.append({
            "seed": int(seed), "frame_id": frame,
            "P_count": int(np.sum(frame_labels == "POSITIVE")),
            "N1_count": int(np.sum(frame_labels == "N1")),
            "N0_count": int(np.sum(frame_labels == "N0")),
            "Layer_A": None if a is None else {
                **a, "N0_FP_at_0_50": int(np.sum(
                    (frame_labels == "N0") & (frame_scores >= 0.50)
                )),
            },
            "Layer_B": b, "Overall": overall,
        })
    return rows


def _local_size_diagnostic(seed, targets, scores):
    sizes = np.asarray([len(row["neighbor_relations"]) for row in targets], dtype=np.int64)
    labels = np.asarray([row["label"] for row in targets], dtype=object)
    rows = []
    for size in sorted(set(sizes.tolist())):
        mask = sizes == size
        rows.append({
            "seed": int(seed), "local_set_size": int(size),
            "sample_count": int(mask.sum()),
            "label_counts": {
                label: int(np.sum(mask & (labels == label)))
                for label in ("POSITIVE", "N1", "N0")
            },
            "score_median": float(np.median(scores[mask])),
            "positive_score_median": (
                None if not np.any(mask & (labels == "POSITIVE"))
                else float(np.median(scores[mask & (labels == "POSITIVE")]))
            ),
        })
    return rows


def _evaluate_gates(seed_results, fold_results, m2_scores, y, labels, folds):
    m2 = _layer_metrics(y, labels, m2_scores)
    m2_a = m2["Layer_A_P_vs_N0"]["average_precision"]
    m2_b = m2["Layer_B_P_vs_N1"]["average_precision"]
    if not np.isclose(m2_b, M2_P_VS_N1_AP, atol=5e-7):
        raise LFRRTrainingError("mechanically recomputed M2 P-vs-N1 AP changed")
    seed_a_ap = [row["metrics"]["Layer_A_P_vs_N0"]["average_precision"] for row in seed_results]
    seed_b_ap = [row["metrics"]["Layer_B_P_vs_N1"]["average_precision"] for row in seed_results]
    seed_n0_fp = [row["metrics"]["Layer_A_P_vs_N0"]["N0_FP_at_0_50"] for row in seed_results]
    seed_all_fp = [row["metrics"]["Overall"]["FP"] for row in seed_results]
    layer_a_checks = {
        "A1_AP_preservation": float(np.median(seed_a_ap)) >= m2_a - 0.02,
        "A2_N0_FP_preservation": float(np.median(seed_n0_fp)) <= 237,
        "A3_overall_FP_preservation": float(np.median(seed_all_fp)) <= 265,
    }
    layer_b_checks = {
        "B1_median_AP_gain_at_least_0_05": float(np.median(seed_b_ap)) - m2_b >= 0.05,
        "B2_at_least_4_of_5_seeds_improve": sum(value > m2_b for value in seed_b_ap) >= 4,
    }
    fold_summary = []
    for fold in FOLDS:
        mask = folds == fold
        m2_fold = _layer_metrics(y[mask], labels[mask], m2_scores[mask])
        selected = [row for row in fold_results if row["fold"] == fold]
        median_a = float(np.median([
            row["metrics"]["Layer_A_P_vs_N0"]["average_precision"] for row in selected
        ]))
        median_b = float(np.median([
            row["metrics"]["Layer_B_P_vs_N1"]["average_precision"] for row in selected
        ]))
        fold_summary.append({
            "fold": fold, "median_seed_LFRR_P_vs_N0_AP": median_a,
            "M2_P_vs_N0_AP": m2_fold["Layer_A_P_vs_N0"]["average_precision"],
            "delta_AP_A": median_a - m2_fold["Layer_A_P_vs_N0"]["average_precision"],
            "median_seed_LFRR_P_vs_N1_AP": median_b,
            "M2_P_vs_N1_AP": m2_fold["Layer_B_P_vs_N1"]["average_precision"],
            "delta_AP_B": median_b - m2_fold["Layer_B_P_vs_N1"]["average_precision"],
        })
    cross_checks = {
        "Layer_B_at_least_4_of_5_positive": sum(row["delta_AP_B"] > 0.0 for row in fold_summary) >= 4,
        "Layer_B_no_fold_below_minus_0_05": all(row["delta_AP_B"] >= -0.05 for row in fold_summary),
        "Layer_A_at_least_4_of_5_within_minus_0_05": sum(row["delta_AP_A"] >= -0.05 for row in fold_summary) >= 4,
        "Layer_A_no_fold_below_minus_0_10": all(row["delta_AP_A"] >= -0.10 for row in fold_summary),
    }
    stable_seeds = [
        row for row, a_ap, b_ap in zip(seed_results, seed_a_ap, seed_b_ap)
        if a_ap >= m2_a - 0.02 and b_ap > m2_b
    ]
    seed_check = len(stable_seeds) >= 4
    gates = {
        "LAYER_A_PRESERVATION_GATE": {
            "result": "PASS" if all(layer_a_checks.values()) else "FAIL",
            "checks": layer_a_checks,
            "M2_P_vs_N0_AP": m2_a,
            "median_seed_LFRR_P_vs_N0_AP": float(np.median(seed_a_ap)),
            "median_seed_N0_FP": float(np.median(seed_n0_fp)),
            "median_seed_overall_FP": float(np.median(seed_all_fp)),
        },
        "LAYER_B_IMPROVEMENT_GATE": {
            "result": "PASS" if all(layer_b_checks.values()) else "FAIL",
            "checks": layer_b_checks,
            "M2_P_vs_N1_AP": m2_b,
            "median_seed_LFRR_P_vs_N1_AP": float(np.median(seed_b_ap)),
            "median_seed_AP_gain": float(np.median(seed_b_ap)) - m2_b,
            "improving_seed_count": sum(value > m2_b for value in seed_b_ap),
        },
        "CROSS_FOLD_STABILITY_GATE": {
            "result": "PASS" if all(cross_checks.values()) else "FAIL",
            "checks": cross_checks, "folds": fold_summary,
        },
        "SEED_STABILITY": {
            "result": "PASS" if seed_check else "FAIL",
            "stable_seed_count": len(stable_seeds),
            "stable_seeds": [row["seed"] for row in stable_seeds],
        },
    }
    conclusion = map_development_signal(gates)
    return m2, gates, conclusion


def map_development_signal(gates):
    """Mechanically map the four frozen gates to the only allowed conclusion."""
    a_pass = gates["LAYER_A_PRESERVATION_GATE"]["result"] == "PASS"
    b_pass = gates["LAYER_B_IMPROVEMENT_GATE"]["result"] == "PASS"
    stability = (
        gates["CROSS_FOLD_STABILITY_GATE"]["result"] == "PASS"
        and gates["SEED_STABILITY"]["result"] == "PASS"
    )
    if a_pass and b_pass and stability:
        return "SUPPORTED"
    if a_pass and b_pass:
        return "UNSTABLE"
    return "NOT_SUPPORTED"


def _aggregate_seed_metrics(seed_results):
    """Return median/min/max summaries without hiding any per-seed result."""
    fields = {
        "Overall_AP": lambda row: row["metrics"]["Overall"]["average_precision"],
        "Overall_ROC_AUC": lambda row: row["metrics"]["Overall"]["roc_auc"],
        "Overall_FP_at_0_50": lambda row: row["metrics"]["Overall"]["FP"],
        "Layer_A_AP": lambda row: row["metrics"]["Layer_A_P_vs_N0"]["average_precision"],
        "Layer_A_ROC_AUC": lambda row: row["metrics"]["Layer_A_P_vs_N0"]["roc_auc"],
        "Layer_A_N0_FP_at_0_50": lambda row: row["metrics"]["Layer_A_P_vs_N0"]["N0_FP_at_0_50"],
        "Layer_A_N0_FPR_at_0_50": lambda row: row["metrics"]["Layer_A_P_vs_N0"]["N0_FPR_at_0_50"],
        "Layer_B_AP": lambda row: row["metrics"]["Layer_B_P_vs_N1"]["average_precision"],
        "Layer_B_ROC_AUC": lambda row: row["metrics"]["Layer_B_P_vs_N1"]["roc_auc"],
        "Layer_B_precision_at_0_50": lambda row: row["metrics"]["Layer_B_P_vs_N1"]["precision_at_0_50"],
        "Layer_B_recall_at_0_50": lambda row: row["metrics"]["Layer_B_P_vs_N1"]["recall_at_0_50"],
        "Layer_B_f1_at_0_50": lambda row: row["metrics"]["Layer_B_P_vs_N1"]["f1_at_0_50"],
    }
    output = {}
    for name, getter in fields.items():
        values = np.asarray([getter(row) for row in seed_results], dtype=np.float64)
        output[name] = {
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return output


def train_lfrr_v1_development(output_dir, *, progress_callback=None):
    """Validate first, then execute exactly 25 sequential fixed development models."""
    require_torch()
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    validator_report, _ = run_pretraining_validators(output_dir)
    if validator_report["TRAINING_READINESS"] != "PASS":
        raise LFRRTrainingError("pre-training hard validators are not all PASS")
    catalog, targets = load_compact_representation(output_dir)
    targets = sorted(targets, key=lambda row: int(row["sample_row"]))
    if [int(row["sample_row"]) for row in targets] != list(range(3252)):
        raise LFRRTrainingError("target sample_row identity changed")
    labels = np.asarray([row["label"] for row in targets], dtype=object)
    y = (labels == "POSITIVE").astype(np.int64)
    folds = np.asarray([int(row["validation_fold"]) for row in targets], dtype=np.int64)
    m2_scores = _load_frozen_m2(output_dir, targets)

    prepared = {}
    for fold in FOLDS:
        normalizer = FoldNormalizer.fit(catalog, targets, fold)
        prepared[fold] = {
            "normalization": normalizer.payload(),
            "training": _tensorize(materialize_fold_samples(
                catalog, targets, normalizer, validation_fold=fold, training=True
            )),
            "validation": _tensorize(materialize_fold_samples(
                catalog, targets, normalizer, validation_fold=fold, training=False
            )),
        }

    model_dir = output_dir / "lfrr_v1_development_models"
    seed_results = []
    fold_results = []
    prediction_rows = []
    per_frame_rows = []
    size_rows = []
    execution_status = []
    for seed in TRAINING_SEEDS:
        oof_scores = np.full(3252, np.nan, dtype=np.float64)
        oof_logits = np.full(3252, np.nan, dtype=np.float64)
        seed_folds = []
        for fold in FOLDS:
            checkpoint = model_dir / f"seed_{seed}_fold_{fold}_epoch_100.pt"
            result = _train_one_model(
                prepared[fold], seed, fold, checkpoint,
                progress_callback=progress_callback,
            )
            rows = np.asarray(result.pop("validation_sample_rows"), dtype=np.int64)
            oof_scores[rows] = result.pop("validation_probabilities")
            oof_logits[rows] = result.pop("validation_logits")
            mask = folds == fold
            metrics = _layer_metrics(y[mask], labels[mask], oof_scores[mask])
            record = {
                **result, "normalization": prepared[fold]["normalization"],
                "train_frame_ids": sorted({
                    str(targets[index]["frame_id"]).zfill(6)
                    for index in np.flatnonzero(~mask)
                }),
                "validation_frame_ids": sorted({
                    str(targets[index]["frame_id"]).zfill(6)
                    for index in np.flatnonzero(mask)
                }),
                "metrics": metrics,
            }
            seed_folds.append(record)
            fold_results.append(record)
            execution_status.append({"seed": seed, "fold": fold, "status": "SUCCESS"})
        if not np.isfinite(oof_scores).all() or not np.isfinite(oof_logits).all():
            raise LFRRTrainingError(f"incomplete OOF coverage for seed {seed}")
        metrics = _layer_metrics(y, labels, oof_scores)
        seed_results.append({
            "seed": seed, "OOF_sample_count": 3252,
            "metrics": metrics, "folds": seed_folds,
        })
        per_frame_rows.extend(_per_frame(seed, targets, y, labels, oof_scores))
        size_rows.extend(_local_size_diagnostic(seed, targets, oof_scores))
        for index, target in enumerate(targets):
            prediction_rows.append({
                "seed": seed, "sample_row": index,
                "frame_id": str(target["frame_id"]).zfill(6),
                "canonical_fragment_identity": int(target["canonical_fragment_identity"]),
                "label": target["label"], "validation_fold": int(target["validation_fold"]),
                "logit": float(oof_logits[index]), "probability": float(oof_scores[index]),
                "prediction_at_0_50": int(oof_scores[index] >= 0.50),
                "local_set_size": len(target["neighbor_relations"]),
            })

    m2_metrics, gates, conclusion = _evaluate_gates(
        seed_results, fold_results, m2_scores, y, labels, folds
    )
    day1 = json.loads(
        (output_dir / "lfrr_v1_day1_summary.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_identity": _git_identity(Path.cwd()),
        "training_protocol": {
            "model": "Small Target-Conditioned Deep Sets", "parameter_count": 2353,
            "seeds": list(TRAINING_SEEDS), "folds": list(FOLDS),
            "model_count": 25, "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "betas": list(BETAS), "eps": ADAM_EPS,
            "loss": "BCEWithLogitsLoss", "batch_size_targets": BATCH_SIZE,
            "epochs": EPOCHS, "early_stopping": False, "lr_scheduler": None,
            "checkpoint_policy": "epoch 100",
        },
        "pretraining_validators": validator_report,
        "model_execution_status": execution_status,
        "model_success_count": sum(row["status"] == "SUCCESS" for row in execution_status),
        "M2_baseline_metrics": m2_metrics,
        "per_seed": seed_results,
        "seed_metric_aggregate": _aggregate_seed_metrics(seed_results),
        "local_set_diagnostics": day1["local_set"],
        "score_vs_local_set_size": size_rows,
        "Development_Gates": gates,
        "LFRR_V1_DEVELOPMENT_SIGNAL": conclusion,
        "FINAL_FIT_PERFORMED": False,
        "INDEPENDENT_RESULTS_OBSERVED": False,
        "FIXED_100_EXECUTED": False,
    }
    result_path = output_dir / "lfrr_v1_development_result.json"
    oof_path = output_dir / "lfrr_v1_all_seed_oof_predictions.csv"
    frame_path = output_dir / "lfrr_v1_per_frame_diagnostics.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(prediction_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    frame_path.write_text(json.dumps(per_frame_rows, indent=2) + "\n", encoding="utf-8")
    return report, result_path, oof_path, frame_path
