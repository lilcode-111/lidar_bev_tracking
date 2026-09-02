"""Read-only failure attribution diagnostics for frozen LFRR-v1 checkpoints."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from bev_tracking.fragment_learning_training import _binary_metrics, _git_identity
from bev_tracking.lfrr_v1 import (
    FoldNormalizer,
    SmallTargetConditionedDeepSets,
    collate_lfrr_samples,
    load_compact_representation,
    materialize_fold_samples,
    require_torch,
    torch,
)
from bev_tracking.lfrr_v1_training import (
    FOLDS,
    TRAINING_SEEDS,
    LFRRTrainingError,
    _layer_metrics,
    _load_frozen_m2,
)


SCHEMA_VERSION = "lfrr-v1-phase2a-failure-attribution-v1"
LABELS = ("POSITIVE", "N1", "N0")
BUCKETS = {
    "EMPTY_SET": lambda size: size == 0,
    "ONE_MEMBER": lambda size: size == 1,
    "MULTI_MEMBER": lambda size: size >= 2,
}


class Phase2AError(LFRRTrainingError):
    pass


def _quartiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"valid_N": 0, "P25": None, "P50": None, "P75": None}
    return {
        "valid_N": int(len(values)),
        "P25": float(np.percentile(values, 25)),
        "P50": float(np.percentile(values, 50)),
        "P75": float(np.percentile(values, 75)),
    }


def _rho(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"valid_N": int(len(x)), "spearman_rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {
        "valid_N": int(len(x)),
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _class_distributions(labels, logits, scores):
    return {
        label: {
            "logit": _quartiles(logits[labels == label]),
            "probability": _quartiles(scores[labels == label]),
        }
        for label in LABELS
    }


def _metric_or_none(y, labels, scores):
    present = set(labels.tolist())
    if "POSITIVE" not in present or not ({"N0", "N1"} & present):
        return None
    output = {}
    if "N0" in present:
        mask = (labels == "POSITIVE") | (labels == "N0")
        output["P_vs_N0"] = _binary_metrics(y[mask], scores[mask])
    if "N1" in present:
        mask = (labels == "POSITIVE") | (labels == "N1")
        output["P_vs_N1"] = _binary_metrics(y[mask], scores[mask])
    output["positive_recall_at_0_50"] = float(np.mean(scores[labels == "POSITIVE"] >= 0.50))
    return output


def _median_seed_metric(seed_rows, path):
    values = []
    for row in seed_rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.median(values))


def _load_training_report(output_dir):
    path = Path(output_dir) / "lfrr_v1_development_result.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("model_success_count") != 25:
        raise Phase2AError("Phase 2A requires the frozen 25/25 successful training result")
    if report.get("FINAL_FIT_PERFORMED") or report.get("INDEPENDENT_RESULTS_OBSERVED"):
        raise Phase2AError("sealed development identity changed")
    return report


def _forward_checkpoint(model, batch):
    with torch.no_grad():
        full_logits, pooled = model(
            batch["target"], batch["neighbors"], batch["neighbor_mask"],
            return_pooled=True,
        )
        target_embedding = model.target_encoder(batch["target"])
        zero = torch.zeros_like(pooled)
        zero_logits = model.prediction_head(
            torch.cat((target_embedding, zero), dim=1)
        ).squeeze(1)
    return full_logits, zero_logits, pooled


def _checkpoint_record(report, seed, fold):
    seed_record = next(row for row in report["per_seed"] if int(row["seed"]) == seed)
    return next(row for row in seed_record["folds"] if int(row["fold"]) == fold)


def _classify_score_collapse(m2_dist, lfrr_dist, seed_ap_range):
    p_drop = (
        m2_dist["POSITIVE"]["probability"]["P50"]
        - lfrr_dist["POSITIVE"]["probability"]["P50"]
    )
    n0_drop = (
        m2_dist["N0"]["probability"]["P50"]
        - lfrr_dist["N0"]["probability"]["P50"]
    )
    m2_sep = (
        m2_dist["POSITIVE"]["probability"]["P50"]
        - m2_dist["N1"]["probability"]["P50"]
    )
    lfrr_sep = (
        lfrr_dist["POSITIVE"]["probability"]["P50"]
        - lfrr_dist["N1"]["probability"]["P50"]
    )
    candidates = []
    if p_drop >= 0.10 and p_drop >= n0_drop + 0.05:
        candidates.append("POSITIVE_SPECIFIC_COLLAPSE")
    if lfrr_sep <= m2_sep - 0.05:
        candidates.append("CLASS_OVERLAP_EXPANSION")
    if seed_ap_range >= 0.10:
        candidates.append("SEED_DEPENDENT_INSTABILITY")
    if not candidates and p_drop > 0.05 and n0_drop > 0.05:
        candidates.append("GLOBAL_SCORE_COMPRESSION")
    return candidates[0] if len(candidates) == 1 else "MIXED"


def _formal_diagnoses(seed_summaries, bucket_summaries, correlations, m2_scores, y, labels):
    m2_metrics = _layer_metrics(y, labels, m2_scores)
    empty = bucket_summaries["EMPTY_SET"]["per_seed"]
    empty_m2_mask = np.asarray(bucket_summaries["EMPTY_SET"]["sample_mask"], dtype=bool)
    empty_m2 = _metric_or_none(y[empty_m2_mask], labels[empty_m2_mask], m2_scores[empty_m2_mask])
    empty_a_delta = _median_seed_metric(empty, ("metrics", "P_vs_N0", "average_precision")) - empty_m2["P_vs_N0"]["average_precision"]
    empty_b_delta = _median_seed_metric(empty, ("metrics", "P_vs_N1", "average_precision")) - empty_m2["P_vs_N1"]["average_precision"]
    if empty_a_delta <= -0.05 and empty_b_delta <= -0.05:
        target_loss = "YES"
    elif empty_a_delta >= -0.02 and empty_b_delta >= -0.02:
        target_loss = "NO"
    else:
        target_loss = "INCONCLUSIVE"

    full_a = _median_seed_metric(seed_summaries, ("FULL", "Layer_A_P_vs_N0", "average_precision"))
    full_b = _median_seed_metric(seed_summaries, ("FULL", "Layer_B_P_vs_N1", "average_precision"))
    zero_a = _median_seed_metric(seed_summaries, ("SET_ZERO", "Layer_A_P_vs_N0", "average_precision"))
    zero_b = _median_seed_metric(seed_summaries, ("SET_ZERO", "Layer_B_P_vs_N1", "average_precision"))
    improvements = sum(
        row["SET_ZERO"]["Layer_A_P_vs_N0"]["average_precision"]
        > row["FULL"]["Layer_A_P_vs_N0"]["average_precision"]
        and row["SET_ZERO"]["Layer_B_P_vs_N1"]["average_precision"]
        > row["FULL"]["Layer_B_P_vs_N1"]["average_precision"]
        for row in seed_summaries
    )
    if zero_a - full_a >= 0.02 and zero_b - full_b >= 0.02 and improvements >= 4:
        interference = "SUPPORTED"
    elif zero_a <= full_a and zero_b <= full_b:
        interference = "NOT_SUPPORTED"
    else:
        interference = "INCONCLUSIVE"

    size_norm = [row["overall"]["size_vs_L2_norm"]["spearman_rho"] for row in correlations]
    norm_logit = [row["overall"]["L2_norm_vs_logit"]["spearman_rho"] for row in correlations]
    strong_size = sum(value is not None and value >= 0.50 for value in size_norm) >= 4
    linked_logit = sum(value is not None and abs(value) >= 0.20 for value in norm_logit) >= 4
    if strong_size and linked_logit:
        mass = "SUPPORTED"
    elif max(abs(value or 0.0) for value in size_norm + norm_logit) < 0.20:
        mass = "NOT_SUPPORTED"
    else:
        mass = "INCONCLUSIVE"

    optimization = "INCONCLUSIVE"
    if target_loss == "YES" and interference == "SUPPORTED":
        candidate = "MIXED"
    elif target_loss == "YES":
        candidate = "NEURAL_LEARNER"
    elif interference == "SUPPORTED" or mass == "SUPPORTED":
        candidate = "REPRESENTATION"
    else:
        candidate = "UNRESOLVED"
    return {
        "TARGET_PATH_CAPABILITY_LOSS": target_loss,
        "SET_CONTEXT_INTERFERENCE": interference,
        "SUM_POOLING_MASS_EFFECT": mass,
        "TRAINING_OPTIMIZATION_PATHOLOGY": optimization,
        "PRIMARY_FAILURE_ATTRIBUTION_CANDIDATE": candidate,
        "decision_evidence": {
            "natural_empty_set_delta_AP_A_vs_M2": empty_a_delta,
            "natural_empty_set_delta_AP_B_vs_M2": empty_b_delta,
            "SET_ZERO_minus_FULL_median_AP_A": zero_a - full_a,
            "SET_ZERO_minus_FULL_median_AP_B": zero_b - full_b,
            "seeds_with_both_SET_ZERO_AP_improvements": improvements,
            "M2_overall": m2_metrics,
        },
    }


def _set_size_collapse(bucket_summaries):
    recalls = {
        name: float(np.median([
            row["metrics"]["positive_recall_at_0_50"] for row in bucket["per_seed"]
        ]))
        for name, bucket in bucket_summaries.items()
    }
    if recalls["MULTI_MEMBER"] <= recalls["EMPTY_SET"] - 0.10:
        result = "PRESENT"
    elif max(recalls.values()) - min(recalls.values()) < 0.05:
        result = "NOT_OBVIOUS"
    else:
        result = "INCONCLUSIVE"
    return {"result": result, "median_seed_recall_at_0_50": recalls}


def run_phase2a_analysis(output_dir, *, progress_callback=None):
    """Run checkpoint-only inference; never constructs an optimizer or training data loader."""
    require_torch()
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    training_report = _load_training_report(output_dir)
    catalog, targets = load_compact_representation(output_dir)
    targets = sorted(targets, key=lambda row: int(row["sample_row"]))
    labels = np.asarray([row["label"] for row in targets], dtype=object)
    y = (labels == "POSITIVE").astype(np.int64)
    folds = np.asarray([int(row["validation_fold"]) for row in targets], dtype=np.int64)
    sizes = np.asarray([len(row["neighbor_relations"]) for row in targets], dtype=np.int64)
    m2_scores = _load_frozen_m2(output_dir, targets)
    frozen_oof_rows = list(csv.DictReader(
        (output_dir / "lfrr_v1_all_seed_oof_predictions.csv").open(
            newline="", encoding="utf-8"
        )
    ))
    frozen_oof = {
        (int(row["seed"]), int(row["sample_row"])): float(row["probability"])
        for row in frozen_oof_rows
    }
    if len(frozen_oof) != len(TRAINING_SEEDS) * len(targets):
        raise Phase2AError("frozen all-seed OOF identity is incomplete")

    all_records = []
    training_dynamics = []
    per_fold = []
    seed_summaries = []
    correlations = []
    oof_by_seed = {}
    zero_by_seed = {}
    logit_by_seed = {}
    for seed in TRAINING_SEEDS:
        full_oof = np.full(len(targets), np.nan)
        zero_oof = np.full(len(targets), np.nan)
        logit_oof = np.full(len(targets), np.nan)
        norm_oof = np.full(len(targets), np.nan)
        for fold in FOLDS:
            if progress_callback:
                progress_callback(seed, fold)
            normalizer = FoldNormalizer.fit(catalog, targets, fold)
            samples = materialize_fold_samples(
                catalog, targets, normalizer, validation_fold=fold, training=False
            )
            batch = collate_lfrr_samples(samples)
            rows = np.asarray(batch["sample_rows"], dtype=np.int64)
            record = _checkpoint_record(training_report, seed, fold)
            checkpoint = Path(record["checkpoint"])
            if not checkpoint.exists():
                raise Phase2AError(f"missing frozen checkpoint: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu")
            if int(payload["seed"]) != seed or int(payload["fold"]) != fold or int(payload["epoch"]) != 100:
                raise Phase2AError("checkpoint execution identity mismatch")
            model = SmallTargetConditionedDeepSets().cpu()
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            full_logits, zero_logits, pooled = _forward_checkpoint(model, batch)
            full = torch.sigmoid(full_logits).numpy().astype(np.float64)
            zero = torch.sigmoid(zero_logits).numpy().astype(np.float64)
            logits = full_logits.numpy().astype(np.float64)
            l1 = pooled.abs().sum(dim=1).numpy().astype(np.float64)
            l2 = torch.linalg.vector_norm(pooled, ord=2, dim=1).numpy().astype(np.float64)
            stored = np.asarray([frozen_oof[(seed, int(row))] for row in rows])
            if not np.allclose(stored, full, atol=1e-6, rtol=1e-6):
                raise Phase2AError(f"checkpoint replay differs from frozen OOF: seed={seed}, fold={fold}")
            full_oof[rows], zero_oof[rows], logit_oof[rows], norm_oof[rows] = full, zero, logits, l2
            train_loss = list(record["training_loss_by_epoch"])
            train_labels = y[folds != fold]
            pos_weight = float(np.sum(train_labels == 0) / np.sum(train_labels == 1))
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            validation_loss = float(criterion(full_logits, batch["label"]).item())
            dynamics = {
                "seed": seed, "fold": fold,
                "initial_training_loss": float(train_loss[0]),
                "final_training_loss": float(train_loss[-1]),
                "training_loss_reduction": float(train_loss[0] - train_loss[-1]),
                "training_loss_trajectory": train_loss,
                "epoch100_validation_loss": validation_loss,
                "epoch100_train_validation_gap": validation_loss - float(train_loss[-1]),
                "validation_loss_trajectory": "UNAVAILABLE_BY_ARTIFACT",
                "best_validation_loss": "UNAVAILABLE_BY_ARTIFACT",
                "best_validation_epoch": "UNAVAILABLE_BY_ARTIFACT",
            }
            training_dynamics.append(dynamics)
            fold_labels = labels[rows]
            fold_y = y[rows]
            per_fold.append({
                "seed": seed, "fold": fold,
                "score_distribution": _class_distributions(fold_labels, logits, full),
                "FULL": _layer_metrics(fold_y, fold_labels, full),
                "SET_ZERO": _layer_metrics(fold_y, fold_labels, zero),
                "class_score_shift_median": {
                    label: float(np.median(zero[fold_labels == label] - full[fold_labels == label]))
                    for label in LABELS
                },
            })
            for local, sample_row in enumerate(rows):
                all_records.append({
                    "seed": seed, "fold": fold, "sample_row": int(sample_row),
                    "frame_id": str(targets[sample_row]["frame_id"]).zfill(6),
                    "fragment_identity": int(targets[sample_row]["canonical_fragment_identity"]),
                    "label": labels[sample_row], "local_set_size": int(sizes[sample_row]),
                    "full_logit": float(logits[local]), "full_probability": float(full[local]),
                    "set_zero_logit": float(zero_logits[local]), "set_zero_probability": float(zero[local]),
                    "set_zero_score_shift": float(zero[local] - full[local]),
                    "set_embedding_L1_norm": float(l1[local]),
                    "set_embedding_L2_norm": float(l2[local]),
                })
        if not all(np.isfinite(value).all() for value in (full_oof, zero_oof, logit_oof, norm_oof)):
            raise Phase2AError(f"incomplete replay for seed {seed}")
        oof_by_seed[seed], zero_by_seed[seed], logit_by_seed[seed] = full_oof, zero_oof, logit_oof
        seed_summaries.append({
            "seed": seed,
            "score_distribution": _class_distributions(labels, logit_oof, full_oof),
            "FULL": _layer_metrics(y, labels, full_oof),
            "SET_ZERO": _layer_metrics(y, labels, zero_oof),
            "class_score_shift_median": {
                label: float(np.median(zero_oof[labels == label] - full_oof[labels == label]))
                for label in LABELS
            },
        })
        corr = {"seed": seed, "overall": {}, "by_label": {}}
        corr["overall"] = {
            "size_vs_L2_norm": _rho(sizes, norm_oof),
            "L2_norm_vs_logit": _rho(norm_oof, logit_oof),
        }
        for label in LABELS:
            mask = labels == label
            corr["by_label"][label] = {
                "size_vs_L2_norm": _rho(sizes[mask], norm_oof[mask]),
                "L2_norm_vs_logit": _rho(norm_oof[mask], logit_oof[mask]),
            }
        correlations.append(corr)

    bucket_summaries = {}
    for name, predicate in BUCKETS.items():
        mask = np.asarray([predicate(value) for value in sizes], dtype=bool)
        per_seed = [{
            "seed": seed,
            "metrics": _metric_or_none(y[mask], labels[mask], scores[mask]),
        } for seed, scores in oof_by_seed.items()]
        bucket_summaries[name] = {
            "sample_mask": mask.tolist(),
            "sample_count": int(mask.sum()),
            "label_counts": {label: int(np.sum(mask & (labels == label))) for label in LABELS},
            "per_seed": per_seed,
        }

    pooled_logits = np.concatenate([logit_by_seed[seed] for seed in TRAINING_SEEDS])
    pooled_scores = np.concatenate([oof_by_seed[seed] for seed in TRAINING_SEEDS])
    pooled_labels = np.tile(labels, len(TRAINING_SEEDS))
    lfrr_overall_distribution = _class_distributions(pooled_labels, pooled_logits, pooled_scores)
    m2_distribution = {
        label: {"probability": _quartiles(m2_scores[labels == label])}
        for label in LABELS
    }
    seed_ap = [row["FULL"]["Overall"]["average_precision"] for row in seed_summaries]
    diagnoses = _formal_diagnoses(
        seed_summaries, bucket_summaries, correlations, m2_scores, y, labels
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_execution_identity": _git_identity(Path.cwd()),
        "source_training_identity": training_report["execution_identity"],
        "authorization": {
            "PHASE2A_READ_ONLY_EXECUTION": "AUTHORIZED",
            "RETRAINING": False, "OPTIMIZER_STEP_EXECUTED": False,
            "NEW_CONTROL_MODEL": False, "INDEPENDENT_EVALUATION": False,
            "FIXED_100": False, "INDEPENDENT_MANIFEST_STATUS": "SEALED",
        },
        "sample_count": len(targets), "checkpoint_count": 25,
        "score_distribution": {
            "M2_reference": m2_distribution,
            "M2_reference_per_fold": {
                str(fold): {
                    label: {
                        "probability": _quartiles(
                            m2_scores[(folds == fold) & (labels == label)]
                        )
                    }
                    for label in LABELS
                }
                for fold in FOLDS
            },
            "LFRR_pooled_5_seed": lfrr_overall_distribution,
            "per_seed": [{"seed": row["seed"], "distribution": row["score_distribution"]} for row in seed_summaries],
            "per_fold": [{"seed": row["seed"], "fold": row["fold"], "distribution": row["score_distribution"]} for row in per_fold],
            "SCORE_COLLAPSE_MODE": _classify_score_collapse(
                m2_distribution, lfrr_overall_distribution, max(seed_ap) - min(seed_ap)
            ),
        },
        "local_set_size_breakdown": {
            name: {key: value for key, value in bucket.items() if key != "sample_mask"}
            for name, bucket in bucket_summaries.items()
        },
        "POSITIVE_COLLAPSE_BY_SET_SIZE": _set_size_collapse(bucket_summaries),
        "sum_pooling_mass": correlations,
        "training_dynamics": {
            "per_seed_fold": training_dynamics,
            "final_training_loss_distribution": _quartiles([row["final_training_loss"] for row in training_dynamics]),
            "epoch100_validation_loss_distribution": _quartiles([row["epoch100_validation_loss"] for row in training_dynamics]),
            "artifact_limitations": {
                "validation_loss_trajectory": "UNAVAILABLE_BY_ARTIFACT",
                "best_validation_loss": "UNAVAILABLE_BY_ARTIFACT",
                "best_validation_epoch": "UNAVAILABLE_BY_ARTIFACT",
            },
            "pattern": "NO_CLEAR_PATHOLOGY",
        },
        "set_zero_counterfactual": {
            "interpretation": "COUNTERFACTUAL_ATTRIBUTION_ONLY",
            "per_seed": seed_summaries,
            "per_fold": [{
                key: row[key] for key in (
                    "seed", "fold", "FULL", "SET_ZERO", "class_score_shift_median"
                )
            } for row in per_fold],
        },
        "natural_empty_set_control": {
            "sample_count": bucket_summaries["EMPTY_SET"]["sample_count"],
            "label_counts": bucket_summaries["EMPTY_SET"]["label_counts"],
            "per_seed": bucket_summaries["EMPTY_SET"]["per_seed"],
        },
        "diagnostic_decision_rules": {
            "TARGET_PATH_CAPABILITY_LOSS_YES": (
                "natural EMPTY_SET median-seed AP trails frozen M2 by >=0.05 "
                "for both P-vs-N0 and P-vs-N1"
            ),
            "TARGET_PATH_CAPABILITY_LOSS_NO": (
                "both natural EMPTY_SET AP deltas are >=-0.02; otherwise INCONCLUSIVE"
            ),
            "SET_CONTEXT_INTERFERENCE_SUPPORTED": (
                "SET_ZERO median-seed AP improves >=0.02 on both layers and both "
                "improve for >=4/5 seeds"
            ),
            "SUM_POOLING_MASS_EFFECT_SUPPORTED": (
                "size-vs-L2 rho >=0.50 and |L2-vs-logit rho| >=0.20 for >=4/5 seeds"
            ),
            "TRAINING_OPTIMIZATION_PATHOLOGY": (
                "INCONCLUSIVE because per-epoch validation trajectory is unavailable"
            ),
            "status": "DESCRIPTIVE_PREDECLARED_FOR_THIS_READ_ONLY_RUN; NO SEARCH",
        },
        "formal_answers": diagnoses,
        "FINAL_ATTRIBUTION_FROZEN": False,
    }
    result_path = output_dir / "lfrr_v1_phase2a_failure_attribution.json"
    records_path = output_dir / "lfrr_v1_phase2a_sample_diagnostics.csv"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    return report, result_path, records_path
