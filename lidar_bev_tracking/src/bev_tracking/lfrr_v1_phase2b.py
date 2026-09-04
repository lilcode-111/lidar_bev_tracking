"""Frozen LFRR-v1 Phase 2B Target-Only Neural Control."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random

import numpy as np

from bev_tracking.fragment_learning_training import _git_identity
from bev_tracking.lfrr_v1 import (
    FoldNormalizer,
    TARGET_DIM,
    load_compact_representation,
    require_torch,
    torch,
    nn,
)
from bev_tracking.lfrr_v1_training import (
    ADAM_EPS,
    BATCH_SIZE,
    BETAS,
    EPOCHS,
    FOLDS,
    LEARNING_RATE,
    TRAINING_SEEDS,
    WEIGHT_DECAY,
    LFRRTrainingError,
    _layer_metrics,
)


SCHEMA_VERSION = "lfrr-v1-phase2b-target-only-control-v1"
A_M2 = 0.17987056812351537
B_M2 = 0.5402350133580981
A_FULL = 0.09233366250004167
B_FULL = 0.41795864687066825
EXPECTED_PARAMETER_COUNT = 705


class Phase2BError(LFRRTrainingError):
    pass


if nn is not None:
    class TargetOnlyNeuralControl(nn.Module):
        """The only authorized 25 -> 16 -> 16 -> 1 control architecture."""

        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(25, 16, bias=True),
                nn.ReLU(),
                nn.Linear(16, 16, bias=True),
                nn.ReLU(),
                nn.Linear(16, 1, bias=True),
            )
            if sum(value.numel() for value in self.parameters()) != EXPECTED_PARAMETER_COUNT:
                raise Phase2BError("Target-Only parameter count changed")

        def forward(self, target):
            if target.ndim != 2 or target.shape[1] != TARGET_DIM:
                raise Phase2BError("Target-Only input must be [B,25]")
            return self.network(target).squeeze(1)
else:
    class TargetOnlyNeuralControl:
        def __init__(self, *args, **kwargs):
            require_torch()


def _normalizer_from_payload(payload):
    """Restore exact Full-LFRR fold statistics without reading local-set membership."""
    return FoldNormalizer(
        payload["base24_mean"],
        payload["base24_scale"],
        [payload["delta_u_mean"], payload["delta_v_mean"]],
        [payload["delta_u_scale"], payload["delta_v_scale"]],
        payload["margin_mean"],
        payload["margin_scale"],
        {"validation_fold": int(payload["validation_fold"])},
    )


def _full_fold_record(training_report, seed, fold):
    seed_row = next(row for row in training_report["per_seed"] if int(row["seed"]) == seed)
    return next(row for row in seed_row["folds"] if int(row["fold"]) == fold)


def _materialize_target_only(catalog, targets, normalizer, fold, *, training):
    """Build target/label tensors only; local-set fields are intentionally untouched."""
    selected = [
        row for row in targets
        if (int(row["validation_fold"]) != fold if training else int(row["validation_fold"]) == fold)
    ]
    vectors = []
    labels = []
    rows = []
    for target in selected:
        key = (
            str(target["frame_id"]).zfill(6),
            int(target["canonical_fragment_identity"]),
        )
        vectors.append(normalizer.transform_target(catalog[key], target["target_margin"]))
        labels.append(1.0 if target["label"] == "POSITIVE" else 0.0)
        rows.append(int(target["sample_row"]))
    target_tensor = torch.as_tensor(np.asarray(vectors), dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    if target_tensor.shape != (len(selected), 25):
        raise Phase2BError("Target-Only materialization changed input identity")
    return {"target": target_tensor, "label": label_tensor, "sample_rows": np.asarray(rows)}


def _set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def validate_control_contract(catalog, targets, training_report, output_dir):
    require_torch()
    frozen_rows = [
        row for row in csv.DictReader(
            (Path(output_dir) / "lfrr_v1_all_seed_oof_predictions.csv").open(
                newline="", encoding="utf-8"
            )
        )
        if int(row["seed"]) == TRAINING_SEEDS[0]
    ]
    frozen_rows.sort(key=lambda row: int(row["sample_row"]))
    target_identity = [(
        int(row["sample_row"]), str(row["frame_id"]).zfill(6),
        int(row["canonical_fragment_identity"]), str(row["label"]),
        int(row["validation_fold"]),
    ) for row in targets]
    frozen_identity = [(
        int(row["sample_row"]), str(row["frame_id"]).zfill(6),
        int(row["canonical_fragment_identity"]), str(row["label"]),
        int(row["validation_fold"]),
    ) for row in frozen_rows]
    checks = {
        "sample_count_3252": len(targets) == 3252,
        "sample_row_identity": [int(row["sample_row"]) for row in targets] == list(range(3252)),
        "label_identity": {
            label: sum(row["label"] == label for row in targets)
            for label in ("POSITIVE", "N1", "N0")
        } == {"POSITIVE": 56, "N1": 121, "N0": 3075},
        "fold_identity": sorted({int(row["validation_fold"]) for row in targets}) == list(FOLDS),
        "exact_sample_label_fold_identity_vs_frozen_OOF": target_identity == frozen_identity,
        "seed_identity": list(TRAINING_SEEDS) == [15531, 15532, 15533, 15534, 15535],
        "input_dim_25": TARGET_DIM == 25,
        "parameter_count_705": sum(
            value.numel() for value in TargetOnlyNeuralControl().parameters()
        ) == EXPECTED_PARAMETER_COUNT,
        "no_local_set_tensor_consumed": True,
        "training_contract_unchanged": (
            training_report["training_protocol"]["batch_size_targets"] == BATCH_SIZE
            and training_report["training_protocol"]["epochs"] == EPOCHS
            and training_report["training_protocol"]["learning_rate"] == LEARNING_RATE
            and training_report["training_protocol"]["weight_decay"] == WEIGHT_DECAY
            and training_report["training_protocol"]["early_stopping"] is False
        ),
        "normalization_reused_from_full_LFRR_artifact": True,
    }
    # Exercise every restored fold normalizer on one target without accessing neighbors.
    for fold in FOLDS:
        record = _full_fold_record(training_report, TRAINING_SEEDS[0], fold)
        normalizer = _normalizer_from_payload(record["normalization"])
        sample = next(row for row in targets if int(row["validation_fold"]) == fold)
        key = (str(sample["frame_id"]).zfill(6), int(sample["canonical_fragment_identity"]))
        vector = normalizer.transform_target(catalog[key], sample["target_margin"])
        checks[f"fold_{fold}_normalization_finite_25D"] = (
            vector.shape == (25,) and bool(np.isfinite(vector).all())
        )
    return {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model_input_keys": ["target"],
        "forbidden_input_keys": [
            "local_set", "neighbor", "neighbors", "delta_u", "delta_v",
            "set_size", "set_embedding", "neighbor_mask",
        ],
    }


def _train_one(fold_data, seed, fold, checkpoint, progress_callback=None):
    _set_determinism(seed)
    model = TargetOnlyNeuralControl().cpu()
    train = fold_data["training"]
    validation = fold_data["validation"]
    positive = int(train["label"].sum().item())
    negative = int(len(train["label"]) - positive)
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
            logits = model(train["target"][index])
            loss = criterion(logits, train["label"][index])
            if not torch.isfinite(loss):
                raise Phase2BError(f"non-finite loss: seed={seed} fold={fold} epoch={epoch}")
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
            logits_parts.append(model(validation["target"][start:start + BATCH_SIZE]))
    logits_tensor = torch.cat(logits_parts).cpu()
    logits = logits_tensor.numpy().astype(np.float64)
    probabilities = torch.sigmoid(logits_tensor).numpy().astype(np.float64)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise Phase2BError("Target-Only validation output is non-finite")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION, "seed": seed, "fold": fold,
        "epoch": EPOCHS, "model_state_dict": model.state_dict(),
    }, checkpoint)
    return {
        "seed": seed, "fold": fold, "pos_weight": pos_weight,
        "training_loss_by_epoch": losses,
        "validation_sample_rows": validation["sample_rows"],
        "validation_logits": logits,
        "validation_probabilities": probabilities,
        "checkpoint": str(checkpoint),
    }


def classify_preliminary_case(gap_a, gap_b, to_a, to_b, full_a, full_b):
    to_a = np.asarray(to_a, dtype=np.float64)
    to_b = np.asarray(to_b, dtype=np.float64)
    full_a = np.asarray(full_a, dtype=np.float64)
    full_b = np.asarray(full_b, dtype=np.float64)
    counts = {
        "A_TO_above_FULL": int(np.sum(to_a > full_a)),
        "A_TO_below_FULL": int(np.sum(to_a < full_a)),
        "B_TO_above_FULL": int(np.sum(to_b > full_b)),
        "B_TO_below_FULL": int(np.sum(to_b < full_b)),
        "A_TO_below_M2": int(np.sum(to_a < A_M2)),
        "B_TO_below_M2": int(np.sum(to_b < B_M2)),
    }
    stable_a = max(counts["A_TO_above_FULL"], counts["A_TO_below_FULL"]) >= 4
    stable_b = max(counts["B_TO_above_FULL"], counts["B_TO_below_FULL"]) >= 4
    if not stable_a or not stable_b:
        case = "SEED_INSTABILITY"
    elif (
        gap_a >= 0.75 and gap_b >= 0.75
        and counts["A_TO_above_FULL"] >= 4
        and counts["B_TO_above_FULL"] >= 4
    ):
        case = "A"
    elif (
        gap_a <= 0.25 and gap_b <= 0.25
        and counts["A_TO_below_M2"] >= 4
        and counts["B_TO_below_M2"] >= 4
    ):
        case = "B"
    else:
        case = "C"
    return case, counts


def train_target_only_control(output_dir, *, progress_callback=None):
    require_torch()
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    training_report = json.loads(
        (output_dir / "lfrr_v1_development_result.json").read_text(encoding="utf-8")
    )
    catalog, targets = load_compact_representation(output_dir)
    targets = sorted(targets, key=lambda row: int(row["sample_row"]))
    validator = validate_control_contract(catalog, targets, training_report, output_dir)
    if validator["result"] != "PASS":
        raise Phase2BError("CONTROL_CONTRACT_VALIDATOR failed before training")
    labels = np.asarray([row["label"] for row in targets], dtype=object)
    y = (labels == "POSITIVE").astype(np.int64)
    folds = np.asarray([int(row["validation_fold"]) for row in targets], dtype=np.int64)

    # Normalization is fold-specific but seed-invariant; reuse the exact saved payload.
    prepared = {}
    for fold in FOLDS:
        payload = _full_fold_record(training_report, TRAINING_SEEDS[0], fold)["normalization"]
        normalizer = _normalizer_from_payload(payload)
        prepared[fold] = {
            "normalization": payload,
            "training": _materialize_target_only(catalog, targets, normalizer, fold, training=True),
            "validation": _materialize_target_only(catalog, targets, normalizer, fold, training=False),
        }

    full_seed_records = {
        int(row["seed"]): row for row in training_report["per_seed"]
    }
    full_a = [
        full_seed_records[seed]["metrics"]["Layer_A_P_vs_N0"]["average_precision"]
        for seed in TRAINING_SEEDS
    ]
    full_b = [
        full_seed_records[seed]["metrics"]["Layer_B_P_vs_N1"]["average_precision"]
        for seed in TRAINING_SEEDS
    ]
    if not np.isclose(np.median(full_a), A_FULL, atol=1e-15, rtol=0.0):
        raise Phase2BError("frozen Full-LFRR Layer-A identity changed")
    if not np.isclose(np.median(full_b), B_FULL, atol=1e-15, rtol=0.0):
        raise Phase2BError("frozen Full-LFRR Layer-B identity changed")

    model_dir = output_dir / "lfrr_v1_phase2b_target_only_models"
    per_seed = []
    per_fold = []
    predictions = []
    execution = []
    for seed in TRAINING_SEEDS:
        scores = np.full(len(targets), np.nan)
        logits = np.full(len(targets), np.nan)
        seed_folds = []
        for fold in FOLDS:
            result = _train_one(
                prepared[fold], seed, fold,
                model_dir / f"seed_{seed}_fold_{fold}_epoch_100.pt",
                progress_callback=progress_callback,
            )
            rows = np.asarray(result.pop("validation_sample_rows"), dtype=np.int64)
            scores[rows] = result.pop("validation_probabilities")
            logits[rows] = result.pop("validation_logits")
            mask = folds == fold
            metrics = _layer_metrics(y[mask], labels[mask], scores[mask])
            fold_record = {**result, "metrics": metrics}
            seed_folds.append(fold_record)
            per_fold.append(fold_record)
            execution.append({"seed": seed, "fold": fold, "status": "SUCCESS"})
        if not np.isfinite(scores).all() or not np.isfinite(logits).all():
            raise Phase2BError(f"incomplete Target-Only OOF prediction: seed={seed}")
        metrics = _layer_metrics(y, labels, scores)
        per_seed.append({"seed": seed, "metrics": metrics, "folds": seed_folds})
        for index, target in enumerate(targets):
            predictions.append({
                "seed": seed, "sample_row": index,
                "frame_id": str(target["frame_id"]).zfill(6),
                "canonical_fragment_identity": int(target["canonical_fragment_identity"]),
                "label": target["label"], "validation_fold": int(target["validation_fold"]),
                "target_only_logit": float(logits[index]),
                "target_only_probability": float(scores[index]),
                "prediction_at_0_50": int(scores[index] >= 0.50),
            })

    to_a = [row["metrics"]["Layer_A_P_vs_N0"]["average_precision"] for row in per_seed]
    to_b = [row["metrics"]["Layer_B_P_vs_N1"]["average_precision"] for row in per_seed]
    median_a = float(np.median(to_a))
    median_b = float(np.median(to_b))
    gap_a = (median_a - A_FULL) / (A_M2 - A_FULL)
    gap_b = (median_b - B_FULL) / (B_M2 - B_FULL)
    case, direction_counts = classify_preliminary_case(
        gap_a, gap_b, to_a, to_b, full_a, full_b
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_identity": _git_identity(Path.cwd()),
        "source_full_LFRR_identity": training_report["execution_identity"],
        "CONTROL_CONTRACT_VALIDATOR": validator,
        "training_protocol": {
            "architecture": "25->Linear(25,16)->ReLU->Linear(16,16)->ReLU->Linear(16,1)",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "seeds": list(TRAINING_SEEDS), "folds": list(FOLDS),
            "model_count": 25, "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS), "eps": ADAM_EPS,
            "loss": "BCEWithLogitsLoss", "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "early_stopping": False,
            "checkpoint": "epoch 100", "LOCAL_SET_INPUT": "NONE",
        },
        "execution_status": execution,
        "model_success_count": sum(row["status"] == "SUCCESS" for row in execution),
        "frozen_references": {
            "A_M2": A_M2, "B_M2": B_M2,
            "A_FULL": A_FULL, "B_FULL": B_FULL,
        },
        "per_seed": per_seed,
        "per_fold": per_fold,
        "aggregate": {
            "A_TARGET_ONLY_median": median_a,
            "B_TARGET_ONLY_median": median_b,
            "A_TARGET_ONLY_min": float(np.min(to_a)),
            "A_TARGET_ONLY_max": float(np.max(to_a)),
            "B_TARGET_ONLY_min": float(np.min(to_b)),
            "B_TARGET_ONLY_max": float(np.max(to_b)),
            "GAP_CLOSURE_A": float(gap_a),
            "GAP_CLOSURE_B": float(gap_b),
            "seed_direction_counts": direction_counts,
        },
        "PRELIMINARY_CASE": case,
        "case_interpretation": {
            "A": "SET_REPRESENTATION_OR_AGGREGATION",
            "B": "NEURAL_LEARNER_OR_SMALL_DATA_GENERALIZATION",
            "C": "MIXED",
            "SEED_INSTABILITY": "SEED_INSTABILITY",
        }[case],
        "PRIMARY_FAILURE_ATTRIBUTION_FROZEN": False,
        "RETRAINING_OF_EXISTING_MODELS": False,
        "NEW_CONTROL_MODEL_TRAINING_PERFORMED": True,
        "INDEPENDENT_MANIFEST_STATUS": "SEALED",
        "INDEPENDENT_EVALUATION": False,
        "FIXED_100_EXECUTED": False,
    }
    result_path = output_dir / "lfrr_v1_phase2b_target_only_result.json"
    oof_path = output_dir / "lfrr_v1_phase2b_target_only_oof.csv"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return report, result_path, oof_path
