"""Frozen Phase-3 M2-anchored incremental context development."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from bev_tracking.fragment_learning_dataset import MODEL_FEATURE_FIELDS
from bev_tracking.fragment_learning_training import (
    M1_PARAMETERS, _default_m1_factory, _git_identity, _load_inputs,
)
from bev_tracking.lfrr_v1 import load_compact_representation, require_torch, torch
from bev_tracking.lfrr_v1_training import (
    ADAM_EPS, BATCH_SIZE, BETAS, EPOCHS, FOLDS, LEARNING_RATE,
    TRAINING_SEEDS, WEIGHT_DECAY, _layer_metrics, _per_frame,
)


SCHEMA_VERSION = "lfrr-phase3-m2-anchored-context-v1"
MARGIN_FIELD = "seed_relation_endpoint_gap_margin"
NEIGHBOR_DIM = 26
PARAMETER_COUNT = 277
TOLERANCE = 1.0e-12
STD_EPSILON = 1.0e-12


class Phase3Error(ValueError):
    pass


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _identity(row):
    return (
        str(row["frame_id"]).zfill(6),
        int(row["canonical_fragment_identity"]),
    )


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _load_phase3_inputs(output_dir):
    output_dir = Path(output_dir)
    loaded = _load_inputs(output_dir)
    summary, split, prerequisites, assignments, base, y, labels, folds, frames = loaded
    if base.shape != (3252, 24) or tuple(MODEL_FEATURE_FIELDS) != tuple(summary.get(
        "model_feature_fields", MODEL_FEATURE_FIELDS
    )):
        # Old summaries do not necessarily repeat the field list; X_model was checked by _load_inputs.
        if base.shape != (3252, 24):
            raise Phase3Error("frozen F_BASE24 identity changed")

    margin_rows = sorted(
        _read_csv(output_dir / "lightgbm_v2_margin_feature.csv"),
        key=lambda row: int(row["sample_row"]),
    )
    if len(margin_rows) != 3252:
        raise Phase3Error("frozen M2 margin row count changed")
    margins = np.full(3252, np.nan, dtype=np.float64)
    for index, (row, assignment) in enumerate(zip(margin_rows, assignments)):
        if int(row["sample_row"]) != index or _identity(row) != _identity(assignment):
            raise Phase3Error(f"M2 margin identity mismatch at sample {index}")
        if str(row.get("margin_valid", "")).lower() == "true":
            margins[index] = float(row[MARGIN_FIELD])
    x_m2 = np.column_stack((base, margins))

    historical_rows = sorted(
        _read_csv(output_dir / "lightgbm_v2_oof_predictions.csv"),
        key=lambda row: int(row["sample_row"]),
    )
    if len(historical_rows) != 3252:
        raise Phase3Error("historical M2 OOF count changed")
    historical = np.empty(3252, dtype=np.float64)
    seen = set()
    for index, (row, assignment) in enumerate(zip(historical_rows, assignments)):
        actual = (_identity(row), row["label"], int(row["validation_fold"]))
        expected = (_identity(assignment), assignment["label"], int(assignment["validation_fold"]))
        if actual != expected or int(row["sample_row"]) != index or actual[0] in seen:
            raise Phase3Error(f"historical M2 identity mismatch at sample {index}")
        seen.add(actual[0])
        historical[index] = float(row["M2_score"])
    if not np.isfinite(historical).all():
        raise Phase3Error("historical M2 probability contains non-finite values")

    m2_result = json.loads(
        (output_dir / "lightgbm_v2_development_result.json").read_text(encoding="utf-8")
    )
    if m2_result.get("M2_parameters") != M1_PARAMETERS:
        raise Phase3Error("frozen M2 configuration identity changed")

    catalog, targets = load_compact_representation(output_dir)
    targets = sorted(targets, key=lambda row: int(row["sample_row"]))
    if len(targets) != 3252:
        raise Phase3Error("compact target count changed")
    for index, (target, assignment) in enumerate(zip(targets, assignments)):
        if int(target["sample_row"]) != index or _identity(target) != _identity(assignment):
            raise Phase3Error(f"local-set target identity mismatch at sample {index}")
        if target["label"] != assignment["label"] or int(target["validation_fold"]) != int(assignment["validation_fold"]):
            raise Phase3Error(f"local-set label/fold identity mismatch at sample {index}")
    return {
        "summary": summary, "split": split, "prerequisites": prerequisites,
        "assignments": assignments, "X_m2": x_m2, "y": y,
        "labels": labels, "folds": folds, "frames": frames,
        "historical_probability": historical, "catalog": catalog, "targets": targets,
    }


def _fit_m2(x, y, folds, excluded_folds, factory):
    excluded_folds = tuple(sorted(int(value) for value in excluded_folds))
    training = ~np.isin(folds, excluded_folds)
    positive = int(y[training].sum())
    negative = int(training.sum() - positive)
    if positive <= 0 or negative <= 0:
        raise Phase3Error(f"invalid M2 training support after excluding {excluded_folds}")
    model = factory(negative / positive)
    model.fit(x[training], y[training])
    raw = np.asarray(model.predict(x, raw_score=True), dtype=np.float64)
    probability = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(probability).all():
        raise Phase3Error("M2 replica produced non-finite output")
    sigmoid_ok = bool(np.allclose(_sigmoid(raw), probability, atol=TOLERANCE, rtol=TOLERANCE))
    return model, raw, probability, {
        "excluded_folds": list(excluded_folds),
        "train_folds": sorted(set(folds[training].tolist())),
        "train_sample_count": int(training.sum()),
        "positive_train": positive, "negative_train": negative,
        "scale_pos_weight": float(negative / positive),
        "sigmoid_raw_margin_matches_probability": sigmoid_ok,
    }


def generate_m2_anchors(data, *, m2_factory=None, progress_callback=None):
    factory = m2_factory or _default_m1_factory
    x, y, folds = data["X_m2"], data["y"], data["folds"]
    outer_raw = np.full(len(y), np.nan, dtype=np.float64)
    outer_probability = np.full(len(y), np.nan, dtype=np.float64)
    pair_raw = {}
    records = []
    historical_checks = []
    for fold in FOLDS:
        if progress_callback:
            progress_callback("M2_OUTER", fold, 5)
        _, raw, probability, evidence = _fit_m2(x, y, folds, (fold,), factory)
        validation = folds == fold
        outer_raw[validation] = raw[validation]
        outer_probability[validation] = probability[validation]
        aligned = bool(np.allclose(
            probability[validation], data["historical_probability"][validation],
            atol=TOLERANCE, rtol=TOLERANCE,
        ))
        historical_checks.append(aligned)
        records.append({
            "replica": f"outer_exclude_{fold}", "kind": "OUTER", **evidence,
            "historical_probability_alignment": aligned,
            "historical_raw_margin_reference": "UNAVAILABLE_BY_EXISTING_ARTIFACT",
        })

    pair_index = 0
    pair_valid = []
    for first in FOLDS:
        for second in FOLDS:
            if second <= first:
                continue
            pair_index += 1
            pair = (first, second)
            if progress_callback:
                progress_callback("M2_PAIR", pair_index, 10)
            _, raw, probability, evidence = _fit_m2(x, y, folds, pair, factory)
            # A true deterministic replay uses a fresh model with the same fixed spec.
            _, replay_raw, replay_probability, _ = _fit_m2(x, y, folds, pair, factory)
            replay_ok = bool(
                np.allclose(raw, replay_raw, atol=TOLERANCE, rtol=TOLERANCE)
                and np.allclose(probability, replay_probability, atol=TOLERANCE, rtol=TOLERANCE)
            )
            expected_train = sorted(set(FOLDS) - set(pair))
            valid = bool(
                evidence["train_folds"] == expected_train
                and evidence["sigmoid_raw_margin_matches_probability"]
                and replay_ok
            )
            pair_valid.append(valid)
            pair_raw[pair] = raw
            records.append({
                "replica": f"pair_exclude_{first}_{second}", "kind": "PAIR_EXCLUSION",
                **evidence, "deterministic_replay_consistent": replay_ok,
                "historical_pair_exclusion_reference": "NONE", "valid": valid,
            })
    if not np.isfinite(outer_raw).all() or not np.isfinite(outer_probability).all():
        raise Phase3Error("outer M2 OOF anchor incomplete")
    validity = {
        "OUTER_M2_HISTORICAL_ALIGNMENT": "PASS" if all(historical_checks) else "FAIL",
        "OUTER_RAW_MARGIN_ANCHOR_VALIDITY": "PASS" if all(
            row["sigmoid_raw_margin_matches_probability"] for row in records if row["kind"] == "OUTER"
        ) else "FAIL",
        "PAIR_EXCLUSION_M2_VALIDITY": "PASS" if all(pair_valid) else "FAIL",
    }
    validity["M2_COMPARATOR_ALIGNMENT_VALIDATOR"] = (
        "PASS" if all(value == "PASS" for value in validity.values()) else "FAIL"
    )
    return {
        "outer_raw": outer_raw, "outer_probability": outer_probability,
        "pair_raw": pair_raw, "replicas": records, "validity": validity,
    }


class ContextNormalizer:
    def __init__(self, base_mean, base_scale, delta_mean, delta_scale, evidence):
        self.base_mean = np.asarray(base_mean, dtype=np.float64)
        self.base_scale = np.asarray(base_scale, dtype=np.float64)
        self.delta_mean = np.asarray(delta_mean, dtype=np.float64)
        self.delta_scale = np.asarray(delta_scale, dtype=np.float64)
        self.evidence = evidence

    @staticmethod
    def _stats(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise Phase3Error("normalization universe must be finite and non-empty")
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        return mean, np.where(std < STD_EPSILON, 1.0, std)

    @classmethod
    def fit(cls, catalog, targets, outer_fold):
        selected = [
            row for row in targets
            if int(row["validation_fold"]) != int(outer_fold)
            and row["label"] in {"POSITIVE", "N1"}
        ]
        unique_neighbors = set()
        deltas = []
        for target in selected:
            frame = str(target["frame_id"]).zfill(6)
            for relation in target["neighbor_relations"]:
                unique_neighbors.add((frame, int(relation["fragment_identity"])))
                deltas.append([float(relation["delta_u"]), float(relation["delta_v"])])
        missing = unique_neighbors - set(catalog)
        if missing:
            raise Phase3Error(f"normalization neighbor missing from catalog: {sorted(missing)[:3]}")
        base = [catalog[key]["base24"] for key in sorted(unique_neighbors)]
        base_mean, base_scale = cls._stats(base)
        delta_mean, delta_scale = cls._stats(deltas)
        evidence = {
            "outer_fold": int(outer_fold), "P_N1_training_target_count": len(selected),
            "unique_context_neighbor_count": len(unique_neighbors),
            "relation_count": len(deltas),
            "outer_validation_frames_excluded": True,
        }
        return cls(base_mean, base_scale, delta_mean, delta_scale, evidence)

    def transform(self, catalog_row, delta_u, delta_v):
        base = (np.asarray(catalog_row["base24"], dtype=np.float64) - self.base_mean) / self.base_scale
        delta = (np.asarray([delta_u, delta_v], dtype=np.float64) - self.delta_mean) / self.delta_scale
        value = np.r_[base, delta]
        if value.shape != (NEIGHBOR_DIM,) or not np.isfinite(value).all():
            raise Phase3Error("context relation must normalize to finite 26-D")
        return value

    def payload(self):
        return {
            "neighbor_base24_mean": self.base_mean.tolist(),
            "neighbor_base24_scale": self.base_scale.tolist(),
            "delta_mean": self.delta_mean.tolist(), "delta_scale": self.delta_scale.tolist(),
            "zero_variance_epsilon": STD_EPSILON, **self.evidence,
        }


if torch is not None:
    class M2AnchoredContextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.neighbor_encoder = torch.nn.Sequential(
                torch.nn.Linear(26, 8), torch.nn.ReLU(),
                torch.nn.Linear(8, 4), torch.nn.ReLU(),
            )
            self.residual_head = torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1),
            )
            torch.nn.init.zeros_(self.residual_head[-1].weight)
            torch.nn.init.zeros_(self.residual_head[-1].bias)
            if sum(parameter.numel() for parameter in self.parameters()) != PARAMETER_COUNT:
                raise Phase3Error("context model parameter count changed")

        def forward(self, neighbors, mask):
            if neighbors.ndim != 3 or neighbors.shape[2] != 26:
                raise Phase3Error("neighbors must be [B,N,26]")
            if mask.shape != neighbors.shape[:2] or mask.dtype != torch.bool:
                raise Phase3Error("context mask must be boolean [B,N]")
            batch = neighbors.shape[0]
            delta = torch.zeros(batch, dtype=neighbors.dtype, device=neighbors.device)
            counts = mask.sum(dim=1)
            nonempty = counts > 0
            if torch.any(nonempty):
                encoded = self.neighbor_encoder(neighbors[nonempty])
                selected_mask = mask[nonempty].unsqueeze(-1).to(encoded.dtype)
                pooled = (encoded * selected_mask).sum(dim=1) / counts[nonempty].unsqueeze(1).to(encoded.dtype)
                delta[nonempty] = self.residual_head(pooled).squeeze(1)
            return delta
else:
    class M2AnchoredContextModel:
        def __init__(self):
            require_torch()


def _context_samples(data, normalizer, outer_fold):
    output = []
    for target in data["targets"]:
        frame = str(target["frame_id"]).zfill(6)
        neighbors = [
            normalizer.transform(
                data["catalog"][(frame, int(relation["fragment_identity"]))],
                relation["delta_u"], relation["delta_v"],
            )
            for relation in target["neighbor_relations"]
        ]
        output.append({
            "sample_row": int(target["sample_row"]), "fold": int(target["validation_fold"]),
            "label": target["label"],
            "neighbors": np.asarray(neighbors, dtype=np.float64).reshape(-1, 26),
        })
    return output


def _collate(samples, anchor_raw):
    maximum = max((len(row["neighbors"]) for row in samples), default=0)
    neighbors = torch.zeros((len(samples), maximum, 26), dtype=torch.float32)
    mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
    for index, row in enumerate(samples):
        count = len(row["neighbors"])
        if count:
            neighbors[index, :count] = torch.as_tensor(row["neighbors"], dtype=torch.float32)
            mask[index, :count] = True
    rows = np.asarray([row["sample_row"] for row in samples], dtype=np.int64)
    labels = torch.as_tensor(
        [1.0 if row["label"] == "POSITIVE" else 0.0 for row in samples], dtype=torch.float32
    )
    anchors = torch.as_tensor(anchor_raw[rows], dtype=torch.float32)
    return {"neighbors": neighbors, "mask": mask, "label": labels, "anchor": anchors, "rows": rows}


def _set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _post_model_validators(model):
    model.eval()
    with torch.no_grad():
        empty_neighbors = torch.zeros((3, 0, 26), dtype=torch.float32)
        empty_mask = torch.zeros((3, 0), dtype=torch.bool)
        empty = model(empty_neighbors, empty_mask)
        values = torch.arange(4 * 3 * 26, dtype=torch.float32).reshape(4, 3, 26) / 100.0
        mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
        original = model(values, mask)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(values[:, permutation], mask[:, permutation])
        probability = torch.sigmoid(original)
    return {
        "EMPTY_SET_EXACT_ZERO": bool(torch.equal(empty, torch.zeros_like(empty))),
        "PERMUTATION_INVARIANCE": bool(torch.allclose(original, permuted, atol=1e-7, rtol=1e-7)),
        "FINITE_DELTA_CONTEXT": bool(torch.isfinite(original).all()),
        "FINITE_PROBABILITY": bool(torch.isfinite(probability).all()),
    }


def _run_pretrain_validators(data, anchors, normalizers, samples_by_fold):
    split_ok = bool(
        len(data["assignments"]) == 3252
        and set(data["folds"].tolist()) == set(FOLDS)
        and len(set(map(_identity, data["assignments"]))) == 3252
    )
    feature_ok = bool(
        data["X_m2"].shape == (3252, 25)
        and len(MODEL_FEATURE_FIELDS) == 24
        and np.isfinite(data["X_m2"][:, :24]).all()
        and M1_PARAMETERS["random_state"] == 15531
        and M1_PARAMETERS["n_jobs"] == 1
    )
    nested_ok = all(
        sorted(row["train_folds"]) == sorted(set(FOLDS) - set(row["excluded_folds"]))
        for row in anchors["replicas"]
    )
    local_ok = True
    for target in data["targets"]:
        frame = str(target["frame_id"]).zfill(6)
        for relation in target["neighbor_relations"]:
            if (frame, int(relation["fragment_identity"])) not in data["catalog"]:
                local_ok = False
    normalization_ok = all(
        np.isfinite(normalizer.base_mean).all() and np.isfinite(normalizer.base_scale).all()
        and np.isfinite(normalizer.delta_mean).all() and np.isfinite(normalizer.delta_scale).all()
        and np.all(normalizer.base_scale > 0) and np.all(normalizer.delta_scale > 0)
        for normalizer in normalizers.values()
    )
    finite_ok = all(
        np.isfinite(row["neighbors"]).all()
        for samples in samples_by_fold.values() for row in samples
    )
    _set_determinism(15531)
    model = M2AnchoredContextModel().cpu()
    empty_check = _post_model_validators(model)["EMPTY_SET_EXACT_ZERO"]
    permutation_check = _post_model_validators(model)["PERMUTATION_INVARIANCE"]
    residual_init = bool(
        torch.count_nonzero(model.residual_head[-1].weight).item() == 0
        and torch.count_nonzero(model.residual_head[-1].bias).item() == 0
    )
    # Zero final layer means every initial residual, including non-empty sets, is exactly zero.
    with torch.no_grad():
        test_values = torch.randn(4, 3, 26)
        test_mask = torch.ones(4, 3, dtype=torch.bool)
        residual_init = residual_init and bool(torch.equal(model(test_values, test_mask), torch.zeros(4)))
    checks = {
        "FROZEN_SPLIT_IDENTITY_VALIDATOR": split_ok,
        "M2_FEATURE_CONFIG_IDENTITY_VALIDATOR": feature_ok,
        "NESTED_CROSSFIT_VALIDATOR": nested_ok,
        "M2_COMPARATOR_ALIGNMENT_VALIDATOR": anchors["validity"]["M2_COMPARATOR_ALIGNMENT_VALIDATOR"] == "PASS",
        "LOCAL_SET_IDENTITY_VALIDATOR": local_ok,
        "CONTEXT_NORMALIZATION_VALIDATOR": normalization_ok,
        "FINITE_PREPROCESSING_VALIDATOR": finite_ok,
        "EMPTY_SET_HARD_VALIDATOR": empty_check,
        "PERMUTATION_VALIDATOR": permutation_check,
        "RESIDUAL_INITIALIZATION_VALIDATOR": residual_init,
    }
    return {
        name: "PASS" if passed else "FAIL" for name, passed in checks.items()
    }


def _train_context_model(samples, anchors, seed, fold, checkpoint, progress_callback=None):
    _set_determinism(seed)
    model = M2AnchoredContextModel().cpu()
    training_samples = [
        row for row in samples if row["fold"] != fold and row["label"] in {"POSITIVE", "N1"}
    ]
    validation_samples = [row for row in samples if row["fold"] == fold]
    training_rows = np.asarray([row["sample_row"] for row in training_samples], dtype=np.int64)
    train_anchor = np.empty(3252, dtype=np.float64)
    for row in training_samples:
        pair = tuple(sorted((int(fold), int(row["fold"]))))
        train_anchor[row["sample_row"]] = anchors["pair_raw"][pair][row["sample_row"]]
    train = _collate(training_samples, train_anchor)
    validation = _collate(validation_samples, anchors["outer_raw"])
    positive = int(train["label"].sum().item())
    negative = int(len(train["label"]) - positive)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(negative / positive), dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=BETAS, eps=ADAM_EPS,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 100 + fold)
    losses, optimizer_steps, all_empty_batches = [], 0, 0
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train["label"]), generator=generator)
        weighted_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            delta = model(train["neighbors"][index], train["mask"][index])
            loss = criterion(train["anchor"][index] + delta, train["label"][index])
            if not torch.isfinite(loss):
                raise Phase3Error("context training loss is non-finite")
            weighted_loss += float(loss.detach()) * len(index)
            if bool(train["mask"][index].any()):
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
            else:
                all_empty_batches += 1
        losses.append(weighted_loss / len(order))
        if progress_callback and (epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS):
            progress_callback("CONTEXT", seed, fold, epoch, EPOCHS, losses[-1])

    model.eval()
    deltas = []
    with torch.no_grad():
        for start in range(0, len(validation["label"]), BATCH_SIZE):
            stop = start + BATCH_SIZE
            deltas.append(model(validation["neighbors"][start:stop], validation["mask"][start:stop]))
    delta = torch.cat(deltas).cpu().numpy().astype(np.float64)
    raw = anchors["outer_raw"][validation["rows"]] + delta
    probability = _sigmoid(raw)
    post = _post_model_validators(model)
    valid = all(post.values()) and np.isfinite(raw).all() and np.isfinite(probability).all()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION, "seed": seed, "fold": fold,
        "epoch": EPOCHS, "model_state_dict": model.state_dict(),
    }, checkpoint)
    return {
        "seed": seed, "fold": fold, "validation_rows": validation["rows"],
        "delta_context": delta, "raw_margin": raw, "probability": probability,
        "training_loss_by_epoch": losses, "optimizer_step_count": optimizer_steps,
        "all_empty_batch_count": all_empty_batches, "post_model_validators": post,
        "valid": bool(valid), "checkpoint": str(checkpoint),
        "pos_weight": float(negative / positive),
    }


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"valid_N": 0}
    return {
        "valid_N": int(len(values)), "min": float(np.min(values)),
        "P25": float(np.percentile(values, 25)), "P50": float(np.median(values)),
        "P75": float(np.percentile(values, 75)), "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _evaluate_gates(seed_results, fold_results, comparator, y, labels, folds):
    m2_a = comparator["Layer_A_P_vs_N0"]
    m2_b = comparator["Layer_B_P_vs_N1"]
    a_fp_allowance = max(5, math.ceil(0.05 * m2_a["N0_FP_at_0_50"]))
    all_fp_allowance = max(5, math.ceil(0.05 * m2_a["overall_FP_at_0_50"]))
    a_ap = np.asarray([row["metrics"]["Layer_A_P_vs_N0"]["average_precision"] for row in seed_results])
    b_ap = np.asarray([row["metrics"]["Layer_B_P_vs_N1"]["average_precision"] for row in seed_results])
    n0_fp = np.asarray([row["metrics"]["Layer_A_P_vs_N0"]["N0_FP_at_0_50"] for row in seed_results])
    recall = np.asarray([row["metrics"]["Layer_A_P_vs_N0"]["P_recall_at_0_50"] for row in seed_results])
    overall_fp = np.asarray([row["metrics"]["Layer_A_P_vs_N0"]["overall_FP_at_0_50"] for row in seed_results])
    a_checks = {
        "A1_median_AP": float(np.median(a_ap)) >= m2_a["average_precision"] - 0.02,
        "A2_median_N0_FP": float(np.median(n0_fp)) <= m2_a["N0_FP_at_0_50"] + a_fp_allowance,
        "A3_median_P_recall": float(np.median(recall)) >= m2_a["P_recall_at_0_50"] - 0.05,
        "A4_median_overall_FP": float(np.median(overall_fp)) <= m2_a["overall_FP_at_0_50"] + all_fp_allowance,
    }
    b_checks = {
        "B1_median_AP_gain": float(np.median(b_ap)) - m2_b["average_precision"] >= 0.05,
        "B2_improving_seed_count": int(np.sum(b_ap > m2_b["average_precision"])) >= 4,
    }
    stable_seeds = 0
    for index in range(5):
        stable_seeds += int(
            a_ap[index] >= m2_a["average_precision"] - 0.02
            and n0_fp[index] <= m2_a["N0_FP_at_0_50"] + a_fp_allowance
            and recall[index] >= m2_a["P_recall_at_0_50"] - 0.05
            and overall_fp[index] <= m2_a["overall_FP_at_0_50"] + all_fp_allowance
            and b_ap[index] > m2_b["average_precision"]
        )
    fold_summary = []
    for fold in FOLDS:
        mask = folds == fold
        m2_fold = _layer_metrics(y[mask], labels[mask], comparator["scores"][mask])
        rows = [row for row in fold_results if row["fold"] == fold]
        median_a = float(np.median([row["metrics"]["Layer_A_P_vs_N0"]["average_precision"] for row in rows]))
        median_b = float(np.median([row["metrics"]["Layer_B_P_vs_N1"]["average_precision"] for row in rows]))
        median_fpr = float(np.median([row["metrics"]["Layer_A_P_vs_N0"]["N0_FPR_at_0_50"] for row in rows]))
        fold_summary.append({
            "fold": fold, "delta_AP_A": median_a - m2_fold["Layer_A_P_vs_N0"]["average_precision"],
            "delta_AP_B": median_b - m2_fold["Layer_B_P_vs_N1"]["average_precision"],
            "delta_N0_FPR": median_fpr - m2_fold["Layer_A_P_vs_N0"]["N0_FPR_at_0_50"],
        })
    fold_checks = {
        "B_positive_at_least_4_folds": sum(row["delta_AP_B"] > 0 for row in fold_summary) >= 4,
        "B_no_fold_below_minus_0_05": all(row["delta_AP_B"] >= -0.05 for row in fold_summary),
        "A_within_minus_0_05_at_least_4_folds": sum(row["delta_AP_A"] >= -0.05 for row in fold_summary) >= 4,
        "A_no_fold_below_minus_0_10": all(row["delta_AP_A"] >= -0.10 for row in fold_summary),
        "N0_FPR_within_plus_0_02_at_least_4_folds": sum(row["delta_N0_FPR"] <= 0.02 for row in fold_summary) >= 4,
        "N0_FPR_no_fold_above_plus_0_05": all(row["delta_N0_FPR"] <= 0.05 for row in fold_summary),
    }
    gates = {
        "LAYER_A_PRESERVATION_GATE": {"result": "PASS" if all(a_checks.values()) else "FAIL", "checks": a_checks},
        "LAYER_B_INCREMENTAL_GAIN_GATE": {"result": "PASS" if all(b_checks.values()) else "FAIL", "checks": b_checks},
        "SEED_STABILITY_GATE": {"result": "PASS" if stable_seeds >= 4 else "FAIL", "stable_seed_count": stable_seeds},
        "FOLD_STABILITY_GATE": {"result": "PASS" if all(fold_checks.values()) else "FAIL", "checks": fold_checks, "folds": fold_summary},
    }
    return gates


def map_phase3_signal(gates, execution_valid=True):
    if not execution_valid:
        return "NOT_EVALUATED"
    a = gates["LAYER_A_PRESERVATION_GATE"]["result"] == "PASS"
    b = gates["LAYER_B_INCREMENTAL_GAIN_GATE"]["result"] == "PASS"
    stable = gates["SEED_STABILITY_GATE"]["result"] == "PASS" and gates["FOLD_STABILITY_GATE"]["result"] == "PASS"
    if not a and not b:
        return "NOT_SUPPORTED"
    if not a and b:
        return "NOT_ACCEPTABLE_LAYER_A_REGRESSION"
    if a and not b:
        return "NO_INCREMENTAL_CONTEXT_VALUE"
    if not stable:
        return "UNSTABLE"
    return "SUPPORTED"


def train_phase3(output_dir, *, progress_callback=None, m2_factory=None):
    require_torch()
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    data = _load_phase3_inputs(output_dir)
    anchors = generate_m2_anchors(data, m2_factory=m2_factory, progress_callback=progress_callback)
    normalizers = {fold: ContextNormalizer.fit(data["catalog"], data["targets"], fold) for fold in FOLDS}
    samples_by_fold = {fold: _context_samples(data, normalizers[fold], fold) for fold in FOLDS}
    validators = _run_pretrain_validators(data, anchors, normalizers, samples_by_fold)
    readiness = "PASS" if all(value == "PASS" for value in validators.values()) else "FAIL"
    if readiness != "PASS":
        raise Phase3Error(f"PHASE3_TRAINING_READINESS=FAIL: {validators}")

    labels, y, folds = data["labels"], data["y"], data["folds"]
    comparator_metrics = _layer_metrics(y, labels, anchors["outer_probability"])
    comparator = {**comparator_metrics, "scores": anchors["outer_probability"]}
    model_dir = output_dir / "phase3_context_models"
    seed_results, fold_results, prediction_rows, per_frame_rows = [], [], [], []
    residual_by_label = defaultdict(list)
    residual_by_size = defaultdict(list)
    transitions = {name: Counter() for name in ("M2_wrong_to_Phase3_correct", "M2_correct_to_Phase3_wrong")}
    execution_valid = True
    for seed in TRAINING_SEEDS:
        oof_probability = np.full(3252, np.nan)
        oof_raw = np.full(3252, np.nan)
        oof_delta = np.full(3252, np.nan)
        seed_folds = []
        for fold in FOLDS:
            result = _train_context_model(
                samples_by_fold[fold], anchors, seed, fold,
                model_dir / f"seed_{seed}_fold_{fold}_epoch_100.pt",
                progress_callback=progress_callback,
            )
            rows = result.pop("validation_rows")
            oof_probability[rows] = result.pop("probability")
            oof_raw[rows] = result.pop("raw_margin")
            oof_delta[rows] = result.pop("delta_context")
            execution_valid = execution_valid and result["valid"]
            mask = folds == fold
            fold_record = {**result, "metrics": _layer_metrics(y[mask], labels[mask], oof_probability[mask])}
            fold_results.append(fold_record)
            seed_folds.append(fold_record)
        if not np.isfinite(oof_probability).all() or not np.isfinite(oof_delta).all():
            execution_valid = False
        metrics = _layer_metrics(y, labels, oof_probability)
        seed_results.append({"seed": seed, "metrics": metrics, "folds": seed_folds})
        per_frame_rows.extend(_per_frame(seed, data["targets"], y, labels, oof_probability))
        for index in range(3252):
            size = len(data["targets"][index]["neighbor_relations"])
            bucket = "EMPTY" if size == 0 else ("ONE_MEMBER" if size == 1 else "MULTI_MEMBER")
            residual_by_label[str(labels[index])].append(oof_delta[index])
            residual_by_size[bucket].append(oof_delta[index])
            m2_correct = (anchors["outer_probability"][index] >= 0.5) == bool(y[index])
            phase3_correct = (oof_probability[index] >= 0.5) == bool(y[index])
            if not m2_correct and phase3_correct:
                transitions["M2_wrong_to_Phase3_correct"][str(labels[index])] += 1
            if m2_correct and not phase3_correct:
                transitions["M2_correct_to_Phase3_wrong"][str(labels[index])] += 1
            prediction_rows.append({
                "seed": seed, "sample_row": index, "frame_id": data["frames"][index],
                "canonical_fragment_identity": data["assignments"][index]["canonical_fragment_identity"],
                "label": labels[index], "validation_fold": int(folds[index]),
                "M2_probability": anchors["outer_probability"][index],
                "Phase3_probability": oof_probability[index], "delta_context": oof_delta[index],
                "Phase3_raw_margin": oof_raw[index], "local_set_size": size,
            })

    gates = _evaluate_gates(seed_results, fold_results, comparator, y, labels, folds)
    signal = map_phase3_signal(gates, execution_valid)
    result = {
        "schema_version": SCHEMA_VERSION,
        "execution_identity": _git_identity(Path.cwd()),
        "historical_artifact_limitation": {
            "HISTORICAL_M2_RAW_MARGIN_REFERENCE": "UNAVAILABLE_BY_EXISTING_ARTIFACT",
            "historical_probability_alignment": anchors["validity"]["OUTER_M2_HISTORICAL_ALIGNMENT"],
            "Phase3_outer_raw_margin_anchor_generation": anchors["validity"]["OUTER_RAW_MARGIN_ANCHOR_VALIDITY"],
            "pair_exclusion_M2_replica_validation": anchors["validity"]["PAIR_EXCLUSION_M2_VALIDITY"],
        },
        "M2_replica_execution": "15/15 SUCCESS",
        "M2_replicas": anchors["replicas"],
        "pretrain_validators": validators,
        "PHASE3_TRAINING_READINESS": readiness,
        "context_model_execution": "25/25 SUCCESS" if execution_valid else "INVALID",
        "PHASE3_EXECUTION_VALIDITY": "PASS" if execution_valid else "FAIL",
        "M2_comparator": comparator_metrics,
        "seed_results": seed_results,
        "fold_results": fold_results,
        "per_frame": per_frame_rows,
        "normalization_by_outer_fold": {str(key): value.payload() for key, value in normalizers.items()},
        "residual_diagnostics": {
            "by_label": {
                label: {"delta_context": _stats(values), "abs_delta_context": _stats(np.abs(values))}
                for label, values in residual_by_label.items()
            },
            "by_local_set_bucket": {
                bucket: {"delta_context": _stats(values), "abs_delta_context": _stats(np.abs(values))}
                for bucket, values in residual_by_size.items()
            },
            "classification_transitions": {name: dict(counts) for name, counts in transitions.items()},
        },
        "Development_Gates": gates,
        "PHASE3_DEVELOPMENT_SIGNAL": signal,
        "INDEPENDENT_MANIFEST_STATUS": "SEALED",
        "FINAL_FIT_PERFORMED": False,
        "FIXED_100_EXECUTED": False,
        "INDEPENDENT_EVALUATION_PERFORMED": False,
    }
    result = _to_builtin(result)
    result_path = output_dir / "phase3_m2_anchored_context_result.json"
    oof_path = output_dir / "phase3_context_oof_predictions.csv"
    anchor_path = output_dir / "phase3_m2_outer_anchors.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    with anchor_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_row", "frame_id", "canonical_fragment_identity", "validation_fold", "historical_M2_probability", "regenerated_M2_probability", "Phase3_outer_raw_margin_anchor"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(data["assignments"]):
            writer.writerow({
                "sample_row": index, "frame_id": row["frame_id"],
                "canonical_fragment_identity": row["canonical_fragment_identity"],
                "validation_fold": row["validation_fold"],
                "historical_M2_probability": data["historical_probability"][index],
                "regenerated_M2_probability": anchors["outer_probability"][index],
                "Phase3_outer_raw_margin_anchor": anchors["outer_raw"][index],
            })
    return result, result_path, oof_path, anchor_path
