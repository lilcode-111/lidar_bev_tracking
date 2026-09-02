"""Frozen LFRR-v1 representation, preprocessing, batching, and manifest semantics.

This module deliberately contains no optimizer or training loop.  Day 1 builds
the immutable development representation and provides the objects exercised by
the pre-training hard validators.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from bev_tracking.fragment_learning_dataset import (
    MODEL_FEATURE_FIELDS,
    extract_runtime_fragment_features,
)
from bev_tracking.fragment_learning_failure_analysis import _rebuild_frame_fragments
from bev_tracking.fragment_learning_training import FRAGMENT_TYPE_ENCODING
from bev_tracking.fragment_phase0 import best_endpoint_relation

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Allows manifest/data validation before torch install.
    torch = None
    nn = None


SCHEMA_VERSION = "lfrr-v1-development-representation-v1"
MODEL_NAME = "Small Target-Conditioned Deep Sets"
TARGET_DIM = 25
NEIGHBOR_DIM = 26
EMBEDDING_DIM = 16
EXPECTED_PARAMETER_COUNT = 2353
SUPERVISED_LABELS = {"POSITIVE", "N1", "N0"}
INDEPENDENT_SELECTION_SEED = 15601
INDEPENDENT_FRAME_COUNT = 100
STD_EPSILON = 1.0e-12


class LFRRV1Error(ValueError):
    pass


def _read_ids(path):
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            values.append(value.zfill(6))
    return values


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base24_vector(features):
    if tuple(features) != MODEL_FEATURE_FIELDS:
        raise LFRRV1Error("F_BASE24 schema/order changed")
    values = []
    for field in MODEL_FEATURE_FIELDS:
        if field == "fragment_type":
            try:
                value = FRAGMENT_TYPE_ENCODING[features[field]]
            except KeyError as exc:
                raise LFRRV1Error(
                    f"unknown frozen fragment_type: {features[field]}"
                ) from exc
        else:
            value = float(features[field])
        values.append(float(value))
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (24,) or not np.isfinite(array).all():
        raise LFRRV1Error("F_BASE24 must be a finite 24-D numeric vector")
    return array


def _load_assignments(output_dir):
    output_dir = Path(output_dir)
    split = json.loads(
        (output_dir / "fragment_learning_splits.json").read_text(encoding="utf-8")
    )
    if split.get("SPLIT_VALIDITY") != "PASS":
        raise LFRRV1Error("frozen 5-fold split is not valid")
    rows = sorted(
        split.get("sample_assignments", []), key=lambda row: int(row["sample_row"])
    )
    if len(rows) != 3252 or [int(row["sample_row"]) for row in rows] != list(range(3252)):
        raise LFRRV1Error("frozen supervised sample identity changed")
    for row in rows:
        if row["label"] not in SUPERVISED_LABELS:
            raise LFRRV1Error("unexpected supervised label")
    return rows


def _load_dataset_identities(output_dir):
    identities = defaultdict(set)
    path = Path(output_dir) / "fragment_dataset.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row["metadata_fields"]
            key = (
                str(metadata["frame_id"]).zfill(6),
                int(metadata["canonical_fragment_identity"]),
            )
            if key[1] in identities[key[0]]:
                raise LFRRV1Error(f"duplicate frozen fragment identity: {key}")
            identities[key[0]].add(key[1])
    return identities


def _load_frozen_margins(output_dir, assignments):
    rows = _read_csv(Path(output_dir) / "lightgbm_v2_margin_feature.csv")
    if len(rows) != len(assignments):
        raise LFRRV1Error("frozen margin/sample count mismatch")
    output = {}
    for assignment, row in zip(assignments, rows):
        expected = (
            int(assignment["sample_row"]),
            str(assignment["frame_id"]).zfill(6),
            int(assignment["canonical_fragment_identity"]),
        )
        actual = (
            int(row["sample_row"]), str(row["frame_id"]).zfill(6),
            int(row["canonical_fragment_identity"]),
        )
        if actual != expected:
            raise LFRRV1Error(f"frozen margin identity mismatch: {actual} != {expected}")
        valid = str(row["margin_valid"]).lower() == "true"
        value = float(row["seed_relation_endpoint_gap_margin"]) if valid else None
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise LFRRV1Error("valid target margin must be finite and non-negative")
        output[actual[1:]] = {
            "value": value,
            "valid": valid,
            "best_seed_component_runtime_id": (
                None if row["best_seed_component_runtime_id"] == ""
                else int(row["best_seed_component_runtime_id"])
            ),
            "second_seed_component_runtime_id": (
                None if row["second_seed_component_runtime_id"] == ""
                else int(row["second_seed_component_runtime_id"])
            ),
        }
    return output


def _fragment_record(fragment, valid_components):
    features = extract_runtime_fragment_features(fragment, valid_components)
    relation = best_endpoint_relation(fragment, valid_components)
    best_seed = (
        int(relation["seed_component_runtime_id"])
        if relation.get("has_computable_seed_relation") else None
    )
    return {
        "fragment_identity": int(fragment["runtime_id"]),
        "base24": _base24_vector(features).tolist(),
        "best_seed_component_runtime_id": best_seed,
        "center_xy": [float(value) for value in fragment["center"]],
    }


def construct_frame_representation(
    frame_id, fragments, component_by_id, valid_components, target_rows, frozen_margins
):
    """Construct one frame without GT, labels, scores, radius, or neighbor truncation."""
    frame_id = str(frame_id).zfill(6)
    catalog = {
        int(identity): _fragment_record(fragment, valid_components)
        for identity, fragment in sorted(fragments.items())
    }
    by_seed = defaultdict(list)
    for identity, item in catalog.items():
        seed_id = item["best_seed_component_runtime_id"]
        if seed_id is not None:
            by_seed[seed_id].append(identity)
    for members in by_seed.values():
        members.sort()

    targets = []
    for assignment in sorted(target_rows, key=lambda row: int(row["sample_row"])):
        identity = int(assignment["canonical_fragment_identity"])
        if identity not in catalog:
            raise LFRRV1Error(f"target fragment replay missing: {frame_id}/{identity}")
        target = catalog[identity]
        seed_id = target["best_seed_component_runtime_id"]
        if seed_id is None:
            neighbor_ids = []
            major = minor = None
        else:
            neighbor_ids = [value for value in by_seed[seed_id] if value != identity]
            if seed_id not in component_by_id:
                raise LFRRV1Error("target best seed component unavailable")
            geometry = component_by_id[seed_id].geometry
            major = np.asarray(geometry.major, dtype=np.float64)
            minor = np.asarray(geometry.minor, dtype=np.float64)
            if major.shape != (2,) or minor.shape != (2,):
                raise LFRRV1Error("seed PCA frame must be 2-D")
        neighbors = []
        target_center = np.asarray(target["center_xy"], dtype=np.float64)
        for neighbor_id in neighbor_ids:
            neighbor = catalog[neighbor_id]
            delta = np.asarray(neighbor["center_xy"], dtype=np.float64) - target_center
            delta_u = float(delta @ major)
            delta_v = float(delta @ minor)
            if not np.isfinite([delta_u, delta_v]).all():
                raise LFRRV1Error("non-finite local relation")
            neighbors.append({
                "fragment_identity": int(neighbor_id),
                "delta_u": delta_u,
                "delta_v": delta_v,
            })
        margin = frozen_margins[(frame_id, identity)]
        if margin["best_seed_component_runtime_id"] != seed_id:
            raise LFRRV1Error(
                f"best-seed identity differs from frozen M2: {frame_id}/{identity}"
            )
        targets.append({
            "sample_row": int(assignment["sample_row"]),
            "frame_id": frame_id,
            "canonical_fragment_identity": identity,
            "label": str(assignment["label"]),
            "validation_fold": int(assignment["validation_fold"]),
            "target_margin": margin["value"],
            "target_margin_valid": bool(margin["valid"]),
            "best_seed_component_runtime_id": seed_id,
            "neighbor_relations": neighbors,
        })
    return catalog, targets


def build_lfrr_development_representation(
    output_dir, data_root, *, progress_callback=None
):
    """Replay the frozen 64 frames and write a compact, GT-free LFRR representation."""
    output_dir = Path(output_dir)
    assignments = _load_assignments(output_dir)
    frozen_ids = _load_dataset_identities(output_dir)
    margins = _load_frozen_margins(output_dir, assignments)
    by_frame = defaultdict(list)
    for row in assignments:
        by_frame[str(row["frame_id"]).zfill(6)].append(row)
    if set(by_frame) != set(frozen_ids) or len(by_frame) != 64:
        raise LFRRV1Error("frozen 64-frame identity changed")

    catalog_path = output_dir / "lfrr_v1_fragment_catalog.jsonl"
    target_path = output_dir / "lfrr_v1_development_targets.jsonl"
    temporary_catalog = catalog_path.with_suffix(".jsonl.tmp")
    temporary_target = target_path.with_suffix(".jsonl.tmp")
    local_sizes = []
    label_sizes = defaultdict(list)
    fragment_count = 0
    try:
        with temporary_catalog.open("w", encoding="utf-8") as catalog_handle, temporary_target.open(
            "w", encoding="utf-8"
        ) as target_handle:
            for position, frame_id in enumerate(sorted(by_frame), start=1):
                if progress_callback:
                    progress_callback(position, len(by_frame), frame_id)
                _, fragments, component_by_id, valid_components = _rebuild_frame_fragments(
                    data_root, frame_id
                )
                if set(fragments) != frozen_ids[frame_id]:
                    missing = sorted(frozen_ids[frame_id] - set(fragments))[:10]
                    extra = sorted(set(fragments) - frozen_ids[frame_id])[:10]
                    raise LFRRV1Error(
                        f"REPLAY_IDENTITY_MISMATCH: {frame_id}: missing={missing}, extra={extra}"
                    )
                catalog, targets = construct_frame_representation(
                    frame_id, fragments, component_by_id, valid_components,
                    by_frame[frame_id], margins,
                )
                for identity, item in sorted(catalog.items()):
                    catalog_handle.write(json.dumps({
                        "frame_id": frame_id,
                        "canonical_fragment_identity": identity,
                        **item,
                    }, separators=(",", ":")) + "\n")
                    fragment_count += 1
                for target in targets:
                    size = len(target["neighbor_relations"])
                    local_sizes.append(size)
                    label_sizes[target["label"]].append(size)
                    target_handle.write(json.dumps(target, separators=(",", ":")) + "\n")
        temporary_catalog.replace(catalog_path)
        temporary_target.replace(target_path)
    except Exception:
        temporary_catalog.unlink(missing_ok=True)
        temporary_target.unlink(missing_ok=True)
        raise

    if fragment_count != sum(len(values) for values in frozen_ids.values()):
        raise LFRRV1Error("catalog fragment count changed")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": 64,
        "fragment_catalog_count": fragment_count,
        "supervised_target_count": len(assignments),
        "target_dim": TARGET_DIM,
        "neighbor_dim": NEIGHBOR_DIM,
        "local_set": {
            "empty": int(sum(value == 0 for value in local_sizes)),
            "one_member": int(sum(value == 1 for value in local_sizes)),
            "multi_member": int(sum(value >= 2 for value in local_sizes)),
            "median": float(np.median(local_sizes)),
            "p95": float(np.percentile(local_sizes, 95)),
            "max": int(max(local_sizes)),
            "by_label": {
                label: {
                    "count": len(values),
                    "empty": int(sum(value == 0 for value in values)),
                    "one_member": int(sum(value == 1 for value in values)),
                    "multi_member": int(sum(value >= 2 for value in values)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max": int(max(values)),
                }
                for label, values in sorted(label_sizes.items())
            },
        },
        "artifacts": {
            "fragment_catalog": str(catalog_path),
            "development_targets": str(target_path),
        },
        "TRAINING_STARTED": False,
        "OPTIMIZER_STEP_EXECUTED": False,
        "INDEPENDENT_RESULTS_OBSERVED": False,
    }
    summary_path = output_dir / "lfrr_v1_day1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, summary_path


def load_compact_representation(output_dir):
    output_dir = Path(output_dir)
    catalog = {}
    with (output_dir / "lfrr_v1_fragment_catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["frame_id"]).zfill(6), int(row["canonical_fragment_identity"]))
            if key in catalog:
                raise LFRRV1Error(f"duplicate compact catalog identity: {key}")
            catalog[key] = row
    targets = []
    with (output_dir / "lfrr_v1_development_targets.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            targets.append(json.loads(line))
    if len(targets) != 3252:
        raise LFRRV1Error("compact target count changed")
    return catalog, targets


class FoldNormalizer:
    """Train-fold-only z-score statistics for the frozen LFRR tensors."""

    def __init__(self, base_mean, base_scale, delta_mean, delta_scale, margin_mean, margin_scale, evidence):
        self.base_mean = np.asarray(base_mean, dtype=np.float64)
        self.base_scale = np.asarray(base_scale, dtype=np.float64)
        self.delta_mean = np.asarray(delta_mean, dtype=np.float64)
        self.delta_scale = np.asarray(delta_scale, dtype=np.float64)
        self.margin_mean = float(margin_mean)
        self.margin_scale = float(margin_scale)
        self.evidence = dict(evidence)

    @staticmethod
    def _stats(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
            raise LFRRV1Error("normalization universe must be finite and non-empty")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0)
        scale = np.where(std < STD_EPSILON, 1.0, std)
        return mean, scale

    @classmethod
    def fit(cls, catalog, targets, validation_fold):
        training = [row for row in targets if int(row["validation_fold"]) != int(validation_fold)]
        if not training:
            raise LFRRV1Error("empty training fold")
        used = set()
        deltas = []
        margins = []
        for target in training:
            target_key = (str(target["frame_id"]).zfill(6), int(target["canonical_fragment_identity"]))
            used.add(target_key)
            if target["target_margin"] is not None:
                margins.append(float(target["target_margin"]))
            for relation in target["neighbor_relations"]:
                used.add((target_key[0], int(relation["fragment_identity"])))
                deltas.append([float(relation["delta_u"]), float(relation["delta_v"])])
        if not margins:
            raise LFRRV1Error("training fold has zero valid margins")
        missing = sorted(used - set(catalog))[:10]
        if missing:
            raise LFRRV1Error(f"normalization catalog identity missing: {missing}")
        base = np.asarray([catalog[key]["base24"] for key in sorted(used)], dtype=np.float64)
        base_mean, base_scale = cls._stats(base)
        delta_mean, delta_scale = cls._stats(deltas)
        margin_mean, margin_scale = cls._stats(np.asarray(margins)[:, None])
        return cls(
            base_mean, base_scale, delta_mean, delta_scale,
            margin_mean[0], margin_scale[0],
            {
                "validation_fold": int(validation_fold),
                "training_target_count": len(training),
                "unique_base24_fragment_count": len(used),
                "relation_edge_count": len(deltas),
                "valid_margin_count": len(margins),
            },
        )

    def transform_target(self, catalog_row, margin):
        base = (np.asarray(catalog_row["base24"], dtype=np.float64) - self.base_mean) / self.base_scale
        raw_margin = self.margin_mean if margin is None else float(margin)
        value = np.r_[base, (raw_margin - self.margin_mean) / self.margin_scale]
        if value.shape != (TARGET_DIM,) or not np.isfinite(value).all():
            raise LFRRV1Error("normalized target must be finite 25-D")
        return value

    def transform_neighbor(self, catalog_row, delta_u, delta_v):
        base = (np.asarray(catalog_row["base24"], dtype=np.float64) - self.base_mean) / self.base_scale
        delta = (
            np.asarray([delta_u, delta_v], dtype=np.float64) - self.delta_mean
        ) / self.delta_scale
        value = np.r_[base, delta]
        if value.shape != (NEIGHBOR_DIM,) or not np.isfinite(value).all():
            raise LFRRV1Error("normalized neighbor must be finite 26-D")
        return value

    def payload(self):
        return {
            "base24_mean": self.base_mean.tolist(),
            "base24_scale": self.base_scale.tolist(),
            "delta_u_mean": float(self.delta_mean[0]),
            "delta_u_scale": float(self.delta_scale[0]),
            "delta_v_mean": float(self.delta_mean[1]),
            "delta_v_scale": float(self.delta_scale[1]),
            "margin_mean": self.margin_mean,
            "margin_scale": self.margin_scale,
            "zero_variance_epsilon": STD_EPSILON,
            **self.evidence,
        }


def materialize_fold_samples(catalog, targets, normalizer, *, validation_fold=None, training=False):
    selected = [
        row for row in targets
        if (int(row["validation_fold"]) != int(validation_fold) if training else int(row["validation_fold"]) == int(validation_fold))
    ] if validation_fold is not None else list(targets)
    samples = []
    for target in selected:
        key = (str(target["frame_id"]).zfill(6), int(target["canonical_fragment_identity"]))
        target_vector = normalizer.transform_target(catalog[key], target["target_margin"])
        neighbors = [
            normalizer.transform_neighbor(
                catalog[(key[0], int(relation["fragment_identity"]))],
                relation["delta_u"], relation["delta_v"],
            )
            for relation in target["neighbor_relations"]
        ]
        samples.append({
            "sample_row": int(target["sample_row"]),
            "frame_id": key[0],
            "canonical_fragment_identity": key[1],
            "label": target["label"],
            "target": target_vector,
            "neighbors": np.asarray(neighbors, dtype=np.float64).reshape(-1, NEIGHBOR_DIM),
        })
    return samples


def require_torch():
    if torch is None:
        raise LFRRV1Error(
            "PyTorch is required for LFRR batching/model validation; install a CPU build first"
        )


def collate_lfrr_samples(samples):
    require_torch()
    if not samples:
        raise LFRRV1Error("cannot collate an empty target batch")
    batch = len(samples)
    maximum = max(len(item["neighbors"]) for item in samples)
    targets = torch.as_tensor(
        np.asarray([item["target"] for item in samples]), dtype=torch.float32
    )
    neighbors = torch.zeros((batch, maximum, NEIGHBOR_DIM), dtype=torch.float32)
    mask = torch.zeros((batch, maximum), dtype=torch.bool)
    for index, item in enumerate(samples):
        count = len(item["neighbors"])
        if count:
            neighbors[index, :count] = torch.as_tensor(item["neighbors"], dtype=torch.float32)
            mask[index, :count] = True
    labels = torch.as_tensor(
        [1.0 if item["label"] == "POSITIVE" else 0.0 for item in samples],
        dtype=torch.float32,
    )
    return {
        "target": targets,
        "neighbors": neighbors,
        "neighbor_mask": mask,
        "label": labels,
        "sample_rows": [item["sample_row"] for item in samples],
    }


if nn is not None:
    class SmallTargetConditionedDeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.neighbor_encoder = nn.Sequential(
                nn.Linear(NEIGHBOR_DIM, 32, bias=True), nn.ReLU(),
                nn.Linear(32, EMBEDDING_DIM, bias=True), nn.ReLU(),
            )
            self.target_encoder = nn.Sequential(
                nn.Linear(TARGET_DIM, EMBEDDING_DIM, bias=True), nn.ReLU()
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(32, 16, bias=True), nn.ReLU(), nn.Linear(16, 1, bias=True)
            )
            if sum(parameter.numel() for parameter in self.parameters()) != EXPECTED_PARAMETER_COUNT:
                raise LFRRV1Error("frozen model parameter count changed")

        def forward(self, target, neighbors, neighbor_mask, *, return_pooled=False):
            if target.ndim != 2 or target.shape[1] != TARGET_DIM:
                raise LFRRV1Error("target tensor must be [B,25]")
            if neighbors.ndim != 3 or neighbors.shape[0] != target.shape[0] or neighbors.shape[2] != NEIGHBOR_DIM:
                raise LFRRV1Error("neighbor tensor must be [B,N,26]")
            if neighbor_mask.shape != neighbors.shape[:2] or neighbor_mask.dtype != torch.bool:
                raise LFRRV1Error("neighbor mask must be boolean [B,N]")
            if neighbors.shape[1] == 0:
                pooled = torch.zeros(
                    (target.shape[0], EMBEDDING_DIM), dtype=target.dtype, device=target.device
                )
            else:
                encoded = self.neighbor_encoder(neighbors)
                encoded = encoded * neighbor_mask.unsqueeze(-1).to(encoded.dtype)
                pooled = encoded.sum(dim=1)
            target_embedding = self.target_encoder(target)
            logits = self.prediction_head(torch.cat((target_embedding, pooled), dim=1)).squeeze(1)
            if return_pooled:
                return logits, pooled
            return logits
else:
    class SmallTargetConditionedDeepSets:
        def __init__(self, *args, **kwargs):
            require_torch()


def _phase0_frame_ids(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = _read_csv(path)
        values = {str(row["frame_id"]).zfill(6) for row in rows}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = set()
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if "frame_id" in item:
                    values.add(str(item["frame_id"]).zfill(6))
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    if len(values) != 9:
        raise LFRRV1Error(
            f"frozen Fragment Phase-0 evidence must contain 9 unique frames, found {len(values)}"
        )
    return values


def register_independent_manifest(
    complete_frame_ids,
    learning_manifest,
    fixed100_manifest,
    phase0_evidence,
    output_path,
    *,
    selection_seed=INDEPENDENT_SELECTION_SEED,
    frame_count=INDEPENDENT_FRAME_COUNT,
):
    """Register frame identities only; this API has no label or model input."""
    learning = set(_read_ids(learning_manifest))
    fixed100 = set(_read_ids(fixed100_manifest))
    phase0 = _phase0_frame_ids(phase0_evidence)
    if len(learning) != 64 or len(fixed100) != 100:
        raise LFRRV1Error("frozen learning64/fixed100 manifest identity changed")
    exclusion = learning | fixed100 | phase0
    complete = sorted({str(value).zfill(6) for value in complete_frame_ids})
    eligible = [value for value in complete if value not in exclusion]
    if len(eligible) < frame_count:
        raise LFRRV1Error("insufficient independent candidate frame identities")
    selected = list(eligible)
    random.Random(selection_seed).shuffle(selected)
    selected = sorted(selected[:frame_count])
    if len(selected) != frame_count or set(selected) & exclusion:
        raise LFRRV1Error("independent manifest selection invariant failed")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(selected) + "\n"
    if output_path.exists() and output_path.read_text(encoding="utf-8") != content:
        raise LFRRV1Error("existing independent manifest differs from frozen selection")
    output_path.write_text(content, encoding="utf-8")
    return {
        "manifest": str(output_path),
        "frame_count": len(selected),
        "selection_seed": int(selection_seed),
        "complete_frame_identity_count": len(complete),
        "exclusion_count": len(exclusion),
        "learning64_count": len(learning),
        "historical_fixed100_count": len(fixed100),
        "phase0_frame_count": len(phase0),
        "phase0_extra_exclusion_frames": sorted(phase0 - learning - fixed100),
        "label_or_result_inspection_performed": False,
        "INDEPENDENT_MANIFEST_GENERATED": True,
        "INDEPENDENT_RESULTS_OBSERVED": False,
    }
