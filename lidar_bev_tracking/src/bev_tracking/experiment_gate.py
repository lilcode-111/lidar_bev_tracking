import hashlib
import json
from pathlib import Path


A0_PRIME_BASELINE_ID = "A0_prime_geometry_corrected"
REQUIRED_IOU_KEYS = ("0.50", "0.25")
REQUIRED_METRIC_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1", "neutralized_detections")


class ExperimentGateError(ValueError):
    pass


def sha256_file(path):
    """Hash UTF-8 manifest content after normalizing platform line endings."""
    text = Path(path).read_text(encoding="utf-8")
    canonical_text = text.replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256()
    digest.update(canonical_text.encode("utf-8"))
    return digest.hexdigest()


def load_a0_prime_declaration(path, repo_root="."):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        declaration = json.load(f)
    validate_a0_prime_declaration(declaration, repo_root=repo_root)
    return declaration


def validate_a0_prime_declaration(declaration, repo_root="."):
    require_dict(declaration, "baseline declaration")
    if declaration.get("baseline_id") != A0_PRIME_BASELINE_ID:
        raise ExperimentGateError(f"baseline_id must be {A0_PRIME_BASELINE_ID}")
    if declaration.get("status") != "frozen":
        raise ExperimentGateError("A0 prime baseline status must be frozen")
    if declaration.get("role") != "unique_algorithm_experiment_baseline":
        raise ExperimentGateError("A0 prime baseline role is invalid")

    source_run = require_dict(declaration.get("source_run"), "source_run")
    required_source_fields = {
        "run_id",
        "run_directory",
        "batch_status",
        "git_commit",
        "git_dirty",
        "config_hash",
        "num_frames",
        "metric_valid_frames",
        "failed_frames",
        "skipped_frames",
    }
    require_fields(source_run, required_source_fields, "source_run")
    if source_run["batch_status"] != "success" or source_run["git_dirty"] is not False:
        raise ExperimentGateError("A0 prime source run must be successful and git clean")
    if source_run["num_frames"] != 100 or source_run["metric_valid_frames"] != 100:
        raise ExperimentGateError("A0 prime source run must contain 100 metric-valid frames")
    if source_run["failed_frames"] != 0 or source_run["skipped_frames"] != 0:
        raise ExperimentGateError("A0 prime source run must not contain failed or skipped frames")

    dataset = require_dict(declaration.get("dataset"), "dataset")
    require_fields(
        dataset,
        {
            "data_root",
            "frame_manifest_path",
            "frame_manifest_sha256",
            "diagnostic_manifest_path",
            "diagnostic_manifest_sha256",
            "diagnostic_num_frames",
        },
        "dataset",
    )
    validate_manifest_hash(repo_root, dataset["frame_manifest_path"], dataset["frame_manifest_sha256"])
    validate_manifest_hash(repo_root, dataset["diagnostic_manifest_path"], dataset["diagnostic_manifest_sha256"])
    if dataset["diagnostic_num_frames"] != 25:
        raise ExperimentGateError("diagnostic manifest must contain 25 frames")

    algorithm_config = require_dict(declaration.get("algorithm_config"), "algorithm_config")
    detector = require_dict(algorithm_config.get("detector"), "algorithm_config.detector")
    require_fields(detector, {"eps", "min_points", "oriented", "z_min", "intensity_min"}, "detector")
    if float(detector["intensity_min"]) != 0.38:
        raise ExperimentGateError("A0 prime intensity_min must be 0.38")

    metrics_by_iou = require_dict(declaration.get("metrics_by_iou"), "metrics_by_iou")
    for iou_key in REQUIRED_IOU_KEYS:
        metrics = require_dict(metrics_by_iou.get(iou_key), f"metrics_by_iou.{iou_key}")
        require_fields(metrics, set(REQUIRED_METRIC_FIELDS), f"metrics_by_iou.{iou_key}")
        if int(metrics["tp"]) + int(metrics["fn"]) != 322:
            raise ExperimentGateError(f"TP + FN conservation failed for IoU {iou_key}")

    totals = require_dict(declaration.get("totals"), "totals")
    require_fields(
        totals,
        {
            "num_positive_gt",
            "num_raw_detections",
            "num_car_detections_before_nms",
            "effective_car_detection_count_iou_0_50",
            "effective_car_detection_count_iou_0_25",
        },
        "totals",
    )
    for iou_key, total_key in (
        ("0.50", "effective_car_detection_count_iou_0_50"),
        ("0.25", "effective_car_detection_count_iou_0_25"),
    ):
        metrics = metrics_by_iou[iou_key]
        effective = int(metrics["tp"]) + int(metrics["fp"]) + int(metrics["neutralized_detections"])
        if effective != int(totals[total_key]):
            raise ExperimentGateError(f"effective detection count mismatch for IoU {iou_key}")


def validate_manifest_hash(repo_root, relative_path, expected_hash):
    path = Path(repo_root) / relative_path
    if not path.is_file():
        raise ExperimentGateError(f"declared manifest not found: {path}")
    actual_hash = sha256_file(path)
    if actual_hash.lower() != str(expected_hash).lower():
        raise ExperimentGateError(f"manifest hash mismatch: {relative_path}")


def require_dict(value, description):
    if not isinstance(value, dict):
        raise ExperimentGateError(f"{description} must be an object")
    return value


def require_fields(mapping, required_fields, description):
    missing = sorted(required_fields - set(mapping))
    if missing:
        raise ExperimentGateError(f"{description} missing fields: {', '.join(missing)}")
