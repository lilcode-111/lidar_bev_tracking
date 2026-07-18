import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result, iou_suffix
from bev_tracking.error_codes import BatchStatus, ErrorCode, ErrorStage, FrameStatus
from bev_tracking.result_types import BatchResult, FrameError, FrameResult, NUMERIC_COUNT_FIELDS, TIME_FIELDS, to_json_compatible


SCHEMA_VERSION = "13.0"
REPORT_IOS = ("0.50", "0.25")
CSV_FIELDNAMES = [
    "frame_id",
    "status",
    "metric_valid",
    "error_code",
    "error_stage",
    "error_message",
    "warning_codes",
    "num_labels_raw",
    "num_positive_gt",
    "num_neutral_gt",
    "num_excluded_gt",
    "num_dontcare",
    "num_gt_outside_roi",
    "num_invalid_gt",
    "num_raw_detections",
    "num_car_detections_before_nms",
    "num_detections_after_nms",
    "num_ignored_detection_class",
    "num_detections_outside_roi",
    "num_suppressed_by_nms",
    "tp_iou_0_50",
    "fp_iou_0_50",
    "fn_iou_0_50",
    "precision_iou_0_50",
    "recall_iou_0_50",
    "f1_iou_0_50",
    "neutralized_iou_0_50",
    "tp_iou_0_25",
    "fp_iou_0_25",
    "fn_iou_0_25",
    "precision_iou_0_25",
    "recall_iou_0_25",
    "f1_iou_0_25",
    "neutralized_iou_0_25",
    "load_time_ms",
    "parse_time_ms",
    "gt_transform_time_ms",
    "detection_time_ms",
    "nms_time_ms",
    "evaluation_time_ms",
    "total_time_ms",
    "frame_report_path",
]


def write_batch_report(
    batch_result,
    output_root="outputs/kitti_batch_eval",
    config_input_path=None,
    config_effective=None,
    task_name="kitti_car_batch",
    command=None,
    started_at=None,
    finished_at=None,
):
    config_hash = compute_config_hash(config_effective or {})
    run_id = make_run_id(task_name, config_hash)
    run_dir = create_run_dir(output_root, run_id)
    paths = report_paths(run_dir)
    started_at = started_at or utc_now_iso()

    try:
        write_config_input(config_input_path, paths["config_input"])
        atomic_write_json(paths["config_effective"], config_effective or {})
        git_metadata = capture_git_metadata()
        atomic_write_json(paths["git_metadata"], git_metadata)
        atomic_write_json(paths["frame_manifest"], build_frame_manifest(batch_result))

        updated_frames = []
        for frame_result in batch_result.frame_results:
            updated_frames.append(write_frame_report(frame_result, paths["per_frame_report_dir"]))

        final_batch = rebuild_batch_after_report_write(
            batch_result,
            updated_frames,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at or utc_now_iso(),
            config_hash=config_hash,
            git_metadata=git_metadata,
            artifacts={key: str(value) for key, value in paths.items()},
        )
        atomic_write_csv(paths["frames_csv"], [frame_to_csv_row(frame) for frame in final_batch.frame_results], CSV_FIELDNAMES)
        summary = build_summary(final_batch, command=command)
        atomic_write_json(paths["summary_json"], summary)
        return final_batch, paths
    except Exception:
        raise


def make_run_id(task_name, config_hash, now=None):
    now = now or datetime.now(timezone.utc)
    safe_task_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in task_name).strip("_")
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{safe_task_name}_{config_hash[:8]}"


def create_run_dir(output_root, run_id):
    output_root = Path(output_root)
    candidate = output_root / run_id
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{run_id}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    (candidate / "frames").mkdir()
    return candidate


def report_paths(run_dir):
    run_dir = Path(run_dir)
    return {
        "summary_json": run_dir / "summary.json",
        "frames_csv": run_dir / "frames.csv",
        "config_input": run_dir / "config_input.yaml",
        "config_effective": run_dir / "config_effective.json",
        "git_metadata": run_dir / "git.json",
        "frame_manifest": run_dir / "frame_manifest.json",
        "per_frame_report_dir": run_dir / "frames",
    }


def write_config_input(config_input_path, output_path):
    output_path = Path(output_path)
    if config_input_path is None:
        atomic_write_text(output_path, "")
        return
    config_input_path = Path(config_input_path)
    if config_input_path.exists():
        atomic_write_text(output_path, config_input_path.read_text(encoding="utf-8"))
    else:
        atomic_write_text(output_path, str(config_input_path))


def write_frame_report(frame_result, per_frame_report_dir):
    frame_report_path = Path(per_frame_report_dir) / f"{str(frame_result.frame_id).zfill(6)}.json"
    try:
        frame_result.artifacts["frame_report_path"] = frame_report_path
        atomic_write_json(frame_report_path, frame_result.to_dict())
        return frame_result
    except Exception as exc:
        if frame_result.metric_valid:
            frame_result.status = FrameStatus.PARTIAL_SUCCESS
            frame_result.error = FrameError(
                error_code=ErrorCode.FRAME_REPORT_WRITE_FAILED,
                error_stage=ErrorStage.FRAME_REPORT,
                error_message=str(exc),
                exception_type=type(exc).__name__,
                input_path=frame_report_path,
            )
        frame_result.warnings.append({"code": ErrorCode.FRAME_REPORT_WRITE_FAILED.value, "message": str(exc)})
        return frame_result


def rebuild_batch_after_report_write(batch_result, frame_results, run_id, started_at, finished_at, config_hash, git_metadata, artifacts):
    parameters = batch_result.totals.get("parameters", {})
    rebuilt = build_batch_result(
        frame_results=frame_results,
        data_root=batch_result.totals.get("data_root", "data/kitti"),
        frame_ids=batch_result.artifacts.get("requested_frame_ids") or [frame.frame_id for frame in frame_results],
        eps=parameters.get("eps", 0.6),
        min_points=parameters.get("min_points", 20),
        oriented=batch_result.totals.get("box_mode") == "oriented_pca",
        nms_iou_threshold=parameters.get("nms_iou_threshold", 0.3),
        eval_iou_threshold=parameters.get("eval_iou_threshold", 0.5),
        auxiliary_iou_thresholds=parameters.get("auxiliary_iou_thresholds", [0.25]),
    )
    rebuilt.schema_version = SCHEMA_VERSION
    rebuilt.run_id = run_id
    rebuilt.started_at = started_at
    rebuilt.finished_at = finished_at
    rebuilt.config_hash = config_hash
    rebuilt.git_commit = git_metadata.get("commit")
    rebuilt.git_dirty = git_metadata.get("dirty")
    rebuilt.artifacts.update(artifacts)
    return rebuilt


def build_summary(batch_result, command=None):
    batch_dict = batch_result.to_dict()
    artifacts = batch_dict["artifacts"]
    requested_frame_ids = batch_result.artifacts.get("requested_frame_ids", [])
    status_by_frame = {frame.frame_id: frame.status.value for frame in batch_result.frame_results}

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": batch_result.run_id,
            "batch_status": batch_result.status.value,
            "started_at": batch_result.started_at,
            "finished_at": batch_result.finished_at,
            "duration_ms": batch_result.total_time_ms,
            "command": command if command is not None else " ".join(sys.argv),
        },
        "reproducibility": {
            "config_input_path": artifacts.get("config_input"),
            "config_effective_path": artifacts.get("config_effective"),
            "config_hash": batch_result.config_hash,
            "git_commit": batch_result.git_commit,
            "git_dirty": batch_result.git_dirty,
            "data_root": batch_result.totals.get("data_root"),
            "frame_manifest_path": artifacts.get("frame_manifest"),
            "requested_frame_ids": requested_frame_ids,
            "evaluation_policy_version": "12.1",
            "primary_iou_threshold": 0.5,
            "iou_thresholds": list(REPORT_IOS),
        },
        "frames": {
            **batch_result.frame_counts,
            "status_by_frame": status_by_frame,
            "csv_path": artifacts.get("frames_csv"),
            "manifest_path": artifacts.get("frame_manifest"),
        },
        "metrics_by_iou": batch_dict["metrics_by_iou"],
        "totals": batch_dict["totals"],
        "errors": {
            "count": sum(batch_result.error_counts.values()),
            "counts_by_code": batch_result.error_counts,
            "counts_by_stage": batch_result.error_stage_counts,
            "items": [frame_error_item(frame) for frame in batch_result.frame_results if frame.error is not None],
        },
        "artifacts": artifacts,
        "warnings": batch_dict["warnings"],
    }


def build_frame_manifest(batch_result):
    return {
        "requested_frame_ids": batch_result.artifacts.get("requested_frame_ids", []),
        "frames": [
            {
                "frame_id": frame.frame_id,
                "status": frame.status.value,
                "metric_valid": frame.metric_valid,
            }
            for frame in batch_result.frame_results
        ],
    }


def frame_error_item(frame_result):
    error = frame_result.error.to_dict() if hasattr(frame_result.error, "to_dict") else frame_result.error
    return {
        "frame_id": frame_result.frame_id,
        "status": frame_result.status.value,
        **error,
    }


def frame_to_csv_row(frame_result):
    frame = frame_result.to_dict()
    error = frame.get("error") or {}
    row = {
        "frame_id": frame["frame_id"],
        "status": frame["status"],
        "metric_valid": frame["metric_valid"],
        "error_code": error.get("error_code", ""),
        "error_stage": error.get("error_stage", ""),
        "error_message": csv_safe(error.get("error_message", "")),
        "warning_codes": ",".join(str(warning.get("code", "")) for warning in frame.get("warnings", []) if isinstance(warning, dict)),
        "frame_report_path": frame.get("frame_report_path", ""),
    }
    for field_name in NUMERIC_COUNT_FIELDS + TIME_FIELDS:
        row[field_name] = csv_metric(frame.get(field_name))
    for iou_key in REPORT_IOS:
        suffix = iou_suffix(iou_key)
        metrics = frame_result.metrics_by_iou.get(iou_key)
        if frame_result.metric_valid and metrics is not None:
            row[f"tp_{suffix}"] = metrics.tp
            row[f"fp_{suffix}"] = metrics.fp
            row[f"fn_{suffix}"] = metrics.fn
            row[f"precision_{suffix}"] = csv_metric(metrics.precision)
            row[f"recall_{suffix}"] = csv_metric(metrics.recall)
            row[f"f1_{suffix}"] = csv_metric(metrics.f1)
            row[f"neutralized_{suffix}"] = metrics.neutralized_detections
        else:
            row[f"tp_{suffix}"] = ""
            row[f"fp_{suffix}"] = ""
            row[f"fn_{suffix}"] = ""
            row[f"precision_{suffix}"] = ""
            row[f"recall_{suffix}"] = ""
            row[f"f1_{suffix}"] = ""
            row[f"neutralized_{suffix}"] = ""
    return row


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(to_json_compatible(data), indent=2) + "\n")


def atomic_write_csv(path, rows, fieldnames):
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp_path, path)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def compute_config_hash(config):
    text = json.dumps(to_json_compatible(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capture_git_metadata(repo_root=None):
    repo_root = repo_root or Path.cwd()
    try:
        commit = run_git(["rev-parse", "HEAD"], repo_root)
        dirty = bool(run_git(["status", "--porcelain"], repo_root))
        return {"commit": commit, "dirty": dirty, "available": True}
    except Exception as exc:
        return {"commit": None, "dirty": None, "available": False, "error": str(exc)}


def run_git(args, repo_root):
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()


def csv_metric(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def csv_safe(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
