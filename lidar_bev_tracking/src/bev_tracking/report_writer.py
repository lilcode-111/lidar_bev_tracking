import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from bev_tracking.batch_pipeline import build_batch_result, iou_suffix
from bev_tracking.error_codes import ErrorCode, ErrorStage, FrameStatus
from bev_tracking.result_types import FailureCategory, FrameError, NUMERIC_COUNT_FIELDS, TIME_FIELDS, to_json_compatible


SCHEMA_VERSION = "13.0"
FAILURE_ANALYSIS_SCHEMA_VERSION = "14.0"
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


class ReportWriteError(RuntimeError):
    def __init__(self, error_code, output_path, cause):
        self.error_code = ErrorCode(error_code)
        self.output_path = Path(output_path)
        self.cause = cause
        super().__init__(f"{self.error_code.value}: {self.output_path}: {cause}")


def write_batch_report(
    batch_result,
    output_root="outputs/kitti_batch_eval",
    config_input_path=None,
    config_effective=None,
    task_name="kitti_car_batch",
    command=None,
    started_at=None,
    finished_at=None,
    total_start_time=None,
):
    config_hash = compute_config_hash(config_effective or {})
    run_id = make_run_id(task_name, config_hash)
    run_dir = create_run_dir(output_root, run_id)
    paths = report_paths(run_dir)
    started_at = started_at or utc_now_iso()
    total_start_time = total_start_time if total_start_time is not None else perf_counter()

    write_config_input(config_input_path, paths["config_input"])
    write_required_json(paths["config_effective"], config_effective or {}, ErrorCode.CONFIG_SNAPSHOT_WRITE_FAILED)
    git_metadata = capture_git_metadata()
    write_required_json(paths["git_metadata"], git_metadata, ErrorCode.GIT_METADATA_UNAVAILABLE)

    updated_frames = []
    for frame_result in batch_result.frame_results:
        updated_frames.append(write_frame_report(frame_result, paths["per_frame_report_dir"]))

    final_batch = rebuild_batch_after_report_write(
        batch_result,
        updated_frames,
        run_id=run_id,
        started_at=started_at,
        finished_at=None,
        total_time_ms=None,
        config_hash=config_hash,
        git_metadata=git_metadata,
        artifacts={key: str(value) for key, value in paths.items()},
    )
    write_required_json(paths["frame_manifest"], build_frame_manifest(final_batch), ErrorCode.FRAME_MANIFEST_WRITE_FAILED)
    write_required_csv(
        paths["frames_csv"],
        [frame_to_csv_row(frame) for frame in final_batch.frame_results],
        CSV_FIELDNAMES,
        ErrorCode.CSV_WRITE_FAILED,
    )
    final_batch.finished_at = finished_at or utc_now_iso()
    final_batch.total_time_ms = elapsed_ms(total_start_time) if total_start_time is not None else None
    summary = build_summary(final_batch, command=command)
    write_required_json(paths["summary_json"], summary, ErrorCode.SUMMARY_WRITE_FAILED)
    return final_batch, paths


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
    try:
        candidate.mkdir(parents=True)
        (candidate / "frames").mkdir()
    except Exception as exc:
        raise ReportWriteError(ErrorCode.OUTPUT_DIR_CREATE_FAILED, candidate, exc) from exc
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


def write_failure_cases_report(run_directory, failure_cases, top_k=5):
    run_directory = Path(run_directory)
    output_path = run_directory / "failure_cases.json"
    payload = build_failure_cases_report(run_directory, failure_cases, top_k)
    write_required_json(path=output_path, data=payload, error_code=ErrorCode.FAILURE_CASES_WRITE_FAILED)
    return payload, output_path


def build_failure_cases_report(run_directory, failure_cases, top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    serialized_cases = [case.to_dict() if hasattr(case, "to_dict") else dict(case) for case in failure_cases]
    category_order = [category.value for category in FailureCategory]
    counts_by_category = {category: 0 for category in category_order}
    for case in serialized_cases:
        category = case.get("category")
        if category not in counts_by_category:
            raise ValueError(f"unknown failure category: {category}")
        counts_by_category[category] += 1

    source_run_ids = {case.get("source_run_id") for case in serialized_cases if case.get("source_run_id") is not None}
    if len(source_run_ids) > 1:
        raise ValueError("failure cases contain multiple source run ids")
    source_run_id = next(iter(source_run_ids), Path(run_directory).name)

    thresholds = {float(case.get("primary_iou_threshold", 0.5)) for case in serialized_cases}
    if len(thresholds) > 1:
        raise ValueError("failure cases contain multiple primary IoU thresholds")
    primary_iou_threshold = next(iter(thresholds), 0.5)

    return {
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "source": {
            "run_id": source_run_id,
            "run_directory": Path(run_directory),
            "primary_iou_threshold": primary_iou_threshold,
        },
        "selection": {
            "top_k": top_k,
            "category_order": category_order,
            "effective_car_detection_count": "tp + fp + neutralized_detections @ IoU=0.50",
        },
        "summary": {
            "total_failure_cases": len(serialized_cases),
            "counts_by_category": counts_by_category,
        },
        "failure_cases": serialized_cases,
    }


def write_config_input(config_input_path, output_path):
    output_path = Path(output_path)
    try:
        if config_input_path is None:
            atomic_write_text(output_path, "")
            return
        config_input_path = Path(config_input_path)
        if config_input_path.exists():
            atomic_write_text(output_path, config_input_path.read_text(encoding="utf-8"))
        else:
            atomic_write_text(output_path, str(config_input_path))
    except Exception as exc:
        raise ReportWriteError(ErrorCode.CONFIG_SNAPSHOT_WRITE_FAILED, output_path, exc) from exc


def write_frame_report(frame_result, per_frame_report_dir):
    frame_report_path = Path(per_frame_report_dir) / f"{str(frame_result.frame_id).zfill(6)}.json"
    try:
        atomic_write_json(frame_report_path, frame_result.to_dict())
        frame_result.artifacts["frame_report_path"] = frame_report_path
        return frame_result
    except Exception as exc:
        frame_result.artifacts.pop("frame_report_path", None)
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


def rebuild_batch_after_report_write(batch_result, frame_results, run_id, started_at, finished_at, total_time_ms, config_hash, git_metadata, artifacts):
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
    rebuilt.total_time_ms = total_time_ms
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


def write_required_json(path, data, error_code):
    try:
        atomic_write_json(path, data)
    except ReportWriteError:
        raise
    except Exception as exc:
        raise ReportWriteError(error_code, path, exc) from exc


def write_required_csv(path, rows, fieldnames, error_code):
    try:
        atomic_write_csv(path, rows, fieldnames)
    except ReportWriteError:
        raise
    except Exception as exc:
        raise ReportWriteError(error_code, path, exc) from exc


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
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


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


def elapsed_ms(start_time):
    return float((perf_counter() - start_time) * 1000.0)
