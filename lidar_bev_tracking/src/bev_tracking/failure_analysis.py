import csv
import json
from pathlib import Path

from bev_tracking.batch_pipeline import build_batch_result, iou_suffix
from bev_tracking.error_codes import FrameStatus
from bev_tracking.result_types import (
    NUMERIC_COUNT_FIELDS,
    TIME_FIELDS,
    BatchResult,
    FailureCase,
    FailureCategory,
    FrameError,
    FrameMetrics,
    FrameResult,
)


DEFAULT_TOP_K = 5
PRIMARY_IOU_KEY = "0.50"
PRIMARY_IOU_THRESHOLD = 0.5
REQUIRED_RUN_PATHS = {
    "summary": "summary.json",
    "frames_csv": "frames.csv",
    "manifest": "frame_manifest.json",
    "frames_dir": "frames",
}


class FailureAnalysisError(ValueError):
    pass


class SourceContractError(FailureAnalysisError):
    pass


def generate_failure_cases(batch_result, top_k=DEFAULT_TOP_K, primary_iou_key=PRIMARY_IOU_KEY):
    top_k = validate_top_k(top_k)
    frames = collect_eligible_frames(batch_result, primary_iou_key)
    cases = []

    cases.extend(rank_category(frames, FailureCategory.MOST_FALSE_NEGATIVES, top_k, most_false_negatives_key, has_false_negatives))
    cases.extend(rank_category(frames, FailureCategory.MOST_FALSE_POSITIVES, top_k, most_false_positives_key, has_false_positives))
    cases.extend(rank_category(frames, FailureCategory.LOWEST_RECALL, top_k, lowest_recall_key, has_low_recall))
    cases.extend(rank_category(frames, FailureCategory.ZERO_DETECTION_WITH_GT, top_k, zero_detection_with_gt_key, has_zero_detection_with_gt))
    cases.extend(
        rank_category(
            frames,
            FailureCategory.HIGHEST_EFFECTIVE_CAR_DETECTIONS,
            top_k,
            highest_effective_car_detections_key,
            has_effective_car_detections,
            diagnostic_only=True,
        )
    )
    return cases


def generate_failure_cases_from_run_directory(
    run_directory,
    top_k=DEFAULT_TOP_K,
    primary_iou_key=PRIMARY_IOU_KEY,
):
    batch_result = load_batch_result_from_run_directory(run_directory, primary_iou_key=primary_iou_key)
    return generate_failure_cases(batch_result, top_k=top_k, primary_iou_key=primary_iou_key)


def load_batch_result_from_run_directory(run_directory, primary_iou_key=PRIMARY_IOU_KEY):
    run_directory = Path(run_directory)
    paths = resolve_run_paths(run_directory)
    summary = read_json_object(paths["summary"], "summary")
    manifest = read_json_object(paths["manifest"], "frame manifest")
    csv_rows = read_csv_rows(paths["frames_csv"])

    requested_frame_ids = requested_ids_from_summary(summary)
    validate_requested_sources(requested_frame_ids, summary, manifest, csv_rows)

    manifest_by_id = index_records(manifest.get("frames"), "manifest frames")
    csv_by_id = index_records(csv_rows, "frames.csv rows")
    report_paths = expected_frame_report_paths(paths["frames_dir"], requested_frame_ids)
    validate_frame_file_set(paths["frames_dir"], requested_frame_ids)

    frame_results = []
    status_by_frame = require_dict(require_dict(summary.get("frames"), "summary.frames").get("status_by_frame"), "summary.frames.status_by_frame")
    run_id = require_dict(summary.get("run"), "summary.run").get("run_id")

    for frame_id in requested_frame_ids:
        frame_payload = read_json_object(report_paths[frame_id], f"frame report {frame_id}")
        frame_result = frame_result_from_payload(
            frame_payload,
            report_paths[frame_id],
            run_id=run_id,
            run_directory=run_directory,
        )
        validate_frame_sources(
            frame_id,
            frame_result,
            frame_payload,
            manifest_by_id[frame_id],
            csv_by_id[frame_id],
            status_by_frame,
            report_paths[frame_id],
            primary_iou_key,
        )
        frame_results.append(frame_result)

    batch_result = rebuild_batch_from_disk(summary, frame_results, requested_frame_ids, run_directory, primary_iou_key)
    validate_summary_contract(summary, batch_result, primary_iou_key)
    return batch_result


def validate_top_k(top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def resolve_run_paths(run_directory):
    if not run_directory.is_dir():
        raise SourceContractError(f"run directory not found: {run_directory}")

    paths = {name: run_directory / relative_path for name, relative_path in REQUIRED_RUN_PATHS.items()}
    for name, path in paths.items():
        expected = path.is_dir() if name == "frames_dir" else path.is_file()
        if not expected:
            raise SourceContractError(f"required run artifact not found: {path}")
    return paths


def read_json_object(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"unable to read {description}: {path}: {exc}") from exc
    return require_dict(value, description)


def read_csv_rows(path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        raise SourceContractError(f"unable to read frames CSV: {path}: {exc}") from exc


def require_dict(value, description):
    if not isinstance(value, dict):
        raise SourceContractError(f"{description} must be an object")
    return value


def requested_ids_from_summary(summary):
    reproducibility = require_dict(summary.get("reproducibility"), "summary.reproducibility")
    requested = reproducibility.get("requested_frame_ids")
    if not isinstance(requested, list):
        raise SourceContractError("summary.reproducibility.requested_frame_ids must be a list")
    return normalize_unique_frame_ids(requested, "summary requested_frame_ids")


def normalize_unique_frame_ids(frame_ids, description):
    normalized = [str(frame_id).zfill(6) for frame_id in frame_ids]
    if len(set(normalized)) != len(normalized):
        raise SourceContractError(f"duplicate frame_id in {description}")
    return normalized


def validate_requested_sources(requested_frame_ids, summary, manifest, csv_rows):
    summary_frames = require_dict(summary.get("frames"), "summary.frames")
    if summary_frames.get("requested") != len(requested_frame_ids):
        raise SourceContractError("summary requested count does not match requested_frame_ids")

    manifest_requested = manifest.get("requested_frame_ids")
    if not isinstance(manifest_requested, list):
        raise SourceContractError("manifest requested_frame_ids must be a list")
    manifest_requested = normalize_unique_frame_ids(manifest_requested, "manifest requested_frame_ids")
    if manifest_requested != requested_frame_ids:
        raise SourceContractError("manifest requested_frame_ids do not match summary")

    manifest_frames = manifest.get("frames")
    if not isinstance(manifest_frames, list) or len(manifest_frames) != len(requested_frame_ids):
        raise SourceContractError("manifest frame count does not match requested count")
    if len(csv_rows) != len(requested_frame_ids):
        raise SourceContractError("frames.csv row count does not match requested count")

    manifest_ids = set(index_records(manifest_frames, "manifest frames"))
    csv_ids = set(index_records(csv_rows, "frames.csv rows"))
    requested_set = set(requested_frame_ids)
    if manifest_ids != requested_set or csv_ids != requested_set:
        raise SourceContractError("frame_id sets differ across summary, manifest, and frames.csv")


def index_records(records, description):
    if not isinstance(records, list):
        raise SourceContractError(f"{description} must be a list")
    indexed = {}
    for record in records:
        if not isinstance(record, dict) or "frame_id" not in record:
            raise SourceContractError(f"{description} contains an invalid record")
        frame_id = str(record["frame_id"]).zfill(6)
        if frame_id in indexed:
            raise SourceContractError(f"duplicate frame_id in {description}: {frame_id}")
        indexed[frame_id] = record
    return indexed


def expected_frame_report_paths(frames_dir, frame_ids):
    return {frame_id: Path(frames_dir) / f"{frame_id}.json" for frame_id in frame_ids}


def validate_frame_file_set(frames_dir, requested_frame_ids):
    actual_ids = {path.stem for path in Path(frames_dir).glob("*.json")}
    expected_ids = set(requested_frame_ids)
    if actual_ids != expected_ids:
        raise SourceContractError("per-frame JSON file set does not match requested frame_ids")


def frame_result_from_payload(payload, frame_report_path, run_id, run_directory):
    try:
        metrics_by_iou = {
            str(iou_key): FrameMetrics(**require_dict(metrics, f"metrics {iou_key}"))
            for iou_key, metrics in require_dict(payload.get("metrics_by_iou", {}), "frame metrics_by_iou").items()
        }
        error_payload = payload.get("error")
        error = FrameError(**error_payload) if isinstance(error_payload, dict) else None
        artifacts = dict(require_dict(payload.get("artifacts", {}), "frame artifacts"))
        artifacts.update(
            {
                "frame_report_path": Path(frame_report_path),
                "source_run_id": run_id,
                "source_run_directory": Path(run_directory),
            }
        )
        field_values = {name: payload.get(name) for name in NUMERIC_COUNT_FIELDS + TIME_FIELDS}
        return FrameResult(
            frame_id=payload.get("frame_id"),
            status=payload.get("status"),
            metrics_by_iou=metrics_by_iou,
            error=error,
            warnings=list(payload.get("warnings", [])),
            artifacts=artifacts,
            **field_values,
        )
    except (TypeError, ValueError) as exc:
        raise SourceContractError(f"invalid frame report {frame_report_path}: {exc}") from exc


def validate_frame_sources(
    frame_id,
    frame_result,
    frame_payload,
    manifest_record,
    csv_record,
    status_by_frame,
    expected_report_path,
    primary_iou_key,
):
    if str(frame_payload.get("frame_id")).zfill(6) != frame_id:
        raise SourceContractError(f"per-frame report frame_id mismatch: {frame_id}")

    status = frame_result.status.value
    metric_valid = frame_result.metric_valid
    expected_statuses = [manifest_record.get("status"), csv_record.get("status"), status_by_frame.get(frame_id)]
    if any(value != status for value in expected_statuses):
        raise SourceContractError(f"status mismatch across reports for frame {frame_id}")

    payload_metric_valid = frame_payload.get("metric_valid")
    manifest_metric_valid = manifest_record.get("metric_valid")
    csv_metric_valid = parse_csv_bool(csv_record.get("metric_valid"), frame_id)
    if any(value is not metric_valid for value in [payload_metric_valid, manifest_metric_valid, csv_metric_valid]):
        raise SourceContractError(f"metric_valid mismatch across reports for frame {frame_id}")

    recorded_path = csv_record.get("frame_report_path")
    if not path_matches_expected(recorded_path, expected_report_path):
        raise SourceContractError(f"frame report path mismatch for frame {frame_id}")

    validate_csv_counts(frame_result, csv_record, frame_id)
    validate_primary_metrics(frame_result, csv_record, frame_id, primary_iou_key)


def parse_csv_bool(value, frame_id):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise SourceContractError(f"invalid metric_valid value in frames.csv for frame {frame_id}")


def path_matches_expected(recorded_path, expected_path):
    if not recorded_path:
        return False
    recorded = Path(recorded_path)
    expected = Path(expected_path)
    if recorded.is_absolute():
        return recorded.resolve() == expected.resolve()
    recorded_parts = tuple(recorded.parts[-3:])
    expected_parts = tuple(expected.parts[-3:])
    return recorded_parts == expected_parts


def validate_csv_counts(frame_result, csv_record, frame_id):
    for field_name in NUMERIC_COUNT_FIELDS + TIME_FIELDS:
        if field_name not in csv_record:
            continue
        expected = getattr(frame_result, field_name)
        actual = csv_record.get(field_name, "")
        if expected is None:
            if actual != "":
                raise SourceContractError(f"CSV field mismatch for frame {frame_id}: {field_name}")
            continue
        if not numeric_values_equal(actual, expected):
            raise SourceContractError(f"CSV field mismatch for frame {frame_id}: {field_name}")


def validate_primary_metrics(frame_result, csv_record, frame_id, primary_iou_key):
    suffix = iou_suffix(primary_iou_key)
    fields = {
        f"tp_{suffix}": "tp",
        f"fp_{suffix}": "fp",
        f"fn_{suffix}": "fn",
        f"precision_{suffix}": "precision",
        f"recall_{suffix}": "recall",
        f"f1_{suffix}": "f1",
        f"neutralized_{suffix}": "neutralized_detections",
    }

    if not frame_result.metric_valid:
        if any(csv_record.get(csv_name, "") != "" for csv_name in fields):
            raise SourceContractError(f"invalid frame has primary metrics in frames.csv: {frame_id}")
        return

    metrics = get_primary_metrics(frame_result, primary_iou_key)
    for csv_name, metric_name in fields.items():
        if csv_name not in csv_record:
            raise SourceContractError(f"frames.csv missing primary metric column: {csv_name}")
        expected = getattr(metrics, metric_name)
        actual = csv_record.get(csv_name, "")
        if expected is None:
            if actual != "":
                raise SourceContractError(f"primary metric mismatch for frame {frame_id}: {metric_name}")
        elif not numeric_values_equal(actual, expected):
            raise SourceContractError(f"primary metric mismatch for frame {frame_id}: {metric_name}")


def numeric_values_equal(actual, expected):
    try:
        return abs(float(actual) - float(expected)) <= 1e-6
    except (TypeError, ValueError):
        return False


def rebuild_batch_from_disk(summary, frame_results, requested_frame_ids, run_directory, primary_iou_key):
    totals = require_dict(summary.get("totals"), "summary.totals")
    parameters = require_dict(totals.get("parameters", {}), "summary.totals.parameters")
    reproducibility = require_dict(summary.get("reproducibility"), "summary.reproducibility")
    iou_keys = reproducibility.get("iou_thresholds")
    if not isinstance(iou_keys, list) or primary_iou_key not in [str(value) for value in iou_keys]:
        raise SourceContractError("summary does not declare the primary IoU threshold")

    auxiliary_thresholds = [float(value) for value in iou_keys if str(value) != primary_iou_key]
    batch_result = build_batch_result(
        frame_results=frame_results,
        data_root=reproducibility.get("data_root", "data/kitti"),
        frame_ids=requested_frame_ids,
        eps=parameters.get("eps", 0.6),
        min_points=parameters.get("min_points", 20),
        oriented=totals.get("box_mode") == "oriented_pca",
        nms_iou_threshold=parameters.get("nms_iou_threshold", 0.3),
        eval_iou_threshold=float(primary_iou_key),
        auxiliary_iou_thresholds=auxiliary_thresholds,
    )

    run = require_dict(summary.get("run"), "summary.run")
    batch_result.schema_version = str(summary.get("schema_version"))
    batch_result.run_id = run.get("run_id")
    batch_result.started_at = run.get("started_at")
    batch_result.finished_at = run.get("finished_at")
    batch_result.total_time_ms = run.get("duration_ms")
    batch_result.config_hash = reproducibility.get("config_hash")
    batch_result.git_commit = reproducibility.get("git_commit")
    batch_result.git_dirty = reproducibility.get("git_dirty")
    batch_result.artifacts.update(require_dict(summary.get("artifacts", {}), "summary.artifacts"))
    batch_result.artifacts["requested_frame_ids"] = requested_frame_ids
    batch_result.artifacts["source_run_directory"] = Path(run_directory)
    return batch_result


def validate_summary_contract(summary, batch_result, primary_iou_key):
    run = require_dict(summary.get("run"), "summary.run")
    if run.get("batch_status") != batch_result.status.value:
        raise SourceContractError("summary batch status does not match frame results")

    summary_frames = require_dict(summary.get("frames"), "summary.frames")
    for field_name, expected in batch_result.frame_counts.items():
        if summary_frames.get(field_name) != expected:
            raise SourceContractError(f"summary frame count mismatch: {field_name}")

    expected_statuses = {frame.frame_id: frame.status.value for frame in batch_result.frame_results}
    if summary_frames.get("status_by_frame") != expected_statuses:
        raise SourceContractError("summary status_by_frame does not match per-frame reports")

    summary_metrics = require_dict(summary.get("metrics_by_iou"), "summary.metrics_by_iou").get(primary_iou_key)
    if not isinstance(summary_metrics, dict):
        raise SourceContractError(f"summary missing primary IoU metrics: {primary_iou_key}")
    expected_metrics = batch_result.metrics_by_iou[primary_iou_key].to_dict()
    for field_name in ["tp", "fp", "fn", "precision", "recall", "f1", "neutralized_detections"]:
        expected = expected_metrics.get(field_name)
        actual = summary_metrics.get(field_name)
        if expected is None:
            if actual is not None:
                raise SourceContractError(f"summary primary metric mismatch: {field_name}")
        elif not numeric_values_equal(actual, expected):
            raise SourceContractError(f"summary primary metric mismatch: {field_name}")


def collect_eligible_frames(batch_result, primary_iou_key):
    seen = set()
    frames = []
    for frame in batch_result.frame_results:
        frame_id = str(frame.frame_id).zfill(6)
        if frame_id in seen:
            raise SourceContractError(f"duplicate frame_id in batch result: {frame_id}")
        seen.add(frame_id)

        if not frame.metric_valid:
            continue

        metrics = get_primary_metrics(frame, primary_iou_key)
        frames.append(
            {
                "frame": frame,
                "frame_id": frame_id,
                "metrics": metrics,
                "effective_car_detection_count": effective_car_detection_count(metrics),
                "gt_counts": gt_counts(frame),
                "detection_counts": detection_counts(frame),
                "source_run_id": frame_source_run_id(frame) or batch_result.run_id,
                "source_run_directory": frame_source_run_directory(frame) or batch_source_run_directory(batch_result),
            }
        )
    return frames


def get_primary_metrics(frame, primary_iou_key):
    metrics = frame.metrics_by_iou.get(primary_iou_key)
    if metrics is None:
        raise SourceContractError(f"metric-valid frame missing primary IoU metrics: {frame.frame_id} {primary_iou_key}")
    if isinstance(metrics, dict):
        metrics = FrameMetrics(**metrics)
    required = {
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "neutralized_detections": metrics.neutralized_detections,
    }
    for key, value in required.items():
        if value is None:
            raise SourceContractError(f"metric-valid frame missing required metric field: {frame.frame_id} {key}")
    return metrics


def effective_car_detection_count(metrics):
    return int(metrics.tp) + int(metrics.fp) + int(metrics.neutralized_detections)


def rank_category(frames, category, top_k, sort_key, eligibility_fn, diagnostic_only=False):
    candidates = [item for item in frames if eligibility_fn(item)]
    candidates.sort(key=sort_key)
    return [
        build_failure_case(category, rank, item, diagnostic_only=diagnostic_only)
        for rank, item in enumerate(candidates[:top_k], start=1)
    ]


def build_failure_case(category, rank, item, diagnostic_only=False):
    frame = item["frame"]
    metrics = item["metrics"]
    return FailureCase(
        category=category,
        rank=rank,
        frame_id=item["frame_id"],
        reason_code=reason_code_for(category),
        ranking_values=ranking_values_for(category, item),
        source_status=frame.status,
        metric_valid=frame.metric_valid,
        primary_iou_threshold=PRIMARY_IOU_THRESHOLD,
        effective_car_detection_count=item["effective_car_detection_count"],
        metrics_by_iou={PRIMARY_IOU_KEY: metrics.to_dict()},
        gt_counts=item["gt_counts"],
        detection_counts=item["detection_counts"],
        frame_report_path=frame.artifacts.get("frame_report_path"),
        source_run_id=item["source_run_id"],
        source_run_directory=item["source_run_directory"],
        diagnostic_only=diagnostic_only,
    )


def reason_code_for(category):
    return {
        FailureCategory.MOST_FALSE_NEGATIVES: "high_false_negative_count",
        FailureCategory.MOST_FALSE_POSITIVES: "high_false_positive_count",
        FailureCategory.LOWEST_RECALL: "low_recall",
        FailureCategory.ZERO_DETECTION_WITH_GT: "positive_gt_without_effective_car_detection",
        FailureCategory.HIGHEST_EFFECTIVE_CAR_DETECTIONS: "high_effective_car_detection_count",
    }[FailureCategory(category)]


def ranking_values_for(category, item):
    metrics = item["metrics"]
    values = {
        "tp": int(metrics.tp),
        "fp": int(metrics.fp),
        "fn": int(metrics.fn),
        "recall": metrics.recall,
        "neutralized_detections": int(metrics.neutralized_detections),
        "effective_car_detection_count": int(item["effective_car_detection_count"]),
    }
    if category == FailureCategory.ZERO_DETECTION_WITH_GT:
        values["num_positive_gt"] = item["gt_counts"]["num_positive_gt"]
    return values


def has_false_negatives(item):
    return item["metrics"].fn > 0


def most_false_negatives_key(item):
    return (-item["metrics"].fn, item["frame_id"])


def has_false_positives(item):
    return item["metrics"].fp > 0


def most_false_positives_key(item):
    return (-item["metrics"].fp, item["frame_id"])


def has_low_recall(item):
    return item["metrics"].recall is not None and item["metrics"].fn > 0


def lowest_recall_key(item):
    return (item["metrics"].recall, -item["metrics"].fn, item["frame_id"])


def has_zero_detection_with_gt(item):
    return item["gt_counts"]["num_positive_gt"] > 0 and item["effective_car_detection_count"] == 0


def zero_detection_with_gt_key(item):
    return (-item["gt_counts"]["num_positive_gt"], -item["metrics"].fn, item["frame_id"])


def has_effective_car_detections(item):
    return item["effective_car_detection_count"] > 0


def highest_effective_car_detections_key(item):
    return (-item["effective_car_detection_count"], -item["metrics"].fp, item["frame_id"])


def gt_counts(frame):
    return {
        "num_positive_gt": int(frame.num_positive_gt or 0),
        "num_neutral_gt": int(frame.num_neutral_gt or 0),
        "num_excluded_gt": int(frame.num_excluded_gt or 0),
        "num_dontcare": int(frame.num_dontcare or 0),
        "num_gt_outside_roi": int(frame.num_gt_outside_roi or 0),
        "num_invalid_gt": int(frame.num_invalid_gt or 0),
    }


def detection_counts(frame):
    return {
        "num_raw_detections": int(frame.num_raw_detections or 0),
        "num_car_detections_before_nms": int(frame.num_car_detections_before_nms or 0),
        "num_detections_after_nms": int(frame.num_detections_after_nms or 0),
        "num_ignored_detection_class": int(frame.num_ignored_detection_class or 0),
        "num_detections_outside_roi": int(frame.num_detections_outside_roi or 0),
        "num_suppressed_by_nms": int(frame.num_suppressed_by_nms or 0),
    }


def frame_source_run_id(frame):
    return frame.artifacts.get("source_run_id") or frame.artifacts.get("run_id")


def frame_source_run_directory(frame):
    run_directory = frame.artifacts.get("source_run_directory") or frame.artifacts.get("run_directory")
    if run_directory is not None:
        return run_directory
    report_path = frame.artifacts.get("frame_report_path")
    if report_path is None:
        return None
    path = Path(report_path)
    if path.parent.name == "frames":
        return path.parent.parent
    return path.parent


def batch_source_run_directory(batch_result):
    run_directory = batch_result.artifacts.get("source_run_directory")
    if run_directory is not None:
        return run_directory
    summary_path = batch_result.artifacts.get("summary_json")
    if summary_path is not None:
        return Path(summary_path).parent
    return None
