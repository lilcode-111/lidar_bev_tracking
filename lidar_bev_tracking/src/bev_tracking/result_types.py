from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path

from bev_tracking.error_codes import BatchStatus, ErrorCode, ErrorStage, FrameStatus, is_metric_valid_status


NUMERIC_COUNT_FIELDS = [
    "num_points",
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
]

TIME_FIELDS = [
    "load_time_ms",
    "parse_time_ms",
    "gt_transform_time_ms",
    "detection_time_ms",
    "nms_time_ms",
    "evaluation_time_ms",
    "total_time_ms",
]


def to_json_compatible(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {item.name: to_json_compatible(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return value.item()
    return value


@dataclass
class FrameMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    neutralized_detections: int = 0

    def to_dict(self):
        return {
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "precision": to_json_compatible(self.precision),
            "recall": to_json_compatible(self.recall),
            "f1": to_json_compatible(self.f1),
            "neutralized_detections": int(self.neutralized_detections),
        }


@dataclass
class FrameError:
    error_code: ErrorCode | str
    error_stage: ErrorStage | str
    error_message: str
    exception_type: str | None = None
    input_path: str | Path | None = None

    def to_dict(self):
        return {
            "error_code": to_json_compatible(self.error_code),
            "error_stage": to_json_compatible(self.error_stage),
            "error_message": str(self.error_message),
            "exception_type": to_json_compatible(self.exception_type),
            "input_path": to_json_compatible(self.input_path),
        }


@dataclass
class FrameResult:
    frame_id: str
    status: FrameStatus | str
    metrics_by_iou: dict = field(default_factory=dict)
    error: FrameError | dict | None = None
    warnings: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    num_points: int | None = None
    num_labels_raw: int | None = None
    num_positive_gt: int | None = None
    num_neutral_gt: int | None = None
    num_excluded_gt: int | None = None
    num_dontcare: int | None = None
    num_gt_outside_roi: int | None = None
    num_invalid_gt: int | None = None
    num_raw_detections: int | None = None
    num_car_detections_before_nms: int | None = None
    num_detections_after_nms: int | None = None
    num_ignored_detection_class: int | None = None
    num_detections_outside_roi: int | None = None
    num_suppressed_by_nms: int | None = None
    load_time_ms: float | None = None
    parse_time_ms: float | None = None
    gt_transform_time_ms: float | None = None
    detection_time_ms: float | None = None
    nms_time_ms: float | None = None
    evaluation_time_ms: float | None = None
    total_time_ms: float | None = None

    def __post_init__(self):
        self.status = FrameStatus(self.status)
        if not self.metric_valid:
            self.metrics_by_iou = {}

    @property
    def metric_valid(self):
        return is_metric_valid_status(self.status)

    def to_dict(self):
        output = {
            "frame_id": str(self.frame_id).zfill(6),
            "status": self.status.value,
            "metric_valid": self.metric_valid,
            "metrics_by_iou": to_json_compatible(self.metrics_by_iou),
            "error": to_json_compatible(self.error),
            "warnings": to_json_compatible(self.warnings),
            "artifacts": to_json_compatible(self.artifacts),
        }
        for field_name in NUMERIC_COUNT_FIELDS + TIME_FIELDS:
            output[field_name] = to_json_compatible(getattr(self, field_name))
        output["frame_report_path"] = to_json_compatible(self.artifacts.get("frame_report_path"))
        return output


@dataclass
class BatchResult:
    schema_version: str = "13.0"
    run_id: str | None = None
    status: BatchStatus | str = BatchStatus.FAILED
    started_at: str | None = None
    finished_at: str | None = None
    total_time_ms: float | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    frame_results: list[FrameResult] = field(default_factory=list)
    frame_counts: dict = field(default_factory=dict)
    metrics_by_iou: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)
    error_counts: dict = field(default_factory=dict)
    error_stage_counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def __post_init__(self):
        self.status = BatchStatus(self.status)

    def to_dict(self):
        return {
            "schema_version": str(self.schema_version),
            "run_id": to_json_compatible(self.run_id),
            "status": self.status.value,
            "started_at": to_json_compatible(self.started_at),
            "finished_at": to_json_compatible(self.finished_at),
            "total_time_ms": to_json_compatible(self.total_time_ms),
            "config_hash": to_json_compatible(self.config_hash),
            "git_commit": to_json_compatible(self.git_commit),
            "git_dirty": to_json_compatible(self.git_dirty),
            "frame_results": to_json_compatible(self.frame_results),
            "frame_counts": to_json_compatible(self.frame_counts),
            "metrics_by_iou": to_json_compatible(self.metrics_by_iou),
            "totals": to_json_compatible(self.totals),
            "error_counts": to_json_compatible(self.error_counts),
            "error_stage_counts": to_json_compatible(self.error_stage_counts),
            "warnings": to_json_compatible(self.warnings),
            "artifacts": to_json_compatible(self.artifacts),
        }
