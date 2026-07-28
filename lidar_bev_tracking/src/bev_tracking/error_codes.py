from enum import Enum


class StableStrEnum(str, Enum):
    def __str__(self):
        return self.value


class FrameStatus(StableStrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"
    FAILED = "failed"


class BatchStatus(StableStrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class ErrorStage(StableStrEnum):
    CONFIG = "config"
    INPUT_CHECK = "input_check"
    POINT_CLOUD_LOAD = "point_cloud_load"
    LABEL_LOAD = "label_load"
    LABEL_PARSE = "label_parse"
    CALIB_LOAD = "calib_load"
    CALIB_PARSE = "calib_parse"
    GT_CONVERSION = "gt_conversion"
    DETECTION = "detection"
    NMS = "nms"
    EVALUATION = "evaluation"
    FRAME_REPORT = "frame_report"
    BATCH_AGGREGATION = "batch_aggregation"
    REPORT_WRITE = "report_write"
    METADATA_CAPTURE = "metadata_capture"
    UNKNOWN = "unknown"


class ErrorCode(StableStrEnum):
    MISSING_BIN = "missing_bin"
    MISSING_LABEL = "missing_label"
    MISSING_CALIB = "missing_calib"
    BIN_READ_FAILED = "bin_read_failed"
    BIN_INVALID_FORMAT = "bin_invalid_format"
    EMPTY_POINT_CLOUD = "empty_point_cloud"
    LABEL_READ_FAILED = "label_read_failed"
    LABEL_PARSE_FAILED = "label_parse_failed"
    CALIB_READ_FAILED = "calib_read_failed"
    CALIB_PARSE_FAILED = "calib_parse_failed"
    CALIB_REQUIRED_FIELD_MISSING = "calib_required_field_missing"
    GT_CONVERSION_FAILED = "gt_conversion_failed"
    DETECTOR_FAILED = "detector_failed"
    NMS_FAILED = "nms_failed"
    EVALUATION_FAILED = "evaluation_failed"
    FRAME_REPORT_WRITE_FAILED = "frame_report_write_failed"
    OUTPUT_DIR_CREATE_FAILED = "output_dir_create_failed"
    CONFIG_SNAPSHOT_WRITE_FAILED = "config_snapshot_write_failed"
    FRAME_MANIFEST_WRITE_FAILED = "frame_manifest_write_failed"
    CSV_WRITE_FAILED = "csv_write_failed"
    SUMMARY_WRITE_FAILED = "summary_write_failed"
    FAILURE_CASES_WRITE_FAILED = "failure_cases_write_failed"
    SOURCE_CONTRACT_ERROR = "source_contract_error"
    SOURCE_CONSISTENCY_ERROR = "source_consistency_error"
    DUPLICATE_FRAME_ID = "duplicate_frame_id"
    INVALID_FAILURE_ANALYSIS_CONFIG = "invalid_failure_analysis_config"
    GIT_METADATA_UNAVAILABLE = "git_metadata_unavailable"
    UNEXPECTED_FRAME_ERROR = "unexpected_frame_error"
    UNEXPECTED_BATCH_ERROR = "unexpected_batch_error"
    UNKNOWN_ERROR = "unknown_error"


def is_metric_valid_status(status):
    status = FrameStatus(status)
    return status in {FrameStatus.SUCCESS, FrameStatus.PARTIAL_SUCCESS}
