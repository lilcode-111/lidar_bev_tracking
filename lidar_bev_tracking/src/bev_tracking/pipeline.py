import json
from pathlib import Path
from time import perf_counter

from bev_tracking.clustering_detector import detect_objects_from_points
from bev_tracking.eval_policy import classify_gt_box, is_center_inside_roi, normalize_class_name
from bev_tracking.error_codes import ErrorCode, ErrorStage, FrameStatus
from bev_tracking.evaluation import evaluate_detections
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.nms import nms_bev
from bev_tracking.result_types import FrameError, FrameMetrics, FrameResult, to_json_compatible


def run_kitti_frame_evaluation(
    data_root="data/kitti",
    frame_id="000000",
    eps=0.6,
    min_points=20,
    oriented=False,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    z_min=-0.9,
    intensity_min=0.38,
):
    frame_id = str(frame_id).zfill(6)
    total_start = perf_counter()
    velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
    calib_path = resolve_kitti_calib_path(data_root, frame_id)

    try:
        load_start = perf_counter()
        points = load_kitti_point_cloud(velodyne_path)
        load_time_ms = elapsed_ms(load_start)
        if len(points) == 0:
            return failed_frame_result(
                frame_id,
                ErrorCode.EMPTY_POINT_CLOUD,
                ErrorStage.POINT_CLOUD_LOAD,
                "KITTI point cloud is empty",
                input_path=velodyne_path,
                load_time_ms=load_time_ms,
                total_start=total_start,
            )
    except FileNotFoundError as exc:
        return skipped_frame_result(
            frame_id,
            ErrorCode.MISSING_BIN,
            ErrorStage.POINT_CLOUD_LOAD,
            str(exc),
            exception=exc,
            input_path=velodyne_path,
            total_start=total_start,
        )
    except ValueError as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.BIN_INVALID_FORMAT,
            ErrorStage.POINT_CLOUD_LOAD,
            str(exc),
            exception=exc,
            input_path=velodyne_path,
            total_start=total_start,
        )
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.BIN_READ_FAILED,
            ErrorStage.POINT_CLOUD_LOAD,
            str(exc),
            exception=exc,
            input_path=velodyne_path,
            total_start=total_start,
        )

    parse_start = perf_counter()
    try:
        labels = load_kitti_labels(label_path)
    except FileNotFoundError as exc:
        return skipped_frame_result(
            frame_id,
            ErrorCode.MISSING_LABEL,
            ErrorStage.LABEL_LOAD,
            str(exc),
            exception=exc,
            input_path=label_path,
            num_points=len(points),
            load_time_ms=load_time_ms,
            total_start=total_start,
        )
    except ValueError as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.LABEL_PARSE_FAILED,
            ErrorStage.LABEL_PARSE,
            str(exc),
            exception=exc,
            input_path=label_path,
            num_points=len(points),
            load_time_ms=load_time_ms,
            total_start=total_start,
        )
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.LABEL_READ_FAILED,
            ErrorStage.LABEL_LOAD,
            str(exc),
            exception=exc,
            input_path=label_path,
            num_points=len(points),
            load_time_ms=load_time_ms,
            total_start=total_start,
        )

    try:
        calib = load_kitti_calib(calib_path)
        parse_time_ms = elapsed_ms(parse_start)
    except FileNotFoundError as exc:
        return skipped_frame_result(
            frame_id,
            ErrorCode.MISSING_CALIB,
            ErrorStage.CALIB_LOAD,
            str(exc),
            exception=exc,
            input_path=calib_path,
            num_points=len(points),
            num_labels_raw=len(labels),
            load_time_ms=load_time_ms,
            total_start=total_start,
        )
    except ValueError as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.CALIB_PARSE_FAILED,
            ErrorStage.CALIB_PARSE,
            str(exc),
            exception=exc,
            input_path=calib_path,
            num_points=len(points),
            num_labels_raw=len(labels),
            load_time_ms=load_time_ms,
            total_start=total_start,
        )
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.CALIB_READ_FAILED,
            ErrorStage.CALIB_LOAD,
            str(exc),
            exception=exc,
            input_path=calib_path,
            num_points=len(points),
            num_labels_raw=len(labels),
            load_time_ms=load_time_ms,
            total_start=total_start,
        )

    try:
        gt_start = perf_counter()
        gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)
        gt_transform_time_ms = elapsed_ms(gt_start)
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.GT_CONVERSION_FAILED,
            ErrorStage.GT_CONVERSION,
            str(exc),
            exception=exc,
            input_path=calib_path,
            num_points=len(points),
            num_labels_raw=len(labels),
            load_time_ms=load_time_ms,
            parse_time_ms=parse_time_ms,
            total_start=total_start,
        )

    try:
        detection_start = perf_counter()
        raw_detections = detect_objects_from_points(
            points,
            eps=eps,
            min_points=min_points,
            oriented=oriented,
            z_min=z_min,
            intensity_min=intensity_min,
        )
        detection_time_ms = elapsed_ms(detection_start)
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.DETECTOR_FAILED,
            ErrorStage.DETECTION,
            str(exc),
            exception=exc,
            num_points=len(points),
            num_labels_raw=len(labels),
            **gt_count_kwargs(gt_boxes),
            load_time_ms=load_time_ms,
            parse_time_ms=parse_time_ms,
            gt_transform_time_ms=gt_transform_time_ms,
            total_start=total_start,
        )

    try:
        nms_start = perf_counter()
        detections = nms_bev(raw_detections, iou_threshold=nms_iou_threshold)
        nms_time_ms = elapsed_ms(nms_start)
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.NMS_FAILED,
            ErrorStage.NMS,
            str(exc),
            exception=exc,
            num_points=len(points),
            num_labels_raw=len(labels),
            num_raw_detections=len(raw_detections),
            **gt_count_kwargs(gt_boxes),
            **detection_count_kwargs(raw_detections, []),
            load_time_ms=load_time_ms,
            parse_time_ms=parse_time_ms,
            gt_transform_time_ms=gt_transform_time_ms,
            detection_time_ms=detection_time_ms,
            total_start=total_start,
        )

    try:
        evaluation_start = perf_counter()
        evaluation = evaluate_detections(
            detections,
            gt_boxes,
            iou_threshold=eval_iou_threshold,
            auxiliary_iou_thresholds=auxiliary_iou_thresholds,
        )
        evaluation_time_ms = elapsed_ms(evaluation_start)
    except Exception as exc:
        return failed_frame_result(
            frame_id,
            ErrorCode.EVALUATION_FAILED,
            ErrorStage.EVALUATION,
            str(exc),
            exception=exc,
            num_points=len(points),
            num_labels_raw=len(labels),
            num_raw_detections=len(raw_detections),
            num_detections_after_nms=len(detections),
            **gt_count_kwargs(gt_boxes),
            **detection_count_kwargs(raw_detections, detections),
            load_time_ms=load_time_ms,
            parse_time_ms=parse_time_ms,
            gt_transform_time_ms=gt_transform_time_ms,
            detection_time_ms=detection_time_ms,
            nms_time_ms=nms_time_ms,
            total_start=total_start,
        )

    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SUCCESS,
        metrics_by_iou=evaluation_to_metrics_by_iou(evaluation),
        num_points=len(points),
        num_labels_raw=len(labels),
        **gt_count_kwargs(gt_boxes),
        num_raw_detections=len(raw_detections),
        num_detections_after_nms=len(detections),
        **detection_count_kwargs(raw_detections, detections),
        load_time_ms=load_time_ms,
        parse_time_ms=parse_time_ms,
        gt_transform_time_ms=gt_transform_time_ms,
        detection_time_ms=detection_time_ms,
        nms_time_ms=nms_time_ms,
        evaluation_time_ms=evaluation_time_ms,
        total_time_ms=elapsed_ms(total_start),
        artifacts={
            "box_mode": "oriented_pca" if oriented else "axis_aligned",
            "parameters": {
                "eps": float(eps),
                "min_points": int(min_points),
                "z_min": float(z_min),
                "intensity_min": float(intensity_min),
                "nms_iou_threshold": float(nms_iou_threshold),
                "eval_iou_threshold": float(eval_iou_threshold),
                "auxiliary_iou_thresholds": [float(threshold) for threshold in auxiliary_iou_thresholds],
            },
            "legacy_evaluation": evaluation,
        },
    )


def run_kitti_bev_evaluation(
    data_root="data/kitti",
    frame_id="000000",
    eps=0.6,
    min_points=20,
    oriented=False,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    report_dir="outputs/reports",
    z_min=-0.9,
    intensity_min=0.38,
):
    frame_result = run_kitti_frame_evaluation(
        data_root=data_root,
        frame_id=frame_id,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
        z_min=z_min,
        intensity_min=intensity_min,
        nms_iou_threshold=nms_iou_threshold,
        eval_iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
    )
    report = frame_result_to_legacy_report(frame_result)

    suffix = "oriented" if oriented else "axis_aligned"
    output_path = Path(report_dir) / f'kitti_eval_{report["frame_id"]}_{suffix}.json'
    save_json_report(report, output_path)
    return report, output_path


def run_kitti_bev_evaluation_from_config(config):
    return run_kitti_bev_evaluation(
        data_root=config["data"]["root"],
        frame_id=config["data"]["frame_id"],
        eps=config["detector"]["eps"],
        min_points=config["detector"]["min_points"],
        oriented=config["detector"]["oriented"],
        z_min=config["detector"].get("z_min", -0.9),
        intensity_min=config["detector"].get("intensity_min", 0.38),
        nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=config["evaluation"]["iou_threshold"],
        auxiliary_iou_thresholds=config["evaluation"].get("auxiliary_iou_thresholds", [0.25]),
        report_dir=config["outputs"]["report_dir"],
    )


def save_json_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_json_compatible(report), f, indent=2)


def format_eval_summary(report, output_path):
    metrics = report["metrics"]
    precision = format_metric(metrics["precision"])
    recall = format_metric(metrics["recall"])
    f1 = format_metric(metrics["f1"])
    return "\n".join(
        [
            f'loaded points: {report["num_points"]}',
            f'gt boxes in lidar frame: {report["num_gt_boxes"]}',
            f'detections after nms: {report["num_detections_after_nms"]}',
            f'box mode: {report["box_mode"]}',
            f'eval iou threshold: {report["iou_threshold"]:.2f}',
            f'tp={metrics["tp"]} fp={metrics["fp"]} fn={metrics["fn"]}',
            f"precision={precision} recall={recall} f1={f1}",
            f"saved {output_path}",
        ]
    )


def format_metric(value):
    if value is None:
        return "undefined"
    return f"{value:.3f}"


def elapsed_ms(start_time):
    return float((perf_counter() - start_time) * 1000.0)


def failed_frame_result(frame_id, error_code, error_stage, error_message, exception=None, input_path=None, total_start=None, **kwargs):
    result = FrameResult(
        frame_id=frame_id,
        status=FrameStatus.FAILED,
        error=FrameError(
            error_code=error_code,
            error_stage=error_stage,
            error_message=error_message,
            exception_type=type(exception).__name__ if exception is not None else None,
            input_path=input_path,
        ),
        **kwargs,
    )
    if total_start is not None:
        result.total_time_ms = elapsed_ms(total_start)
    return result


def skipped_frame_result(frame_id, error_code, error_stage, error_message, exception=None, input_path=None, total_start=None, **kwargs):
    result = FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SKIPPED,
        error=FrameError(
            error_code=error_code,
            error_stage=error_stage,
            error_message=error_message,
            exception_type=type(exception).__name__ if exception is not None else None,
            input_path=input_path,
        ),
        **kwargs,
    )
    if total_start is not None:
        result.total_time_ms = elapsed_ms(total_start)
    return result


def evaluation_to_metrics_by_iou(evaluation):
    metrics_by_iou = {
        f'{evaluation["iou_threshold"]:.2f}': frame_metrics_from_evaluation(evaluation)
    }
    for iou_key, auxiliary in evaluation.get("auxiliary", {}).items():
        metrics_by_iou[iou_key] = frame_metrics_from_evaluation(auxiliary)
    return metrics_by_iou


def frame_metrics_from_evaluation(evaluation):
    metrics = evaluation["metrics"]
    return FrameMetrics(
        tp=metrics["tp"],
        fp=metrics["fp"],
        fn=metrics["fn"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        neutralized_detections=len(evaluation.get("neutralized_detections", [])),
        per_class=metrics.get("per_class", {}),
    )


def gt_count_kwargs(gt_boxes):
    counts = {
        "num_positive_gt": 0,
        "num_neutral_gt": 0,
        "num_excluded_gt": 0,
        "num_dontcare": 0,
        "num_gt_outside_roi": 0,
        "num_invalid_gt": 0,
    }
    for gt in gt_boxes:
        role = classify_gt_box(gt)
        if role == "positive":
            counts["num_positive_gt"] += 1
        elif role == "neutral":
            counts["num_neutral_gt"] += 1
        elif role == "outside_roi":
            counts["num_gt_outside_roi"] += 1
        elif role == "dontcare":
            counts["num_dontcare"] += 1
        elif role == "excluded":
            counts["num_excluded_gt"] += 1
        else:
            counts["num_invalid_gt"] += 1
    return counts


def detection_count_kwargs(raw_detections, detections_after_nms):
    num_car_detections_before_nms = 0
    num_ignored_detection_class = 0
    num_detections_outside_roi = 0
    for det in raw_detections:
        class_name = normalize_class_name(det.get("class_name", ""))
        inside_roi = is_center_inside_roi(det)
        if class_name == "car" and inside_roi:
            num_car_detections_before_nms += 1
        elif not inside_roi:
            num_detections_outside_roi += 1
        else:
            num_ignored_detection_class += 1
    return {
        "num_car_detections_before_nms": num_car_detections_before_nms,
        "num_ignored_detection_class": num_ignored_detection_class,
        "num_detections_outside_roi": num_detections_outside_roi,
        "num_suppressed_by_nms": max(0, len(raw_detections) - len(detections_after_nms)),
    }


def frame_result_to_legacy_report(frame_result):
    frame_result_dict = frame_result.to_dict()
    legacy_evaluation = frame_result.artifacts.get("legacy_evaluation")
    if legacy_evaluation is None:
        legacy_evaluation = {
            "iou_threshold": None,
            "metrics": {},
            "auxiliary": {},
            "policy": {},
        }

    report = {
        "frame_id": frame_result_dict["frame_id"],
        "status": frame_result.status.value,
        "metric_valid": frame_result.metric_valid,
        "error": frame_result_dict["error"],
        "warnings": frame_result_dict["warnings"],
        "num_points": frame_result.num_points,
        "num_labels": frame_result.num_labels_raw,
        "num_labels_raw": frame_result.num_labels_raw,
        "num_gt_boxes": sum(
            value or 0
            for value in [
                frame_result.num_positive_gt,
                frame_result.num_neutral_gt,
                frame_result.num_excluded_gt,
                frame_result.num_dontcare,
                frame_result.num_gt_outside_roi,
            ]
        ),
        "num_raw_detections": frame_result.num_raw_detections,
        "num_detections_after_nms": frame_result.num_detections_after_nms,
        "box_mode": frame_result.artifacts.get("box_mode"),
        "parameters": frame_result.artifacts.get("parameters", {}),
        "metrics_by_iou": frame_result.metrics_by_iou,
        "frame_result": frame_result_dict,
    }
    report.update(legacy_evaluation)
    return report
