import csv
import json
from pathlib import Path
from time import perf_counter

from bev_tracking.eval_policy import safe_divide, safe_f1
from bev_tracking.error_codes import BatchStatus, ErrorCode, ErrorStage, FrameStatus
from bev_tracking.kitti import resolve_kitti_calib_path, resolve_kitti_paths
from bev_tracking.pipeline import frame_result_to_legacy_report, run_kitti_frame_evaluation, save_json_report
from bev_tracking.pipeline import format_metric
from bev_tracking.result_types import BatchResult, FrameError, FrameMetrics, FrameResult


def run_kitti_batch_evaluation(
    data_root="data/kitti",
    frame_ids=None,
    eps=0.6,
    min_points=20,
    oriented=False,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    report_dir="outputs/reports",
    z_min=-0.9,
    intensity_min=0.38,
    gesr_enabled=False,
    gesr_reason_attribution=True,
    gesr_evidence_level="detailed",
    progress_callback=None,
    frame_result_callback=None,
):
    frame_ids = normalize_frame_ids(frame_ids)
    frame_results = run_kitti_batch_frame_results(
        data_root=data_root,
        frame_ids=frame_ids,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
        z_min=z_min,
        intensity_min=intensity_min,
        nms_iou_threshold=nms_iou_threshold,
        eval_iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
        gesr_enabled=gesr_enabled,
        gesr_reason_attribution=gesr_reason_attribution,
        gesr_evidence_level=gesr_evidence_level,
        progress_callback=progress_callback,
        frame_result_callback=frame_result_callback,
    )
    frame_reports = []

    for frame_result in frame_results:
        report = frame_result_to_legacy_report(frame_result)
        report["report_path"] = ""
        frame_reports.append(report)

    summary = summarize_batch_reports(
        frame_reports=frame_reports,
        data_root=data_root,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
        z_min=z_min,
        intensity_min=intensity_min,
        nms_iou_threshold=nms_iou_threshold,
        eval_iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
    )

    suffix = "oriented" if oriented else "axis_aligned"
    report_dir = Path(report_dir)
    summary_path = report_dir / f"kitti_batch_eval_{suffix}.json"
    csv_path = report_dir / f"kitti_batch_eval_frames_{suffix}.csv"
    save_json_report(summary, summary_path)
    save_frame_csv(frame_reports, csv_path)
    return summary, summary_path, csv_path


def run_kitti_batch_frame_results(
    data_root="data/kitti",
    frame_ids=None,
    eps=0.6,
    min_points=20,
    oriented=False,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    z_min=-0.9,
    intensity_min=0.38,
    gesr_enabled=False,
    gesr_reason_attribution=True,
    gesr_evidence_level="detailed",
    progress_callback=None,
    frame_result_callback=None,
    phase2_variant=None,
    phase2_delta22_gt_ids_by_frame=None,
):
    frame_results = []
    normalized_frame_ids = normalize_frame_ids(frame_ids)
    total_frames = len(normalized_frame_ids)
    phase2_delta22_gt_ids_by_frame = phase2_delta22_gt_ids_by_frame or {}
    for frame_index, frame_id in enumerate(normalized_frame_ids, start=1):
        if progress_callback is not None:
            progress_callback(frame_index, total_frames, frame_id)
        skipped = precheck_kitti_frame_inputs(data_root, frame_id)
        if skipped is not None:
            frame_results.append(
                frame_result_callback(skipped)
                if frame_result_callback is not None
                else skipped
            )
            continue

        try:
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
                gesr_enabled=gesr_enabled,
                gesr_reason_attribution=gesr_reason_attribution,
                gesr_evidence_level=gesr_evidence_level,
                phase2_variant=phase2_variant,
                phase2_gate_gt_ids=phase2_delta22_gt_ids_by_frame.get(frame_id, ()),
            )
        except Exception as exc:
            frame_result = unexpected_failed_frame_result(frame_id, exc)
        frame_results.append(
            frame_result_callback(frame_result)
            if frame_result_callback is not None
            else frame_result
        )

    return frame_results


def run_kitti_batch_result(
    data_root="data/kitti",
    frame_ids=None,
    eps=0.6,
    min_points=20,
    oriented=False,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    z_min=-0.9,
    intensity_min=0.38,
    gesr_enabled=False,
    gesr_reason_attribution=True,
    gesr_evidence_level="detailed",
    progress_callback=None,
    frame_result_callback=None,
    phase2_variant=None,
    phase2_delta22_gt_ids_by_frame=None,
):
    frame_ids = normalize_frame_ids(frame_ids)
    frame_results = run_kitti_batch_frame_results(
        data_root=data_root,
        frame_ids=frame_ids,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
        z_min=z_min,
        intensity_min=intensity_min,
        nms_iou_threshold=nms_iou_threshold,
        eval_iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
        gesr_enabled=gesr_enabled,
        gesr_reason_attribution=gesr_reason_attribution,
        gesr_evidence_level=gesr_evidence_level,
        progress_callback=progress_callback,
        frame_result_callback=frame_result_callback,
        phase2_variant=phase2_variant,
        phase2_delta22_gt_ids_by_frame=phase2_delta22_gt_ids_by_frame,
    )
    return build_batch_result(
        frame_results=frame_results,
        data_root=data_root,
        frame_ids=frame_ids,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
        z_min=z_min,
        intensity_min=intensity_min,
        nms_iou_threshold=nms_iou_threshold,
        eval_iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
        gesr_enabled=gesr_enabled,
        gesr_reason_attribution=gesr_reason_attribution,
        gesr_evidence_level=gesr_evidence_level,
    )


def build_batch_result(
    frame_results,
    data_root="data/kitti",
    frame_ids=None,
    eps=0.6,
    min_points=20,
    oriented=False,
    nms_iou_threshold=0.3,
    eval_iou_threshold=0.5,
    auxiliary_iou_thresholds=(0.25,),
    z_min=-0.9,
    intensity_min=0.38,
    gesr_enabled=False,
    gesr_reason_attribution=True,
    gesr_evidence_level="detailed",
):
    frame_ids = normalize_frame_ids(frame_ids) if frame_ids is not None else [result.frame_id for result in frame_results]
    iou_keys = metric_keys(eval_iou_threshold, auxiliary_iou_thresholds)
    frame_counts = count_frame_statuses(frame_results, requested=len(frame_ids))
    metrics_by_iou = aggregate_frame_result_metrics(frame_results, iou_keys)
    totals = aggregate_frame_result_totals(frame_results, iou_keys)
    error_counts, error_stage_counts = count_frame_errors(frame_results)

    return BatchResult(
        status=derive_batch_status(frame_counts),
        frame_results=frame_results,
        frame_counts=frame_counts,
        metrics_by_iou=metrics_by_iou,
        totals={
            **totals,
            "data_root": str(data_root),
            "box_mode": "oriented_pca" if oriented else "axis_aligned",
            "parameters": {
                "eps": float(eps),
                "min_points": int(min_points),
                "z_min": float(z_min),
                "intensity_min": float(intensity_min),
                "nms_iou_threshold": float(nms_iou_threshold),
                "eval_iou_threshold": float(eval_iou_threshold),
                "auxiliary_iou_thresholds": [float(threshold) for threshold in auxiliary_iou_thresholds],
                "gesr_enabled": bool(gesr_enabled),
                "gesr_reason_attribution": bool(gesr_reason_attribution),
                "gesr_evidence_level": str(gesr_evidence_level),
            },
        },
        error_counts=error_counts,
        error_stage_counts=error_stage_counts,
        artifacts={"requested_frame_ids": frame_ids},
    )


def count_frame_statuses(frame_results, requested):
    counts = {
        "requested": int(requested),
        "processed": len(frame_results),
        "metric_valid": 0,
        "success": 0,
        "partial_success": 0,
        "skipped": 0,
        "failed": 0,
        "excluded_from_metrics": 0,
    }

    for result in frame_results:
        status = FrameStatus(result.status)
        counts[status.value] += 1
        if result.metric_valid:
            counts["metric_valid"] += 1
        else:
            counts["excluded_from_metrics"] += 1

    return counts


def derive_batch_status(frame_counts):
    if frame_counts["requested"] == 0:
        return BatchStatus.FAILED
    if frame_counts["metric_valid"] == 0:
        return BatchStatus.FAILED
    if frame_counts["success"] == frame_counts["requested"]:
        return BatchStatus.SUCCESS
    return BatchStatus.PARTIAL_SUCCESS


def aggregate_frame_result_metrics(frame_results, iou_keys):
    valid_results = [result for result in frame_results if result.metric_valid]
    metrics_by_iou = {}
    for iou_key in iou_keys:
        frame_metrics = [result.metrics_by_iou.get(iou_key) for result in valid_results]
        metrics_by_iou[iou_key] = aggregate_frame_metrics(frame_metrics)
    return metrics_by_iou


def aggregate_frame_metrics(metrics_list):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_neutralized = 0
    per_class = {}

    for metrics in metrics_list:
        if metrics is None:
            continue
        if isinstance(metrics, dict):
            metrics = FrameMetrics(**metrics)
        total_tp += metrics.tp
        total_fp += metrics.fp
        total_fn += metrics.fn
        total_neutralized += metrics.neutralized_detections
        merge_per_class_metrics(per_class, metrics.per_class)

    return FrameMetrics(
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        precision=safe_divide(total_tp, total_tp + total_fp),
        recall=safe_divide(total_tp, total_tp + total_fn),
        f1=safe_f1(total_tp, total_fp, total_fn),
        neutralized_detections=total_neutralized,
        per_class=finalize_per_class_metrics(per_class),
    )


def merge_per_class_metrics(target, per_class):
    for class_name, class_metrics in (per_class or {}).items():
        class_target = target.setdefault(class_name, {"tp": 0, "fp": 0, "fn": 0})
        class_target["tp"] += int(class_metrics.get("tp", 0))
        class_target["fp"] += int(class_metrics.get("fp", 0))
        class_target["fn"] += int(class_metrics.get("fn", 0))


def finalize_per_class_metrics(per_class):
    finalized = {}
    for class_name, class_metrics in per_class.items():
        tp = class_metrics["tp"]
        fp = class_metrics["fp"]
        fn = class_metrics["fn"]
        finalized[class_name] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": safe_divide(tp, tp + fp),
            "recall": safe_divide(tp, tp + fn),
            "f1": safe_f1(tp, fp, fn),
        }
    return dict(sorted(finalized.items()))


def aggregate_frame_result_totals(frame_results, iou_keys):
    valid_results = [result for result in frame_results if result.metric_valid]
    totals = {
        "num_points": 0,
        "num_labels_raw": 0,
        "num_positive_gt": 0,
        "num_neutral_gt": 0,
        "num_excluded_gt": 0,
        "num_dontcare": 0,
        "num_gt_outside_roi": 0,
        "num_invalid_gt": 0,
        "num_raw_detections": 0,
        "num_car_detections_before_nms": 0,
        "num_detections_after_nms": 0,
        "num_ignored_detection_class": 0,
        "num_detections_outside_roi": 0,
        "num_suppressed_by_nms": 0,
    }

    for result in valid_results:
        for key in list(totals):
            totals[key] += getattr(result, key) or 0

    for iou_key in iou_keys:
        suffix = iou_suffix(iou_key)
        totals[f"neutralized_{suffix}"] = sum(
            (result.metrics_by_iou.get(iou_key).neutralized_detections if result.metrics_by_iou.get(iou_key) else 0)
            for result in valid_results
        )

    return totals


def count_frame_errors(frame_results):
    counts_by_code = {}
    counts_by_stage = {}
    for result in frame_results:
        if result.error is None:
            continue
        error_code = str(result.error.error_code)
        error_stage = str(result.error.error_stage)
        counts_by_code[error_code] = counts_by_code.get(error_code, 0) + 1
        counts_by_stage[error_stage] = counts_by_stage.get(error_stage, 0) + 1
    return dict(sorted(counts_by_code.items())), dict(sorted(counts_by_stage.items()))


def precheck_kitti_frame_inputs(data_root, frame_id):
    frame_id = str(frame_id).zfill(6)
    velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
    calib_path = resolve_kitti_calib_path(data_root, frame_id)

    if not velodyne_path.exists():
        return skipped_frame_result(frame_id, ErrorCode.MISSING_BIN, "missing KITTI point cloud file", velodyne_path)
    if not label_path.exists():
        return skipped_frame_result(frame_id, ErrorCode.MISSING_LABEL, "missing KITTI label file", label_path)
    if not calib_path.exists():
        return skipped_frame_result(frame_id, ErrorCode.MISSING_CALIB, "missing KITTI calib file", calib_path)
    return None


def skipped_frame_result(frame_id, error_code, message, input_path):
    return FrameResult(
        frame_id=frame_id,
        status=FrameStatus.SKIPPED,
        error=FrameError(
            error_code=error_code,
            error_stage=ErrorStage.INPUT_CHECK,
            error_message=message,
            input_path=input_path,
        ),
    )


def unexpected_failed_frame_result(frame_id, exc):
    return FrameResult(
        frame_id=str(frame_id).zfill(6),
        status=FrameStatus.FAILED,
        error=FrameError(
            error_code=ErrorCode.UNEXPECTED_FRAME_ERROR,
            error_stage=ErrorStage.UNKNOWN,
            error_message=str(exc),
            exception_type=type(exc).__name__,
        ),
    )


def run_kitti_batch_evaluation_from_config(config, progress_callback=None):
    data_config = config["data"]
    frame_ids = resolve_config_frame_ids(data_config)
    return run_kitti_batch_evaluation(
        data_root=data_config["root"],
        frame_ids=frame_ids,
        eps=config["detector"]["eps"],
        min_points=config["detector"]["min_points"],
        oriented=config["detector"]["oriented"],
        z_min=config["detector"].get("z_min", -0.9),
        intensity_min=config["detector"].get("intensity_min", 0.38),
        nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=config["evaluation"]["iou_threshold"],
        auxiliary_iou_thresholds=config["evaluation"].get("auxiliary_iou_thresholds", [0.25]),
        report_dir=config["outputs"]["report_dir"],
        gesr_enabled=config["detector"].get("gesr_enabled", False),
        gesr_reason_attribution=config["detector"].get("gesr_reason_attribution", True),
        gesr_evidence_level=config["detector"].get("gesr_evidence_level", "detailed"),
        progress_callback=progress_callback,
    )


def run_kitti_batch_report_from_config(config, config_input_path=None, command=None, progress_callback=None):
    data_config = config["data"]
    frame_ids = resolve_config_frame_ids(data_config)
    output_root = config["outputs"].get("batch_report_root", "outputs/kitti_batch_eval")
    from bev_tracking.report_writer import (
        finalize_batch_report,
        prepare_batch_report,
        utc_now_iso,
        write_streamed_frame_report,
    )

    started_at = utc_now_iso()
    total_start_time = perf_counter()
    report_context = prepare_batch_report(
        output_root=output_root,
        config_input_path=config_input_path,
        config_effective=config,
        task_name="kitti_car_batch",
        started_at=started_at,
        total_start_time=total_start_time,
    )
    phase2_variant = config.get("phase2", {}).get("variant")
    delta22_gt_ids_by_frame = (
        load_phase2_delta22_gt_ids_by_frame()
        if phase2_variant in {"T0", "T2", "GESR-v1"}
        else {}
    )
    batch_result = run_kitti_batch_result(
        data_root=data_config["root"],
        frame_ids=frame_ids,
        eps=config["detector"]["eps"],
        min_points=config["detector"]["min_points"],
        oriented=config["detector"]["oriented"],
        z_min=config["detector"].get("z_min", -0.9),
        intensity_min=config["detector"].get("intensity_min", 0.38),
        nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=config["evaluation"]["iou_threshold"],
        auxiliary_iou_thresholds=config["evaluation"].get("auxiliary_iou_thresholds", [0.25]),
        gesr_enabled=config["detector"].get("gesr_enabled", False),
        gesr_reason_attribution=config["detector"].get("gesr_reason_attribution", True),
        gesr_evidence_level=config["detector"].get("gesr_evidence_level", "detailed"),
        progress_callback=progress_callback,
        frame_result_callback=lambda frame_result: write_streamed_frame_report(
            frame_result,
            report_context,
        ),
        phase2_variant=phase2_variant,
        phase2_delta22_gt_ids_by_frame=delta22_gt_ids_by_frame,
    )
    return finalize_batch_report(
        batch_result,
        report_context,
        command=command,
    )


def resolve_config_frame_ids(data_config):
    manifest_path = data_config.get("manifest")
    if manifest_path:
        from bev_tracking.failure_evidence_batch import load_diagnostic_frame_ids

        return load_diagnostic_frame_ids(manifest_path)
    return data_config.get("frame_ids") or [data_config["frame_id"]]


def load_phase2_delta22_gt_ids_by_frame(
    identity_path="configs/experiments/v15_4/pre_run_identity.json",
):
    payload = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    identities = payload["delta_22"]["ordered_identity_list"]
    if len(identities) != 22:
        raise ValueError("frozen delta-22 identity list must contain exactly 22 GTs")
    by_frame = {}
    for frame_id, gt_id in identities:
        by_frame.setdefault(str(frame_id).zfill(6), []).append(str(gt_id))
    return {frame_id: tuple(gt_ids) for frame_id, gt_ids in by_frame.items()}


def format_batch_report_summary(batch_result, paths):
    frame_counts = batch_result.frame_counts
    return "\n".join(
        [
            f"batch status: {batch_result.status.value}",
            f'frames requested: {frame_counts["requested"]}',
            f'frames metric valid: {frame_counts["metric_valid"]}',
            f'frames success: {frame_counts["success"]}',
            f'frames skipped: {frame_counts["skipped"]}',
            f'frames failed: {frame_counts["failed"]}',
            f'saved {paths["summary_json"]}',
            f'saved {paths["frames_csv"]}',
            f'run directory: {paths["summary_json"].parent}',
        ]
    )


def normalize_frame_ids(frame_ids):
    if frame_ids is None:
        return ["000000"]
    if isinstance(frame_ids, str):
        frame_ids = [frame_ids]
    return [str(frame_id).zfill(6) for frame_id in frame_ids]


def summarize_batch_reports(
    frame_reports,
    data_root,
    eps,
    min_points,
    oriented,
    nms_iou_threshold,
    eval_iou_threshold,
    auxiliary_iou_thresholds,
    z_min=-0.9,
    intensity_min=0.38,
):
    total_points = 0
    total_gt_boxes = 0
    total_detections = 0
    iou_keys = metric_keys(eval_iou_threshold, auxiliary_iou_thresholds)
    valid_frame_reports = [report for report in frame_reports if is_metric_valid_report(report)]

    for report in valid_frame_reports:
        total_points += report.get("num_points") or 0
        total_gt_boxes += report.get("num_gt_boxes") or 0
        total_detections += report.get("num_detections_after_nms") or 0

    metrics_by_iou = {}
    for iou_key in iou_keys:
        frame_metrics = [get_report_metrics_for_iou(report, iou_key) for report in valid_frame_reports]
        metrics_by_iou[iou_key] = aggregate_metrics(frame_metrics)

    primary_key = format_iou_key(eval_iou_threshold)
    primary_metrics = metrics_by_iou[primary_key]

    return {
        "num_frames": len(frame_reports),
        "frame_ids": [report["frame_id"] for report in frame_reports],
        "data_root": data_root,
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
        "metrics_by_iou": metrics_by_iou,
        "totals": {
            "num_points": int(total_points),
            "num_gt_boxes": int(total_gt_boxes),
            "num_detections_after_nms": int(total_detections),
            "tp": primary_metrics["tp"],
            "fp": primary_metrics["fp"],
            "fn": primary_metrics["fn"],
            "precision": primary_metrics["precision"],
            "recall": primary_metrics["recall"],
            "f1": primary_metrics["f1"],
            "per_class": primary_metrics["per_class"],
        },
        "frames": [
            {
                "frame_id": report["frame_id"],
                "num_points": report["num_points"],
                "num_gt_boxes": report["num_gt_boxes"],
                "num_detections_after_nms": report["num_detections_after_nms"],
                "status": report.get("status"),
                "metric_valid": is_metric_valid_report(report),
                "error": report.get("error"),
                "metrics": report.get("metrics", {}),
                "metrics_by_iou": collect_report_metrics_by_iou(report, iou_keys),
                "report_path": report["report_path"],
            }
            for report in frame_reports
        ],
    }


def metric_keys(eval_iou_threshold, auxiliary_iou_thresholds):
    keys = [format_iou_key(eval_iou_threshold)]
    for threshold in auxiliary_iou_thresholds:
        key = format_iou_key(threshold)
        if key not in keys:
            keys.append(key)
    return keys


def format_iou_key(threshold):
    return f"{float(threshold):.2f}"


def iou_suffix(iou_key):
    return f'iou_{iou_key.replace(".", "_")}'


def get_report_metrics_for_iou(report, iou_key):
    if not is_metric_valid_report(report):
        return None
    if iou_key == format_iou_key(report["iou_threshold"]):
        return report["metrics"]
    return report["auxiliary"][iou_key]["metrics"]


def collect_report_metrics_by_iou(report, iou_keys):
    if not is_metric_valid_report(report):
        return {}
    return {iou_key: get_report_metrics_for_iou(report, iou_key) for iou_key in iou_keys}


def is_metric_valid_report(report):
    return bool(report.get("metric_valid", True))


def aggregate_metrics(metrics_list):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_class = {}

    for metrics in metrics_list:
        if metrics is None:
            continue
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        for class_name, class_metrics in metrics.get("per_class", {}).items():
            target = per_class.setdefault(class_name, {"tp": 0, "fp": 0, "fn": 0})
            target["tp"] += class_metrics["tp"]
            target["fp"] += class_metrics["fp"]
            target["fn"] += class_metrics["fn"]

    for class_metrics in per_class.values():
        tp = class_metrics["tp"]
        fp = class_metrics["fp"]
        fn = class_metrics["fn"]
        class_metrics["precision"] = safe_divide(tp, tp + fp)
        class_metrics["recall"] = safe_divide(tp, tp + fn)
        class_metrics["f1"] = safe_f1(tp, fp, fn)

    return {
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": safe_divide(total_tp, total_tp + total_fp),
        "recall": safe_divide(total_tp, total_tp + total_fn),
        "f1": safe_f1(total_tp, total_fp, total_fn),
        "per_class": dict(sorted(per_class.items())),
    }


def save_frame_csv(frame_reports, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    iou_keys = report_iou_keys(frame_reports)
    metric_fieldnames = []
    for iou_key in iou_keys:
        suffix = iou_suffix(iou_key)
        metric_fieldnames.extend(
            [
                f"tp_{suffix}",
                f"fp_{suffix}",
                f"fn_{suffix}",
                f"precision_{suffix}",
                f"recall_{suffix}",
                f"f1_{suffix}",
            ]
        )

    fieldnames = [
        "frame_id",
        "num_points",
        "num_gt_boxes",
        "num_detections_after_nms",
        *metric_fieldnames,
        "report_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in frame_reports:
            row = {
                "frame_id": report["frame_id"],
                "num_points": report["num_points"],
                "num_gt_boxes": report["num_gt_boxes"],
                "num_detections_after_nms": report["num_detections_after_nms"],
                "report_path": report["report_path"],
            }
            for iou_key in iou_keys:
                metrics = get_report_metrics_for_iou(report, iou_key)
                suffix = iou_suffix(iou_key)
                if metrics is None:
                    row[f"tp_{suffix}"] = ""
                    row[f"fp_{suffix}"] = ""
                    row[f"fn_{suffix}"] = ""
                    row[f"precision_{suffix}"] = ""
                    row[f"recall_{suffix}"] = ""
                    row[f"f1_{suffix}"] = ""
                    continue
                row[f"tp_{suffix}"] = metrics["tp"]
                row[f"fp_{suffix}"] = metrics["fp"]
                row[f"fn_{suffix}"] = metrics["fn"]
                row[f"precision_{suffix}"] = csv_metric(metrics["precision"])
                row[f"recall_{suffix}"] = csv_metric(metrics["recall"])
                row[f"f1_{suffix}"] = csv_metric(metrics["f1"])
            writer.writerow(row)


def report_iou_keys(frame_reports):
    if not frame_reports:
        return []
    first = next((report for report in frame_reports if is_metric_valid_report(report)), None)
    if first is None:
        return ["0.50", "0.25"]
    return metric_keys(
        first["iou_threshold"],
        [float(key) for key in first.get("auxiliary", {}).keys()],
    )


def format_batch_summary(summary, summary_path, csv_path):
    totals = summary["totals"]
    precision = format_metric(totals["precision"])
    recall = format_metric(totals["recall"])
    f1 = format_metric(totals["f1"])
    metric_lines = []
    for iou_key, metrics in summary["metrics_by_iou"].items():
        metric_lines.append(
            f'iou={iou_key} tp={metrics["tp"]} fp={metrics["fp"]} fn={metrics["fn"]} '
            f'precision={format_metric(metrics["precision"])} '
            f'recall={format_metric(metrics["recall"])} '
            f'f1={format_metric(metrics["f1"])}'
        )
    return "\n".join(
        [
            f'frames: {summary["num_frames"]}',
            f'box mode: {summary["box_mode"]}',
            f'total points: {totals["num_points"]}',
            f'total gt boxes: {totals["num_gt_boxes"]}',
            f'total detections after nms: {totals["num_detections_after_nms"]}',
            f'tp={totals["tp"]} fp={totals["fp"]} fn={totals["fn"]}',
            f"precision={precision} recall={recall} f1={f1}",
            *metric_lines,
            f"saved {summary_path}",
            f"saved {csv_path}",
        ]
    )


def csv_metric(value):
    if value is None:
        return ""
    return f"{value:.6f}"
