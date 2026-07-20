from pathlib import Path

from bev_tracking.result_types import BatchResult, FailureCase, FailureCategory, FrameMetrics, FrameResult


DEFAULT_TOP_K = 5
PRIMARY_IOU_KEY = "0.50"
PRIMARY_IOU_THRESHOLD = 0.5


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


def validate_top_k(top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


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
        source_run_id=frame_source_run_id(frame),
        source_run_directory=frame_source_run_directory(frame),
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
