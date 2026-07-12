from collections import defaultdict

from bev_tracking.eval_policy import (
    AUXILIARY_IOU_THRESHOLDS,
    NEUTRAL_IOU_THRESHOLD,
    PRIMARY_IOU_THRESHOLD,
    assign_det_indices,
    classify_gt_box,
    detection_sort_key,
    is_positive_detection,
    normalize_class_name,
    safe_divide,
    safe_f1,
)
from bev_tracking.geometry import bev_iou


def evaluate_detections(
    detections,
    gt_boxes,
    iou_threshold=PRIMARY_IOU_THRESHOLD,
    auxiliary_iou_thresholds=AUXILIARY_IOU_THRESHOLDS,
):
    primary = evaluate_detections_at_iou(
        detections,
        gt_boxes,
        iou_threshold=iou_threshold,
        neutral_iou_threshold=NEUTRAL_IOU_THRESHOLD,
    )
    auxiliary = {
        f"{threshold:.2f}": evaluate_detections_at_iou(
            detections,
            gt_boxes,
            iou_threshold=threshold,
            neutral_iou_threshold=NEUTRAL_IOU_THRESHOLD,
        )
        for threshold in auxiliary_iou_thresholds
    }

    return {
        **primary,
        "policy": {
            "positive_gt_classes": ["car"],
            "neutral_gt_classes": ["van", "truck"],
            "excluded_gt_classes": ["pedestrian", "cyclist", "person_sitting", "tram", "misc"],
            "positive_detection_classes": ["car"],
            "roi": {"x": [0.0, 40.0], "y": [-20.0, 20.0]},
            "neutral_iou_threshold": NEUTRAL_IOU_THRESHOLD,
            "zero_denominator": "undefined_null",
        },
        "auxiliary": auxiliary,
    }


def evaluate_detections_at_iou(
    detections,
    gt_boxes,
    iou_threshold=PRIMARY_IOU_THRESHOLD,
    neutral_iou_threshold=NEUTRAL_IOU_THRESHOLD,
):
    prepared_detections = prepare_detections(detections)
    prepared_gt = prepare_gt_boxes(gt_boxes)
    matches, neutralized_detections, false_positives, false_negatives = match_policy_detections(
        prepared_detections,
        prepared_gt["positive"],
        prepared_gt["neutral"],
        positive_iou_threshold=iou_threshold,
        neutral_iou_threshold=neutral_iou_threshold,
    )
    metrics = summarize_metrics(matches, false_positives, false_negatives)

    return {
        "iou_threshold": float(iou_threshold),
        "neutral_iou_threshold": float(neutral_iou_threshold),
        "metrics": metrics,
        "matches": matches,
        "neutralized_detections": neutralized_detections,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ignored": {
            "detections": prepared_detections["ignored"],
            "gt_boxes": prepared_gt["ignored"],
            "counts": {
                "detections": len(prepared_detections["ignored"]),
                "positive_gt": len(prepared_gt["positive"]),
                "neutral_gt": len(prepared_gt["neutral"]),
                "excluded_gt": len(prepared_gt["ignored"]),
            },
        },
    }


def prepare_detections(detections):
    indexed = assign_det_indices(detections)
    positives = []
    ignored = []

    for det in indexed:
        item = dict(det)
        item["class_name"] = normalize_class_name(item.get("class_name", ""))
        if is_positive_detection(item):
            positives.append(item)
        else:
            ignored.append(
                {
                    "det_id": item.get("id"),
                    "class_name": item["class_name"],
                    "det_index": int(item.get("det_index", 0)),
                    "reason": "not_car_or_outside_roi",
                }
            )

    return {
        "positive": sorted(positives, key=detection_sort_key),
        "ignored": sorted(ignored, key=lambda item: item["det_index"]),
    }


def prepare_gt_boxes(gt_boxes):
    positive = []
    neutral = []
    ignored = []

    for idx, gt in enumerate(gt_boxes):
        item = dict(gt)
        item["gt_index"] = idx
        item["class_name"] = normalize_class_name(item.get("class_name", ""))
        gt_role = classify_gt_box(item)

        if gt_role == "positive":
            positive.append(item)
        elif gt_role == "neutral":
            neutral.append(item)
        else:
            ignored.append(
                {
                    "gt_id": item.get("id"),
                    "class_name": item["class_name"],
                    "gt_index": idx,
                    "reason": gt_role,
                }
            )

    return {"positive": positive, "neutral": neutral, "ignored": ignored}


def match_policy_detections(detections, positive_gt, neutral_gt, positive_iou_threshold, neutral_iou_threshold):
    matched_positive_gt = set()
    matched_neutral_gt = set()
    matches = []
    neutralized_detections = []
    false_positives = []

    for det in detections["positive"]:
        positive_idx, positive_iou = find_best_unmatched_iou(det, positive_gt, matched_positive_gt)
        if positive_idx is not None and positive_iou >= positive_iou_threshold:
            matched_positive_gt.add(positive_idx)
            gt = positive_gt[positive_idx]
            matches.append(
                {
                    "det_id": det.get("id"),
                    "gt_id": gt.get("id"),
                    "class_name": "car",
                    "iou": float(positive_iou),
                    "score": float(det.get("score", 0.0)),
                    "det_index": int(det.get("det_index", 0)),
                }
            )
            continue

        neutral_idx, neutral_iou = find_best_unmatched_iou(det, neutral_gt, matched_neutral_gt)
        if neutral_idx is not None and neutral_iou >= neutral_iou_threshold:
            matched_neutral_gt.add(neutral_idx)
            gt = neutral_gt[neutral_idx]
            neutralized_detections.append(
                {
                    "det_id": det.get("id"),
                    "neutral_gt_id": gt.get("id"),
                    "neutral_class_name": gt.get("class_name"),
                    "iou": float(neutral_iou),
                    "score": float(det.get("score", 0.0)),
                    "det_index": int(det.get("det_index", 0)),
                }
            )
            continue

        false_positives.append(
            {
                "det_id": det.get("id"),
                "class_name": det.get("class_name"),
                "best_positive_iou": float(positive_iou),
                "best_neutral_iou": float(neutral_iou),
                "score": float(det.get("score", 0.0)),
                "det_index": int(det.get("det_index", 0)),
            }
        )

    false_negatives = []
    for gt_idx, gt in enumerate(positive_gt):
        if gt_idx not in matched_positive_gt:
            false_negatives.append({"gt_id": gt.get("id"), "class_name": gt.get("class_name"), "gt_index": gt_idx})

    return matches, neutralized_detections, false_positives, false_negatives


def find_best_unmatched_iou(det, gt_boxes, matched_gt):
    best_iou = 0.0
    best_gt_idx = None

    for gt_idx, gt in enumerate(gt_boxes):
        if gt_idx in matched_gt:
            continue
        iou = bev_iou(det, gt)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx

    return best_gt_idx, best_iou


def _empty_class_metrics():
    return {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}


def summarize_metrics(matches, false_positives, false_negatives):
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_f1(tp, fp, fn)

    per_class = defaultdict(_empty_class_metrics)
    for match in matches:
        per_class[match["class_name"]]["tp"] += 1
    for item in false_positives:
        per_class[item["class_name"]]["fp"] += 1
    for item in false_negatives:
        per_class[item["class_name"]]["fn"] += 1

    for metrics in per_class.values():
        class_tp = metrics["tp"]
        class_fp = metrics["fp"]
        class_fn = metrics["fn"]
        metrics["precision"] = safe_divide(class_tp, class_tp + class_fp)
        metrics["recall"] = safe_divide(class_tp, class_tp + class_fn)
        metrics["f1"] = safe_f1(class_tp, class_fp, class_fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": dict(sorted(per_class.items())),
    }
