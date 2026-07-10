from collections import defaultdict

from bev_tracking.geometry import bev_iou


def match_detections_to_gt(detections, gt_boxes, iou_threshold=0.25):
    sorted_detections = sorted(detections, key=lambda box: box.get("score", 0.0), reverse=True)
    matched_gt = set()
    matches = []
    false_positives = []

    for det in sorted_detections:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if det["class_name"] != gt["class_name"]:
                continue

            iou = bev_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            gt = gt_boxes[best_gt_idx]
            matches.append(
                {
                    "det_id": det["id"],
                    "gt_id": gt["id"],
                    "class_name": det["class_name"],
                    "iou": float(best_iou),
                    "score": float(det.get("score", 0.0)),
                }
            )
        else:
            false_positives.append(
                {
                    "det_id": det["id"],
                    "class_name": det["class_name"],
                    "best_iou": float(best_iou),
                    "score": float(det.get("score", 0.0)),
                }
            )

    false_negatives = []
    for gt_idx, gt in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            false_negatives.append({"gt_id": gt["id"], "class_name": gt["class_name"]})

    return matches, false_positives, false_negatives


def _empty_class_metrics():
    return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0}


def summarize_metrics(matches, false_positives, false_negatives):
    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

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
        metrics["precision"] = class_tp / (class_tp + class_fp) if class_tp + class_fp > 0 else 0.0
        metrics["recall"] = class_tp / (class_tp + class_fn) if class_tp + class_fn > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "per_class": dict(sorted(per_class.items())),
    }


def evaluate_detections(detections, gt_boxes, iou_threshold=0.25):
    matches, false_positives, false_negatives = match_detections_to_gt(
        detections, gt_boxes, iou_threshold=iou_threshold
    )
    metrics = summarize_metrics(matches, false_positives, false_negatives)
    return {
        "iou_threshold": float(iou_threshold),
        "metrics": metrics,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
