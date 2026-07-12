import csv
from pathlib import Path

from bev_tracking.eval_policy import safe_divide
from bev_tracking.pipeline import run_kitti_bev_evaluation, save_json_report
from bev_tracking.pipeline import format_metric


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
):
    frame_ids = normalize_frame_ids(frame_ids)
    frame_reports = []

    for frame_id in frame_ids:
        report, output_path = run_kitti_bev_evaluation(
            data_root=data_root,
            frame_id=frame_id,
            eps=eps,
            min_points=min_points,
            oriented=oriented,
            nms_iou_threshold=nms_iou_threshold,
            eval_iou_threshold=eval_iou_threshold,
            auxiliary_iou_thresholds=auxiliary_iou_thresholds,
            report_dir=report_dir,
        )
        report["report_path"] = str(output_path)
        frame_reports.append(report)

    summary = summarize_batch_reports(
        frame_reports=frame_reports,
        data_root=data_root,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
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


def run_kitti_batch_evaluation_from_config(config):
    data_config = config["data"]
    frame_ids = data_config.get("frame_ids") or [data_config["frame_id"]]
    return run_kitti_batch_evaluation(
        data_root=data_config["root"],
        frame_ids=frame_ids,
        eps=config["detector"]["eps"],
        min_points=config["detector"]["min_points"],
        oriented=config["detector"]["oriented"],
        nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=config["evaluation"]["iou_threshold"],
        auxiliary_iou_thresholds=config["evaluation"].get("auxiliary_iou_thresholds", [0.25]),
        report_dir=config["outputs"]["report_dir"],
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
):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_points = 0
    total_gt_boxes = 0
    total_detections = 0
    per_class = {}

    for report in frame_reports:
        metrics = report["metrics"]
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_points += report["num_points"]
        total_gt_boxes += report["num_gt_boxes"]
        total_detections += report["num_detections_after_nms"]

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

    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)

    return {
        "num_frames": len(frame_reports),
        "frame_ids": [report["frame_id"] for report in frame_reports],
        "data_root": data_root,
        "box_mode": "oriented_pca" if oriented else "axis_aligned",
        "parameters": {
            "eps": float(eps),
            "min_points": int(min_points),
            "nms_iou_threshold": float(nms_iou_threshold),
            "eval_iou_threshold": float(eval_iou_threshold),
            "auxiliary_iou_thresholds": [float(threshold) for threshold in auxiliary_iou_thresholds],
        },
        "totals": {
            "num_points": int(total_points),
            "num_gt_boxes": int(total_gt_boxes),
            "num_detections_after_nms": int(total_detections),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "precision": precision,
            "recall": recall,
            "per_class": dict(sorted(per_class.items())),
        },
        "frames": [
            {
                "frame_id": report["frame_id"],
                "num_points": report["num_points"],
                "num_gt_boxes": report["num_gt_boxes"],
                "num_detections_after_nms": report["num_detections_after_nms"],
                "metrics": report["metrics"],
                "report_path": report["report_path"],
            }
            for report in frame_reports
        ],
    }


def save_frame_csv(frame_reports, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_id",
        "num_points",
        "num_gt_boxes",
        "num_detections_after_nms",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "report_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in frame_reports:
            metrics = report["metrics"]
            writer.writerow(
                {
                    "frame_id": report["frame_id"],
                    "num_points": report["num_points"],
                    "num_gt_boxes": report["num_gt_boxes"],
                    "num_detections_after_nms": report["num_detections_after_nms"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": csv_metric(metrics["precision"]),
                    "recall": csv_metric(metrics["recall"]),
                    "report_path": report["report_path"],
                }
            )


def format_batch_summary(summary, summary_path, csv_path):
    totals = summary["totals"]
    precision = format_metric(totals["precision"])
    recall = format_metric(totals["recall"])
    return "\n".join(
        [
            f'frames: {summary["num_frames"]}',
            f'box mode: {summary["box_mode"]}',
            f'total points: {totals["num_points"]}',
            f'total gt boxes: {totals["num_gt_boxes"]}',
            f'total detections after nms: {totals["num_detections_after_nms"]}',
            f'tp={totals["tp"]} fp={totals["fp"]} fn={totals["fn"]}',
            f"precision={precision} recall={recall}",
            f"saved {summary_path}",
            f"saved {csv_path}",
        ]
    )


def csv_metric(value):
    if value is None:
        return ""
    return f"{value:.6f}"
