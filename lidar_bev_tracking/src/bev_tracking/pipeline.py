import json
from pathlib import Path

from bev_tracking.clustering_detector import detect_objects_from_points
from bev_tracking.evaluation import evaluate_detections
from bev_tracking.kitti import (
    load_kitti_labels,
    load_kitti_point_cloud,
    resolve_kitti_calib_path,
    resolve_kitti_paths,
)
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes, load_kitti_calib
from bev_tracking.nms import nms_bev


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
):
    velodyne_path, label_path = resolve_kitti_paths(data_root, frame_id)
    calib_path = resolve_kitti_calib_path(data_root, frame_id)

    points = load_kitti_point_cloud(velodyne_path)
    labels = load_kitti_labels(label_path)
    calib = load_kitti_calib(calib_path)

    raw_detections = detect_objects_from_points(
        points,
        eps=eps,
        min_points=min_points,
        oriented=oriented,
    )
    detections = nms_bev(raw_detections, iou_threshold=nms_iou_threshold)
    gt_boxes = kitti_labels_to_lidar_boxes(labels, calib)
    evaluation = evaluate_detections(
        detections,
        gt_boxes,
        iou_threshold=eval_iou_threshold,
        auxiliary_iou_thresholds=auxiliary_iou_thresholds,
    )

    frame_id = str(frame_id).zfill(6)
    report = {
        "frame_id": frame_id,
        "num_points": len(points),
        "num_labels": len(labels),
        "num_gt_boxes": len(gt_boxes),
        "num_raw_detections": len(raw_detections),
        "num_detections_after_nms": len(detections),
        "box_mode": "oriented_pca" if oriented else "axis_aligned",
        "parameters": {
            "eps": float(eps),
            "min_points": int(min_points),
            "nms_iou_threshold": float(nms_iou_threshold),
            "eval_iou_threshold": float(eval_iou_threshold),
            "auxiliary_iou_thresholds": [float(threshold) for threshold in auxiliary_iou_thresholds],
        },
        **evaluation,
    }

    suffix = "oriented" if oriented else "axis_aligned"
    output_path = Path(report_dir) / f"kitti_eval_{frame_id}_{suffix}.json"
    save_json_report(report, output_path)
    return report, output_path


def run_kitti_bev_evaluation_from_config(config):
    return run_kitti_bev_evaluation(
        data_root=config["data"]["root"],
        frame_id=config["data"]["frame_id"],
        eps=config["detector"]["eps"],
        min_points=config["detector"]["min_points"],
        oriented=config["detector"]["oriented"],
        nms_iou_threshold=config["nms"]["iou_threshold"],
        eval_iou_threshold=config["evaluation"]["iou_threshold"],
        auxiliary_iou_thresholds=config["evaluation"].get("auxiliary_iou_thresholds", [0.25]),
        report_dir=config["outputs"]["report_dir"],
    )


def save_json_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


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
