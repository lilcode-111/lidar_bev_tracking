import numpy as np

from bev_tracking.kitti_calib import (
    camera_rect_to_lidar_matrix,
    has_valid_3d_box,
    kitti_labels_to_lidar_boxes,
    lidar_to_camera_rect_matrix,
    lidar_yaw_to_camera_rotation_y,
    transform_points,
)


DEFAULT_ROUND_TRIP_TOLERANCE = 1e-4


def normalize_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def angle_error(angle_a, angle_b):
    return abs(normalize_angle(float(angle_a) - float(angle_b)))


def points_in_oriented_3d_box(points, box, tolerance=1e-6):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape N x 3 or N x 4")

    required = ["x", "y", "z", "length", "width", "height", "yaw"]
    missing = [name for name in required if name not in box]
    if missing:
        raise ValueError(f"3D box missing required fields: {', '.join(missing)}")

    center = np.asarray([box["x"], box["y"], box["z"]], dtype=np.float32)
    centered = points[:, :3] - center

    yaw = float(box["yaw"])
    c, s = np.cos(yaw), np.sin(yaw)
    world_to_local_xy = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    local_xy = centered[:, :2] @ world_to_local_xy

    half_length = float(box["length"]) / 2.0
    half_width = float(box["width"]) / 2.0
    half_height = float(box["height"]) / 2.0
    return (
        (np.abs(local_xy[:, 0]) <= half_length + tolerance)
        & (np.abs(local_xy[:, 1]) <= half_width + tolerance)
        & (np.abs(centered[:, 2]) <= half_height + tolerance)
    )


def coordinate_round_trip_metrics(points, calib):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape N x 3 or N x 4")
    if len(points) == 0:
        return {"num_points": 0, "mean_error_m": 0.0, "max_error_m": 0.0}

    lidar_xyz = points[:, :3]
    camera_xyz = transform_points(lidar_xyz, lidar_to_camera_rect_matrix(calib))
    reconstructed = transform_points(camera_xyz, camera_rect_to_lidar_matrix(calib))
    errors = np.linalg.norm(reconstructed - lidar_xyz, axis=1)
    return {
        "num_points": int(len(points)),
        "mean_error_m": float(errors.mean()),
        "max_error_m": float(errors.max()),
    }


def reconstructed_camera_label(box, calib):
    lidar_center = np.asarray([[box["x"], box["y"], box["z"]]], dtype=np.float32)
    camera_center = transform_points(lidar_center, lidar_to_camera_rect_matrix(calib))[0]
    return {
        "location_camera": [
            float(camera_center[0]),
            float(camera_center[1] + float(box["height"]) / 2.0),
            float(camera_center[2]),
        ],
        "rotation_y": lidar_yaw_to_camera_rotation_y(box["yaw"], calib),
    }


def build_geometry_sanity_report(points, labels, calib, frame_id, tolerance=DEFAULT_ROUND_TRIP_TOLERANCE):
    frame_id = str(frame_id).zfill(6)
    round_trip = coordinate_round_trip_metrics(points, calib)
    boxes = kitti_labels_to_lidar_boxes(labels, calib)
    labels_by_gt_id = {
        f"gt_{index}": label
        for index, label in enumerate(labels, start=1)
        if has_valid_3d_box(label)
    }

    box_reports = []
    for box in boxes:
        label = labels_by_gt_id[box["id"]]
        reconstructed = reconstructed_camera_label(box, calib)
        original_location = np.asarray(label["location_camera"], dtype=np.float32)
        reconstructed_location = np.asarray(reconstructed["location_camera"], dtype=np.float32)
        center_error = float(np.linalg.norm(reconstructed_location - original_location))
        yaw_error = angle_error(reconstructed["rotation_y"], label["rotation_y"])
        point_count = int(points_in_oriented_3d_box(points, box).sum())
        box_reports.append(
            {
                "gt_id": box["id"],
                "class_name": box["class_name"],
                "points_in_box": point_count,
                "center_round_trip_error_m": center_error,
                "yaw_round_trip_error_rad": yaw_error,
                "dimensions": {
                    "length": float(box["length"]),
                    "width": float(box["width"]),
                    "height": float(box["height"]),
                },
                "passed": center_error <= tolerance and yaw_error <= tolerance,
            }
        )

    car_reports = [item for item in box_reports if item["class_name"] == "car"]
    invalid_3d_labels = sum(1 for label in labels if not has_valid_3d_box(label))
    passed = round_trip["max_error_m"] <= tolerance and all(item["passed"] for item in box_reports)
    return {
        "frame_id": frame_id,
        "tolerance": float(tolerance),
        "passed": bool(passed),
        "coordinate_round_trip": round_trip,
        "summary": {
            "num_labels_raw": int(len(labels)),
            "num_valid_3d_boxes": int(len(boxes)),
            "num_invalid_3d_labels": int(invalid_3d_labels),
            "num_car_boxes": int(len(car_reports)),
            "car_boxes_with_points": int(sum(item["points_in_box"] > 0 for item in car_reports)),
            "car_boxes_without_points": int(sum(item["points_in_box"] == 0 for item in car_reports)),
        },
        "boxes": box_reports,
    }
