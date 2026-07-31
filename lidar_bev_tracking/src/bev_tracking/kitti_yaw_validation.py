import numpy as np

from bev_tracking.geometry import box_corners_bev
from bev_tracking.geometry_sanity import angle_error
from bev_tracking.kitti_calib import (
    camera_rotation_y_to_lidar_yaw,
    has_valid_3d_box,
    kitti_labels_to_lidar_boxes,
    lidar_to_camera_rect_matrix,
    transform_points,
)


DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD = 1e-4
DEFAULT_CORNER_TOLERANCE_M = 1e-3


def reference_camera_rotation_y_to_lidar_yaw(rotation_y, calib):
    """Convert KITTI rotation_y using the labelled box length axis."""
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    heading_camera = np.asarray([[c, 0.0, -s]], dtype=np.float32)
    camera_to_lidar = np.linalg.inv(lidar_to_camera_rect_matrix(calib))
    heading_lidar = heading_camera @ camera_to_lidar[:3, :3].T
    return float(np.arctan2(heading_lidar[0, 1], heading_lidar[0, 0]))


def reference_kitti_box_corners_in_lidar(label, calib):
    """Build KITTI box BEV corners independently from the project yaw helper."""
    length = float(label["length"])
    width = float(label["width"])
    height = float(label["height"])
    x_camera, y_camera, z_camera = label["location_camera"]
    rotation_y = float(label["rotation_y"])

    local_corners = np.asarray(
        [
            [length / 2.0, 0.0, width / 2.0],
            [length / 2.0, 0.0, -width / 2.0],
            [-length / 2.0, 0.0, -width / 2.0],
            [-length / 2.0, 0.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    rotation_camera_y = np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )
    center_camera = np.asarray(
        [x_camera, y_camera - height / 2.0, z_camera],
        dtype=np.float32,
    )
    corners_camera = local_corners @ rotation_camera_y.T + center_camera
    camera_to_lidar = np.linalg.inv(lidar_to_camera_rect_matrix(calib))
    return transform_points(corners_camera, camera_to_lidar)[:, :2]


def unordered_corner_error(reference_corners, candidate_corners):
    reference_corners = np.asarray(reference_corners, dtype=np.float32)
    candidate_corners = np.asarray(candidate_corners, dtype=np.float32)
    distances = np.linalg.norm(
        reference_corners[:, None, :] - candidate_corners[None, :, :],
        axis=2,
    )
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def build_kitti_yaw_semantic_report(
    labels,
    calib,
    frame_id,
    yaw_tolerance_rad=DEFAULT_YAW_SEMANTIC_TOLERANCE_RAD,
    corner_tolerance_m=DEFAULT_CORNER_TOLERANCE_M,
):
    valid_labels = [label for label in labels if has_valid_3d_box(label)]
    boxes = kitti_labels_to_lidar_boxes(labels, calib)
    box_reports = []

    for label, box in zip(valid_labels, boxes):
        expected_yaw = reference_camera_rotation_y_to_lidar_yaw(label["rotation_y"], calib)
        actual_yaw = camera_rotation_y_to_lidar_yaw(label["rotation_y"], calib)
        yaw_error = angle_error(expected_yaw, actual_yaw)
        reference_corners = reference_kitti_box_corners_in_lidar(label, calib)
        actual_corners = box_corners_bev(box)
        corner_error = unordered_corner_error(reference_corners, actual_corners)
        passed = yaw_error <= yaw_tolerance_rad and corner_error <= corner_tolerance_m
        box_reports.append(
            {
                "gt_id": box["id"],
                "class_name": box["class_name"],
                "rotation_y": float(label["rotation_y"]),
                "expected_lidar_yaw": expected_yaw,
                "actual_lidar_yaw": actual_yaw,
                "yaw_semantic_error_rad": yaw_error,
                "corner_alignment_error_m": corner_error,
                "passed": bool(passed),
            }
        )

    failed_boxes = [item for item in box_reports if not item["passed"]]
    return {
        "schema_version": "15.0-yaw-validation",
        "frame_id": str(frame_id).zfill(6),
        "passed": not failed_boxes,
        "tolerances": {
            "yaw_semantic_error_rad": float(yaw_tolerance_rad),
            "corner_alignment_error_m": float(corner_tolerance_m),
        },
        "summary": {
            "num_valid_boxes": len(box_reports),
            "num_passed_boxes": len(box_reports) - len(failed_boxes),
            "num_failed_boxes": len(failed_boxes),
            "max_yaw_semantic_error_rad": max(
                (item["yaw_semantic_error_rad"] for item in box_reports),
                default=0.0,
            ),
            "max_corner_alignment_error_m": max(
                (item["corner_alignment_error_m"] for item in box_reports),
                default=0.0,
            ),
        },
        "boxes": box_reports,
    }
