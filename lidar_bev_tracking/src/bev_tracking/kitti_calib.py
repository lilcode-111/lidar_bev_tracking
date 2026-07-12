from pathlib import Path

import numpy as np


def _parse_matrix(line):
    values = [float(v) for v in line.split()[1:]]
    return np.asarray(values, dtype=np.float32)


def load_kitti_calib(calib_path):
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"KITTI calib not found: {calib_path}")

    calib = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key = line.split(":", 1)[0]
            values = _parse_matrix(line)
            if key.startswith("P") and values.size == 12:
                calib[key] = values.reshape(3, 4)
            elif key in {"R0_rect", "R_rect"} and values.size == 9:
                calib["R0_rect"] = values.reshape(3, 3)
            elif key in {"Tr_velo_to_cam", "Tr_velo_cam"} and values.size == 12:
                calib["Tr_velo_to_cam"] = values.reshape(3, 4)

    if "R0_rect" not in calib:
        calib["R0_rect"] = np.eye(3, dtype=np.float32)
    if "Tr_velo_to_cam" not in calib:
        raise ValueError(f"KITTI calib missing Tr_velo_to_cam: {calib_path}")

    return calib


def _to_homogeneous_4x4(matrix):
    out = np.eye(4, dtype=np.float32)
    out[: matrix.shape[0], : matrix.shape[1]] = matrix
    return out


def camera_rect_to_lidar_matrix(calib):
    r0 = np.eye(4, dtype=np.float32)
    r0[:3, :3] = calib["R0_rect"]
    tr_velo_to_cam = _to_homogeneous_4x4(calib["Tr_velo_to_cam"])
    velo_to_rect = r0 @ tr_velo_to_cam
    return np.linalg.inv(velo_to_rect)


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float32)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (points_h @ transform.T)[:, :3]


def has_valid_3d_box(label):
    return label["height"] > 0.0 and label["width"] > 0.0 and label["length"] > 0.0


def kitti_labels_to_lidar_boxes(labels, calib):
    cam_to_lidar = camera_rect_to_lidar_matrix(calib)
    boxes = []

    for idx, label in enumerate(labels, start=1):
        if not has_valid_3d_box(label):
            continue

        height = label["height"]
        width = label["width"]
        length = label["length"]
        x_cam, y_cam, z_cam = label["location_camera"]
        rotation_y = label["rotation_y"]

        center_cam = np.asarray([[x_cam, y_cam - height / 2.0, z_cam]], dtype=np.float32)
        center_lidar = transform_points(center_cam, cam_to_lidar)[0]

        heading_cam = np.asarray([[np.sin(rotation_y), 0.0, np.cos(rotation_y)]], dtype=np.float32)
        heading_lidar = heading_cam @ cam_to_lidar[:3, :3].T
        yaw = float(np.arctan2(heading_lidar[0, 1], heading_lidar[0, 0]))

        boxes.append(
            {
                "id": f"gt_{idx}",
                "class_name": label["class_name"].lower(),
                "x": float(center_lidar[0]),
                "y": float(center_lidar[1]),
                "z": float(center_lidar[2]),
                "length": float(length),
                "width": float(width),
                "yaw": yaw,
                "source": "kitti_label",
            }
        )

    return boxes
