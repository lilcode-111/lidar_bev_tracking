from pathlib import Path

import numpy as np


KITTI_KNOWN_CLASSES = {
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
    "DontCare",
}


def resolve_kitti_paths(data_root, frame_id):
    data_root = Path(data_root)
    frame_id = str(frame_id).zfill(6)
    velodyne_path = data_root / "training" / "velodyne" / f"{frame_id}.bin"
    label_path = data_root / "training" / "label_2" / f"{frame_id}.txt"
    return velodyne_path, label_path


def resolve_kitti_calib_path(data_root, frame_id):
    data_root = Path(data_root)
    frame_id = str(frame_id).zfill(6)
    return data_root / "training" / "calib" / f"{frame_id}.txt"


def load_kitti_point_cloud(bin_path):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"KITTI point cloud not found: {bin_path}")

    points = np.fromfile(str(bin_path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Invalid KITTI point cloud size, expected N*4 floats: {bin_path}")

    return points.reshape(-1, 4)


def parse_kitti_label_line(line):
    fields = line.strip().split()
    if len(fields) < 15:
        return None

    class_name = fields[0]

    height, width, length = map(float, fields[8:11])
    x_cam, y_cam, z_cam = map(float, fields[11:14])
    rotation_y = float(fields[14])

    return {
        "class_name": class_name,
        "is_known_class": class_name in KITTI_KNOWN_CLASSES,
        "truncated": float(fields[1]),
        "occluded": int(fields[2]),
        "alpha": float(fields[3]),
        "bbox_2d": [float(v) for v in fields[4:8]],
        "height": height,
        "width": width,
        "length": length,
        "location_camera": [x_cam, y_cam, z_cam],
        "rotation_y": rotation_y,
    }


def load_kitti_labels(label_path):
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"KITTI label file not found: {label_path}")

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            label = parse_kitti_label_line(line)
            if label is not None:
                labels.append(label)

    return labels
