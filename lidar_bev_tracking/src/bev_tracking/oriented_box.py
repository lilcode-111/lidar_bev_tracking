import numpy as np


def estimate_oriented_box_xy(cluster):
    points_xy = cluster[:, :2].astype(np.float32)
    center = points_xy.mean(axis=0)
    centered = points_xy - center

    if len(points_xy) < 3:
        return float(center[0]), float(center[1]), 0.1, 0.1, 0.0

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, int(np.argmax(eigvals))]
    yaw = float(np.arctan2(main_axis[1], main_axis[0]))

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = centered @ rot

    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    size = np.maximum(local_max - local_min, 0.1)
    local_center = (local_min + local_max) / 2.0
    world_center = center + local_center @ rot.T

    length = float(max(size[0], 0.1))
    width = float(max(size[1], 0.1))

    if width > length:
        length, width = width, length
        yaw += np.pi / 2.0

    yaw = normalize_yaw(yaw)
    return float(world_center[0]), float(world_center[1]), length, width, yaw


def normalize_yaw(yaw):
    return float((yaw + np.pi) % (2.0 * np.pi) - np.pi)
