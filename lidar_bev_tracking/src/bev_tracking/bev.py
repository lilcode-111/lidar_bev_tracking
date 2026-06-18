import numpy as np


def points_to_bev(points, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1):
    x_min, x_max = x_range
    y_min, y_max = y_range

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )
    pts = points[mask]

    height = int((x_max - x_min) / resolution)
    width = int((y_max - y_min) / resolution)
    bev = np.zeros((height, width), dtype=np.float32)

    rows = ((pts[:, 0] - x_min) / resolution).astype(np.int32)
    cols = ((pts[:, 1] - y_min) / resolution).astype(np.int32)

    np.maximum.at(bev, (rows, cols), pts[:, 3])
    return (bev * 255).clip(0, 255).astype(np.uint8)
