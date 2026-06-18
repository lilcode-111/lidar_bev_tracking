import cv2
import numpy as np


CLASS_COLORS = {
    "car": (0, 220, 0),
    "pedestrian": (0, 180, 255),
    "cone": (255, 120, 0),
}


def box_corners_bev(obj):
    length, width = obj["length"], obj["width"]
    corners = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )

    c, s = np.cos(obj["yaw"]), np.sin(obj["yaw"])
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([obj["x"], obj["y"]])


def world_to_pixel(xy, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1):
    x_min, _ = x_range
    y_min, _ = y_range
    row = ((xy[:, 0] - x_min) / resolution).astype(np.int32)
    col = ((xy[:, 1] - y_min) / resolution).astype(np.int32)
    return np.stack([col, row], axis=1)


def draw_objects(bev, objects, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1):
    image = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

    for obj in objects:
        corners = box_corners_bev(obj)
        pts = world_to_pixel(corners, x_range, y_range, resolution)
        color = CLASS_COLORS.get(obj["class_name"], (255, 255, 255))

        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        label = f'{obj["class_name"]}:{obj["id"]}'
        cv2.putText(image, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
