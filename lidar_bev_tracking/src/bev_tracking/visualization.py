import cv2
import numpy as np

from bev_tracking.geometry import box_corners_bev


CLASS_COLORS = {
    "car": (0, 220, 0),
    "pedestrian": (0, 180, 255),
    "cone": (255, 120, 0),
}

GT_COLORS = {
    "car": (0, 255, 255),
    "pedestrian": (255, 0, 255),
    "cyclist": (255, 255, 0),
    "van": (0, 255, 255),
    "truck": (0, 255, 255),
    "person_sitting": (255, 0, 255),
}


def world_to_pixel(xy, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1):
    x_min, _ = x_range
    y_min, _ = y_range
    row = ((xy[:, 0] - x_min) / resolution).astype(np.int32)
    col = ((xy[:, 1] - y_min) / resolution).astype(np.int32)
    return np.stack([col, row], axis=1)


def draw_objects(bev, objects, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1, show_score=False):
    image = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

    for obj in objects:
        color = CLASS_COLORS.get(obj["class_name"], (255, 255, 255))
        label = f'{obj["class_name"]}:{obj["id"]}'
        if show_score and "score" in obj:
            label += f' {obj["score"]:.2f}'
        _draw_bev_box(image, obj, color, label, x_range, y_range, resolution)

    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def draw_detections_and_gt(
    bev,
    detections,
    gt_boxes,
    x_range=(0.0, 40.0),
    y_range=(-20.0, 20.0),
    resolution=0.1,
):
    image = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

    for obj in detections:
        color = CLASS_COLORS.get(obj["class_name"], (255, 255, 255))
        label = f'det:{obj["class_name"]}'
        if "score" in obj:
            label += f' {obj["score"]:.2f}'
        _draw_bev_box(image, obj, color, label, x_range, y_range, resolution)

    for obj in gt_boxes:
        color = GT_COLORS.get(obj["class_name"], (0, 255, 255))
        label = f'gt:{obj["class_name"]}'
        _draw_bev_box(image, obj, color, label, x_range, y_range, resolution, thickness=1)

    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _draw_bev_box(image, obj, color, label, x_range, y_range, resolution, thickness=2):
    corners = box_corners_bev(obj)
    pts = world_to_pixel(corners, x_range, y_range, resolution)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.putText(image, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_tracks(bev, tracks, x_range=(0.0, 40.0), y_range=(-20.0, 20.0), resolution=0.1):
    image = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

    for track in tracks:
        obj = track.to_box()
        corners = box_corners_bev(obj)
        pts = world_to_pixel(corners, x_range, y_range, resolution)
        color = CLASS_COLORS.get(obj["class_name"], (255, 255, 255))

        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        if len(track.history) >= 2:
            history = np.asarray(track.history, dtype=np.float32)
            history_pts = world_to_pixel(history, x_range, y_range, resolution)
            cv2.polylines(image, [history_pts], isClosed=False, color=color, thickness=1)

        label = f'T{track.track_id}:{obj["class_name"]}'
        cv2.putText(image, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
