from collections import deque

import numpy as np

from bev_tracking.oriented_box import estimate_oriented_box_xy


DEFAULT_Z_MIN = -0.9
DEFAULT_INTENSITY_MIN = 0.38


def split_obstacle_filter_stages(
    points,
    x_range=(0.0, 40.0),
    y_range=(-20.0, 20.0),
    z_min=DEFAULT_Z_MIN,
    intensity_min=DEFAULT_INTENSITY_MIN,
):
    roi_mask = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] < x_range[1])
        & (points[:, 1] >= y_range[0])
        & (points[:, 1] < y_range[1])
    )
    roi_points = points[roi_mask]
    z_filtered_points = roi_points[roi_points[:, 2] >= z_min]
    intensity_filtered_points = z_filtered_points[z_filtered_points[:, 3] >= intensity_min]
    return {
        "raw": points,
        "roi": roi_points,
        "z_filter": z_filtered_points,
        "intensity_filter": intensity_filtered_points,
    }


def filter_obstacle_points(
    points,
    x_range=(0.0, 40.0),
    y_range=(-20.0, 20.0),
    z_min=DEFAULT_Z_MIN,
    intensity_min=DEFAULT_INTENSITY_MIN,
):
    stages = split_obstacle_filter_stages(
        points,
        x_range=x_range,
        y_range=y_range,
        z_min=z_min,
        intensity_min=intensity_min,
    )
    return stages["intensity_filter"]


def _build_grid(points_xy, cell_size):
    grid = {}
    coords = np.floor(points_xy / cell_size).astype(np.int32)
    for idx, coord in enumerate(coords):
        key = (int(coord[0]), int(coord[1]))
        grid.setdefault(key, []).append(idx)
    return grid, coords


def _region_query(point_idx, points_xy, grid, coords, eps):
    cx, cy = coords[point_idx]
    neighbors = []
    eps2 = eps * eps

    for gx in range(cx - 1, cx + 2):
        for gy in range(cy - 1, cy + 2):
            for other_idx in grid.get((gx, gy), []):
                diff = points_xy[other_idx] - points_xy[point_idx]
                if float(diff @ diff) <= eps2:
                    neighbors.append(other_idx)

    return neighbors


def euclidean_cluster(points, eps=0.6, min_points=20):
    if len(points) == 0:
        return []

    points_xy = points[:, :2]
    grid, coords = _build_grid(points_xy, eps)
    visited = np.zeros(len(points), dtype=bool)
    clustered = np.zeros(len(points), dtype=bool)
    clusters = []

    for start_idx in range(len(points)):
        if visited[start_idx]:
            continue

        visited[start_idx] = True
        neighbors = _region_query(start_idx, points_xy, grid, coords, eps)
        if len(neighbors) < min_points:
            continue

        cluster_indices = []
        queue = deque(neighbors)
        clustered[start_idx] = True
        cluster_indices.append(start_idx)

        while queue:
            idx = queue.popleft()

            if not visited[idx]:
                visited[idx] = True
                idx_neighbors = _region_query(idx, points_xy, grid, coords, eps)
                if len(idx_neighbors) >= min_points:
                    queue.extend(idx_neighbors)

            if not clustered[idx]:
                clustered[idx] = True
                cluster_indices.append(idx)

        clusters.append(points[cluster_indices])

    return clusters


def classify_cluster(length, width, num_points):
    max_dim = max(length, width)
    min_dim = min(length, width)

    if max_dim >= 2.0 or num_points >= 500:
        return "car"
    if max_dim >= 0.65 or min_dim >= 0.55:
        return "pedestrian"
    return "cone"


def cluster_to_box(cluster, det_id):
    x_min, y_min = cluster[:, 0].min(), cluster[:, 1].min()
    x_max, y_max = cluster[:, 0].max(), cluster[:, 1].max()
    z_min, z_max = cluster[:, 2].min(), cluster[:, 2].max()

    length = float(max(x_max - x_min, 0.1))
    width = float(max(y_max - y_min, 0.1))
    num_points = int(len(cluster))
    class_name = classify_cluster(length, width, num_points)
    score = min(0.99, 0.45 + 0.001 * num_points)

    return {
        "id": f"cluster_{det_id}",
        "class_name": class_name,
        "x": float((x_min + x_max) / 2.0),
        "y": float((y_min + y_max) / 2.0),
        "z": float((z_min + z_max) / 2.0),
        "length": length,
        "width": width,
        "yaw": 0.0,
        "score": float(score),
        "num_points": num_points,
        "det_index": int(det_id - 1),
    }


def cluster_to_oriented_box(cluster, det_id):
    x_min, y_min = cluster[:, 0].min(), cluster[:, 1].min()
    x_max, y_max = cluster[:, 0].max(), cluster[:, 1].max()
    z_min, z_max = cluster[:, 2].min(), cluster[:, 2].max()
    x, y, length, width, yaw = estimate_oriented_box_xy(cluster)
    num_points = int(len(cluster))
    axis_length = float(max(x_max - x_min, 0.1))
    axis_width = float(max(y_max - y_min, 0.1))
    class_name = classify_cluster(axis_length, axis_width, num_points)
    score = min(0.99, 0.45 + 0.001 * num_points)

    return {
        "id": f"cluster_{det_id}",
        "class_name": class_name,
        "x": x,
        "y": y,
        "z": float((z_min + z_max) / 2.0),
        "length": length,
        "width": width,
        "yaw": yaw,
        "score": float(score),
        "num_points": num_points,
        "box_type": "oriented_pca",
        "det_index": int(det_id - 1),
    }


def detect_objects_from_points(
    points,
    eps=0.6,
    min_points=20,
    oriented=False,
    z_min=DEFAULT_Z_MIN,
    intensity_min=DEFAULT_INTENSITY_MIN,
    return_trace=False,
    clustering_policy=None,
):
    stages = split_obstacle_filter_stages(
        points,
        z_min=z_min,
        intensity_min=intensity_min,
    )
    obstacle_points = stages["intensity_filter"]
    if clustering_policy is None:
        clusters = euclidean_cluster(obstacle_points, eps=eps, min_points=min_points)
        clustering_mode = "legacy_fixed"
    else:
        from bev_tracking.adaptive_clustering import cluster_points

        clusters = cluster_points(obstacle_points, clustering_policy)
        clustering_mode = clustering_policy.mode
    box_fn = cluster_to_oriented_box if oriented else cluster_to_box
    detections = [box_fn(cluster, idx + 1) for idx, cluster in enumerate(clusters)]
    if not return_trace:
        return detections

    trace = {
        "point_counts": {stage_name: int(len(stage_points)) for stage_name, stage_points in stages.items()},
        "cluster_count": int(len(clusters)),
        "cluster_point_counts": [int(len(cluster)) for cluster in clusters],
        "parameters": {
            "eps": float(eps),
            "min_points": int(min_points),
            "oriented": bool(oriented),
            "z_min": float(z_min),
            "intensity_min": float(intensity_min),
            "clustering_mode": clustering_mode,
        },
    }
    return detections, trace
