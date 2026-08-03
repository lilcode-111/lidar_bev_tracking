"""Reference and grid neighbor implementations for v15.2 clustering.

The brute-force implementation is the correctness oracle. The grid
implementation may be used for acceleration only after its output is proven
identical to the oracle.
"""

from collections import defaultdict
import math

import numpy as np

from bev_tracking.clustering_policy import ClusteringPolicy, pairwise_eps


DISTANCE_TOLERANCE_M = 1e-6


def point_parameters(points, policy):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape N x 2 or N x greater")

    parameters = []
    for point in points:
        range_xy = float(np.hypot(float(point[0]), float(point[1])))
        parameters.append(policy.params_for_range(range_xy))
    return parameters


def _validate_inputs(points, parameters, global_max_eps):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape N x 2 or N x greater")
    if len(parameters) != len(points):
        raise ValueError("one clustering parameter pair is required per point")
    if float(global_max_eps) <= 0 or not math.isfinite(float(global_max_eps)):
        raise ValueError("global_max_eps must be a positive finite number")
    return points[:, :2].astype(np.float64, copy=False)


def _within_pairwise_radius(points_xy, point_idx, other_idx, parameters, tolerance):
    diff = points_xy[other_idx] - points_xy[point_idx]
    distance = float(np.hypot(diff[0], diff[1]))
    radius = pairwise_eps(parameters[point_idx]["eps"], parameters[other_idx]["eps"])
    return distance <= radius + float(tolerance)


def brute_force_neighbors(points, parameters, tolerance=DISTANCE_TOLERANCE_M):
    """Return exact symmetric neighbors, including each query point once."""
    points_xy = _validate_inputs(points, parameters, max(item["eps"] for item in parameters) if parameters else 1.0)
    neighbors = []
    for point_idx in range(len(points_xy)):
        current = [
            other_idx
            for other_idx in range(len(points_xy))
            if _within_pairwise_radius(points_xy, point_idx, other_idx, parameters, tolerance)
        ]
        neighbors.append(current)
    return neighbors


def adaptive_grid_neighbors(
    points,
    parameters,
    global_max_eps,
    tolerance=DISTANCE_TOLERANCE_M,
):
    """Return the same neighbors as brute force using a bounded grid search."""
    points_xy = _validate_inputs(points, parameters, global_max_eps)
    cell_size = float(global_max_eps)
    coords = np.floor(points_xy / cell_size).astype(np.int64)
    grid = defaultdict(list)
    for point_idx, coord in enumerate(coords):
        grid[(int(coord[0]), int(coord[1]))].append(point_idx)

    neighbors = []
    for point_idx, coord in enumerate(coords):
        candidates = set()
        cx, cy = int(coord[0]), int(coord[1])
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                candidates.update(grid.get((gx, gy), ()))

        current = [
            other_idx
            for other_idx in sorted(candidates)
            if _within_pairwise_radius(points_xy, point_idx, other_idx, parameters, tolerance)
        ]
        if point_idx not in current:
            raise AssertionError("grid neighbor query omitted its query point")
        neighbors.append(current)
    return neighbors


def core_mask(neighbors, parameters):
    if len(neighbors) != len(parameters):
        raise ValueError("one neighbor list is required per point")
    return [len(items) >= int(parameters[index]["min_points"]) for index, items in enumerate(neighbors)]


def cluster_from_neighbors(points, neighbors, parameters):
    """Build deterministic connected components from precomputed neighbors."""
    points = np.asarray(points)
    if len(points) != len(neighbors) or len(parameters) != len(points):
        raise ValueError("points, neighbors, and parameters must have equal length")

    core = core_mask(neighbors, parameters)
    visited = [False] * len(points)
    assigned = [False] * len(points)
    clusters = []

    for start_idx in range(len(points)):
        if visited[start_idx]:
            continue
        visited[start_idx] = True
        if not core[start_idx]:
            continue

        queue = [start_idx]
        component = []
        while queue:
            current = queue.pop(0)
            if not assigned[current]:
                assigned[current] = True
                component.append(current)
            if not core[current]:
                continue
            for neighbor_idx in neighbors[current]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    if core[neighbor_idx]:
                        queue.append(neighbor_idx)
                if not assigned[neighbor_idx]:
                    assigned[neighbor_idx] = True
                    component.append(neighbor_idx)

        clusters.append(points[sorted(component)])

    return clusters
