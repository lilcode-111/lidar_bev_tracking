import unittest

import numpy as np

from bev_tracking.adaptive_clustering import (
    DISTANCE_TOLERANCE_M,
    adaptive_grid_neighbors,
    brute_force_neighbors,
    cluster_from_neighbors,
    core_mask,
    point_parameters,
)
from bev_tracking.clustering_policy import ClusteringPolicy, GLOBAL_MAX_EPS, pairwise_eps


def adaptive_policy():
    return ClusteringPolicy(
        mode="adaptive",
        global_max_eps=GLOBAL_MAX_EPS,
        distance_params={
            "near_0_15": {"eps": 0.50, "min_points": 3},
            "mid_15_30": {"eps": 0.65, "min_points": 3},
            "far_30_inf": {"eps": 0.85, "min_points": 2},
        },
    )


class AdaptiveNeighborTest(unittest.TestCase):
    def test_grid_matches_brute_force_exactly(self):
        points = np.asarray(
            [
                [1.0, 1.0],
                [1.49, 1.0],
                [15.0, 0.0],
                [15.64, 0.0],
                [30.0, 0.0],
                [30.84, 0.0],
                [50.0, 50.0],
            ],
            dtype=np.float32,
        )
        policy = adaptive_policy()
        parameters = point_parameters(points, policy)

        expected = brute_force_neighbors(points, parameters)
        actual = adaptive_grid_neighbors(points, parameters, GLOBAL_MAX_EPS)

        self.assertEqual(actual, expected)

    def test_neighbors_are_symmetric_and_include_self_once(self):
        points = np.asarray([[1.0, 1.0], [1.4, 1.0], [10.0, 0.0]], dtype=np.float32)
        parameters = point_parameters(points, adaptive_policy())
        neighbors = adaptive_grid_neighbors(points, parameters, GLOBAL_MAX_EPS)

        for index, items in enumerate(neighbors):
            self.assertEqual(items.count(index), 1)
            for other in items:
                self.assertIn(index, neighbors[other])

    def test_pairwise_boundary_uses_max_radius_with_tolerance(self):
        self.assertEqual(pairwise_eps(0.50, 0.65), 0.65)
        points = np.asarray([[15.0, 0.0], [15.65 + DISTANCE_TOLERANCE_M / 2.0, 0.0]], dtype=np.float32)
        parameters = point_parameters(points, adaptive_policy())
        neighbors = brute_force_neighbors(points, parameters)
        self.assertIn(1, neighbors[0])

    def test_core_mask_uses_min_points_including_query_point(self):
        parameters = point_parameters(np.asarray([[1.0, 0.0], [1.1, 0.0]], dtype=np.float32), adaptive_policy())
        neighbors = [[0, 1], [0, 1]]
        self.assertEqual(core_mask(neighbors, parameters), [False, False])
        parameters = [{"eps": 0.5, "min_points": 2}] * 2
        self.assertEqual(core_mask(neighbors, parameters), [True, True])

    def test_cluster_order_and_membership_are_deterministic(self):
        points = np.asarray([[1.0, 1.0], [1.2, 1.0], [10.0, 0.0], [10.2, 0.0]], dtype=np.float32)
        parameters = [{"eps": 0.5, "min_points": 2}] * len(points)
        neighbors = brute_force_neighbors(points, parameters)
        clusters = cluster_from_neighbors(points, neighbors, parameters)
        self.assertEqual([cluster.tolist() for cluster in clusters], [points[:2].tolist(), points[2:].tolist()])


if __name__ == "__main__":
    unittest.main()
