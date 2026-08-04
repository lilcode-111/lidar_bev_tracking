import unittest

import numpy as np

from bev_tracking.adaptive_clustering import cluster_points, clusters_equal
from bev_tracking.clustering_detector import detect_objects_from_points, euclidean_cluster
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.synthetic import generate_frame


class ClusteringEntryTest(unittest.TestCase):
    def test_fixed_unified_entry_matches_legacy_clusters_exactly(self):
        points, _ = generate_frame(seed=7)
        obstacle_points = points[(points[:, 0] >= 0.0) & (points[:, 0] < 40.0)]
        policy = ClusteringPolicy(mode="fixed", eps=0.6, min_points=20)

        legacy = euclidean_cluster(obstacle_points, eps=0.6, min_points=20)
        unified = cluster_points(obstacle_points, policy)

        self.assertTrue(clusters_equal(legacy, unified))

    def test_detector_default_path_remains_legacy(self):
        points, _ = generate_frame(seed=7)
        detections, trace = detect_objects_from_points(points, oriented=True, return_trace=True)
        self.assertEqual(trace["parameters"]["clustering_mode"], "legacy_fixed")
        self.assertEqual(trace["cluster_count"], len(detections))

    def test_detector_fixed_policy_path_matches_default_output(self):
        points, _ = generate_frame(seed=7)
        policy = ClusteringPolicy(mode="fixed", eps=0.6, min_points=20)
        default = detect_objects_from_points(points, oriented=True)
        unified = detect_objects_from_points(points, oriented=True, clustering_policy=policy)
        self.assertEqual(default, unified)

    def test_adaptive_entry_accepts_brute_force_and_grid(self):
        points = np.asarray(
            [[1.0, 1.0], [1.2, 1.0], [15.0, 0.0], [15.5, 0.0], [30.0, 0.0], [30.7, 0.0]],
            dtype=np.float32,
        )
        policy = ClusteringPolicy(
            mode="adaptive",
            distance_params={
                "near_0_15": {"eps": 0.5, "min_points": 2},
                "mid_15_30": {"eps": 0.65, "min_points": 2},
                "far_30_inf": {"eps": 0.85, "min_points": 2},
            },
        )
        brute_force = cluster_points(points, policy, implementation="brute_force")
        grid = cluster_points(points, policy, implementation="grid")
        self.assertTrue(clusters_equal(brute_force, grid))


if __name__ == "__main__":
    unittest.main()
