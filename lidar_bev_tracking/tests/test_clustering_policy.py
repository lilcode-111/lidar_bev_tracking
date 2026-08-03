import unittest

from bev_tracking.clustering_policy import (
    ClusteringPolicy,
    GLOBAL_MAX_EPS,
    clustering_policy_from_config,
    distance_bin_name,
    pairwise_eps,
)


class ClusteringPolicyTest(unittest.TestCase):
    def test_distance_bins_use_left_closed_right_open_ranges(self):
        self.assertEqual(distance_bin_name(0.0), "near_0_15")
        self.assertEqual(distance_bin_name(14.999), "near_0_15")
        self.assertEqual(distance_bin_name(15.0), "mid_15_30")
        self.assertEqual(distance_bin_name(30.0), "far_30_inf")

    def test_pairwise_eps_is_symmetric_max_rule(self):
        self.assertEqual(pairwise_eps(0.55, 0.8), 0.8)
        self.assertEqual(pairwise_eps(0.8, 0.55), 0.8)

    def test_fixed_policy_reads_legacy_detector_values(self):
        policy = clustering_policy_from_config({"detector": {"eps": 0.6, "min_points": 20}})
        self.assertEqual(policy.mode, "fixed")
        self.assertEqual(policy.params_for_range(29.0), {"eps": 0.6, "min_points": 20})

    def test_adaptive_policy_requires_all_distance_bins(self):
        with self.assertRaises(ValueError):
            ClusteringPolicy(
                mode="adaptive",
                distance_params={"near_0_15": {"eps": 0.5, "min_points": 20}},
            )

    def test_adaptive_eps_cannot_exceed_global_bound(self):
        params = {
            "near_0_15": {"eps": GLOBAL_MAX_EPS, "min_points": 20},
            "mid_15_30": {"eps": 0.7, "min_points": 15},
            "far_30_inf": {"eps": 0.8, "min_points": 8},
        }
        policy = ClusteringPolicy(mode="adaptive", distance_params=params)
        self.assertEqual(policy.params_for_range(40.0)["eps"], 0.8)


if __name__ == "__main__":
    unittest.main()
