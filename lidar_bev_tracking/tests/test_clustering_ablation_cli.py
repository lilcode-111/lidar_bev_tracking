import tempfile
import unittest
from pathlib import Path

from bev_tracking.adaptive_experiment import variant_specs_from_config


class ClusteringAblationConfigTest(unittest.TestCase):
    def test_config_parser_requires_all_four_variants(self):
        with self.assertRaises(ValueError):
            variant_specs_from_config({"variants": {"C0": {"mode": "fixed"}}})

    def test_config_parser_keeps_adaptive_distance_params(self):
        adaptive = {
            "mode": "adaptive",
            "distance_params": {
                "near_0_15": {"eps": 0.5, "min_points": 20},
                "mid_15_30": {"eps": 0.65, "min_points": 15},
                "far_30_inf": {"eps": 0.85, "min_points": 8},
            },
        }
        config = {
            "variants": {
                "C0": {"mode": "fixed", "eps": 0.6, "min_points": 20},
                "C1": adaptive,
                "C2": adaptive,
                "C3": adaptive,
            }
        }
        specs = variant_specs_from_config(config)
        self.assertEqual(specs["C2"].policy.distance_params["far_30_inf"]["eps"], 0.85)


if __name__ == "__main__":
    unittest.main()
