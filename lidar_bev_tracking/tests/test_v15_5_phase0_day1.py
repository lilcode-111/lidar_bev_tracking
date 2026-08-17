import unittest

import numpy as np

from bev_tracking.v15_5_seed_support import (
    V155SeedSupportError,
    build_band_identity,
    compute_seed_metrics,
    range_segment,
    select_identity_metrics,
    update_intensity_audit,
    validate_band_partition,
)


class V155Phase0Day1Test(unittest.TestCase):
    def test_band_identity_uses_source_index_not_coordinate(self):
        result = build_band_identity(
            np.asarray([10, 11, 12], dtype=np.int64),
            np.asarray([10], dtype=np.int64),
        )
        self.assertEqual(result.tolist(), [11, 12])

    def test_adjacent_bands_exactly_partition_toff_minus_t0(self):
        frame_sets = {
            "T0": {"global": np.asarray([1])},
            "T1": {"global": np.asarray([1, 2])},
            "T2": {"global": np.asarray([1, 2, 3])},
            "T_off": {"global": np.asarray([1, 2, 3, 4])},
        }
        bands = validate_band_partition(frame_sets, category="global")
        self.assertEqual(bands["H"].tolist(), [2])
        self.assertEqual(bands["M"].tolist(), [3])
        self.assertEqual(bands["L"].tolist(), [4])

    def test_seed_distance_and_radius_count_are_exact(self):
        points = np.asarray([
            [0.0, 0.0, 0.0, 0.50],
            [1.0, 0.0, 0.0, 0.45],
            [0.3, 0.0, 0.0, 0.20],
            [0.5, 0.0, 0.0, 0.20],
            [0.2, 0.0, 0.0, 0.20],
            [0.2, 0.0, 0.0, 0.20],
        ], dtype=np.float32)
        metrics = compute_seed_metrics(points, np.asarray([2, 3, 4, 5]), points[:2, :2])
        np.testing.assert_allclose(metrics["d_seed"], [0.3, 0.5, 0.2, 0.2], atol=1e-7)
        self.assertEqual(metrics["n_seed_0p6"].tolist(), [1, 2, 1, 1])
        self.assertEqual(metrics["source_point_indices"].tolist(), [2, 3, 4, 5])

    def test_category_selection_occurs_after_global_support(self):
        points = np.asarray([
            [0.0, 0.0, 0.0, 0.50],
            [0.2, 0.0, 0.0, 0.20],
            [0.8, 0.0, 0.0, 0.20],
        ], dtype=np.float32)
        global_indices = np.asarray([1, 2], dtype=np.int64)
        global_metrics = compute_seed_metrics(points, global_indices, points[:1, :2])
        vehicle = select_identity_metrics(global_metrics, global_indices, np.asarray([1]))
        self.assertEqual(vehicle["source_point_indices"].tolist(), [1])
        self.assertAlmostEqual(float(vehicle["d_seed"][0]), 0.2, places=6)

    def test_no_seed_is_not_misreported_as_supported(self):
        points = np.asarray([[1.0, 0.0, 0.0, 0.20]], dtype=np.float32)
        metrics = compute_seed_metrics(points, np.asarray([0]), np.empty((0, 2)))
        self.assertTrue(np.isinf(metrics["d_seed"][0]))
        self.assertEqual(int(metrics["n_seed_0p6"][0]), 0)

    def test_intensity_intervals_are_half_open(self):
        audit = {"min": None, "max": None, "violation_count": 0}
        contract = {"intensity_min": 0.30, "intensity_max_exclusive": 0.38}
        update_intensity_audit(audit, np.asarray([0.30, 0.379], dtype=np.float32), contract)
        self.assertEqual(audit["violation_count"], 0)
        update_intensity_audit(audit, np.asarray([0.38], dtype=np.float32), contract)
        self.assertEqual(audit["violation_count"], 1)

    def test_range_boundaries_are_frozen(self):
        self.assertEqual(range_segment(14.999), "near")
        self.assertEqual(range_segment(15.0), "mid")
        self.assertEqual(range_segment(29.999), "mid")
        self.assertEqual(range_segment(30.0), "far")
        with self.assertRaises(V155SeedSupportError):
            range_segment(-0.1)


if __name__ == "__main__":
    unittest.main()
