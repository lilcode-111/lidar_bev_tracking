import unittest

from bev_tracking.relational_support_phase4 import _top_shares, _percentiles


class RelationalSupportPhase4Test(unittest.TestCase):
    def test_top_shares_use_explicit_population_denominator(self):
        result = _top_shares([4, 3, 2, 1, 0])
        self.assertEqual(result["support_count"], 10)
        self.assertEqual(result["nonzero_unit_count"], 4)
        self.assertAlmostEqual(result["Top1_share"], 0.4)
        self.assertAlmostEqual(result["Top3_share"], 0.9)
        self.assertAlmostEqual(result["Top5_share"], 1.0)

    def test_empty_concentration_is_not_fabricated(self):
        result = _top_shares([])
        self.assertEqual(result["support_count"], 0)
        self.assertIsNone(result["Top1_share"])

    def test_local_size_percentiles_are_mechanical(self):
        result = _percentiles([0, 1, 2, 3])
        self.assertEqual(result["valid_N"], 4)
        self.assertAlmostEqual(result["P50"], 1.5)


if __name__ == "__main__":
    unittest.main()
