import copy
import unittest

from bev_tracking.v15_4_formal import validate_both_t0_replays
from bev_tracking.v15_4_materialization import V154MaterializationError


class V154Phase2Day2Test(unittest.TestCase):
    def test_both_t0_replays_must_match(self):
        reference_25 = {"schema_version": "25", "records": [{"iou": 0.2}]}
        reference_100 = {"schema_version": "100", "tp": 4}
        result = validate_both_t0_replays(reference_25, copy.deepcopy(reference_25), reference_100, copy.deepcopy(reference_100))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["non_t0_interpretation_allowed"])

    def test_25_mismatch_blocks_gate(self):
        with self.assertRaises(V154MaterializationError):
            validate_both_t0_replays({"value": 1}, {"value": 2}, {"value": 1}, {"value": 1})

    def test_100_mismatch_blocks_gate(self):
        with self.assertRaises(V154MaterializationError):
            validate_both_t0_replays({"value": 1}, {"value": 1}, {"value": 1}, {"value": 2})


if __name__ == "__main__":
    unittest.main()
