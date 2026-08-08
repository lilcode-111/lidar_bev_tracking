import unittest

import numpy as np

from bev_tracking.failure_evidence import build_failure_evidence_report


def car_box():
    return {
        "id": "gt_1",
        "class_name": "car",
        "x": 10.0,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": 0.0,
    }


class CandidateConversionDay5Test(unittest.TestCase):
    def test_existing_replay_can_emit_candidate_conversion_report(self):
        rng = np.random.default_rng(7)
        points = np.column_stack([
            rng.uniform(9.0, 11.0, 80),
            rng.uniform(-0.5, 0.5, 80),
            rng.uniform(-0.3, 0.3, 80),
            np.ones(80),
        ]).astype(np.float32)
        report = build_failure_evidence_report(
            points,
            [car_box()],
            frame_id="42",
            min_points=20,
            oriented=False,
            candidate_variant="C0",
        )
        conversion = report["candidate_conversion"]
        self.assertEqual(conversion["variant"], "C0")
        self.assertEqual(conversion["num_positive_gt"], 1)
        self.assertEqual(len(conversion["evidence"]), 1)
        self.assertIn("downstream_attribution", conversion["evidence"][0])

    def test_legacy_call_keeps_optional_conversion_absent(self):
        report = build_failure_evidence_report(
            np.empty((0, 4), dtype=np.float32),
            [car_box()],
            frame_id="42",
            oriented=False,
        )
        self.assertIsNone(report["candidate_conversion"])

    def test_candidate_conversion_uses_evaluation_roi_policy(self):
        points = np.ones((30, 4), dtype=np.float32)
        points[:, 0] = 10.0
        points[:, 1] = 0.0
        report = build_failure_evidence_report(
            points,
            [car_box(), {**car_box(), "id": "gt_outside", "x": 45.0}],
            frame_id="42",
            min_points=20,
            oriented=False,
            candidate_variant="C0",
        )
        conversion = report["candidate_conversion"]
        self.assertEqual(conversion["num_positive_gt"], 1)
        self.assertEqual([item["gt_id"] for item in conversion["evidence"]], ["gt_1"])


if __name__ == "__main__":
    unittest.main()
