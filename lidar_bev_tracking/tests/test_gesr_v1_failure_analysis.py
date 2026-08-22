import unittest

import numpy as np

from bev_tracking.gesr_v1 import CandidateDecision, GESRReferenceResult
from bev_tracking.gesr_v1_failure_analysis import (
    GESRFailureAnalysisError,
    analyze_gt_failure,
    compact_oracle,
    select_t2_material_recovery_targets,
)


class GESRV1FailureAnalysisTest(unittest.TestCase):
    def test_selects_only_t2_material_recovery_targets_after_valid_gate0(self):
        payload = {
            "Gate0_Phase2": {"result": "PASS"},
            "Gate_A": {
                "delta22": [
                    {
                        "frame_id": "1",
                        "gt_id": "gt_1",
                        "IoU_T0": 0.1,
                        "IoU_T2": 0.3,
                        "IoU_GESR": 0.12,
                        "T2_minus_T0": 0.2,
                        "GESR_minus_T0": 0.02,
                    },
                    {
                        "frame_id": "2",
                        "gt_id": "gt_2",
                        "IoU_T0": 0.1,
                        "IoU_T2": 0.15,
                        "IoU_GESR": 0.11,
                        "T2_minus_T0": 0.05,
                        "GESR_minus_T0": 0.01,
                    },
                ]
            },
        }
        targets = select_t2_material_recovery_targets(payload)
        self.assertEqual([(item["frame_id"], item["gt_id"]) for item in targets], [("000001", "gt_1")])

        payload["Gate0_Phase2"]["result"] = "FAIL"
        with self.assertRaises(GESRFailureAnalysisError):
            select_t2_material_recovery_targets(payload)

    def test_gt_analysis_partitions_t2_added_points_and_replays_pca(self):
        points = np.asarray(
            [
                [-1.0, -1.0, 0.0, 0.50],
                [-1.0, 1.0, 0.0, 0.50],
                [1.0, -1.0, 0.0, 0.50],
                [1.0, 1.0, 0.0, 0.50],
                [2.0, 0.0, 0.0, 0.20],
                [0.0, 2.0, 0.0, 0.20],
                [-2.0, 0.0, 0.0, 0.20],
            ],
            dtype=np.float64,
        )
        gt_box = {
            "id": "gt_1",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "length": 6.0,
            "width": 6.0,
            "height": 4.0,
            "yaw": 0.0,
        }
        t0 = np.asarray([0, 1, 2, 3], dtype=np.int64)
        t2 = np.arange(7, dtype=np.int64)
        gesr_indices = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        target = {
            "frame_id": "000001",
            "gt_id": "gt_1",
            "formal_iou": {
                "T0": compact_oracle(points, t0, gt_box)["iou"],
                "T2": compact_oracle(points, t2, gt_box)["iou"],
                "GESR-v1": compact_oracle(points, gesr_indices, gt_box)["iou"],
            },
            "formal_gain": {"T2_minus_T0": 0.2, "GESR_minus_T0": 0.01},
        }
        decisions = (
            CandidateDecision(4, True, 0, "component", "ACCEPTED", ()),
            CandidateDecision(5, False, None, None, "INSIDE_CURRENT_EXTENT", ()),
            CandidateDecision(6, False, None, None, "NO_VALID_COMPONENT", ()),
        )
        result = GESRReferenceResult(
            frame_id="000001",
            seed_source_indices=(0, 1, 2, 3),
            candidate_source_indices=(4, 5, 6),
            discarded_source_indices=(),
            accepted_source_indices=(4,),
            expanded_source_indices=(0, 1, 2, 3, 4),
            components=(),
            candidate_decisions=decisions,
        )

        record = analyze_gt_failure(
            points=points,
            gt_box=gt_box,
            t0_source_indices=t0,
            t2_source_indices=t2,
            gesr_result=result,
            target=target,
        )

        self.assertEqual(record["point_counts"]["T2_added_GT"], 3)
        self.assertEqual(record["point_counts"]["GESR_accepted_from_T2_added"], 1)
        self.assertEqual(record["point_counts"]["GESR_missed_from_T2_added"], 2)
        self.assertEqual(record["missed_terminal_reason_counts"]["INSIDE_CURRENT_EXTENT"], 1)
        self.assertEqual(record["missed_terminal_reason_counts"]["NO_VALID_COMPONENT"], 1)
        self.assertEqual(len(record["top_missed_single_point_marginals"]), 2)
        self.assertEqual(
            record["representations"]["T2"]["iou"],
            compact_oracle(points, t2, gt_box)["iou"],
        )


if __name__ == "__main__":
    unittest.main()
