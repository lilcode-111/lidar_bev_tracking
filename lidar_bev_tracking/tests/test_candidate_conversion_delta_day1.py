import unittest

from bev_tracking.candidate_conversion import build_candidate_conversion_delta


def record(gt_id, clusters, terminal, car_before=None):
    return {
        "frame_id": "000001",
        "gt_id": gt_id,
        "cluster_ids": clusters,
        "terminal_state": terminal,
        "car_detection_ids_before_nms": car_before or [],
        "car_detection_ids_after_nms": [],
        "best_iou_after_nms": 0.0,
        "matched_by_iou": {"0.50": False, "0.25": False},
    }


def report(records):
    return [{"frame_id": "000001", "candidate_conversion": {"evidence": records}}]


class CandidateConversionDeltaDay1Test(unittest.TestCase):
    def test_new_c1_associated_gt_are_counted_by_terminal_state(self):
        c0 = report([
            record("gt_1", [], "no_associated_cluster"),
            record("gt_2", [], "no_associated_cluster"),
            record("gt_3", ["cluster_3"], "rejected_by_car_classifier"),
        ])
        c1 = report([
            record("gt_1", ["cluster_1"], "rejected_by_car_classifier"),
            record("gt_2", ["cluster_2"], "box_iou_0_25_to_0_50", ["det_2"]),
            record("gt_3", ["cluster_3"], "rejected_by_car_classifier"),
        ])
        result = build_candidate_conversion_delta({"C0": c0, "C1": c1})
        self.assertEqual(result["new_associated_gt_count"], 2)
        self.assertEqual(result["terminal_state_counts"], {
            "box_iou_0_25_to_0_50": 1,
            "rejected_by_car_classifier": 1,
        })
        self.assertEqual(result["conversion_stage_counts"]["rejected_by_car_classifier"], 1)
        self.assertEqual(result["conversion_stage_counts"]["car_classification_pass"], 1)
        self.assertEqual([item["gt_id"] for item in result["records"]], ["gt_1", "gt_2"])

    def test_delta_rejects_duplicate_or_mismatched_gt_sets(self):
        duplicate = report([record("gt_1", [], "no_associated_cluster"), record("gt_1", [], "no_associated_cluster")])
        with self.assertRaises(ValueError):
            build_candidate_conversion_delta({"C0": duplicate, "C1": report([record("gt_1", [], "no_associated_cluster")])})

        with self.assertRaises(ValueError):
            build_candidate_conversion_delta({"C0": report([record("gt_1", [], "no_associated_cluster")]), "C1": report([record("gt_2", [], "no_associated_cluster")])})


if __name__ == "__main__":
    unittest.main()
