import unittest

from bev_tracking.candidate_conversion import build_downstream_attribution, trace_nms_suppression


def box(box_id, x, score, class_name="car"):
    return {
        "id": box_id,
        "class_name": class_name,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "yaw": 0.0,
        "score": score,
        "det_index": 0 if box_id == "det_a" else 1,
    }


class CandidateConversionDay4Test(unittest.TestCase):
    def test_nms_trace_identifies_suppressor(self):
        boxes = [box("det_a", 10.0, 0.9), box("det_b", 10.1, 0.8)]
        trace = trace_nms_suppression(boxes, [boxes[0]], iou_threshold=0.3)
        self.assertEqual(trace["kept_ids"], ["det_a"])
        self.assertEqual(trace["suppressed"][0]["suppressed_id"], "det_b")
        self.assertEqual(trace["suppressed"][0]["suppressor_id"], "det_a")

    def test_different_class_is_not_suppressed(self):
        boxes = [box("det_a", 10.0, 0.9), box("det_b", 10.1, 0.8, "pedestrian")]
        trace = trace_nms_suppression(boxes, boxes, iou_threshold=0.3)
        self.assertEqual(trace["suppressed"], [])

    def test_geometry_and_evaluation_competition_are_recorded(self):
        gt = {"id": "gt_1", "x": 10.0, "y": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0}
        det = box("det_a", 10.0, 0.9)
        attribution = build_downstream_attribution(
            gt_box=gt,
            associated_before=[det],
            associated_after=[det],
            nms_trace={"suppressed": []},
            evaluation={"matches": []},
            matched_ids=set(),
        )
        self.assertEqual(attribution["geometry"]["best_iou_candidate"]["iou"], 1.0)
        self.assertTrue(attribution["evaluation"]["iou_ge_0_50_but_unmatched"])


if __name__ == "__main__":
    unittest.main()
