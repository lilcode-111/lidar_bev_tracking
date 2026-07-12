import tempfile
import unittest
from pathlib import Path

from bev_tracking.evaluation import evaluate_detections
from bev_tracking.kitti import load_kitti_labels, parse_kitti_label_line
from bev_tracking.nms import nms_bev


def box(box_id, class_name="car", x=10.0, y=0.0, score=0.9, det_index=0):
    return {
        "id": box_id,
        "class_name": class_name,
        "x": x,
        "y": y,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "yaw": 0.0,
        "score": score,
        "det_index": det_index,
    }


class KittiParserPolicyTest(unittest.TestCase):
    def test_parser_preserves_kitti_policy_classes(self):
        classes = ["Car", "Van", "Truck", "DontCare", "Tram", "Misc"]
        for class_name in classes:
            line = f"{class_name} 0.00 0 0.00 0 0 50 50 1.50 1.80 4.00 1.00 1.50 15.00 0.00"
            label = parse_kitti_label_line(line)
            self.assertIsNotNone(label)
            self.assertEqual(label["class_name"], class_name)

    def test_missing_and_empty_label_files_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.txt"
            with self.assertRaises(FileNotFoundError):
                load_kitti_labels(missing_path)

            empty_path = Path(tmp) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")
            self.assertEqual(load_kitti_labels(empty_path), [])


class EvaluationPolicyTest(unittest.TestCase):
    def test_neutral_vehicle_absorbs_one_detection_only(self):
        detections = [
            box("det_1", class_name="car", score=0.9, det_index=0),
            box("det_2", class_name="car", score=0.8, det_index=1),
        ]
        gt_boxes = [box("gt_van", class_name="van")]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertEqual(result["metrics"]["tp"], 0)
        self.assertEqual(result["metrics"]["fp"], 1)
        self.assertEqual(result["metrics"]["fn"], 0)
        self.assertEqual(len(result["neutralized_detections"]), 1)
        self.assertEqual(result["false_positives"][0]["det_id"], "det_2")

    def test_non_car_detection_does_not_enter_car_evaluation(self):
        detections = [box("det_ped", class_name="pedestrian")]
        gt_boxes = [box("gt_car", class_name="car")]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertEqual(result["metrics"]["tp"], 0)
        self.assertEqual(result["metrics"]["fp"], 0)
        self.assertEqual(result["metrics"]["fn"], 1)
        self.assertEqual(len(result["ignored"]["detections"]), 1)

    def test_primary_and_auxiliary_iou_are_matched_independently(self):
        detections = [box("det_car", class_name="car", x=12.0)]
        gt_boxes = [box("gt_car", class_name="car", x=10.0)]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[0.25])

        self.assertEqual(result["metrics"]["tp"], 0)
        self.assertEqual(result["metrics"]["fp"], 1)
        self.assertEqual(result["metrics"]["fn"], 1)
        aux = result["auxiliary"]["0.25"]
        self.assertEqual(aux["metrics"]["tp"], 1)
        self.assertEqual(aux["metrics"]["fp"], 0)
        self.assertEqual(aux["metrics"]["fn"], 0)

    def test_zero_denominator_metrics_are_undefined(self):
        result = evaluate_detections([], [], iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertIsNone(result["metrics"]["precision"])
        self.assertIsNone(result["metrics"]["recall"])

    def test_det_index_makes_matching_order_deterministic(self):
        detections_a = [
            box("det_fp", class_name="car", x=30.0, score=0.9, det_index=1),
            box("det_tp", class_name="car", x=10.0, score=0.9, det_index=0),
        ]
        detections_b = list(reversed(detections_a))
        gt_boxes = [box("gt_car", class_name="car", x=10.0)]

        result_a = evaluate_detections(detections_a, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])
        result_b = evaluate_detections(detections_b, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertEqual(result_a["matches"], result_b["matches"])
        self.assertEqual(result_a["false_positives"], result_b["false_positives"])

    def test_nms_uses_stable_det_index_tie_break(self):
        detections = [
            box("det_late", score=0.9, det_index=1),
            box("det_early", score=0.9, det_index=0),
        ]

        kept = nms_bev(detections, iou_threshold=0.3)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], "det_early")


if __name__ == "__main__":
    unittest.main()
