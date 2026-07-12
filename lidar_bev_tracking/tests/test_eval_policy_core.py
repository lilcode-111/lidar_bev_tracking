import tempfile
import unittest
from pathlib import Path

from bev_tracking.evaluation import evaluate_detections
from bev_tracking.kitti import load_kitti_labels, parse_kitti_label_line
from bev_tracking.nms import nms_bev
from bev_tracking.batch_pipeline import save_frame_csv, summarize_batch_reports


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

    def test_positive_gt_has_priority_over_neutral_gt(self):
        detections = [box("det_car", class_name="car", x=10.0, score=0.9, det_index=0)]
        gt_boxes = [
            box("gt_van", class_name="van", x=10.0),
            box("gt_car", class_name="car", x=10.0),
        ]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertEqual(result["metrics"]["tp"], 1)
        self.assertEqual(result["metrics"]["fp"], 0)
        self.assertEqual(result["metrics"]["fn"], 0)
        self.assertEqual(len(result["neutralized_detections"]), 0)
        self.assertEqual(result["matches"][0]["gt_id"], "gt_car")

    def test_neutral_iou_threshold_stays_fixed_for_auxiliary_iou(self):
        detections = [box("det_car", class_name="car", x=12.0, score=0.9, det_index=0)]
        gt_boxes = [box("gt_van", class_name="van", x=10.0)]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[0.25])
        aux = result["auxiliary"]["0.25"]

        self.assertEqual(result["metrics"]["tp"], 0)
        self.assertEqual(result["metrics"]["fp"], 1)
        self.assertEqual(result["metrics"]["fn"], 0)
        self.assertEqual(len(result["neutralized_detections"]), 0)

        self.assertEqual(aux["metrics"]["tp"], 0)
        self.assertEqual(aux["metrics"]["fp"], 1)
        self.assertEqual(aux["metrics"]["fn"], 0)
        self.assertEqual(len(aux["neutralized_detections"]), 0)
    
    def test_excluded_and_dontcare_gt_do_not_create_false_negatives(self):
        detections = []
        gt_boxes = [
            box("gt_ped", class_name="pedestrian", x=10.0),
            box("gt_cyclist", class_name="cyclist", x=12.0),
            box("gt_dontcare", class_name="dontcare", x=14.0),
        ]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertEqual(result["metrics"]["tp"], 0)
        self.assertEqual(result["metrics"]["fp"], 0)
        self.assertEqual(result["metrics"]["fn"], 0)
        self.assertEqual(len(result["ignored"]["gt_boxes"]), 3)

    def test_roi_boundary_is_left_closed_right_open(self):
        detections = []
        gt_boxes = [
            box("gt_inside_left", class_name="car", x=0.0, y=0.0),
            box("gt_outside_right", class_name="car", x=40.0, y=0.0),
            box("gt_outside_y", class_name="car", x=10.0, y=20.0),
        ]

        result = evaluate_detections(detections, gt_boxes, iou_threshold=0.5, auxiliary_iou_thresholds=[])

        self.assertEqual(result["metrics"]["tp"], 0)
        self.assertEqual(result["metrics"]["fp"], 0)
        self.assertEqual(result["metrics"]["fn"], 1)
        self.assertEqual(len(result["ignored"]["gt_boxes"]), 2)
        ignored_reasons = [item["reason"] for item in result["ignored"]["gt_boxes"]]
        self.assertEqual(ignored_reasons, ["outside_roi", "outside_roi"])

    def test_zero_denominator_metric_matrix(self):
        empty = evaluate_detections([], [], iou_threshold=0.5, auxiliary_iou_thresholds=[])
        self.assertIsNone(empty["metrics"]["precision"])
        self.assertIsNone(empty["metrics"]["recall"])
        self.assertIsNone(empty["metrics"]["f1"])

        only_gt = evaluate_detections([], [box("gt_car", class_name="car")], iou_threshold=0.5, auxiliary_iou_thresholds=[])
        self.assertIsNone(only_gt["metrics"]["precision"])
        self.assertEqual(only_gt["metrics"]["recall"], 0.0)
        self.assertEqual(only_gt["metrics"]["f1"], 0.0)

        only_det = evaluate_detections([box("det_car", class_name="car")], [], iou_threshold=0.5, auxiliary_iou_thresholds=[])
        self.assertEqual(only_det["metrics"]["precision"], 0.0)
        self.assertIsNone(only_det["metrics"]["recall"])
        self.assertEqual(only_det["metrics"]["f1"], 0.0)

    def test_batch_summary_and_csv_include_primary_and_auxiliary_iou_metrics(self):
        frame_reports = [
            {
                "frame_id": "000000",
                "num_points": 100,
                "num_gt_boxes": 1,
                "num_detections_after_nms": 1,
                "iou_threshold": 0.5,
                "metrics": {
                    "tp": 0,
                    "fp": 1,
                    "fn": 1,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "per_class": {},
                },
                "auxiliary": {
                    "0.25": {
                        "metrics": {
                            "tp": 1,
                            "fp": 0,
                            "fn": 0,
                            "precision": 1.0,
                            "recall": 1.0,
                            "f1": 1.0,
                            "per_class": {},
                        }
                    }
                },
                "report_path": "outputs/reports/frame_000000.json",
            },
            {
                "frame_id": "000001",
                "num_points": 120,
                "num_gt_boxes": 1,
                "num_detections_after_nms": 1,
                "iou_threshold": 0.5,
                "metrics": {
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "per_class": {},
                },
                "auxiliary": {
                    "0.25": {
                        "metrics": {
                            "tp": 1,
                            "fp": 0,
                            "fn": 0,
                            "precision": 1.0,
                            "recall": 1.0,
                            "f1": 1.0,
                            "per_class": {},
                        }
                    }
                },
                "report_path": "outputs/reports/frame_000001.json",
            },
        ]

        summary = summarize_batch_reports(
            frame_reports=frame_reports,
            data_root="data/kitti",
            eps=0.6,
            min_points=20,
            oriented=True,
            nms_iou_threshold=0.3,
            eval_iou_threshold=0.5,
            auxiliary_iou_thresholds=[0.25],
        )

        self.assertEqual(summary["metrics_by_iou"]["0.50"]["tp"], 1)
        self.assertEqual(summary["metrics_by_iou"]["0.50"]["fp"], 1)
        self.assertEqual(summary["metrics_by_iou"]["0.50"]["fn"], 1)
        self.assertEqual(summary["metrics_by_iou"]["0.50"]["f1"], 0.5)

        self.assertEqual(summary["metrics_by_iou"]["0.25"]["tp"], 2)
        self.assertEqual(summary["metrics_by_iou"]["0.25"]["fp"], 0)
        self.assertEqual(summary["metrics_by_iou"]["0.25"]["fn"], 0)
        self.assertEqual(summary["metrics_by_iou"]["0.25"]["f1"], 1.0)

        self.assertEqual(summary["totals"]["tp"], 1)
        self.assertEqual(summary["frames"][0]["metrics_by_iou"]["0.50"]["tp"], 0)
        self.assertEqual(summary["frames"][0]["metrics_by_iou"]["0.25"]["tp"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "frames.csv"
            save_frame_csv(frame_reports, csv_path)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("tp_iou_0_50", header)
        self.assertIn("tp_iou_0_25", header)
        self.assertIn("f1_iou_0_50", header)
        self.assertIn("f1_iou_0_25", header)


if __name__ == "__main__":
    unittest.main()
