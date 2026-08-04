import unittest

from bev_tracking.c0_replay import compare_c0_reports, run_c0_frame_replay
from bev_tracking.synthetic import generate_frame
from bev_tracking.kitti_calib import kitti_labels_to_lidar_boxes


class C0ReplayTest(unittest.TestCase):
    def test_c0_fixed_replay_has_zero_mismatches(self):
        points, objects = generate_frame(seed=7)
        gt_boxes = [
            {
                "id": f"gt_{index}",
                "class_name": obj["class_name"],
                "x": obj["x"],
                "y": obj["y"],
                "z": obj["z"],
                "length": obj["length"],
                "width": obj["width"],
                "height": obj.get("height", 1.5),
                "yaw": obj["yaw"],
            }
            for index, obj in enumerate(objects, start=1)
        ]
        result = run_c0_frame_replay(
            points,
            gt_boxes,
            "7",
            eps=0.6,
            min_points=20,
            oriented=True,
            z_min=-0.9,
            intensity_min=0.38,
            nms_iou_threshold=0.3,
            eval_iou_threshold=0.5,
            auxiliary_iou_thresholds=(0.25,),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["mismatches"], [])

    def test_compare_c0_reports_exposes_metric_mismatch(self):
        points, objects = generate_frame(seed=7)
        gt_boxes = [
            {
                "id": f"gt_{index}",
                "class_name": obj["class_name"],
                "x": obj["x"],
                "y": obj["y"],
                "z": obj["z"],
                "length": obj["length"],
                "width": obj["width"],
                "height": obj.get("height", 1.5),
                "yaw": obj["yaw"],
            }
            for index, obj in enumerate(objects, start=1)
        ]
        result = run_c0_frame_replay(
            points,
            gt_boxes,
            "7",
            eps=0.6,
            min_points=20,
            oriented=True,
            z_min=-0.9,
            intensity_min=0.38,
            nms_iou_threshold=0.3,
            eval_iou_threshold=0.5,
            auxiliary_iou_thresholds=(0.25,),
        )
        legacy = result["legacy_summary"]
        unified = result["unified_summary"]
        unified["metrics_by_iou"]["0.50"]["tp"] += 1
        self.assertIn("summary.metrics_by_iou", compare_c0_reports({"summary": legacy}, {"summary": unified}))


if __name__ == "__main__":
    unittest.main()
