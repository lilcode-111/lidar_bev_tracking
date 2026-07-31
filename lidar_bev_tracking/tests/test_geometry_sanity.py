import json
import unittest

import numpy as np

from bev_tracking.geometry import bev_iou
from bev_tracking.geometry_sanity import (
    angle_error,
    build_geometry_sanity_report,
    coordinate_round_trip_metrics,
    points_in_oriented_3d_box,
)
from bev_tracking.kitti_calib import (
    camera_rotation_y_to_lidar_yaw,
    kitti_labels_to_lidar_boxes,
    lidar_yaw_to_camera_rotation_y,
)


def standard_test_calib():
    return {
        "R0_rect": np.eye(3, dtype=np.float32),
        "Tr_velo_to_cam": np.asarray(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }


def car_label():
    return {
        "class_name": "Car",
        "height": 1.5,
        "width": 1.9,
        "length": 4.5,
        "location_camera": [3.0, 0.75, 12.0],
        "rotation_y": 0.0,
    }


class GeometrySanityTest(unittest.TestCase):
    def test_camera_lidar_coordinate_round_trip(self):
        points = np.asarray(
            [
                [12.0, -3.0, 0.0, 0.8],
                [20.0, 4.0, -1.0, 0.4],
            ],
            dtype=np.float32,
        )

        metrics = coordinate_round_trip_metrics(points, standard_test_calib())

        self.assertEqual(metrics["num_points"], 2)
        self.assertLessEqual(metrics["max_error_m"], 1e-6)

    def test_yaw_round_trip_uses_camera_and_lidar_conventions(self):
        calib = standard_test_calib()

        for rotation_y in [0.0, 0.4, -1.2, np.pi / 2.0]:
            lidar_yaw = camera_rotation_y_to_lidar_yaw(rotation_y, calib)
            reconstructed = lidar_yaw_to_camera_rotation_y(lidar_yaw, calib)
            self.assertLessEqual(angle_error(rotation_y, reconstructed), 1e-6)

    def test_kitti_gt_box_keeps_height_and_uses_geometric_center(self):
        box = kitti_labels_to_lidar_boxes([car_label()], standard_test_calib())[0]

        self.assertAlmostEqual(box["x"], 12.0)
        self.assertAlmostEqual(box["y"], -3.0)
        self.assertAlmostEqual(box["z"], 0.0)
        self.assertAlmostEqual(box["length"], 4.5)
        self.assertAlmostEqual(box["width"], 1.9)
        self.assertAlmostEqual(box["height"], 1.5)
        self.assertAlmostEqual(box["yaw"], 0.0)

    def test_oriented_3d_point_in_box_respects_yaw_and_height(self):
        box = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "length": 4.0,
            "width": 2.0,
            "height": 2.0,
            "yaw": np.pi / 2.0,
        }
        points = np.asarray(
            [
                [0.0, 1.9, 0.9],
                [1.1, 0.0, 0.0],
                [0.0, 0.0, 1.1],
            ],
            dtype=np.float32,
        )

        mask = points_in_oriented_3d_box(points, box)

        self.assertEqual(mask.tolist(), [True, False, False])

    def test_height_does_not_change_bev_iou(self):
        box = {
            "x": 10.0,
            "y": 0.0,
            "z": 0.0,
            "length": 4.0,
            "width": 2.0,
            "yaw": 0.2,
        }

        self.assertAlmostEqual(bev_iou(box, {**box, "height": 1.5}), 1.0)

    def test_frame_report_contains_point_and_box_sanity_evidence(self):
        points = np.asarray(
            [
                [12.0, -3.0, 0.0, 0.8],
                [13.0, -3.0, 0.0, 0.7],
                [30.0, 10.0, 0.0, 0.4],
            ],
            dtype=np.float32,
        )

        report = build_geometry_sanity_report(
            points,
            [car_label()],
            standard_test_calib(),
            frame_id="317",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["frame_id"], "000317")
        self.assertEqual(report["summary"]["num_car_boxes"], 1)
        self.assertEqual(report["summary"]["car_boxes_with_points"], 1)
        self.assertEqual(report["boxes"][0]["points_in_box"], 2)
        json.dumps(report)

    def test_center_and_yaw_tolerances_are_independent(self):
        report = build_geometry_sanity_report(
            np.asarray([[12.0, -3.0, 0.0, 0.8]], dtype=np.float32),
            [car_label()],
            standard_test_calib(),
            frame_id="317",
            center_tolerance_m=1e-12,
            yaw_tolerance_rad=1e-3,
        )

        self.assertEqual(report["tolerances"]["center_error_m"], 1e-12)
        self.assertEqual(report["tolerances"]["yaw_error_rad"], 1e-3)
        self.assertIn("center_passed", report["boxes"][0])
        self.assertIn("yaw_passed", report["boxes"][0])


if __name__ == "__main__":
    unittest.main()
