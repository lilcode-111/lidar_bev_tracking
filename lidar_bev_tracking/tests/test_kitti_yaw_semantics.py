import unittest

import numpy as np

from bev_tracking.geometry import box_corners_bev
from bev_tracking.geometry_sanity import angle_error
from bev_tracking.kitti_calib import (
    camera_rotation_y_to_lidar_yaw,
    kitti_labels_to_lidar_boxes,
    lidar_yaw_to_camera_rotation_y,
)
from bev_tracking.kitti_yaw_validation import (
    reference_camera_rotation_y_to_lidar_yaw,
    reference_kitti_box_corners_in_lidar,
    unordered_corner_error,
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


def car_label(rotation_y=0.0):
    return {
        "class_name": "Car",
        "height": 1.5,
        "width": 1.9,
        "length": 4.5,
        "location_camera": [3.0, 0.75, 12.0],
        "rotation_y": rotation_y,
    }


class KittiYawSemanticTest(unittest.TestCase):
    def test_round_trip_control_can_pass_even_with_wrong_semantics(self):
        calib = standard_test_calib()
        rotation_y = 0.4
        lidar_yaw = camera_rotation_y_to_lidar_yaw(rotation_y, calib)
        reconstructed = lidar_yaw_to_camera_rotation_y(lidar_yaw, calib)

        self.assertLessEqual(angle_error(rotation_y, reconstructed), 1e-6)

    def test_rotation_y_zero_follows_kitti_length_axis(self):
        calib = standard_test_calib()
        expected = reference_camera_rotation_y_to_lidar_yaw(0.0, calib)
        actual = camera_rotation_y_to_lidar_yaw(0.0, calib)

        self.assertAlmostEqual(expected, -np.pi / 2.0, places=6)
        self.assertLessEqual(angle_error(expected, actual), 1e-6)

    def test_lidar_box_corners_match_independent_kitti_corner_transform(self):
        calib = standard_test_calib()
        label = car_label(rotation_y=0.4)
        box = kitti_labels_to_lidar_boxes([label], calib)[0]
        reference_corners = reference_kitti_box_corners_in_lidar(label, calib)
        actual_corners = box_corners_bev(box)

        self.assertLessEqual(unordered_corner_error(reference_corners, actual_corners), 1e-5)

    def test_multiple_directions_match_independent_yaw_and_corner_reference(self):
        calib = standard_test_calib()
        rotation_values = [-np.pi + 0.01, -np.pi / 2.0, -0.4, 0.0, 0.4, np.pi / 2.0, np.pi - 0.01]

        for rotation_y in rotation_values:
            with self.subTest(rotation_y=rotation_y):
                expected_yaw = reference_camera_rotation_y_to_lidar_yaw(rotation_y, calib)
                actual_yaw = camera_rotation_y_to_lidar_yaw(rotation_y, calib)
                self.assertLessEqual(angle_error(expected_yaw, actual_yaw), 1e-6)

                label = car_label(rotation_y=rotation_y)
                box = kitti_labels_to_lidar_boxes([label], calib)[0]
                reference_corners = reference_kitti_box_corners_in_lidar(label, calib)
                actual_corners = box_corners_bev(box)
                self.assertLessEqual(unordered_corner_error(reference_corners, actual_corners), 1e-5)


if __name__ == "__main__":
    unittest.main()
