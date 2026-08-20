import inspect
import unittest
from unittest.mock import patch

import numpy as np

from bev_tracking.clustering_detector import detect_objects_from_points
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.pipeline import run_kitti_frame_evaluation


def _point(x, y, intensity, z=0.0):
    return [x, y, z, intensity]


def _pipeline_cloud():
    return np.asarray([
        _point(10.00, -0.05, 0.50),
        _point(10.00, 0.05, 0.50),
        _point(10.20, -0.05, 0.50),
        _point(10.20, 0.05, 0.50),
        _point(10.75, 0.00, 0.20),
        _point(10.10, 0.00, 0.20),
        _point(20.00, 0.00, 0.10),
        _point(10.30, 0.00, 0.50, z=-1.0),
    ], dtype=np.float64)


class GESRV1Day4Test(unittest.TestCase):
    def test_disabled_default_never_calls_gesr_runtime(self):
        points = _pipeline_cloud()
        with patch(
            "bev_tracking.gesr_v1.run_gesr_v1_optimized",
            side_effect=AssertionError("disabled path called GESR"),
        ):
            implicit = detect_objects_from_points(
                points, min_points=2, oriented=True, return_trace=True
            )
            explicit = detect_objects_from_points(
                points,
                min_points=2,
                oriented=True,
                return_trace=True,
                gesr_enabled=False,
            )
        self.assertEqual(implicit, explicit)
        self.assertNotIn("gesr", implicit[1])
        self.assertNotIn("detector_input", implicit[1]["point_counts"])

    def test_enabled_pipeline_uses_expanded_source_identity(self):
        detections, trace = detect_objects_from_points(
            _pipeline_cloud(),
            min_points=2,
            oriented=True,
            return_trace=True,
            gesr_enabled=True,
            gesr_frame_id="7",
        )
        self.assertTrue(detections)
        self.assertEqual(trace["gesr"]["frame_id"], "000007")
        self.assertEqual(trace["gesr"]["obstacle_source_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(trace["point_counts"]["intensity_filter"], 4)
        self.assertEqual(trace["point_counts"]["detector_input"], 5)
        evidence = trace["gesr"]["runtime_evidence"]
        self.assertEqual(evidence["point_sets"]["accepted_source_indices"], [4])
        invariants = evidence["invariants"]
        self.assertTrue(invariants["accepted_iff_terminal_ACCEPTED"])
        self.assertTrue(invariants["exactly_one_SELECTED_per_accepted_candidate"])
        self.assertEqual(invariants["rejected_candidate_SELECTED_count"], 0)
        self.assertTrue(invariants["accepted_subset_of_candidate_universe"])
        self.assertTrue(invariants["expanded_equals_seed_union_accepted"])

    def test_reason_attribution_on_off_preserves_pipeline_input_and_detections(self):
        points = _pipeline_cloud()
        enabled = detect_objects_from_points(
            points,
            min_points=2,
            oriented=True,
            return_trace=True,
            gesr_enabled=True,
            gesr_frame_id="7",
            gesr_reason_attribution=True,
        )
        disabled = detect_objects_from_points(
            points,
            min_points=2,
            oriented=True,
            return_trace=True,
            gesr_enabled=True,
            gesr_frame_id="7",
            gesr_reason_attribution=False,
        )
        self.assertEqual(enabled[0], disabled[0])
        self.assertEqual(
            enabled[1]["gesr"]["obstacle_source_indices"],
            disabled[1]["gesr"]["obstacle_source_indices"],
        )
        self.assertIsNotNone(enabled[1]["gesr"]["runtime_evidence"])
        self.assertIsNone(disabled[1]["gesr"]["runtime_evidence"])

    def test_post_z_pre_intensity_stage_is_the_gesr_input(self):
        _, trace = detect_objects_from_points(
            _pipeline_cloud(),
            min_points=2,
            return_trace=True,
            gesr_enabled=True,
            gesr_frame_id="7",
        )
        evidence = trace["gesr"]["runtime_evidence"]
        candidate_indices = evidence["point_sets"]["candidate_source_indices"]
        discarded_indices = evidence["point_sets"]["discarded_source_indices"]
        self.assertEqual(candidate_indices, [4, 5])
        self.assertEqual(discarded_indices, [6])
        self.assertNotIn(7, evidence["point_sets"]["seed_source_indices"])

    def test_enabled_pipeline_rejects_non_frozen_base_threshold(self):
        with self.assertRaisesRegex(ValueError, "intensity_min=0.38"):
            detect_objects_from_points(
                _pipeline_cloud(),
                intensity_min=0.30,
                gesr_enabled=True,
                gesr_frame_id="7",
            )

    def test_enabled_pipeline_requires_frame_identity(self):
        with self.assertRaisesRegex(ValueError, "gesr_frame_id"):
            detect_objects_from_points(_pipeline_cloud(), gesr_enabled=True)

    def test_enabled_pipeline_rejects_non_c0_clustering_policy(self):
        policy = ClusteringPolicy(mode="fixed", eps=0.6, min_points=2)
        with self.assertRaisesRegex(ValueError, "frozen C0 clustering"):
            detect_objects_from_points(
                _pipeline_cloud(),
                gesr_enabled=True,
                gesr_frame_id="7",
                clustering_policy=policy,
            )

    def test_pipeline_interfaces_add_no_gt_runtime_argument(self):
        detector_parameters = inspect.signature(detect_objects_from_points).parameters
        frame_parameters = inspect.signature(run_kitti_frame_evaluation).parameters
        forbidden = {"gt", "labels", "evaluation", "oracle", "failure_evidence"}
        self.assertTrue(forbidden.isdisjoint(detector_parameters))
        self.assertTrue(forbidden.isdisjoint(frame_parameters))

    def test_synthetic_pipeline_is_deterministic(self):
        kwargs = {
            "min_points": 2,
            "oriented": True,
            "return_trace": True,
            "gesr_enabled": True,
            "gesr_frame_id": "7",
        }
        first = detect_objects_from_points(_pipeline_cloud(), **kwargs)
        second = detect_objects_from_points(_pipeline_cloud(), **kwargs)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
