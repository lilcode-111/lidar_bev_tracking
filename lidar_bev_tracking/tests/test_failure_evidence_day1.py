import json
import unittest
from pathlib import Path

import numpy as np

from bev_tracking.clustering_detector import (
    DEFAULT_INTENSITY_MIN,
    DEFAULT_Z_MIN,
    detect_objects_from_points,
    filter_obstacle_points,
    split_obstacle_filter_stages,
)
from bev_tracking.config import DEFAULT_KITTI_EVAL_CONFIG
from bev_tracking.result_types import FailureEvidence, FailureReason, FilterStageCounts
from bev_tracking.synthetic import generate_frame


class FilterConfigurationTest(unittest.TestCase):
    def test_explicit_a0_filter_values_match_legacy_defaults(self):
        points, _ = generate_frame(seed=7)

        default_points = filter_obstacle_points(points)
        explicit_points = filter_obstacle_points(
            points,
            z_min=-0.9,
            intensity_min=0.38,
        )

        np.testing.assert_array_equal(default_points, explicit_points)

    def test_filter_stage_counts_are_monotonic(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, 0.8],
                [50.0, 0.0, 0.0, 0.8],
                [10.0, 0.0, -1.0, 0.8],
                [10.0, 0.0, 0.0, 0.2],
            ],
            dtype=np.float32,
        )

        stages = split_obstacle_filter_stages(points)

        self.assertEqual(len(stages["raw"]), 4)
        self.assertEqual(len(stages["roi"]), 3)
        self.assertEqual(len(stages["z_filter"]), 2)
        self.assertEqual(len(stages["intensity_filter"]), 1)

    def test_trace_mode_does_not_change_detections(self):
        points, _ = generate_frame(seed=7)

        detections = detect_objects_from_points(points, oriented=True)
        traced_detections, trace = detect_objects_from_points(
            points,
            oriented=True,
            return_trace=True,
        )

        self.assertEqual(traced_detections, detections)
        self.assertEqual(trace["point_counts"]["raw"], len(points))
        self.assertEqual(trace["parameters"]["z_min"], DEFAULT_Z_MIN)
        self.assertEqual(trace["parameters"]["intensity_min"], DEFAULT_INTENSITY_MIN)
        self.assertEqual(trace["cluster_count"], len(trace["cluster_point_counts"]))

    def test_default_config_freezes_a0_filter_values(self):
        detector = DEFAULT_KITTI_EVAL_CONFIG["detector"]

        self.assertEqual(detector["z_min"], -0.9)
        self.assertEqual(detector["intensity_min"], 0.38)


class FailureEvidenceTypesTest(unittest.TestCase):
    def test_failure_evidence_is_json_serializable(self):
        evidence = FailureEvidence(
            frame_id="317",
            gt_id="gt_1",
            primary_reason=FailureReason.REMOVED_BY_INTENSITY_FILTER,
            stage_point_counts=FilterStageCounts(raw=40, roi=40, z_filter=32, intensity_filter=0),
            supporting_flags=[FailureReason.INSUFFICIENT_POINTS_FOR_CLUSTERING],
            source_run_id="run-test",
        )

        output = evidence.to_dict()

        self.assertEqual(output["frame_id"], "000317")
        self.assertEqual(output["primary_reason"], "removed_by_intensity_filter")
        self.assertEqual(output["stage_point_counts"]["z_filter"], 32)
        self.assertEqual(output["supporting_flags"], ["insufficient_points_for_clustering"])
        json.dumps(output)

    def test_diagnostic_manifest_is_frozen_unique_subset(self):
        diagnostic_path = Path("configs/kitti_diagnostic_frames.txt")
        baseline_path = Path("configs/kitti_100_frames.txt")

        diagnostic_ids = [
            line.strip()
            for line in diagnostic_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        baseline_ids = {
            line.strip()
            for line in baseline_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        self.assertEqual(len(diagnostic_ids), 25)
        self.assertEqual(len(set(diagnostic_ids)), 25)
        self.assertEqual(diagnostic_ids, sorted(diagnostic_ids))
        self.assertTrue(set(diagnostic_ids).issubset(baseline_ids))


if __name__ == "__main__":
    unittest.main()
