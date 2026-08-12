import copy
import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.clustering_detector import cluster_to_oriented_box
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.geometry import bev_iou
from bev_tracking.point_retention import (
    FRAGMENT_ORACLE_SCHEMA_VERSION,
    POINT_RETENTION_SCHEMA_VERSION,
    build_frame_fragment_oracles,
    build_point_retention_day2,
)


def car_box():
    return {
        "id": "gt_1", "class_name": "car", "x": 10.0, "y": 0.0, "z": 0.0,
        "length": 4.0, "width": 2.0, "height": 2.0, "yaw": 0.0,
    }


def rectangle(x_min, x_max, y_min=-0.8, y_max=0.8):
    return np.asarray([
        [x_min, y_min, 0.0, 0.8], [x_min, y_max, 0.0, 0.8],
        [x_max, y_min, 0.0, 0.8], [x_max, y_max, 0.0, 0.8],
    ], dtype=np.float32)


def canonical_payload(clusters, detections, iou_offset=0.0):
    branches = [
        {
            "cluster_id": detection["id"],
            "candidate_iou": float(bev_iou(car_box(), detection) + iou_offset),
        }
        for detection in detections
    ]
    return {
        "evidence": [{
            "frame_id": "000001", "gt_id": "gt_1", "distance_bin": "near_0_15",
            "cluster_ids": [detection["id"] for detection in detections],
            "candidate_branches": branches,
        }]
    }


def provenance(clusters):
    raw_points = np.vstack(clusters)
    source_indices = []
    start = 0
    for cluster in clusters:
        source_indices.append(np.arange(start, start + len(cluster), dtype=np.int64))
        start += len(cluster)
    return raw_points, source_indices


class PointRetentionDay2Test(unittest.TestCase):
    def test_fragment_union_improves_over_best_single_fragment(self):
        clusters = [rectangle(8.2, 9.6), rectangle(10.4, 11.8)]
        detections = [cluster_to_oriented_box(cluster, index + 1) for index, cluster in enumerate(clusters)]
        raw_points, source_indices = provenance(clusters)
        payload = build_frame_fragment_oracles(
            frame_id="1", variant="C1", gt_boxes=[car_box()], clusters=clusters,
            raw_detections=detections,
            candidate_conversion=canonical_payload(clusters, detections),
            raw_points=raw_points, cluster_source_point_indices=source_indices,
        )
        record = payload["records"][0]
        self.assertEqual(record["associated_cluster_count"], 2)
        self.assertGreater(record["O2"]["iou"], record["O1"]["iou"])
        self.assertGreater(record["signed_deltas"]["O2_minus_O1"], 0.0)

    def test_union_deduplicates_exact_source_points(self):
        left = rectangle(8.2, 9.6)
        right = rectangle(10.4, 11.8)
        raw_points = np.vstack([left, right])
        clusters = [raw_points[:4], np.vstack([raw_points[4:], raw_points[0]])]
        source_indices = [np.arange(4), np.asarray([4, 5, 6, 7, 0])]
        detections = [cluster_to_oriented_box(cluster, index + 1) for index, cluster in enumerate(clusters)]
        record = build_frame_fragment_oracles(
            frame_id="1", variant="C1", gt_boxes=[car_box()], clusters=clusters,
            raw_detections=detections, candidate_conversion=canonical_payload(clusters, detections),
            raw_points=raw_points, cluster_source_point_indices=source_indices,
        )["records"][0]
        self.assertEqual(record["duplicate_union_point_count"], 1)
        self.assertEqual(record["associated_union_point_count"], 8)

    def test_gt_clipped_control_separates_cluster_contamination(self):
        cluster = np.vstack([rectangle(8.2, 11.8), rectangle(13.0, 14.0)])
        detection = cluster_to_oriented_box(cluster, 1)
        raw_points, source_indices = provenance([cluster])
        record = build_frame_fragment_oracles(
            frame_id="1", variant="C1", gt_boxes=[car_box()], clusters=[cluster],
            raw_detections=[detection], candidate_conversion=canonical_payload([cluster], [detection]),
            raw_points=raw_points, cluster_source_point_indices=source_indices,
        )["records"][0]
        self.assertLess(record["associated_union_gt_clipped_point_count"], record["associated_union_point_count"])
        self.assertGreater(record["O2_gt_clipped"]["iou"], record["O2"]["iou"])

    def test_o1_must_match_canonical_candidate_iou(self):
        cluster = rectangle(8.5, 9.5)
        detection = cluster_to_oriented_box(cluster, 1)
        raw_points, source_indices = provenance([cluster])
        with self.assertRaisesRegex(ValueError, "O1 differs"):
            build_frame_fragment_oracles(
                frame_id="1", variant="C1", gt_boxes=[car_box()], clusters=[cluster],
                raw_detections=[detection],
                candidate_conversion=canonical_payload([cluster], [detection], iou_offset=0.01),
                raw_points=raw_points, cluster_source_point_indices=source_indices,
            )

    def test_run_variant_frame_emits_fragment_oracle_without_mutating_canonical(self):
        points = rectangle(9.2, 10.8, -0.4, 0.4)
        report = run_variant_frame(
            points, [car_box()], frame_id="1",
            variant=VariantSpec("C1", ClusteringPolicy(mode="fixed", eps=2.0, min_points=3, global_max_eps=2.0)),
        )
        canonical_before = copy.deepcopy(report["candidate_conversion"])
        self.assertEqual(report["fragment_recoverability"]["schema_version"], FRAGMENT_ORACLE_SCHEMA_VERSION)
        self.assertEqual(report["fragment_recoverability"]["associated_gt_count"], 1)
        self.assertEqual(report["candidate_conversion"], canonical_before)

    def test_day2_consumes_day1_keys_without_rebuilding_them(self):
        selected = {
            "frame_id": "000001", "gt_id": "gt_1", "O1": {"iou": 0.1},
            "O2": {"iou": 0.2}, "O2_gt_clipped": {"iou": 0.3},
        }
        day1 = {
            "schema_version": POINT_RETENTION_SCHEMA_VERSION,
            "candidate_variant": "C1",
            "delta_gt_keys": [{"frame_id": "000001", "gt_id": "gt_1"}],
            "p1_control_gt_keys": [],
        }
        reports = {"C1": [{
            "frame_id": "000001",
            "fragment_recoverability": {
                "schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
                "records": [selected],
            },
        }]}
        result = build_point_retention_day2(day1, reports)
        self.assertEqual(result["delta_gt_count"], 1)
        self.assertEqual(result["delta_records"], [selected])


if __name__ == "__main__":
    unittest.main()
