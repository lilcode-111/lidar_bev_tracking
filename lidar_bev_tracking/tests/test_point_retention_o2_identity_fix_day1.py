import copy
import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.clustering_detector import cluster_to_oriented_box
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.geometry import bev_iou
from bev_tracking.point_retention import build_frame_fragment_oracles


def car_box():
    return {
        "id": "gt_1", "class_name": "car", "x": 10.0, "y": 0.0, "z": 0.0,
        "length": 4.0, "width": 2.0, "height": 2.0, "yaw": 0.0,
    }


def canonical(clusters, detections):
    return {"evidence": [{
        "frame_id": "000001", "gt_id": "gt_1", "distance_bin": "near_0_15",
        "cluster_ids": [item["id"] for item in detections],
        "candidate_branches": [
            {"cluster_id": item["id"], "candidate_iou": float(bev_iou(car_box(), item))}
            for item in detections
        ],
    }]}


def build(raw_points, source_indices):
    clusters = [raw_points[np.asarray(indices)] for indices in source_indices]
    detections = [cluster_to_oriented_box(cluster, index + 1) for index, cluster in enumerate(clusters)]
    return build_frame_fragment_oracles(
        frame_id="1", variant="C1", gt_boxes=[car_box()], clusters=clusters,
        raw_detections=detections, candidate_conversion=canonical(clusters, detections),
        raw_points=raw_points, cluster_source_point_indices=source_indices,
    )["records"][0]


class O2PointIdentityFixDay1Test(unittest.TestCase):
    def test_distinct_indices_with_identical_rows_are_both_retained(self):
        raw = np.asarray([
            [9.0, -0.5, 0.0, 0.8], [9.0, -0.5, 0.0, 0.8],
            [10.0, 0.5, 0.0, 0.8], [11.0, -0.5, 0.0, 0.8],
        ], dtype=np.float32)
        record = build(raw, [np.asarray([0, 2]), np.asarray([1, 3])])
        self.assertEqual(record["associated_union_point_count"], 4)
        self.assertEqual(record["associated_union_source_point_indices"], [0, 1, 2, 3])

    def test_same_source_index_across_fragments_is_retained_once(self):
        raw = np.asarray([
            [9.0, -0.5, 0.0, 0.8], [10.0, 0.5, 0.0, 0.8],
            [11.0, -0.5, 0.0, 0.8],
        ], dtype=np.float32)
        record = build(raw, [np.asarray([0, 1]), np.asarray([1, 2])])
        self.assertEqual(record["associated_union_point_count"], 3)
        self.assertEqual(record["duplicate_union_point_count"], 1)
        self.assertEqual(record["associated_union_source_point_indices"], [0, 1, 2])

    def test_union_order_does_not_change_pca_or_iou(self):
        raw = np.asarray([
            [8.5, -0.7, 0.0, 0.8], [8.5, 0.7, 0.0, 0.8],
            [11.5, -0.7, 0.0, 0.8], [11.5, 0.7, 0.0, 0.8],
        ], dtype=np.float32)
        forward = build(raw, [np.asarray([0, 2]), np.asarray([1, 3])])
        reverse = build(raw, [np.asarray([3, 1]), np.asarray([2, 0])])
        self.assertEqual(forward["associated_union_source_point_indices"], [0, 1, 2, 3])
        self.assertEqual(forward["O2"], reverse["O2"])
        self.assertEqual(forward["O2_gt_clipped"], reverse["O2_gt_clipped"])

    def test_o3_o6_and_canonical_count_gate_are_unchanged(self):
        points = np.asarray([
            [9.0, -0.5, 0.0, 0.8], [9.0, 0.5, 0.0, 0.8],
            [10.0, -0.5, 0.0, 0.8], [10.0, 0.5, 0.0, 0.8],
        ], dtype=np.float32)
        report = run_variant_frame(
            points, [car_box()], frame_id="1",
            variant=VariantSpec(
                "C1", ClusteringPolicy(
                    mode="fixed", eps=2.0, min_points=3, global_max_eps=2.0
                ),
            ),
        )
        stage = copy.deepcopy(report["stage_recoverability"])
        self.assertTrue(stage["canonical_count_gate_passed"])
        record = stage["records"][0]
        self.assertEqual([record[name]["num_points"] for name in ("O3", "O4", "O5", "O6")], [4, 4, 4, 4])
        self.assertTrue(all(record[name]["canonical_count_match"] for name in ("O3", "O4", "O5", "O6")))

    def test_cluster_source_indices_must_reconstruct_cluster_exactly(self):
        raw = np.asarray([
            [9.0, -0.5, 0.0, 0.8], [10.0, 0.5, 0.0, 0.8],
            [11.0, -0.5, 0.0, 0.8],
        ], dtype=np.float32)
        clusters = [raw[[0, 1, 2]]]
        detections = [cluster_to_oriented_box(clusters[0], 1)]
        with self.assertRaisesRegex(ValueError, "differ from raw LiDAR"):
            build_frame_fragment_oracles(
                frame_id="1", variant="C1", gt_boxes=[car_box()], clusters=clusters,
                raw_detections=detections, candidate_conversion=canonical(clusters, detections),
                raw_points=raw, cluster_source_point_indices=[np.asarray([2, 1, 0])],
            )


if __name__ == "__main__":
    unittest.main()
