import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.cluster_separability import (
    build_cluster_group_day1,
    build_frame_cluster_groups,
    validate_cluster_group_partition,
)
from bev_tracking.clustering_policy import ClusteringPolicy


def box(box_id, class_name, x):
    return {
        "id": box_id,
        "class_name": class_name,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": 0.0,
    }


def cluster(x, count=10):
    return np.column_stack(
        [
            np.linspace(x - 0.5, x + 0.5, count),
            np.linspace(-0.3, 0.3, count),
            np.zeros(count),
            np.ones(count),
        ]
    ).astype(np.float32)


class ClusterSeparabilityDay1Test(unittest.TestCase):
    def test_run_variant_frame_emits_conserved_cluster_groups(self):
        frame_points = cluster(10.0, count=12)
        report = run_variant_frame(
            frame_points,
            [box("gt_1", "car", 10.0)],
            frame_id="1",
            variant=VariantSpec(
                "C1", ClusteringPolicy(mode="fixed", eps=0.6, min_points=3)
            ),
        )
        payload = report["cluster_separability"]
        self.assertEqual(payload["cluster_count"], 1)
        self.assertEqual(sum(payload["group_counts"].values()), 1)
        self.assertEqual(len(payload["records"]), 1)

    def test_frame_groups_partition_clusters_into_p1_p2_and_n(self):
        clusters = [cluster(10.0), cluster(20.0), cluster(35.0)]
        detections = [
            {"id": "cluster_1", "class_name": "car", "x": 10.0, "y": 0.0, "length": 3.0, "width": 1.5, "yaw": 0.0},
            {"id": "cluster_2", "class_name": "pedestrian", "x": 20.0, "y": 0.0, "length": 1.0, "width": 0.6, "yaw": 0.0},
            {"id": "cluster_3", "class_name": "cone", "x": 35.0, "y": 0.0, "length": 0.5, "width": 0.3, "yaw": 0.0},
        ]
        payload = build_frame_cluster_groups(
            frame_id="1",
            variant="C1",
            filtered_points=np.vstack(clusters),
            clusters=clusters,
            raw_detections=detections,
            gt_boxes=[box("gt_1", "car", 10.0), box("gt_2", "car", 20.0)],
        )
        self.assertEqual(payload["group_counts"], {"P1": 1, "P2": 1, "N": 1})
        self.assertEqual([item["group"] for item in payload["records"]], ["P1", "P2", "N"])
        self.assertEqual(payload["records"][1]["associated_gt_ids"], ["gt_2"])
        self.assertEqual(payload["records"][2]["cluster_distance_bin"], "far_30_inf")

    def test_n_retains_non_car_annotation_context(self):
        background = cluster(20.0)
        payload = build_frame_cluster_groups(
            frame_id="1",
            variant="C1",
            filtered_points=background,
            clusters=[background],
            raw_detections=[{"id": "cluster_1", "class_name": "pedestrian", "x": 20.0, "y": 0.0}],
            gt_boxes=[box("ped_1", "pedestrian", 20.0)],
        )
        record = payload["records"][0]
        self.assertEqual(record["group"], "N")
        self.assertIn("excluded", record["annotation_overlap_categories"])
        self.assertFalse(record["is_strict_background"])

    def test_day1_marks_only_new_c1_rejected_associations(self):
        def evidence(frame_id, gt_id, cluster_ids):
            return {"frame_id": frame_id, "gt_id": gt_id, "cluster_ids": cluster_ids}

        def report(variant, gt_records, group_records):
            return {
                "frame_id": "000001",
                "candidate_conversion": {"evidence": gt_records},
                "cluster_separability": {
                    "schema_version": "15.3.1-cluster-groups-day1",
                    "records": group_records,
                },
            }

        base = report("C0", [evidence("000001", "gt_1", [])], [])
        p2 = {
            "frame_id": "000001", "variant": "C1", "cluster_id": "cluster_1",
            "group": "P2", "associated_gt_ids": ["gt_1"],
        }
        candidate = report("C1", [evidence("000001", "gt_1", ["cluster_1"])], [p2])
        result = build_cluster_group_day1({"C0": [base], "C1": [candidate]})
        self.assertEqual(result["delta_associated_gt_count"], 1)
        self.assertEqual(result["delta_p2_gt_count"], 1)
        self.assertTrue(result["records_by_variant"]["C1"][0]["is_delta_22"])

    def test_partition_validation_rejects_duplicates_and_group_mismatch(self):
        with self.assertRaises(ValueError):
            validate_cluster_group_partition(
                [
                    {"cluster_id": "cluster_1", "group": "N", "associated_gt_ids": []},
                    {"cluster_id": "cluster_1", "group": "N", "associated_gt_ids": []},
                ],
                2,
            )
        with self.assertRaises(ValueError):
            validate_cluster_group_partition(
                [{"cluster_id": "cluster_1", "group": "P2", "associated_gt_ids": []}],
                1,
            )


if __name__ == "__main__":
    unittest.main()
