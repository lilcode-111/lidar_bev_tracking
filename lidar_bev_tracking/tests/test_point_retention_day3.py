import copy
import unittest

import numpy as np

from bev_tracking.adaptive_experiment import VariantSpec, run_variant_frame
from bev_tracking.clustering_policy import ClusteringPolicy
from bev_tracking.point_retention import (
    FRAGMENT_ORACLE_SCHEMA_VERSION,
    POINT_RETENTION_SCHEMA_VERSION,
    STAGE_ORACLE_SCHEMA_VERSION,
    build_frame_stage_oracles,
    build_point_retention_day3,
)


def car_box():
    return {
        "id": "gt_1", "class_name": "car", "x": 10.0, "y": 0.0, "z": 0.0,
        "length": 4.0, "width": 2.0, "height": 2.0, "yaw": 0.0,
    }


def nested_stage_points():
    intensity = np.asarray([
        [9.0, -0.5, 0.0, 0.8], [9.0, 0.5, 0.0, 0.8],
        [10.0, -0.5, 0.0, 0.8], [10.0, 0.5, 0.0, 0.8],
    ], dtype=np.float32)
    z_filter = np.vstack([intensity, [[11.0, -0.5, 0.0, 0.1]]]).astype(np.float32)
    roi = np.vstack([z_filter, [[11.0, 0.5, 0.0, 0.1]]]).astype(np.float32)
    raw = np.vstack([roi, [[8.5, 0.0, 0.0, 0.1]]]).astype(np.float32)
    return {"raw": raw, "roi": roi, "z_filter": z_filter, "intensity_filter": intensity}


def canonical_conversion(counts=None):
    if counts is None:
        counts = {"raw": 7, "roi": 6, "z_filter": 5, "intensity_filter": 4}
    return {"evidence": [{
        "frame_id": "000001", "gt_id": "gt_1", "distance_bin": "near_0_15",
        "stage_point_counts": counts,
    }]}


def oracle(iou, num_points):
    return {"status": "valid", "num_points": num_points, "box": {}, "iou": iou}


class PointRetentionDay3Test(unittest.TestCase):
    def test_stage_oracles_map_frozen_stages_and_match_canonical_counts(self):
        result = build_frame_stage_oracles(
            frame_id="1", variant="C1", stages=nested_stage_points(),
            gt_boxes=[car_box()], candidate_conversion=canonical_conversion(),
        )
        record = result["records"][0]
        self.assertTrue(result["canonical_count_gate_passed"])
        self.assertEqual(record["O3"]["source_stage"], "intensity_filter")
        self.assertEqual(record["O4"]["source_stage"], "z_filter")
        self.assertEqual(record["O5"]["source_stage"], "roi")
        self.assertEqual(record["O6"]["source_stage"], "raw")
        self.assertEqual(
            [record[name]["num_points"] for name in ("O3", "O4", "O5", "O6")],
            [4, 5, 6, 7],
        )
        self.assertTrue(all(record[name]["canonical_count_match"] for name in ("O3", "O4", "O5", "O6")))

    def test_stage_oracle_rejects_count_mismatch(self):
        counts = {"raw": 8, "roi": 6, "z_filter": 5, "intensity_filter": 4}
        with self.assertRaisesRegex(ValueError, "differs from canonical evidence"):
            build_frame_stage_oracles(
                frame_id="1", variant="C1", stages=nested_stage_points(),
                gt_boxes=[car_box()], candidate_conversion=canonical_conversion(counts),
            )

    def test_sparse_and_degenerate_stage_inputs_keep_iou_null(self):
        intensity = np.asarray([[9.0, 0.0, 0.0, 0.8], [10.0, 0.0, 0.0, 0.8]], dtype=np.float32)
        z_filter = np.vstack([intensity, [[11.0, 0.0, 0.0, 0.8]]]).astype(np.float32)
        roi = np.vstack([z_filter, [[10.0, 0.5, 0.0, 0.8]]]).astype(np.float32)
        stages = {"raw": roi, "roi": roi, "z_filter": z_filter, "intensity_filter": intensity}
        counts = {"raw": 4, "roi": 4, "z_filter": 3, "intensity_filter": 2}
        record = build_frame_stage_oracles(
            frame_id="1", variant="C1", stages=stages, gt_boxes=[car_box()],
            candidate_conversion=canonical_conversion(counts),
        )["records"][0]
        self.assertEqual(record["O3"]["status"], "insufficient_points")
        self.assertEqual(record["O4"]["status"], "degenerate_geometry")
        self.assertIsNone(record["O3"]["iou"])
        self.assertIsNone(record["O4"]["iou"])
        self.assertIsNone(record["signed_deltas"]["O4_minus_O3"])

    def test_formal_pipeline_emits_new_oracles_only_for_c1(self):
        points = nested_stage_points()["raw"]
        c0 = run_variant_frame(
            points, [car_box()], frame_id="1",
            variant=VariantSpec(
                "C0", ClusteringPolicy(mode="fixed", eps=2.0, min_points=3, global_max_eps=2.0)
            ),
        )
        c1 = run_variant_frame(
            points, [car_box()], frame_id="1",
            variant=VariantSpec(
                "C1", ClusteringPolicy(mode="fixed", eps=2.0, min_points=3, global_max_eps=2.0)
            ),
        )
        self.assertIsNone(c0["fragment_recoverability"])
        self.assertIsNone(c0["stage_recoverability"])
        self.assertEqual(c1["fragment_recoverability"]["schema_version"], FRAGMENT_ORACLE_SCHEMA_VERSION)
        self.assertEqual(c1["stage_recoverability"]["schema_version"], STAGE_ORACLE_SCHEMA_VERSION)

    def test_day3_joins_full_ladder_and_preserves_negative_delta(self):
        day1 = {
            "schema_version": POINT_RETENTION_SCHEMA_VERSION,
            "candidate_variant": "C1",
            "delta_records": [{
                "frame_id": "000001", "gt_id": "gt_1", "distance_bin": "near_0_15",
                "stage_point_counts": {"raw": 7, "roi": 6, "z_filter": 5, "intensity_filter": 4},
            }],
            "p1_control_records": [],
        }
        fragment = {
            "frame_id": "000001", "gt_id": "gt_1",
            "associated_cluster_point_count": 3, "associated_union_point_count": 3,
            "associated_union_gt_clipped_point_count": 3,
            "O1": oracle(0.10, 3), "O2": oracle(0.20, 3),
            "O2_gt_clipped": oracle(0.25, 3),
            "signed_deltas": {"O2_minus_O1": 0.10, "O2_gt_clipped_minus_O2": 0.05},
        }
        day2 = {
            "schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
            "candidate_variant": "C1", "delta_records": [fragment], "p1_control_records": [],
        }
        stage = {
            "frame_id": "000001", "gt_id": "gt_1",
            "O3": oracle(0.40, 4), "O4": oracle(0.50, 5),
            "O5": oracle(0.45, 6), "O6": oracle(0.60, 7),
            "signed_deltas": {"O4_minus_O3": 0.10, "O5_minus_O4": -0.05, "O6_minus_O5": 0.15},
        }
        reports = {"C1": [{
            "frame_id": "000001",
            "stage_recoverability": {
                "schema_version": STAGE_ORACLE_SCHEMA_VERSION,
                "canonical_count_gate_passed": True, "records": [stage],
            },
        }]}
        before = copy.deepcopy((day1, day2, reports))
        result = build_point_retention_day3(day1, day2, reports)
        record = result["delta_records"][0]
        self.assertAlmostEqual(record["signed_deltas"]["O3_minus_O2"], 0.20)
        self.assertAlmostEqual(record["signed_deltas"]["O3_minus_O2_gt_clipped"], 0.15)
        self.assertAlmostEqual(record["signed_deltas"]["O5_minus_O4"], -0.05)
        self.assertEqual((day1, day2, reports), before)


if __name__ == "__main__":
    unittest.main()
