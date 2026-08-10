import copy
import unittest

from bev_tracking.cluster_separability import (
    CLUSTER_FEATURE_SCHEMA_VERSION,
    CLUSTER_GROUP_SCHEMA_VERSION,
    build_cluster_separability_day3,
)


def features(value, associated=True, oracle=None):
    return {
        "num_points": value,
        "axis_length": float(value),
        "axis_width": 1.0,
        "height_span": 0.5,
        "point_density_xy": 2.0,
        "pca_length": float(value),
        "pca_width": 1.0,
        "target_gt_id": "gt_1" if associated else None,
        "gt_coverage_ratio": 0.5 if associated else None,
        "cluster_purity_ratio": 0.75 if associated else None,
        "pca_iou_to_target_gt": 0.1 if associated else None,
        "oracle_car_iou": oracle,
    }


def record(cluster_id, group, frame_id="000001", distance_bin="mid_15_30", strict=False, delta=False):
    associated = group in {"P1", "P2"}
    return {
        "frame_id": frame_id,
        "variant": "C1",
        "cluster_id": cluster_id,
        "group": group,
        "cluster_distance_bin": distance_bin,
        "associated_gt_ids": ["gt_1"] if associated else [],
        "is_strict_background": strict,
        "is_delta_22": delta,
        "delta_22_gt_ids": ["gt_1"] if delta else [],
        "feature_schema_version": CLUSTER_FEATURE_SCHEMA_VERSION,
        "features": features(10 if group == "P2" else 20, associated, 0.1 if group == "P2" else None),
    }


class ClusterSeparabilityDay3Test(unittest.TestCase):
    def test_day3_matches_only_strict_background_in_same_frame_and_bin(self):
        records = [
            record("p1", "P1"),
            record("p2", "P2", delta=True),
            record("n_match", "N", strict=True),
            record("n_labeled", "N", strict=False),
            record("n_wrong_frame", "N", frame_id="000002", strict=True),
            record("n_wrong_bin", "N", distance_bin="far_30_inf", strict=True),
        ]
        grouped = {
            "schema_version": CLUSTER_GROUP_SCHEMA_VERSION,
            "candidate_variant": "C1",
            "records_by_variant": {"C1": records},
        }
        featured = {
            "schema_version": CLUSTER_FEATURE_SCHEMA_VERSION,
            "candidate_variant": "C1",
            "candidate_p2_oracle": {"count": 1, "iou_ge_0_25_count": 0, "iou_ge_0_50_count": 0, "mean": 0.1, "max": 0.1},
            "delta_22_p2_oracle": {"count": 1, "iou_ge_0_25_count": 0, "iou_ge_0_50_count": 0, "mean": 0.1, "max": 0.1},
        }
        before_grouped = copy.deepcopy(grouped)
        before_featured = copy.deepcopy(featured)
        result = build_cluster_separability_day3(grouped, featured)
        self.assertEqual(result["set_counts"], {
            "P1": 1, "P2": 1, "N": 1, "delta_22_P2": 1, "delta_22_N": 1,
        })
        self.assertEqual(
            result["delta_22_context"]["P2"]["total"]["features"]["num_points"]["median"],
            10.0,
        )
        self.assertEqual(len(result["delta_22_records"]), 1)
        self.assertEqual(grouped, before_grouped)
        self.assertEqual(featured, before_featured)

    def test_day3_requires_completed_day1_and_day2_inputs(self):
        with self.assertRaisesRegex(ValueError, "valid Day 1"):
            build_cluster_separability_day3({}, {})
        with self.assertRaisesRegex(ValueError, "valid Day 2"):
            build_cluster_separability_day3(
                {"schema_version": CLUSTER_GROUP_SCHEMA_VERSION}, {}
            )


if __name__ == "__main__":
    unittest.main()
