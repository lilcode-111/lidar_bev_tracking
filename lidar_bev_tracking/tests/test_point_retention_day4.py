import copy
import json
import unittest

from bev_tracking.point_retention import (
    FRAGMENT_ORACLE_SCHEMA_VERSION,
    MATERIAL_GAIN_IOU,
    POINT_RETENTION_SCHEMA_VERSION,
    ROOT_CAUSE_SCHEMA_VERSION,
    STAGE_ORACLE_SCHEMA_VERSION,
    attribute_point_retention_record,
    build_point_retention_day4,
)


DELTA_NAMES = (
    "O2_minus_O1", "O2_gt_clipped_minus_O2", "O3_minus_O2",
    "O3_minus_O2_gt_clipped", "O4_minus_O3", "O5_minus_O4", "O6_minus_O5",
)


def oracle(iou, num_points=10):
    return {
        "status": "valid" if iou is not None else "insufficient_points",
        "num_points": num_points, "box": {} if iou is not None else None, "iou": iou,
    }


def ladder_record(frame_id="000001", gt_id="gt_1", raw_iou=0.40, **deltas):
    signed = {name: 0.0 for name in DELTA_NAMES}
    signed.update(deltas)
    return {
        "frame_id": frame_id, "gt_id": gt_id, "distance_bin": "near_0_15",
        "stage_point_counts": {"raw": 12, "roi": 11, "z_filter": 10, "intensity_filter": 9},
        "associated_cluster_point_count": 6,
        "associated_union_point_count": 6,
        "associated_union_gt_clipped_point_count": 5,
        "O1": oracle(0.10, 6), "O2": oracle(0.10, 6),
        "O2_gt_clipped": oracle(0.10, 5), "O3": oracle(0.10, 9),
        "O4": oracle(0.10, 10), "O5": oracle(0.10, 11), "O6": oracle(raw_iou, 12),
        "signed_deltas": signed,
    }


def day1(delta_count, control_count=0):
    labels = [
        "CLUSTER_FRAGMENTATION_LIMITED", "CLUSTER_FORMATION_LIMITED",
        "INTENSITY_FILTER_LIMITED", "Z_FILTER_LIMITED", "ROI_FILTER_LIMITED",
        "RAW_GEOMETRY_OBSERVABILITY_LIMITED", "MIXED", "UNRESOLVED",
    ]
    return {
        "schema_version": POINT_RETENTION_SCHEMA_VERSION,
        "candidate_variant": "C1", "delta_gt_count": delta_count,
        "p1_control_gt_count": control_count,
        "attribution_contract": {"material_gain_iou": MATERIAL_GAIN_IOU, "labels": labels},
    }


def day3(delta_records, control_records=None):
    control_records = control_records or []
    return {
        "schema_version": STAGE_ORACLE_SCHEMA_VERSION,
        "source_day2_schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
        "candidate_variant": "C1", "delta_gt_count": len(delta_records),
        "p1_control_gt_count": len(control_records),
        "delta_records": delta_records, "p1_control_records": control_records,
    }


class PointRetentionDay4Test(unittest.TestCase):
    def test_each_material_stage_signal_maps_to_preregistered_label(self):
        cases = {
            "O2_minus_O1": "CLUSTER_FRAGMENTATION_LIMITED",
            "O2_gt_clipped_minus_O2": "CLUSTER_FORMATION_LIMITED",
            "O3_minus_O2_gt_clipped": "CLUSTER_FORMATION_LIMITED",
            "O4_minus_O3": "INTENSITY_FILTER_LIMITED",
            "O5_minus_O4": "Z_FILTER_LIMITED",
            "O6_minus_O5": "ROI_FILTER_LIMITED",
        }
        for delta_name, expected in cases.items():
            with self.subTest(delta_name=delta_name):
                result = attribute_point_retention_record(
                    ladder_record(**{delta_name: MATERIAL_GAIN_IOU})
                )
                self.assertEqual(result["root_cause"], expected)
                self.assertEqual(result["material_signals"], [expected])

    def test_multiple_distinct_stage_signals_are_mixed(self):
        result = attribute_point_retention_record(ladder_record(
            O2_minus_O1=0.11, O4_minus_O3=0.12,
        ))
        self.assertEqual(result["root_cause"], "MIXED")
        self.assertEqual(len(result["material_signals"]), 2)

    def test_raw_limit_and_unresolved_use_frozen_point25_boundary(self):
        limited = attribute_point_retention_record(ladder_record(raw_iou=0.249999))
        boundary = attribute_point_retention_record(ladder_record(raw_iou=0.25))
        missing = ladder_record(raw_iou=None)
        self.assertEqual(limited["root_cause"], "RAW_GEOMETRY_OBSERVABILITY_LIMITED")
        self.assertEqual(boundary["root_cause"], "UNRESOLVED")
        self.assertEqual(
            attribute_point_retention_record(missing)["root_cause"],
            "RAW_GEOMETRY_OBSERVABILITY_LIMITED",
        )

    def test_single_stage_signal_takes_precedence_over_low_raw_iou(self):
        result = attribute_point_retention_record(ladder_record(
            raw_iou=0.10, O4_minus_O3=0.15,
        ))
        self.assertEqual(result["root_cause"], "INTENSITY_FILTER_LIMITED")
        self.assertFalse(result["raw_recoverable"])

    def test_day4_aggregates_delta_and_control_without_mutating_day3(self):
        delta_records = [
            ladder_record("000001", "gt_1", O4_minus_O3=0.15),
            ladder_record("000002", "gt_2", O4_minus_O3=0.10),
            ladder_record("000003", "gt_3", raw_iou=0.10, O5_minus_O4=-0.05),
        ]
        controls = [ladder_record("000004", "gt_4")]
        source = day3(delta_records, controls)
        before = copy.deepcopy(source)
        result = build_point_retention_day4(day1(3, 1), source)
        summary = result["delta_summary"]
        self.assertEqual(result["schema_version"], ROOT_CAUSE_SCHEMA_VERSION)
        self.assertEqual(summary["root_cause_counts"]["INTENSITY_FILTER_LIMITED"], 2)
        self.assertEqual(summary["root_cause_counts"]["RAW_GEOMETRY_OBSERVABILITY_LIMITED"], 1)
        self.assertEqual(summary["dominant_root_cause"], "INTENSITY_FILTER_LIMITED")
        self.assertTrue(summary["majority_reached"])
        self.assertEqual(summary["signed_deltas"]["O4_minus_O3"]["material_gain_count"], 2)
        self.assertEqual(summary["signed_deltas"]["O5_minus_O4"]["negative_count"], 1)
        self.assertEqual(summary["point_counts"]["raw"]["median"], 12.0)
        self.assertEqual(source, before)
        json.dumps(result)

    def test_day4_rejects_cohort_count_drift(self):
        with self.assertRaisesRegex(ValueError, "frozen Day 1 cohort"):
            build_point_retention_day4(day1(2), day3([ladder_record()]))


if __name__ == "__main__":
    unittest.main()
