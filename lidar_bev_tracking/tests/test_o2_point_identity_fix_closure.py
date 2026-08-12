import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bev_tracking.o2_point_identity_fix import (
    O2_FIX_SCHEMA_VERSION, build_o2_fix_result,
    rebuild_day3_with_corrected_o2, write_sha256_sidecar,
)
from bev_tracking.point_retention import (
    FRAGMENT_ORACLE_SCHEMA_VERSION, MATERIAL_GAIN_IOU,
    POINT_RETENTION_SCHEMA_VERSION, STAGE_ORACLE_SCHEMA_VERSION,
    build_point_retention_day4,
)


LABELS = [
    "CLUSTER_FRAGMENTATION_LIMITED", "CLUSTER_FORMATION_LIMITED",
    "INTENSITY_FILTER_LIMITED", "Z_FILTER_LIMITED", "ROI_FILTER_LIMITED",
    "RAW_GEOMETRY_OBSERVABILITY_LIMITED", "MIXED", "UNRESOLVED",
]


def oracle(iou, count):
    return {"status": "valid", "num_points": count, "box": {}, "iou": iou}


def ladder(index):
    deltas = {
        "O2_minus_O1": 0.0, "O2_gt_clipped_minus_O2": 0.0,
        "O3_minus_O2": 0.0, "O3_minus_O2_gt_clipped": 0.0,
        "O4_minus_O3": 0.2 if index < 20 else 0.0,
        "O5_minus_O4": 0.2 if index == 20 else 0.0,
        "O6_minus_O5": 0.0,
    }
    if index < 4:
        deltas["O3_minus_O2_gt_clipped"] = 0.2
    raw_iou = 0.1 if index == 21 else 0.5
    return {
        "frame_id": f"{index + 1:06d}", "gt_id": "gt_1",
        "distance_bin": "mid_15_30",
        "stage_point_counts": {"raw": 20, "roi": 20, "z_filter": 15, "intensity_filter": 8},
        "associated_cluster_point_count": 6, "associated_union_point_count": 6,
        "associated_union_gt_clipped_point_count": 6,
        "O1": oracle(0.1, 6), "O2": oracle(0.1, 6),
        "O2_gt_clipped": oracle(0.1, 6), "O3": oracle(0.1, 8),
        "O4": oracle(0.3, 15), "O5": oracle(0.5, 20), "O6": oracle(raw_iou, 20),
        "signed_deltas": deltas,
    }


def day1():
    return {
        "schema_version": POINT_RETENTION_SCHEMA_VERSION, "candidate_variant": "C1",
        "delta_gt_count": 22, "p1_control_gt_count": 0,
        "attribution_contract": {"material_gain_iou": MATERIAL_GAIN_IOU, "labels": LABELS},
    }


def day3():
    return {
        "schema_version": STAGE_ORACLE_SCHEMA_VERSION, "candidate_variant": "C1",
        "canonical_count_gate_passed": True, "delta_gt_count": 22,
        "p1_control_gt_count": 0, "delta_records": [ladder(i) for i in range(22)],
        "p1_control_records": [],
    }


class O2PointIdentityFixClosureTest(unittest.TestCase):
    def test_rebuild_day3_changes_only_o2_dependent_fields(self):
        source = day3()
        fragment = [{
            "frame_id": item["frame_id"], "gt_id": item["gt_id"],
            "associated_cluster_point_count": 7, "associated_union_point_count": 7,
            "associated_union_gt_clipped_point_count": 7,
            "O2": oracle(0.15, 7), "O2_gt_clipped": oracle(0.14, 7),
            "signed_deltas": {"O2_minus_O1": 0.05, "O2_gt_clipped_minus_O2": -0.01},
        } for item in source["delta_records"]]
        corrected_day2 = {
            "schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION,
            "delta_records": fragment, "p1_control_records": [],
        }
        result = rebuild_day3_with_corrected_o2(source, corrected_day2)
        self.assertEqual(result["delta_records"][0]["O1"], source["delta_records"][0]["O1"])
        for name in ("O3", "O4", "O5", "O6"):
            self.assertEqual(result["delta_records"][0][name], source["delta_records"][0][name])
        self.assertEqual(result["delta_records"][0]["O2"]["iou"], 0.15)
        self.assertAlmostEqual(
            result["delta_records"][0]["signed_deltas"]["O3_minus_O2_gt_clipped"], -0.04
        )

    def test_closure_builds_requested_diff_and_preserves_core_result(self):
        frozen_day3 = day3()
        frozen_day4 = build_point_retention_day4(day1(), frozen_day3)
        source = {
            "point_retention_day1": day1(), "point_retention_day3": frozen_day3,
            "point_retention_day4": frozen_day4,
        }
        corrected_day2 = {"schema_version": FRAGMENT_ORACLE_SCHEMA_VERSION}
        result = build_o2_fix_result(source, corrected_day2, frozen_day3)
        self.assertEqual(result["schema_version"], O2_FIX_SCHEMA_VERSION)
        self.assertEqual(len(result["before_after"]["delta_22_O2_iou"]), 22)
        self.assertEqual(
            result["before_after"]["O4_minus_O3_material_count"],
            {"before": 20, "after": 20},
        )
        self.assertEqual(
            len(result["before_after"]["original_cluster_formation_plus_intensity_samples"]), 4
        )
        json.dumps(result)

    def test_sha256_sidecar_identifies_exact_json_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrected.json"
            payload = b'{"result":"ok"}\n'
            path.write_bytes(payload)
            digest, sidecar = write_sha256_sidecar(path)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(sidecar.read_text(encoding="utf-8"), f"{digest}  corrected.json\n")


if __name__ == "__main__":
    unittest.main()
