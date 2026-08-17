from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from bev_tracking.v15_4_materialization import raw_file_sha256
from bev_tracking.v15_5_seed_support import (
    BANDS,
    V155SeedSupportError,
    build_frame_stability,
    build_selectivity_record,
    load_verified_metric_cache,
    segment_masks,
    summarize_metric_values,
    validate_day1_for_day2,
)


class V155Phase0Day2Test(unittest.TestCase):
    def test_range_masks_freeze_15m_and_30m_boundaries(self):
        masks = segment_masks(np.asarray([0.0, 14.999, 15.0, 29.999, 30.0]))
        self.assertEqual(masks["near"].tolist(), [True, True, False, False, False])
        self.assertEqual(masks["mid"].tolist(), [False, False, True, True, False])
        self.assertEqual(masks["far"].tolist(), [False, False, False, False, True])

    def test_summary_reports_required_percentiles_and_support_ratio(self):
        result = summarize_metric_values({
            "d_seed": [np.asarray([0.1, 0.2, 0.7, 1.0])],
            "n_seed_0p6": [np.asarray([2, 1, 0, 0])],
        })
        self.assertEqual(result["point_count"], 4)
        self.assertEqual(result["seed_supported_count"], 2)
        self.assertEqual(result["seed_supported_ratio"], 0.5)
        self.assertEqual(set(result["d_seed_percentiles_m"]), {"P25", "P50", "P75", "P90"})
        self.assertEqual(result["n_seed_0p6_explanation"]["mean"], 0.75)

    def test_empty_stratum_is_null_not_zero_ratio(self):
        result = summarize_metric_values({"d_seed": [], "n_seed_0p6": []})
        self.assertEqual(result["point_count"], 0)
        self.assertIsNone(result["seed_supported_ratio"])
        self.assertTrue(all(value is None for value in result["d_seed_percentiles_m"].values()))

    def test_infinite_no_seed_distance_is_json_safe_null(self):
        result = summarize_metric_values({
            "d_seed": [np.asarray([np.inf])],
            "n_seed_0p6": [np.asarray([0])],
        })
        self.assertEqual(result["no_finite_seed_count"], 1)
        self.assertTrue(all(value is None for value in result["d_seed_percentiles_m"].values()))

    def test_selectivity_keeps_vrr_and_brr_separate(self):
        vehicle = {"seed_supported_ratio": 0.7}
        background = {"seed_supported_ratio": 0.2}
        result = build_selectivity_record(vehicle, background)
        self.assertEqual(result["VRR"], 0.7)
        self.assertEqual(result["BRR"], 0.2)
        self.assertAlmostEqual(result["VRR_minus_BRR"], 0.5)
        self.assertAlmostEqual(result["VRR_div_BRR"], 3.5)

    def test_frame_stability_uses_h_plus_m_and_reports_concentration(self):
        frames = {
            "000001": self._frame_counts(vehicle=(10, 8), background=(20, 10)),
            "000002": self._frame_counts(vehicle=(10, 2), background=(20, 10)),
        }
        records, summary = build_frame_stability(frames)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["frames_VRR_gt_BRR"], 1)
        self.assertEqual(summary["frames_VRR_lt_BRR"], 1)
        self.assertEqual(summary["max_frame_vehicle_rescue_share"], 0.8)
        self.assertEqual(summary["top5_frame_vehicle_rescue_share"], 1.0)

    def test_cache_hash_and_source_identity_are_verified(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "metrics.npz"
            np.savez_compressed(
                path,
                source_point_indices=np.asarray([10, 11]),
                range_xy=np.asarray([5.0, 20.0]),
                d_seed=np.asarray([0.2, 0.8]),
                n_seed_0p6=np.asarray([1, 0]),
            )
            record = {"path": "metrics.npz", "sha256": raw_file_sha256(path)}
            loaded = load_verified_metric_cache(root, record)
            self.assertEqual(loaded["source_point_indices"].tolist(), [10, 11])
            record["sha256"] = "0" * 64
            with self.assertRaisesRegex(V155SeedSupportError, "hash mismatch"):
                load_verified_metric_cache(root, record)

    def test_day2_rejects_radius_search_or_gt_leakage(self):
        day1 = {
            "schema_version": "15.5-phase0-seed-support-day1-v1",
            "day1_complete": True,
            "identity_validation": {"status": "PASS", "source_point_monotonicity": "PASS"},
            "contracts": {
                "point_identity": "(frame_id, raw_lidar_point_index)",
                "coordinate_row_dedup_used": False,
                "r_seed_m": 0.60,
                "r_seed_search_allowed": True,
                "gt_oracle_leakage": False,
            },
            "formal_pipeline_rerun": False,
        }
        with self.assertRaisesRegex(V155SeedSupportError, "r_seed"):
            validate_day1_for_day2(day1)

    @staticmethod
    def _frame_counts(vehicle, background):
        def split(total):
            count, rescued = total
            return {
                "H": {"point_count": count // 2, "seed_supported_count": rescued // 2},
                "M": {"point_count": count - count // 2, "seed_supported_count": rescued - rescued // 2},
                "L": {"point_count": 0, "seed_supported_count": 0},
            }
        vehicle_bands, background_bands = split(vehicle), split(background)
        return {
            "frame_id": "unused",
            "by_band": {
                band: {
                    "vehicle": vehicle_bands[band],
                    "background": background_bands[band],
                }
                for band in BANDS
            },
        }


if __name__ == "__main__":
    unittest.main()
