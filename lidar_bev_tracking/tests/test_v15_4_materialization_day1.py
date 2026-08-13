import copy
from pathlib import Path
import unittest

from bev_tracking.v15_4_materialization import (
    V154MaterializationError,
    canonical_identity_sha256,
    load_json,
    validate_day1_materialization,
    validate_pre_run_identity,
    validate_threshold_schedule,
)


SCHEDULE_PATH = Path("configs/experiments/v15_4/threshold_schedule.json")
IDENTITY_PATH = Path("configs/experiments/v15_4/pre_run_identity.json")


class V154MaterializationDay1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = load_json(SCHEDULE_PATH)
        cls.identity = load_json(IDENTITY_PATH)

    def test_frozen_day1_material_is_valid(self):
        result = validate_day1_materialization(SCHEDULE_PATH, IDENTITY_PATH)
        self.assertEqual(result["pre_run_materialization_day1"], "COMPLETE")
        self.assertFalse(result["formal_run_authorized"])

    def test_thresholds_and_release_eligibility_are_frozen(self):
        result = validate_threshold_schedule(self.schedule)
        self.assertEqual(result["variant_count"], 4)
        self.assertEqual(
            [self.schedule["variants"][name]["intensity_min"] for name in ("T0", "T1", "T2", "T_off")],
            [0.38, 0.30, 0.15, 0.00],
        )
        self.assertFalse(self.schedule["variants"]["T_off"]["release_candidate_eligible"])

    def test_only_intensity_min_is_an_allowed_config_difference(self):
        self.assertEqual(self.schedule["allowed_config_diff_paths"], ["detector.intensity_min"])
        changed = copy.deepcopy(self.schedule)
        changed["allowed_config_diff_paths"].append("detector.z_min")
        with self.assertRaises(V154MaterializationError):
            validate_threshold_schedule(changed)

    def test_manifest_and_delta_identities_are_frozen(self):
        result = validate_pre_run_identity(self.identity)
        self.assertEqual(result["diagnostic_num_frames"], 25)
        self.assertEqual(result["formal_num_frames"], 100)
        self.assertEqual(result["delta_22_count"], 22)
        delta = self.identity["delta_22"]
        self.assertEqual(canonical_identity_sha256(delta["ordered_identity_list"]), delta["ordered_identity_sha256"])

    def test_delta_cohort_cannot_be_rederived_at_runtime(self):
        self.assertFalse(self.identity["delta_22"]["runtime_rederivation_allowed"])

    def test_hash_modes_are_explicit(self):
        contract = self.identity["hash_contract"]
        self.assertEqual(contract["manifest"], "canonical_utf8_lf")
        self.assertEqual(contract["ordered_identity"], "canonical_compact_json_utf8")
        self.assertEqual(contract["json_artifact"], "raw_file_bytes")

    def test_missing_historical_identity_evidence_is_not_fabricated(self):
        t0_25 = self.identity["t0_25_reference"]
        t0_100 = self.identity["t0_100_reference"]
        self.assertFalse(t0_25["historical_fields_available"]["post_intensity_source_point_index_set"])
        self.assertFalse(t0_100["historical_fields_available"]["candidate_identity_set"])
        self.assertTrue(t0_25["reference_status"].startswith("REQUIRES_"))
        self.assertTrue(t0_100["reference_status"].startswith("REQUIRES_"))

    def test_day1_does_not_authorize_formal_or_non_t0_runs(self):
        state = self.identity["phase_1_state"]
        self.assertFalse(state["t0_exact_reference_complete"])
        self.assertFalse(state["formal_run_authorized"])
        self.assertFalse(state["non_t0_variant_run_allowed"])


if __name__ == "__main__":
    unittest.main()
