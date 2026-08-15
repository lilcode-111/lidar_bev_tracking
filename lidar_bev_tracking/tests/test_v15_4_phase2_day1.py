import unittest
from unittest.mock import patch

from bev_tracking.v15_4_authorization import V154AuthorizationError, build_formal_run_authorization, validate_formal_run_authorization, validate_t0_reference_registry
from bev_tracking.v15_4_formal import apply_t0_replay_result, build_formal_run_plan
from bev_tracking.v15_4_materialization import load_json


class V154Phase2Day1Test(unittest.TestCase):
    def test_authorization_requires_clean_full_commit(self):
        with self.assertRaises(V154AuthorizationError):
            build_formal_run_authorization(repo_root=".", commit="a" * 40, working_tree_clean=False, schedule_path="x", gate_path="x", identity_path="x", reference_registry_path="x")

    def test_authorization_binds_materialized_t0_hashes(self):
        registry = {
            "schema_version": "15.4-t0-reference-index-v1",
            "status": "MATERIALIZED",
            "formal_variants_run": ["T0"],
            "non_t0_results_observed": False,
            "artifacts": {
                "t0_25_reference": {"path": "t0_25.json", "sha256": "a" * 64},
                "t0_100_reference": {"path": "t0_100.json", "sha256": "b" * 64},
            },
        }
        with patch("bev_tracking.v15_4_authorization.raw_file_sha256", side_effect=["a" * 64, "b" * 64]):
            validate_t0_reference_registry(".", registry)
        registry["artifacts"]["t0_25_reference"]["sha256"] = "0" * 64
        with patch("bev_tracking.v15_4_authorization.raw_file_sha256", return_value="a" * 64):
            with self.assertRaises(V154AuthorizationError):
                validate_t0_reference_registry(".", registry)

    def test_formal_plan_is_t0_first_and_blocks_non_t0_interpretation(self):
        schedule = load_json("configs/experiments/v15_4/threshold_schedule.json")
        base = {"detector": {"intensity_min": 0.38}}
        authorization = {"formal_run_authorized": True, "formal_comparison_commit": "a" * 40}
        plan = build_formal_run_plan(schedule, base, authorization)
        self.assertEqual(plan["execution_order"], ["T0", "T1", "T2", "T_off"])
        self.assertFalse(plan["non_t0_interpretation_allowed"])
        self.assertEqual([plan["variants"][name]["intensity_min"] for name in plan["execution_order"]], [0.38, 0.30, 0.15, 0.0])

    def test_non_t0_opens_only_after_both_t0_replays_pass(self):
        plan = {"t0_replay_gate": {}, "non_t0_interpretation_allowed": False}
        self.assertFalse(apply_t0_replay_result(plan, replay_25_passed=True, replay_100_passed=False)["non_t0_interpretation_allowed"])
        self.assertTrue(apply_t0_replay_result(plan, replay_25_passed=True, replay_100_passed=True)["non_t0_interpretation_allowed"])

    def test_authorization_rejects_commit_or_dirty_tree_change(self):
        authorization = {"schema_version": "15.4-formal-run-authorization-v1", "status": "AUTHORIZED", "formal_run_authorized": True, "formal_comparison_commit": "a" * 40, "source": {}}
        with self.assertRaises(V154AuthorizationError):
            validate_formal_run_authorization(authorization, current_commit="b" * 40, working_tree_clean=True)
        with self.assertRaises(V154AuthorizationError):
            validate_formal_run_authorization(authorization, current_commit="a" * 40, working_tree_clean=False)


if __name__ == "__main__":
    unittest.main()
