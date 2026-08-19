import unittest
from pathlib import Path

from bev_tracking.gesr_v1_preregistration import (
    ASSOCIATION_OUTCOMES,
    TERMINAL_CODES,
    build_preregistration_payloads,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_preregistration_artifacts,
)


class GESRV1PreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.payloads = build_preregistration_payloads(cls.root)

    def test_repository_artifacts_match_frozen_source_of_truth(self):
        result = validate_preregistration_artifacts(self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["all_SHA_bindings_verified"])

    def test_canonical_hash_sorts_object_keys_and_preserves_array_order(self):
        left = {"b": 2, "a": ["first", "second"]}
        right = {"a": ["first", "second"], "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(canonical_json_sha256(left), canonical_json_sha256(right))
        reversed_array = {"a": ["second", "first"], "b": 2}
        self.assertNotEqual(canonical_json_sha256(left), canonical_json_sha256(reversed_array))

    def test_gate0_has_two_phases_and_is_not_evaluated(self):
        gate0 = self.payloads["gesr_v1_evaluation_spec.json"]["Gate0"]
        self.assertEqual(gate0["formula"], "Gate0_Phase2 AND Gate0_Phase3")
        self.assertEqual(gate0["Gate0_Phase2"]["initial_status"], "not_evaluated")
        self.assertEqual(gate0["Gate0_Phase3"]["initial_status"], "not_evaluated")
        self.assertEqual(gate0["initial_Final_Gate0"], "not_evaluated")

    def test_point_and_association_semantics_are_separate(self):
        evaluation = self.payloads["gesr_v1_evaluation_spec.json"]
        terminal = evaluation["point_terminal_decision_policy"]
        association = evaluation["association_arbitration"]
        self.assertEqual(terminal["point_terminal_decision_codes"], list(TERMINAL_CODES))
        self.assertEqual(terminal["terminal_reason_precedence"], list(TERMINAL_CODES))
        self.assertEqual(association["outcomes"], list(ASSOCIATION_OUTCOMES))
        self.assertNotIn("MULTI_COMPONENT_LOST", terminal["point_terminal_decision_codes"])
        self.assertFalse(association["MULTI_COMPONENT_LOST_is_point_reject_reason"])

    def test_registration_contains_no_results_or_runtime_authorization(self):
        state = self.payloads["comparison_identity.json"]["registration_state"]
        self.assertFalse(state["results_observed_at_registration"])
        self.assertFalse(state["GESR_runtime_implemented"])
        self.assertFalse(state["GESR_25_frame_treatment_run"])
        self.assertFalse(state["GESR_100_frame_treatment_run"])
        self.assertFalse(state["GESR_formal_result_generated"])
        self.assertEqual(state["registered_treatment_result_artifacts"], [])

    def test_release_selector_cannot_select_T2(self):
        comparison = self.payloads["comparison_identity.json"]
        release = self.payloads["gesr_v1_release_gate.json"]
        self.assertEqual(comparison["release_selector_allowed_output"], ["GESR-v1", None])
        self.assertFalse(comparison["comparisons"]["T2"]["release_candidate_eligible"])
        self.assertEqual(release["selection_policy"]["allowed_output"], ["GESR-v1", None])


if __name__ == "__main__":
    unittest.main()
