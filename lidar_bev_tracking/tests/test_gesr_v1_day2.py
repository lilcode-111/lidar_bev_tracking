from dataclasses import FrozenInstanceError
import inspect
import unittest

import numpy as np

from bev_tracking.gesr_v1 import (
    GESRV1Error,
    build_gesr_v1_evidence,
    run_gesr_v1_reference,
)


def _point(x, y, intensity, z=0.0):
    return [x, y, z, intensity]


def _two_component_cloud():
    return np.asarray([
        _point(-0.55, -0.05, 0.50), _point(-0.55, 0.05, 0.50),
        _point(-0.50, -0.05, 0.50), _point(-0.50, 0.05, 0.50),
        _point(0.50, -0.05, 0.50), _point(0.50, 0.05, 0.50),
        _point(0.55, -0.05, 0.50), _point(0.55, 0.05, 0.50),
        _point(0.0, 0.0, 0.20),
    ]), np.asarray([1, 2, 3, 4, 10, 11, 12, 13, 50])


class GESRV1Day2Test(unittest.TestCase):
    def test_all_frozen_terminal_reasons(self):
        accepted_points, accepted_indices = _two_component_cloud()
        accepted = run_gesr_v1_reference("1", accepted_points, accepted_indices)
        self.assertEqual(accepted.candidate_decisions[0].point_terminal_decision, "ACCEPTED")

        line_seed = np.asarray([
            _point(0.0, 0.0, 0.5), _point(0.1, 0.0, 0.5),
            _point(0.2, 0.0, 0.5), _point(0.3, 0.0, 0.5),
        ])
        inside = run_gesr_v1_reference(
            "1", np.vstack((line_seed, _point(0.15, 0.0, 0.2))), [1, 2, 3, 4, 20]
        )
        self.assertEqual(inside.candidate_decisions[0].point_terminal_decision, "INSIDE_CURRENT_EXTENT")

        insufficient = run_gesr_v1_reference(
            "1", np.vstack((line_seed, _point(0.85, 0.0, 0.2))), [1, 2, 3, 4, 20]
        )
        self.assertEqual(
            insufficient.candidate_decisions[0].point_terminal_decision,
            "INSUFFICIENT_DIRECT_ANCHORS",
        )

        absent = run_gesr_v1_reference(
            "1", np.vstack((line_seed, _point(5.0, 0.0, 0.2))), [1, 2, 3, 4, 20]
        )
        self.assertEqual(absent.candidate_decisions[0].point_terminal_decision, "NO_VALID_COMPONENT")

        degenerate_seed = np.asarray([_point(0.0, 0.0, 0.5)] * 4)
        invalid = run_gesr_v1_reference(
            "1", np.vstack((degenerate_seed, _point(0.1, 0.0, 0.2))), [1, 2, 3, 4, 20]
        )
        self.assertEqual(
            invalid.candidate_decisions[0].point_terminal_decision,
            "GEOMETRY_EXTENSION_INVALID",
        )

    def test_association_outcomes_are_separate_from_terminal_reason(self):
        points, indices = _two_component_cloud()
        result = run_gesr_v1_reference("1", points, indices)
        decision = result.candidate_decisions[0]
        outcomes = [item.arbitration_outcome for item in decision.associations]
        self.assertEqual(outcomes.count("SELECTED"), 1)
        self.assertEqual(outcomes.count("MULTI_COMPONENT_LOST"), 1)
        self.assertEqual(decision.point_terminal_decision, "ACCEPTED")

    def test_ineligible_association_is_not_multi_component_lost(self):
        points, indices = _two_component_cloud()
        points[-1, :2] = (-1.05, 0.0)
        result = run_gesr_v1_reference("1", points, indices)
        decision = result.candidate_decisions[0]
        ineligible = [item for item in decision.associations if not item.eligible]
        self.assertTrue(ineligible)
        self.assertTrue(all(item.arbitration_outcome is None for item in ineligible))
        self.assertTrue(all(item.arbitration_status == "not_applicable" for item in ineligible))

    def test_reason_attribution_on_off_preserves_runtime_result(self):
        points, indices = _two_component_cloud()
        enabled = run_gesr_v1_reference("1", points, indices, reason_attribution=True)
        disabled = run_gesr_v1_reference("1", points, indices, reason_attribution=False)
        self.assertEqual(enabled.accepted_source_indices, disabled.accepted_source_indices)
        self.assertEqual(enabled.expanded_source_indices, disabled.expanded_source_indices)
        self.assertEqual(
            [item.selected_component_runtime_id for item in enabled.candidate_decisions],
            [item.selected_component_runtime_id for item in disabled.candidate_decisions],
        )
        self.assertTrue(all(item.point_terminal_decision is None for item in disabled.candidate_decisions))

    def test_runtime_result_is_immutable(self):
        points, indices = _two_component_cloud()
        result = run_gesr_v1_reference("1", points, indices)
        with self.assertRaises(FrozenInstanceError):
            result.frame_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.candidate_decisions[0].accepted = False

    def test_evidence_builder_resolves_frozen_invariants(self):
        points, indices = _two_component_cloud()
        evidence = build_gesr_v1_evidence(run_gesr_v1_reference("1", points, indices))
        invariants = evidence["invariants"]
        self.assertTrue(invariants["accepted_iff_terminal_ACCEPTED"])
        self.assertTrue(invariants["exactly_one_SELECTED_per_accepted_candidate"])
        self.assertEqual(invariants["rejected_candidate_SELECTED_count"], 0)
        self.assertTrue(invariants["accepted_subset_of_candidate_universe"])
        self.assertTrue(invariants["expanded_equals_seed_union_accepted"])
        self.assertEqual(evidence["arbitration_metrics"]["selected_association_count"], 1)
        self.assertEqual(evidence["arbitration_metrics"]["lost_association_count"], 1)
        self.assertFalse(evidence["GT_runtime_input"])
        self.assertFalse(evidence["formal_result"])

    def test_evidence_requires_attribution_but_cannot_change_runtime(self):
        points, indices = _two_component_cloud()
        result = run_gesr_v1_reference("1", points, indices, reason_attribution=False)
        with self.assertRaisesRegex(GESRV1Error, "attribution must be enabled"):
            build_gesr_v1_evidence(result)

    def test_runtime_and_evidence_signatures_accept_no_gt(self):
        runtime_parameters = inspect.signature(run_gesr_v1_reference).parameters
        evidence_parameters = inspect.signature(build_gesr_v1_evidence).parameters
        forbidden = {"gt", "labels", "evaluation", "oracle", "failure_evidence"}
        self.assertTrue(forbidden.isdisjoint(runtime_parameters))
        self.assertTrue(forbidden.isdisjoint(evidence_parameters))

    def test_gt_metadata_metamorphic_variants_cannot_reach_runtime(self):
        points, indices = _two_component_cloud()
        metadata_variants = (
            {"gt": "normal"},
            {},
            {"gt": "order_changed"},
            {"gt": "content_perturbed"},
        )
        results = []
        for metadata in metadata_variants:
            self.assertIsInstance(metadata, dict)
            result = run_gesr_v1_reference("1", points, indices)
            results.append(
                (
                    result.accepted_source_indices,
                    tuple(component.signature for component in result.components),
                    tuple(
                        (
                            item.source_index,
                            item.accepted,
                            item.selected_component_signature,
                            item.point_terminal_decision,
                        )
                        for item in result.candidate_decisions
                    ),
                )
            )
        self.assertTrue(all(item == results[0] for item in results[1:]))


if __name__ == "__main__":
    unittest.main()
