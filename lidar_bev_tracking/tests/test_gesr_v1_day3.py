from dataclasses import replace
import inspect
import math
import unittest

import numpy as np

from bev_tracking.gesr_v1 import (
    GESRV1Error,
    build_seed_components_optimized,
    build_seed_components_reference,
    run_gesr_v1_optimized,
    run_gesr_v1_reference,
    validate_gesr_v1_semantic_equivalence,
)


def _point(x, y, intensity, z=0.0):
    return [x, y, z, intensity]


def _assert_full_equivalence(test_case, points, indices, reason_attribution=True):
    reference = run_gesr_v1_reference(
        "1", points, indices, reason_attribution=reason_attribution
    )
    optimized = run_gesr_v1_optimized(
        "1", points, indices, reason_attribution=reason_attribution
    )
    test_case.assertEqual(reference, optimized)
    audit = validate_gesr_v1_semantic_equivalence(reference, optimized)
    test_case.assertEqual(audit["status"], "PASS")
    test_case.assertTrue(audit["full_runtime_result_equal"])
    return optimized


def _tie_cloud():
    return np.asarray([
        _point(-0.55, -0.05, 0.50), _point(-0.55, 0.05, 0.50),
        _point(-0.50, -0.05, 0.50), _point(-0.50, 0.05, 0.50),
        _point(0.50, -0.05, 0.50), _point(0.50, 0.05, 0.50),
        _point(0.55, -0.05, 0.50), _point(0.55, 0.05, 0.50),
        _point(0.0, 0.0, 0.20),
    ]), np.asarray([1, 2, 3, 4, 10, 11, 12, 13, 50])


class GESRV1Day3Test(unittest.TestCase):
    def test_positive_grid_boundary_does_not_split_component(self):
        points = np.asarray([
            _point(0.40, 0.0, 0.5), _point(0.59, 0.0, 0.5),
            _point(0.61, 0.0, 0.5), _point(0.80, 0.0, 0.5),
        ])
        reference = build_seed_components_reference(points, [1, 2, 3, 4])
        optimized = build_seed_components_optimized(points, [1, 2, 3, 4])
        self.assertEqual(reference, optimized)
        self.assertEqual(len(optimized), 1)

    def test_negative_grid_boundary_does_not_split_component(self):
        points = np.asarray([
            _point(-0.80, 0.0, 0.5), _point(-0.61, 0.0, 0.5),
            _point(-0.59, 0.0, 0.5), _point(-0.40, 0.0, 0.5),
        ])
        reference = build_seed_components_reference(points, [1, 2, 3, 4])
        optimized = build_seed_components_optimized(points, [1, 2, 3, 4])
        self.assertEqual(reference, optimized)
        self.assertEqual(len(optimized), 1)

    def test_grid_coarse_candidates_still_receive_exact_distance_filter(self):
        points = np.asarray([
            _point(0.01, 0.01, 0.5),
            _point(0.61, 0.61, 0.5),
        ])
        reference = build_seed_components_reference(points, [1, 2])
        optimized = build_seed_components_optimized(points, [1, 2])
        self.assertEqual(reference, optimized)
        self.assertEqual(len(optimized), 2)

    def test_exact_radius_and_just_above_radius_match_reference(self):
        exact = np.asarray([_point(0.0, 0.0, 0.5), _point(0.6, 0.0, 0.5)])
        above = exact.copy()
        above[1, 0] = math.nextafter(0.6, math.inf)
        self.assertEqual(
            build_seed_components_reference(exact, [1, 2]),
            build_seed_components_optimized(exact, [1, 2]),
        )
        self.assertEqual(
            build_seed_components_reference(above, [1, 2]),
            build_seed_components_optimized(above, [1, 2]),
        )
        self.assertEqual(len(build_seed_components_optimized(exact, [1, 2])), 1)
        self.assertEqual(len(build_seed_components_optimized(above, [1, 2])), 2)

    def test_exact_arbitration_tie_matches_reference(self):
        points, indices = _tie_cloud()
        result = _assert_full_equivalence(self, points, indices)
        self.assertEqual(result.candidate_decisions[0].selected_component_runtime_id, 1)

    def test_reason_attribution_off_matches_reference(self):
        points, indices = _tie_cloud()
        result = _assert_full_equivalence(
            self, points, indices, reason_attribution=False
        )
        self.assertIsNone(result.candidate_decisions[0].point_terminal_decision)

    def test_input_reorder_preserves_optimized_result(self):
        points, indices = _tie_cloud()
        permutation = np.asarray([8, 3, 0, 7, 4, 2, 6, 1, 5])
        original = run_gesr_v1_optimized("1", points, indices)
        reordered = run_gesr_v1_optimized(
            "1", points[permutation], indices[permutation]
        )
        self.assertEqual(original, reordered)

    def test_repeated_optimized_run_is_identical(self):
        points, indices = _tie_cloud()
        self.assertEqual(
            run_gesr_v1_optimized("1", points, indices),
            run_gesr_v1_optimized("1", points, indices),
        )

    def test_same_coordinate_different_identity_is_preserved(self):
        points, indices = _tie_cloud()
        points = np.vstack((points, points[-1]))
        indices = np.append(indices, 51)
        result = _assert_full_equivalence(self, points, indices)
        self.assertIn(50, result.candidate_source_indices)
        self.assertIn(51, result.candidate_source_indices)

    def test_structured_fixed_random_corpus_matches_exactly(self):
        rng = np.random.default_rng(20260820)
        for case in range(40):
            rows = []
            source_indices = []
            next_index = case * 1000
            centers = rng.uniform(-8.0, 8.0, size=(3, 2))
            for center in centers:
                for _ in range(7):
                    xy = center + rng.normal(0.0, 0.12, size=2)
                    rows.append(_point(xy[0], xy[1], rng.uniform(0.38, 0.8)))
                    source_indices.append(next_index)
                    next_index += 1
                for _ in range(9):
                    xy = center + rng.uniform(-0.85, 0.85, size=2)
                    rows.append(_point(xy[0], xy[1], rng.uniform(0.15, 0.38)))
                    source_indices.append(next_index)
                    next_index += 1
            for _ in range(8):
                xy = rng.uniform(-10.0, 10.0, size=2)
                rows.append(_point(xy[0], xy[1], rng.uniform(0.0, 0.15)))
                source_indices.append(next_index)
                next_index += 1
            points = np.asarray(rows, dtype=np.float64)
            indices = np.asarray(source_indices, dtype=np.int64)
            permutation = rng.permutation(len(points))
            with self.subTest(case=case):
                _assert_full_equivalence(
                    self, points[permutation], indices[permutation]
                )

    def test_equivalence_validator_rejects_any_identity_difference(self):
        points, indices = _tie_cloud()
        reference = run_gesr_v1_reference("1", points, indices)
        altered = replace(reference, accepted_source_indices=())
        with self.assertRaisesRegex(GESRV1Error, "accepted_source_indices"):
            validate_gesr_v1_semantic_equivalence(reference, altered)

    def test_optimized_runtime_signature_contains_no_gt(self):
        parameters = inspect.signature(run_gesr_v1_optimized).parameters
        forbidden = {"gt", "labels", "evaluation", "oracle", "failure_evidence"}
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
