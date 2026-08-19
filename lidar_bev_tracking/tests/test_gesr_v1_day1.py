import math
import unittest

import numpy as np

from bev_tracking.gesr_v1 import (
    DeterministicPCA2D,
    EXTENSION_DISTANCE_EPSILON,
    PCA_EIGENVALUE_EPSILON,
    build_seed_components_reference,
    component_signature,
    deterministic_pca_2d,
    extension_score,
    is_geometry_extension,
    is_valid_eigenvalue,
    run_gesr_v1_reference,
)


def _point(x, y, intensity, z=0.0):
    return [x, y, z, intensity]


def _arbitration_cloud(left_indices=(1, 2, 3, 4), candidate_index=50):
    points = [
        _point(-0.55, -0.05, 0.50),
        _point(-0.55, 0.05, 0.50),
        _point(-0.50, -0.05, 0.50),
        _point(-0.50, 0.05, 0.50),
        _point(0.50, -0.05, 0.50),
        _point(0.50, 0.05, 0.50),
        _point(0.55, -0.05, 0.50),
        _point(0.55, 0.05, 0.50),
        _point(0.0, 0.0, 0.20),
    ]
    indices = list(left_indices) + [10, 11, 12, 13, candidate_index]
    return np.asarray(points, dtype=np.float64), np.asarray(indices, dtype=np.int64)


class GESRV1Day1Test(unittest.TestCase):
    def test_point_reorder_preserves_pca(self):
        points = np.asarray([
            _point(0.0, 0.0, 0.5), _point(0.2, 0.0, 0.5),
            _point(0.1, 0.05, 0.5), _point(0.3, 0.025, 0.5),
        ])
        indices = np.asarray([10, 20, 30, 40])
        first = build_seed_components_reference(points, indices)[0].geometry
        permutation = np.asarray([2, 0, 3, 1])
        second = build_seed_components_reference(points[permutation], indices[permutation])[0].geometry
        for field in ("center", "major", "minor"):
            np.testing.assert_allclose(getattr(first, field), getattr(second, field), rtol=0.0, atol=1e-15)
        for field in ("lambda1", "lambda2", "u_min", "u_max", "v_min", "v_max"):
            self.assertAlmostEqual(getattr(first, field), getattr(second, field), places=15)

    def test_point_reorder_preserves_component_signature(self):
        points = np.asarray([_point(0.0, 0.0, 0.5), _point(0.1, 0.0, 0.5), _point(0.2, 0.0, 0.5), _point(0.3, 0.0, 0.5)])
        first = build_seed_components_reference(points, [3, 17, 42, 99])[0]
        second = build_seed_components_reference(points[[2, 0, 3, 1]], [42, 3, 99, 17])[0]
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(first.signature, component_signature([3, 17, 42, 99]))

    def test_exact_isotropic_uses_frozen_axes(self):
        geometry = deterministic_pca_2d(np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]))
        self.assertTrue(geometry.exact_isotropic)
        self.assertEqual(geometry.major, (1.0, 0.0))
        self.assertEqual(geometry.minor, (0.0, 1.0))

    def test_near_equal_non_equal_does_not_use_isotropic_branch(self):
        geometry = deterministic_pca_2d(np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0 + 1e-12]]))
        self.assertFalse(geometry.exact_isotropic)

    def test_eigenvalue_boundary_is_strict(self):
        self.assertFalse(is_valid_eigenvalue(PCA_EIGENVALUE_EPSILON))
        self.assertTrue(is_valid_eigenvalue(math.nextafter(PCA_EIGENVALUE_EPSILON, math.inf)))

    def test_extension_boundary_is_strict(self):
        geometry = DeterministicPCA2D(
            center=(0.0, 0.0), major=(1.0, 0.0), minor=(0.0, 1.0),
            lambda1=1.0, lambda2=0.5, u_min=0.0, u_max=0.0,
            v_min=0.0, v_max=0.0, exact_isotropic=False, valid=True,
        )
        boundary = extension_score(geometry, (EXTENSION_DISTANCE_EPSILON, 0.0))
        above = extension_score(geometry, (math.nextafter(EXTENSION_DISTANCE_EPSILON, math.inf), 0.0))
        self.assertEqual(boundary, EXTENSION_DISTANCE_EPSILON)
        self.assertFalse(is_geometry_extension(boundary))
        self.assertTrue(is_geometry_extension(above))

    def test_exact_arbitration_tie_uses_minimum_runtime_id(self):
        points, indices = _arbitration_cloud()
        result = run_gesr_v1_reference("1", points, indices)
        decision = result.candidate_decisions[0]
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.selected_component_runtime_id, 1)
        eligible = [item for item in decision.associations if item.eligible]
        self.assertEqual(len(eligible), 2)
        self.assertEqual(eligible[0].two_anchor_mean_distance, eligible[1].two_anchor_mean_distance)

    def test_signature_change_does_not_affect_runtime_arbitration(self):
        points, indices = _arbitration_cloud(left_indices=(1, 2, 3, 4))
        changed_points, changed_indices = _arbitration_cloud(left_indices=(1, 5, 6, 7))
        first = run_gesr_v1_reference("1", points, indices)
        second = run_gesr_v1_reference("1", changed_points, changed_indices)
        self.assertNotEqual(first.components[0].signature, second.components[0].signature)
        self.assertEqual(first.candidate_decisions[0].selected_component_runtime_id, 1)
        self.assertEqual(second.candidate_decisions[0].selected_component_runtime_id, 1)

    def test_candidate_processing_reorder_preserves_accepted_set(self):
        points, indices = _arbitration_cloud()
        points = np.vstack((points, _point(0.0, 2.0, 0.20)))
        indices = np.append(indices, 51)
        permutation = np.asarray([8, 3, 9, 0, 6, 1, 7, 4, 2, 5])
        first = run_gesr_v1_reference("1", points, indices)
        second = run_gesr_v1_reference("1", points[permutation], indices[permutation])
        self.assertEqual(first.accepted_source_indices, second.accepted_source_indices)
        self.assertEqual(first.expanded_source_indices, second.expanded_source_indices)

    def test_repeated_run_is_identical(self):
        points, indices = _arbitration_cloud()
        first = run_gesr_v1_reference("1", points, indices)
        second = run_gesr_v1_reference("1", points, indices)
        self.assertEqual(first, second)

    def test_source_identity_is_not_coordinate_deduplicated(self):
        points = np.asarray([
            _point(0.0, 0.0, 0.5), _point(0.0, 0.0, 0.5),
            _point(0.1, 0.0, 0.5), _point(0.2, 0.0, 0.5),
        ])
        component = build_seed_components_reference(points, [1, 2, 3, 4])[0]
        self.assertEqual(component.source_indices, (1, 2, 3, 4))

    def test_runtime_core_signature_contains_no_gt_argument(self):
        import inspect

        parameters = inspect.signature(run_gesr_v1_reference).parameters
        self.assertEqual(tuple(parameters), ("frame_id", "points", "source_indices"))

    def test_frozen_intensity_boundaries_partition_points(self):
        points = np.asarray([
            _point(0.0, 0.0, 0.38),
            _point(0.1, 0.0, math.nextafter(0.38, -math.inf)),
            _point(0.2, 0.0, 0.15),
            _point(0.3, 0.0, math.nextafter(0.15, -math.inf)),
        ])
        result = run_gesr_v1_reference("1", points, [1, 2, 3, 4])
        self.assertEqual(result.seed_source_indices, (1,))
        self.assertEqual(result.candidate_source_indices, (2, 3))
        self.assertEqual(result.discarded_source_indices, (4,))

    def test_connectivity_radius_boundary_is_inclusive(self):
        connected = np.asarray([
            _point(0.0, 0.0, 0.5),
            _point(0.6, 0.0, 0.5),
        ])
        separated = connected.copy()
        separated[1, 0] = math.nextafter(0.6, math.inf)
        self.assertEqual(len(build_seed_components_reference(connected, [1, 2])), 1)
        self.assertEqual(len(build_seed_components_reference(separated, [1, 2])), 2)


if __name__ == "__main__":
    unittest.main()
