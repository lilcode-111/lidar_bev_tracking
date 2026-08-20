"""Deterministic brute-force reference core for frozen GESR-v1.

This module contains no GT/evaluation dependency and no formal experiment entry
point.  The reference implementation intentionally uses quadratic XY searches;
an optimized spatial index must later prove exact semantic equivalence to it.
"""

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np


SEED_INTENSITY_MIN = 0.38
CANDIDATE_INTENSITY_MIN = 0.15
CONNECTIVITY_RADIUS_M = 0.60
MIN_SEED_COMPONENT_POINTS = 4
MIN_DIRECT_SEED_ANCHORS = 2
PCA_EIGENVALUE_EPSILON = 1.0e-12
EXTENSION_DISTANCE_EPSILON = 1.0e-9
NUMERICAL_DTYPE = np.float64
POINT_TERMINAL_CODES = (
    "ACCEPTED",
    "GEOMETRY_EXTENSION_INVALID",
    "INSIDE_CURRENT_EXTENT",
    "INSUFFICIENT_DIRECT_ANCHORS",
    "NO_VALID_COMPONENT",
)
ASSOCIATION_OUTCOMES = ("SELECTED", "MULTI_COMPONENT_LOST")


class GESRV1Error(ValueError):
    pass


def component_signature(source_indices):
    """Hash compact canonical JSON of sorted raw-LiDAR point indices."""
    ordered = sorted(int(value) for value in source_indices)
    payload = json.dumps(ordered, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SourcePointBatch:
    frame_id: str
    points: np.ndarray
    source_indices: np.ndarray

    def __post_init__(self):
        points = np.array(self.points, dtype=NUMERICAL_DTYPE, copy=True)
        indices = np.array(self.source_indices, dtype=np.int64, copy=True)
        if points.ndim != 2 or points.shape[1] < 4:
            raise GESRV1Error("points must be an Nx4-or-wider array containing XYZI")
        if indices.ndim != 1 or len(indices) != len(points):
            raise GESRV1Error("source indices must be one-dimensional and aligned with points")
        if len(indices) != len(set(indices.tolist())):
            raise GESRV1Error("raw_lidar_point_index must be unique within a frame")
        if np.any(indices < 0):
            raise GESRV1Error("raw_lidar_point_index must be non-negative")
        order = np.argsort(indices, kind="stable")
        points = np.ascontiguousarray(points[order], dtype=NUMERICAL_DTYPE)
        indices = np.ascontiguousarray(indices[order], dtype=np.int64)
        points.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "frame_id", str(self.frame_id).zfill(6))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "source_indices", indices)


@dataclass(frozen=True)
class DeterministicPCA2D:
    center: tuple
    major: tuple
    minor: tuple
    lambda1: float
    lambda2: float
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    exact_isotropic: bool
    valid: bool


@dataclass(frozen=True)
class SeedComponent:
    runtime_id: int
    signature: str
    source_indices: tuple
    points_xy: tuple
    geometry: DeterministicPCA2D


@dataclass(frozen=True)
class ComponentAssociation:
    component_runtime_id: int
    component_signature: str
    valid_component: bool
    direct_anchor_count: int
    two_anchor_mean_distance: float | None
    extension_score: float | None
    eligible: bool
    arbitration_outcome: str | None = None
    arbitration_status: str = "not_applicable"


@dataclass(frozen=True)
class CandidateDecision:
    source_index: int
    accepted: bool
    selected_component_runtime_id: int | None
    selected_component_signature: str | None
    point_terminal_decision: str | None
    associations: tuple


@dataclass(frozen=True)
class GESRReferenceResult:
    frame_id: str
    seed_source_indices: tuple
    candidate_source_indices: tuple
    discarded_source_indices: tuple
    accepted_source_indices: tuple
    expanded_source_indices: tuple
    components: tuple
    candidate_decisions: tuple


def is_valid_eigenvalue(lambda1):
    return float(lambda1) > PCA_EIGENVALUE_EPSILON


def deterministic_pca_2d(points_xy):
    """Frozen binary64 closed-form population PCA over source-sorted input."""
    xy = np.asarray(points_xy, dtype=NUMERICAL_DTYPE)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise GESRV1Error("PCA input must be a non-empty Nx2 array")

    center_array = np.sum(xy, axis=0, dtype=NUMERICAL_DTYPE) / NUMERICAL_DTYPE(len(xy))
    centered = xy - center_array
    cxx = float(np.sum(centered[:, 0] * centered[:, 0], dtype=NUMERICAL_DTYPE) / len(xy))
    cyy = float(np.sum(centered[:, 1] * centered[:, 1], dtype=NUMERICAL_DTYPE) / len(xy))
    cxy = float(np.sum(centered[:, 0] * centered[:, 1], dtype=NUMERICAL_DTYPE) / len(xy))
    trace = cxx + cyy
    delta = math.sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy)
    lambda1 = (trace + delta) / 2.0
    lambda2 = (trace - delta) / 2.0

    exact_isotropic = cxx == cyy and cxy == 0.0
    if exact_isotropic:
        major = np.array((1.0, 0.0), dtype=NUMERICAL_DTYPE)
    else:
        theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
        major = np.array((math.cos(theta), math.sin(theta)), dtype=NUMERICAL_DTYPE)
        if major[0] < 0.0 or (major[0] == 0.0 and major[1] < 0.0):
            major = -major
    minor = np.array((-major[1], major[0]), dtype=NUMERICAL_DTYPE)
    u = centered @ major
    v = centered @ minor
    return DeterministicPCA2D(
        center=(float(center_array[0]), float(center_array[1])),
        major=(float(major[0]), float(major[1])),
        minor=(float(minor[0]), float(minor[1])),
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        u_min=float(np.min(u)),
        u_max=float(np.max(u)),
        v_min=float(np.min(v)),
        v_max=float(np.max(v)),
        exact_isotropic=bool(exact_isotropic),
        valid=is_valid_eigenvalue(lambda1),
    )


def extension_score(geometry, point_xy):
    point = np.asarray(point_xy, dtype=NUMERICAL_DTYPE)
    centered = point - np.asarray(geometry.center, dtype=NUMERICAL_DTYPE)
    u_p = float(centered @ np.asarray(geometry.major, dtype=NUMERICAL_DTYPE))
    v_p = float(centered @ np.asarray(geometry.minor, dtype=NUMERICAL_DTYPE))
    d_major = max(geometry.u_min - u_p, 0.0, u_p - geometry.u_max)
    d_minor = max(geometry.v_min - v_p, 0.0, v_p - geometry.v_max)
    return float(math.sqrt(d_major * d_major + d_minor * d_minor))


def is_geometry_extension(score):
    return float(score) > EXTENSION_DISTANCE_EPSILON


def _squared_xy_distance(left, right):
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return dx * dx + dy * dy


def _sorted_seed_input(seed_points, seed_source_indices):
    points = np.asarray(seed_points, dtype=NUMERICAL_DTYPE)
    indices = np.asarray(seed_source_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) != len(indices):
        raise GESRV1Error("seed points and source indices must be aligned")
    if len(indices) != len(set(indices.tolist())):
        raise GESRV1Error("seed source indices must be unique")
    order = np.argsort(indices, kind="stable")
    return points[order], indices[order]


def _materialize_seed_components(points, indices, memberships):
    components = []
    for members in memberships:
        members = tuple(sorted(members, key=lambda position: int(indices[position])))
        member_indices = tuple(int(indices[position]) for position in members)
        member_xy_array = np.asarray(
            [points[position, :2] for position in members], dtype=NUMERICAL_DTYPE
        )
        geometry = deterministic_pca_2d(member_xy_array)
        if len(members) < MIN_SEED_COMPONENT_POINTS:
            geometry = replace(geometry, valid=False)
        components.append(
            SeedComponent(
                runtime_id=min(member_indices),
                signature=component_signature(member_indices),
                source_indices=member_indices,
                points_xy=tuple(
                    (float(point[0]), float(point[1])) for point in member_xy_array
                ),
                geometry=geometry,
            )
        )
    return tuple(sorted(components, key=lambda component: component.runtime_id))


def build_seed_components_reference(seed_points, seed_source_indices):
    """Build deterministic XY connected components with an O(n^2) search."""
    points, indices = _sorted_seed_input(seed_points, seed_source_indices)
    radius2 = CONNECTIVITY_RADIUS_M * CONNECTIVITY_RADIUS_M
    visited = np.zeros(len(points), dtype=bool)
    memberships = []

    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque((start,))
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            neighbors = [
                other
                for other in range(len(points))
                if not visited[other]
                and _squared_xy_distance(points[current], points[other]) <= radius2
            ]
            for other in neighbors:
                visited[other] = True
                queue.append(other)
        memberships.append(tuple(members))
    return _materialize_seed_components(points, indices, memberships)


def _grid_cell(point_xy):
    """Map XY to a radius-derived cell; this is not a tunable parameter."""
    return (
        math.floor(float(point_xy[0]) / CONNECTIVITY_RADIUS_M),
        math.floor(float(point_xy[1]) / CONNECTIVITY_RADIUS_M),
    )


def _build_spatial_grid(points_xy):
    buckets = {}
    for position, point in enumerate(points_xy):
        buckets.setdefault(_grid_cell(point), []).append(position)
    return {key: tuple(values) for key, values in buckets.items()}


def _grid_neighbor_positions(point_xy, points_xy, grid):
    """Return exact-radius neighbors after a deterministic 3x3 coarse query."""
    cell_x, cell_y = _grid_cell(point_xy)
    radius2 = CONNECTIVITY_RADIUS_M * CONNECTIVITY_RADIUS_M
    candidates = []
    for grid_x in range(cell_x - 1, cell_x + 2):
        for grid_y in range(cell_y - 1, cell_y + 2):
            candidates.extend(grid.get((grid_x, grid_y), ()))
    return tuple(
        position
        for position in sorted(candidates)
        if _squared_xy_distance(point_xy, points_xy[position]) <= radius2
    )


def build_seed_components_optimized(seed_points, seed_source_indices):
    """Build seed components with a deterministic exact-filtered spatial grid."""
    points, indices = _sorted_seed_input(seed_points, seed_source_indices)
    grid = _build_spatial_grid(points[:, :2])
    visited = np.zeros(len(points), dtype=bool)
    memberships = []
    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque((start,))
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for other in _grid_neighbor_positions(points[current, :2], points[:, :2], grid):
                if not visited[other]:
                    visited[other] = True
                    queue.append(other)
        memberships.append(tuple(members))
    return _materialize_seed_components(points, indices, memberships)


def _association_from_distances(point_xy, component, distances):
    distances = sorted(float(value) for value in distances)
    anchor_count = len(distances)
    mean_distance = float((distances[0] + distances[1]) / 2.0) if anchor_count >= 2 else None
    score = extension_score(component.geometry, point_xy) if component.geometry.valid else None
    eligible = bool(
        component.geometry.valid
        and anchor_count >= MIN_DIRECT_SEED_ANCHORS
        and score is not None
        and is_geometry_extension(score)
    )
    return ComponentAssociation(
        component_runtime_id=component.runtime_id,
        component_signature=component.signature,
        valid_component=component.geometry.valid,
        direct_anchor_count=anchor_count,
        two_anchor_mean_distance=mean_distance,
        extension_score=score,
        eligible=eligible,
    )


def _associate_candidate(point_xy, component):
    distances = (
        math.sqrt(_squared_xy_distance(point_xy, anchor))
        for anchor in component.points_xy
        if _squared_xy_distance(point_xy, anchor)
        <= CONNECTIVITY_RADIUS_M * CONNECTIVITY_RADIUS_M
    )
    return _association_from_distances(point_xy, component, distances)


def _attribute_terminal_reason(accepted, associations):
    """Apply frozen furthest-valid-progress precedence after runtime decision."""
    if accepted:
        return "ACCEPTED"
    directly_related = [item for item in associations if item.direct_anchor_count > 0]
    if any(not item.valid_component for item in directly_related):
        return "GEOMETRY_EXTENSION_INVALID"
    if any(
        item.valid_component
        and item.direct_anchor_count >= MIN_DIRECT_SEED_ANCHORS
        and item.extension_score is not None
        and not is_geometry_extension(item.extension_score)
        for item in associations
    ):
        return "INSIDE_CURRENT_EXTENT"
    if directly_related:
        return "INSUFFICIENT_DIRECT_ANCHORS"
    return "NO_VALID_COMPONENT"


def _apply_arbitration_outcomes(associations, selected_runtime_id):
    output = []
    for association in associations:
        if not association.eligible:
            output.append(association)
        elif association.component_runtime_id == selected_runtime_id:
            output.append(
                replace(
                    association,
                    arbitration_outcome="SELECTED",
                    arbitration_status="applicable",
                )
            )
        else:
            output.append(
                replace(
                    association,
                    arbitration_outcome="MULTI_COMPONENT_LOST",
                    arbitration_status="applicable",
                )
            )
    return tuple(output)


def _optimized_association_builder(seed_points, seed_indices, components):
    grid = _build_spatial_grid(seed_points[:, :2])
    component_by_source_index = {
        source_index: component.runtime_id
        for component in components
        for source_index in component.source_indices
    }
    component_by_runtime_id = {component.runtime_id: component for component in components}

    def build(point_xy):
        distances_by_component = {}
        for position in _grid_neighbor_positions(point_xy, seed_points[:, :2], grid):
            source_index = int(seed_indices[position])
            runtime_id = component_by_source_index[source_index]
            distance = math.sqrt(_squared_xy_distance(point_xy, seed_points[position, :2]))
            distances_by_component.setdefault(runtime_id, []).append(distance)
        return tuple(
            _association_from_distances(
                point_xy,
                component_by_runtime_id[component.runtime_id],
                distances_by_component.get(component.runtime_id, ()),
            )
            for component in components
        )

    return build


def _execute_gesr_v1(
    frame_id,
    points,
    source_indices,
    *,
    reason_attribution=True,
    optimized=False,
):
    batch = SourcePointBatch(frame_id=frame_id, points=points, source_indices=source_indices)
    intensities = batch.points[:, 3]
    seed_mask = intensities >= SEED_INTENSITY_MIN
    candidate_mask = (intensities >= CANDIDATE_INTENSITY_MIN) & (intensities < SEED_INTENSITY_MIN)
    discard_mask = intensities < CANDIDATE_INTENSITY_MIN

    seed_points = batch.points[seed_mask]
    seed_indices = batch.source_indices[seed_mask]
    candidate_points = batch.points[candidate_mask]
    candidate_indices = batch.source_indices[candidate_mask]
    component_builder = (
        build_seed_components_optimized if optimized else build_seed_components_reference
    )
    components = component_builder(seed_points, seed_indices)
    if optimized:
        association_builder = _optimized_association_builder(
            seed_points, seed_indices, components
        )
    else:
        association_builder = lambda point_xy: tuple(
            _associate_candidate(point_xy, component) for component in components
        )

    decisions = []
    accepted = []
    for point, source_index in zip(candidate_points, candidate_indices):
        associations = association_builder(point[:2])
        eligible = [association for association in associations if association.eligible]
        if eligible:
            winner = min(
                eligible,
                key=lambda association: (
                    association.two_anchor_mean_distance,
                    association.component_runtime_id,
                ),
            )
            selected = winner.component_runtime_id
            selected_signature = winner.component_signature
            accepted.append(int(source_index))
        else:
            selected = None
            selected_signature = None
        associations = _apply_arbitration_outcomes(associations, selected)
        terminal = (
            _attribute_terminal_reason(bool(eligible), associations)
            if reason_attribution
            else None
        )
        decisions.append(
            CandidateDecision(
                source_index=int(source_index),
                accepted=bool(eligible),
                selected_component_runtime_id=selected,
                selected_component_signature=selected_signature,
                point_terminal_decision=terminal,
                associations=associations,
            )
        )

    seed_identity = tuple(int(value) for value in seed_indices)
    candidate_identity = tuple(int(value) for value in candidate_indices)
    discarded_identity = tuple(int(value) for value in batch.source_indices[discard_mask])
    accepted_identity = tuple(sorted(accepted))
    expanded_identity = tuple(sorted(seed_identity + accepted_identity))
    return GESRReferenceResult(
        frame_id=batch.frame_id,
        seed_source_indices=seed_identity,
        candidate_source_indices=candidate_identity,
        discarded_source_indices=discarded_identity,
        accepted_source_indices=accepted_identity,
        expanded_source_indices=expanded_identity,
        components=components,
        candidate_decisions=tuple(decisions),
    )


def run_gesr_v1_reference(
    frame_id,
    points,
    source_indices,
    *,
    reason_attribution=True,
):
    """Run frozen single-pass GESR-v1 with brute-force spatial searches."""
    return _execute_gesr_v1(
        frame_id,
        points,
        source_indices,
        reason_attribution=reason_attribution,
        optimized=False,
    )


def run_gesr_v1_optimized(
    frame_id,
    points,
    source_indices,
    *,
    reason_attribution=True,
):
    """Run GESR-v1 with a radius-derived grid and exact distance filtering."""
    return _execute_gesr_v1(
        frame_id,
        points,
        source_indices,
        reason_attribution=reason_attribution,
        optimized=True,
    )


def validate_gesr_v1_semantic_equivalence(reference, optimized):
    """Require full immutable runtime equality, including all evidence fields."""
    if not isinstance(reference, GESRReferenceResult) or not isinstance(
        optimized, GESRReferenceResult
    ):
        raise GESRV1Error("equivalence inputs must be GESR runtime results")
    if reference != optimized:
        mismatches = [
            name
            for name in reference.__dataclass_fields__
            if getattr(reference, name) != getattr(optimized, name)
        ]
        raise GESRV1Error(
            "reference/optimized semantic mismatch: " + ", ".join(mismatches)
        )
    return {
        "status": "PASS",
        "frame_id": reference.frame_id,
        "component_count": len(reference.components),
        "candidate_count": len(reference.candidate_source_indices),
        "accepted_count": len(reference.accepted_source_indices),
        "full_runtime_result_equal": True,
        "source_point_identity_equal": True,
        "component_membership_equal": True,
        "component_geometry_equal": True,
        "candidate_decisions_equal": True,
        "association_outcomes_equal": True,
        "accepted_and_expanded_sets_equal": True,
    }


def _geometry_record(geometry):
    return {
        "center": list(geometry.center),
        "major": list(geometry.major),
        "minor": list(geometry.minor),
        "lambda1": geometry.lambda1,
        "lambda2": geometry.lambda2,
        "extent": {
            "u_min": geometry.u_min,
            "u_max": geometry.u_max,
            "v_min": geometry.v_min,
            "v_max": geometry.v_max,
        },
        "exact_isotropic": geometry.exact_isotropic,
        "valid": geometry.valid,
    }


def build_gesr_v1_evidence(result):
    """Build read-only runtime evidence without GT/evaluation inputs."""
    if not isinstance(result, GESRReferenceResult):
        raise GESRV1Error("evidence input must be a GESRReferenceResult")
    if any(item.point_terminal_decision is None for item in result.candidate_decisions):
        raise GESRV1Error("terminal reason attribution must be enabled for evidence")

    candidate_records = []
    selected_count = 0
    lost_count = 0
    for decision in result.candidate_decisions:
        associations = []
        for association in decision.associations:
            if association.arbitration_outcome == "SELECTED":
                selected_count += 1
            elif association.arbitration_outcome == "MULTI_COMPONENT_LOST":
                lost_count += 1
            associations.append(
                {
                    "frame_id": result.frame_id,
                    "raw_lidar_point_index": decision.source_index,
                    "component_runtime_id": association.component_runtime_id,
                    "component_signature": association.component_signature,
                    "valid_component": association.valid_component,
                    "direct_anchor_count": association.direct_anchor_count,
                    "two_anchor_mean_distance": association.two_anchor_mean_distance,
                    "extension_score": association.extension_score,
                    "arbitration_eligible": association.eligible,
                    "arbitration_outcome": association.arbitration_outcome,
                    "status": association.arbitration_status,
                }
            )
        candidate_records.append(
            {
                "frame_id": result.frame_id,
                "raw_lidar_point_index": decision.source_index,
                "accepted": decision.accepted,
                "point_terminal_decision": decision.point_terminal_decision,
                "selected_component_runtime_id": decision.selected_component_runtime_id,
                "selected_component_signature": decision.selected_component_signature,
                "candidate_component_count": sum(
                    item.direct_anchor_count > 0 for item in decision.associations
                ),
                "arbitration_eligible_component_count": sum(
                    item.eligible for item in decision.associations
                ),
                "associations": associations,
            }
        )

    accepted_set = set(result.accepted_source_indices)
    terminal_accepted_set = {
        item.source_index
        for item in result.candidate_decisions
        if item.point_terminal_decision == "ACCEPTED"
    }
    if accepted_set != terminal_accepted_set:
        raise GESRV1Error("terminal attribution changed runtime accepted identity")
    if selected_count != len(accepted_set):
        raise GESRV1Error("accepted candidates must have exactly one SELECTED association")
    for item in result.candidate_decisions:
        eligible_count = sum(association.eligible for association in item.associations)
        item_lost_count = sum(
            association.arbitration_outcome == "MULTI_COMPONENT_LOST"
            for association in item.associations
        )
        expected_lost = eligible_count - 1 if item.accepted else 0
        if item_lost_count != expected_lost:
            raise GESRV1Error("association arbitration invariant failed")

    return {
        "schema_version": "15.5-gesr-v1-runtime-evidence-v1",
        "algorithm": "GESR-v1 brute-force reference",
        "frame_id": result.frame_id,
        "point_identity": "(frame_id, raw_lidar_point_index)",
        "point_terminal_decision_codes": list(POINT_TERMINAL_CODES),
        "association_arbitration_outcomes": list(ASSOCIATION_OUTCOMES),
        "point_sets": {
            "seed_source_indices": list(result.seed_source_indices),
            "candidate_source_indices": list(result.candidate_source_indices),
            "discarded_source_indices": list(result.discarded_source_indices),
            "accepted_source_indices": list(result.accepted_source_indices),
            "expanded_source_indices": list(result.expanded_source_indices),
        },
        "components": [
            {
                "component_runtime_id": component.runtime_id,
                "component_signature": component.signature,
                "source_point_indices": list(component.source_indices),
                "geometry": _geometry_record(component.geometry),
            }
            for component in result.components
        ],
        "candidate_decisions": candidate_records,
        "arbitration_metrics": {
            "multi_component_candidate_count": sum(
                record["candidate_component_count"] >= 2
                for record in candidate_records
            ),
            "conflict_arbitration_count": sum(
                record["arbitration_eligible_component_count"] >= 2
                for record in candidate_records
            ),
            "selected_association_count": selected_count,
            "lost_association_count": lost_count,
        },
        "invariants": {
            "accepted_iff_terminal_ACCEPTED": True,
            "exactly_one_SELECTED_per_accepted_candidate": True,
            "rejected_candidate_SELECTED_count": 0,
            "accepted_subset_of_candidate_universe": accepted_set.issubset(
                set(result.candidate_source_indices)
            ),
            "expanded_equals_seed_union_accepted": set(result.expanded_source_indices)
            == set(result.seed_source_indices).union(accepted_set),
        },
        "GT_runtime_input": False,
        "formal_result": False,
    }
