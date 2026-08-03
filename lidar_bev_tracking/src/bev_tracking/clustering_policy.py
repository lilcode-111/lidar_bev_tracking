"""Frozen protocol definitions for the v15.2 clustering experiments.

Day 1 deliberately keeps the existing fixed clustering implementation intact.
This module only validates and normalizes the policy that later implementations
will consume, so the A0 prime replay remains unchanged.
"""

from dataclasses import dataclass
import math


GLOBAL_MAX_EPS = 0.85
DISTANCE_BIN_NAMES = ("near_0_15", "mid_15_30", "far_30_inf")


@dataclass(frozen=True)
class DistanceBin:
    name: str
    lower_m: float
    upper_m: float | None

    def contains(self, range_xy_m):
        value = float(range_xy_m)
        if value < self.lower_m:
            return False
        return self.upper_m is None or value < self.upper_m


DISTANCE_BINS = (
    DistanceBin("near_0_15", 0.0, 15.0),
    DistanceBin("mid_15_30", 15.0, 30.0),
    DistanceBin("far_30_inf", 30.0, None),
)


@dataclass(frozen=True)
class ClusteringPolicy:
    mode: str = "fixed"
    eps: float = 0.6
    min_points: int = 20
    global_max_eps: float = GLOBAL_MAX_EPS
    distance_params: dict | None = None

    def __post_init__(self):
        if self.mode not in {"fixed", "adaptive"}:
            raise ValueError("clustering mode must be fixed or adaptive")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0:
            raise ValueError("eps must be a positive finite number")
        if isinstance(self.min_points, bool) or not isinstance(self.min_points, int) or self.min_points <= 0:
            raise ValueError("min_points must be a positive integer")
        if not math.isfinite(float(self.global_max_eps)) or float(self.global_max_eps) <= 0:
            raise ValueError("global_max_eps must be a positive finite number")
        if float(self.eps) > float(self.global_max_eps):
            raise ValueError("eps must not exceed global_max_eps")
        if self.mode == "adaptive":
            validate_distance_params(self.distance_params)

    def params_for_range(self, range_xy_m):
        if self.mode == "fixed":
            return {"eps": float(self.eps), "min_points": int(self.min_points)}
        name = distance_bin_name(range_xy_m)
        params = self.distance_params[name]
        return {"eps": float(params["eps"]), "min_points": int(params["min_points"])}


def distance_bin_name(range_xy_m):
    value = float(range_xy_m)
    if not math.isfinite(value) or value < 0:
        raise ValueError("range_xy_m must be a finite non-negative number")
    for distance_bin in DISTANCE_BINS:
        if distance_bin.contains(value):
            return distance_bin.name
    raise ValueError(f"unsupported range_xy_m: {range_xy_m}")


def pairwise_eps(eps_i, eps_j):
    """Return the frozen symmetric radius for a point pair."""
    left = float(eps_i)
    right = float(eps_j)
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right <= 0:
        raise ValueError("pairwise eps values must be positive finite numbers")
    return max(left, right)


def validate_distance_params(distance_params):
    if not isinstance(distance_params, dict):
        raise ValueError("adaptive distance_params must be a mapping")
    if set(distance_params) != set(DISTANCE_BIN_NAMES):
        raise ValueError("adaptive distance_params must define near, mid, and far bins")

    for name in DISTANCE_BIN_NAMES:
        params = distance_params[name]
        if not isinstance(params, dict) or set(params) != {"eps", "min_points"}:
            raise ValueError(f"invalid clustering parameters for {name}")
        eps = float(params["eps"])
        min_points = params["min_points"]
        if not math.isfinite(eps) or eps <= 0 or eps > GLOBAL_MAX_EPS:
            raise ValueError(f"adaptive eps for {name} must be in (0, {GLOBAL_MAX_EPS}]")
        if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points <= 0:
            raise ValueError(f"adaptive min_points for {name} must be a positive integer")


def clustering_policy_from_config(config):
    if not isinstance(config, dict):
        raise ValueError("clustering config must be a mapping")
    detector = config.get("detector", {})
    clustering = config.get("clustering", {})
    if not isinstance(detector, dict) or not isinstance(clustering, dict):
        raise ValueError("detector and clustering config must be mappings")

    mode = clustering.get("mode", "fixed")
    policy = ClusteringPolicy(
        mode=mode,
        eps=float(clustering.get("eps", detector.get("eps", 0.6))),
        min_points=int(clustering.get("min_points", detector.get("min_points", 20))),
        global_max_eps=float(clustering.get("global_max_eps", GLOBAL_MAX_EPS)),
        distance_params=clustering.get("distance_params"),
    )
    if policy.mode == "adaptive":
        validate_distance_params(policy.distance_params)
    return policy
