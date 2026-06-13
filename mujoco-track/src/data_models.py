"""Goal data models for the humanoid-mujoco-llm-rl system.

Carried over from the Isaac track — pure-Python, no JAX/MJX/torch dependency, so
it imports and unit-tests on the controller host. The robot-frame goal math is
framework-agnostic trigonometry that applies identically to the MJX env.

  - :class:`Goal` — the configurable target (point B) + arrival ``success_radius_m``.
  - :class:`GoalObservation` — robot-frame goal signal (vector-to-goal, distance,
    heading) appended to the proprio obs. No pixels.
  - :class:`GoalRef` — lightweight handle the Reward_Executor binds so the
    generated reward can read the Goal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Goal", "GoalObservation", "GoalRef"]


def _as_xy(value: object, field_name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{field_name}: expected a 2-element (x, y) sequence, "
            f"got {type(value).__name__} ({value!r})"
        )
    if len(value) != 2:
        raise ValueError(
            f"{field_name}: expected exactly 2 elements (x, y), got {len(value)}"
        )
    coerced: list[float] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"{field_name}[{i}]: expected a number, got {type(item).__name__}"
            )
        f = float(item)
        if not math.isfinite(f):
            raise ValueError(f"{field_name}[{i}]: must be finite, got {item!r}")
        coerced.append(f)
    return (coerced[0], coerced[1])


@dataclass
class Goal:
    """The configurable target position (point B) on the flat ground plane."""

    position_xy: tuple[float, float]
    success_radius_m: float

    def __post_init__(self) -> None:
        self.position_xy = _as_xy(self.position_xy, "position_xy")
        if isinstance(self.success_radius_m, bool) or not isinstance(
            self.success_radius_m, (int, float)
        ):
            raise ValueError(
                f"success_radius_m: expected a number, "
                f"got {type(self.success_radius_m).__name__}"
            )
        self.success_radius_m = float(self.success_radius_m)
        if not math.isfinite(self.success_radius_m):
            raise ValueError(
                f"success_radius_m: must be finite, got {self.success_radius_m!r}"
            )
        if self.success_radius_m <= 0.0:
            raise ValueError(
                f"success_radius_m: must be > 0, got {self.success_radius_m}"
            )

    def is_reached(self, distance_m: float) -> bool:
        """Whether a base-to-Goal ``distance_m`` is within the success radius."""
        return distance_m <= self.success_radius_m


@dataclass
class GoalObservation:
    """Robot-frame goal signal appended to the proprio obs (no pixels)."""

    vec_to_goal_xy: tuple[float, float]
    distance_m: float
    heading_error_rad: float

    def __post_init__(self) -> None:
        self.vec_to_goal_xy = _as_xy(self.vec_to_goal_xy, "vec_to_goal_xy")
        self.distance_m = float(self.distance_m)
        self.heading_error_rad = float(self.heading_error_rad)

    @classmethod
    def from_world(
        cls,
        robot_xy: tuple[float, float],
        base_yaw_rad: float,
        goal: "Goal",
    ) -> "GoalObservation":
        """Build a robot-frame ``GoalObservation`` from world-frame inputs.

        Pure trigonometry (no JAX/torch): world-frame vector from robot base to
        Goal, rotated by ``-base_yaw_rad`` into the robot base frame.
        """
        rx, ry = _as_xy(robot_xy, "robot_xy")
        gx, gy = goal.position_xy
        dx_world = gx - rx
        dy_world = gy - ry

        cos_y = math.cos(-base_yaw_rad)
        sin_y = math.sin(-base_yaw_rad)
        vec_x = dx_world * cos_y - dy_world * sin_y
        vec_y = dx_world * sin_y + dy_world * cos_y

        distance_m = math.hypot(dx_world, dy_world)
        heading_error_rad = math.atan2(vec_y, vec_x)
        return cls(
            vec_to_goal_xy=(vec_x, vec_y),
            distance_m=distance_m,
            heading_error_rad=heading_error_rad,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Flatten to the ``(vec_x, vec_y, distance, heading)`` obs packing order."""
        return (
            self.vec_to_goal_xy[0],
            self.vec_to_goal_xy[1],
            self.distance_m,
            self.heading_error_rad,
        )


@dataclass
class GoalRef:
    """A concrete, lightweight handle to a :class:`Goal` used for reward binding.

    Structurally exposes ``position_xy`` and ``success_radius_m`` so it is a
    drop-in for the Reward_Executor's duck-typed goal-ref contract while keeping
    a single source of truth (the wrapped :class:`Goal`).
    """

    goal: Goal = field()

    @property
    def position_xy(self) -> tuple[float, float]:
        return self.goal.position_xy

    @property
    def success_radius_m(self) -> float:
        return self.goal.success_radius_m
