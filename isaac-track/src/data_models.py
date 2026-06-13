"""Goal data models for the YOUR_REPO system (Task 8.1).

This module is the authoritative, pure-Python home for the goal-reaching data
models the rest of the system binds to. It deliberately has **no** Isaac Sim /
Isaac Lab / torch dependency so it can be imported and unit-tested on the
controller host without GPUs, exactly like :mod:`src.config` and
:mod:`src.exceptions`.

Design references:
  - design.md -> Data Models -> ``@dataclass Goal`` / ``@dataclass GoalObservation``
  - design.md -> "Introducing the Goal ... into the env" (robot-frame goal signal)
  - requirements.md -> Glossary (Goal, Success_Radius, Goal_Observation)
  - requirements.md -> Requirement 18.2 (Config exposes Goal position + Success_Radius)
  - requirements.md -> Requirement 6.2 (generated reward reads the Goal from env state)

What lives here:
  - :class:`Goal` — the configurable target (point B) on the ground plane plus the
    arrival ``success_radius_m`` (Req 18.2, 9.2).
  - :class:`GoalObservation` — the robot-frame goal-conditioning signal appended to
    the proprioceptive observation: a vector-to-goal, a scalar distance, and a
    heading error, all expressed in the robot base frame (Req 6.2, 8.1). No pixels.
  - :class:`GoalRef` — a concrete, lightweight handle the Reward_Executor binds onto
    the env so the generated ``compute_reward`` can read the Goal (Req 6.2). It
    structurally satisfies the duck-typed ``GoalRef`` protocol declared in
    :mod:`src.rewards.reward_executor` (it exposes ``position_xy`` and
    ``success_radius_m``), wiring Task 8.1 to the wrapper from Task 4.3 / Task 9.1.

These are intentionally plain data containers (mirroring the design's Data Models
section). Light, cheap invariant checks live in ``__post_init__`` so an obviously
malformed Goal fails fast at construction, but range/default policy belongs to the
Config loader (Req 18.3, 18.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Goal", "GoalObservation", "GoalRef"]


def _as_xy(value: object, field_name: str) -> tuple[float, float]:
    """Coerce ``value`` into a finite ``(x, y)`` float tuple or raise ``ValueError``."""
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


# --------------------------------------------------------------------------- #
# Goal (point B + arrival threshold) — Req 18.2, 9.2
# --------------------------------------------------------------------------- #
@dataclass
class Goal:
    """The configurable target position (point B) on the flat ground plane.

    Attributes:
        position_xy: The ``(x, y)`` target location on the ground plane the robot
            must reach from its start position, point A (Req 18.2).
        success_radius_m: The arrival threshold; an episode is a success when the
            robot's base reaches within this distance of ``position_xy`` without
            falling (Req 18.2, 9.2). Must be a positive, finite distance in meters.
    """

    position_xy: tuple[float, float]
    success_radius_m: float

    def __post_init__(self) -> None:
        # Normalize the position to a finite float tuple regardless of the input
        # sequence type (list/tuple), so downstream consumers get a stable shape.
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
        """Whether a base-to-Goal ``distance_m`` is within the success radius.

        This is the geometric half of the success test (Req 9.2); the "without
        falling" half is evaluated by the Evaluator against episode state.
        """
        return distance_m <= self.success_radius_m


# --------------------------------------------------------------------------- #
# GoalObservation (robot-frame goal-conditioning signal) — Req 6.2, 8.1
# --------------------------------------------------------------------------- #
@dataclass
class GoalObservation:
    """Robot-frame goal signal appended to the ~69-dim proprio obs (no pixels).

    All quantities are expressed in the robot **base frame** so the policy sees a
    pose-invariant goal signal (design.md: rotate the world-frame goal vector by
    the inverse of the base yaw).

    Attributes:
        vec_to_goal_xy: The ``(x, y)`` vector from the robot base to the Goal,
            rotated into the robot base frame.
        distance_m: The straight-line distance from the robot base to the Goal in
            meters (non-negative; frame-invariant).
        heading_error_rad: The heading error in radians: the angle between the
            robot's forward (+x base) axis and the direction to the Goal, in
            ``[-pi, pi]``.
    """

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

        Pure trigonometry (no Isaac/torch): compute the world-frame vector from
        the robot base to the Goal, then rotate it by ``-base_yaw_rad`` into the
        robot base frame. ``distance_m`` is frame-invariant and ``heading_error_rad``
        is the bearing of the Goal relative to the robot's forward axis.

        Args:
            robot_xy: The robot base ``(x, y)`` position in the world frame.
            base_yaw_rad: The robot base yaw in radians in the world frame.
            goal: The :class:`Goal` (point B) to observe.
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


# --------------------------------------------------------------------------- #
# GoalRef (reward-binding handle) — Req 6.2
# --------------------------------------------------------------------------- #
@dataclass
class GoalRef:
    """A concrete, lightweight handle to a :class:`Goal` used for reward binding.

    The Reward_Executor's ``wrap`` step (Task 4.3) binds a "goal ref" onto the env
    (``env.goal``) so the generated ``compute_reward`` can read the Goal alongside
    base pose/velocity, joint pos/vel, joint torques, and foot contact forces
    (Req 6.2). The executor accepts any object that *structurally* exposes
    ``position_xy`` and ``success_radius_m`` (see
    :class:`src.rewards.reward_executor.GoalRef`). This class is the canonical
    concrete implementation of that contract: it wraps a :class:`Goal` and
    re-exposes its fields, so a ``GoalRef`` is a drop-in for the wrapper while
    keeping a single source of truth (the wrapped ``Goal``).

    The live per-env Goal buffer carried by the manager-based env (Task 8.2 / 9.1)
    is broadcast from this same Goal.

    Attributes:
        goal: The authoritative :class:`Goal` (point B + Success_Radius) this ref
            points at.
    """

    goal: Goal = field()

    @property
    def position_xy(self) -> tuple[float, float]:
        """The Goal's ground-plane target position (delegates to :attr:`goal`)."""
        return self.goal.position_xy

    @property
    def success_radius_m(self) -> float:
        """The Goal's arrival threshold in meters (delegates to :attr:`goal`)."""
        return self.goal.success_radius_m
