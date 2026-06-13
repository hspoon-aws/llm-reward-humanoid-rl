"""Goal-reaching metric computation (spec Task 11.2, Req 9).

This module owns the *computation* half of the Evaluator: given recorded
episode trajectories and the :class:`~src.data_models.Goal`, it produces the
:class:`~src.eval.metrics.EvalMetrics` data shape implemented in Task 11.1.

Design references:
  - design.md -> Components -> Evaluator (`src/eval/evaluator.py`) -- Req 9, 10, 17
  - requirements.md -> Requirement 9 (goal-reaching evaluation metrics)
  - src/eval/metrics.py -> EvalMetrics / CapabilityGates (Task 11.1)
  - src/data_models.py -> Goal (Task 8.1)

Scope of THIS file (Task 11.2 -- metric computation)
----------------------------------------------------
Compute, from trajectory/episode data structures rather than a live Isaac Sim
env, every goal-reaching measurement required by Req 9:

  * distance-to-goal at episode end (Req 9.1),
  * per-episode success indicator + success rate (Req 9.2),
  * time-to-goal for successful episodes (Req 9.3),
  * path efficiency (Req 9.4),
  * upright time + fall rate (Req 9.5),
  * optional secondary quality metrics -- average forward speed, energy
    efficiency, gait smoothness, symmetry (Req 9.6).

Because the inputs are plain trajectory arrays (`(x, y)` samples, upright flags,
optional torque/reward/foot signals), the computation has **no** Isaac Sim /
torch dependency and is fully unit-testable on the controller host without a
GPU, exactly like :mod:`src.eval.metrics` and :mod:`src.data_models`.

Best/worst demo-video recording (Task 11.4 -- Req 10)
-----------------------------------------------------
:class:`DemoVideoProducer` records the best and worst evaluation episodes (Req
10.1) using the external, world-frame :class:`~src.sensors.camera_cfg.CameraConfig`
cameras (Req 10.3) at a small env count (Req 10.4). The *selection* half --
choosing which rolled-out episode is best/worst by the configured selection metric
-- is the pure, fully unit-testable :func:`select_best_worst` / :func:`episode_score`
pair (argmax/argmin of a per-episode score). The *rollout* (running the trained
policy in the live env to produce trajectories) and the *renderer* (turning an
episode into RGB video files via the cameras) are **injected** dependencies, both
guarded the same way as the training-capture renderer: Isaac Sim / Isaac Lab is not
importable on the controller/dev host, so when no rollout or no recorder is injected
the producer is a safe no-op (returns ``None``) and nothing here imports the
simulator. This keeps best/worst selection testable with fakes on a CPU-only host.

Staged goal gates (Task 11.3 -- Req 17)
---------------------------------------
:func:`compute_capability_gates` computes the three staged goal gates
(`makes_progress`, `reaches_goal`, `efficient_goal`) from the per-episode metrics
and Config-sourced thresholds (`min_progress_distance_m`, `time_to_goal_threshold_s`,
`path_efficiency_threshold`, plus the Goal's `success_radius_m`) -- Req 17.1-17.4.
:func:`compute_eval_metrics` accepts a :class:`GateThresholds` and computes the
gates in-line; when neither thresholds nor an explicit :class:`CapabilityGates` are
supplied it falls back to the documented all-false placeholder so the produced
:class:`EvalMetrics` is always complete.

The gates are **staged-monotonic** by construction (Req 17): a higher gate implies
every lower one, ``efficient_goal => reaches_goal => makes_progress``. This holds at
the aggregate level because each gate is computed over the same evaluated episodes
and the predicates are nested (see :func:`compute_capability_gates`).

Aggregation contract (single vs. multiple episodes)
---------------------------------------------------
:class:`EvalMetrics` carries a single value per field, but evaluation runs many
episodes. The aggregation rules below are chosen so the field invariants
documented on :class:`EvalMetrics` hold for the aggregate, not just for one
episode:

  * ``success_rate`` / ``fall_rate`` -- mean of the per-episode 0/1 indicators,
    so both lie in ``[0, 1]`` (Req 9.2, 9.5).
  * ``distance_to_goal_m`` / ``path_efficiency`` / ``upright_time_s`` -- mean of
    the per-episode values (non-negative; path efficiency stays in ``(0, 1]``).
  * ``time_to_goal_s`` -- mean over *successful* episodes only, or ``None`` when
    no episode succeeded (Req 9.3).
  * ``success`` -- ``True`` iff at least one episode reached the Goal within the
    Success_Radius without falling (i.e. ``success_rate > 0``). This keeps the
    "time-to-goal present iff success" relationship exact at the aggregate
    level, and for a single episode reduces to that episode's own success.
  * secondary metrics -- mean over the episodes that supplied the required raw
    signal, or ``None`` when none did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Protocol, Sequence, runtime_checkable

from ..data_models import Goal
from ..sensors.camera_cfg import CameraConfig, CameraSpec
from .metrics import CapabilityGates, EvalMetrics

__all__ = [
    "EpisodeTrajectory",
    "EpisodeMetrics",
    "GateThresholds",
    "compute_episode_metrics",
    "compute_capability_gates",
    "compute_eval_metrics",
    # best/worst demo-video recording (Task 11.4 -- Req 10)
    "SELECTION_METRIC_DEFAULT",
    "SELECTION_METRIC_EPISODE_ALIAS",
    "episode_selection_metric",
    "LOWER_IS_BETTER_METRICS",
    "episode_score",
    "select_best_worst",
    "BestWorstSelection",
    "EpisodeRollout",
    "PolicyRollout",
    "DemoVideoRecorder",
    "DemoVideo",
    "DemoVideoResult",
    "DemoVideoProducer",
    # live Isaac-Sim-backed rollout + recorder and the top-level Evaluator
    "EVAL_ENV_COUNT_DEFAULT",
    "UPRIGHT_PROJECTED_GRAVITY_Z_MAX",
    "SECONDARY_METRICS",
    "EvalConfig",
    "build_mjx_policy_rollout",
    "build_mjx_demo_recorder",
    "Evaluator",
]

# Strictly-positive floor used to keep clamped ratios inside the open-lower
# interval ``(0, 1]`` (e.g. path efficiency, gait smoothness) without ever
# returning exactly 0, which the EvalMetrics invariants exclude.
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Trajectory input
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeTrajectory:
    """One evaluated episode, recorded as plain host-side arrays.

    A trajectory is a time series of per-step samples. Sample ``0`` is the start
    (point A); the robot's path is the polyline through ``positions_xy``. All
    fields are pure Python so the trajectory can be constructed synthetically in
    tests with no Isaac Sim present; Task 11.4 will populate it from a live
    policy rollout.

    Attributes:
        positions_xy: Base ``(x, y)`` ground-plane position at each timestep,
            in order, starting at point A. Must contain at least one sample.
        upright_flags: Per-timestep flag, ``True`` while the robot is upright and
            ``False`` once it has fallen at that step. Same length as
            ``positions_xy``.
        dt: Seconds between consecutive samples (timestep). Must be ``> 0``.
        torques: Optional per-timestep joint-torque vectors, used for energy
            efficiency (Req 9.6). When omitted, energy efficiency is ``None``.
        rewards: Optional per-timestep scalar reward, used as the numerator of
            energy efficiency (reward per squared torque, Req 9.6).
        left_contacts / right_contacts: Optional per-timestep left/right foot
            signals (e.g. contact force or boolean contact) used for gait
            symmetry (Req 9.6). When omitted, symmetry is ``None``.

    The straight-line A->Goal reference distance for path efficiency (Req 9.4) is
    derived from ``positions_xy[0]`` and the Goal, matching design.md ("the
    straight-line A->Goal distance is recorded at episode reset").
    """

    positions_xy: Sequence[tuple[float, float]]
    upright_flags: Sequence[bool]
    dt: float
    torques: Sequence[Sequence[float]] | None = None
    rewards: Sequence[float] | None = None
    left_contacts: Sequence[float] | None = None
    right_contacts: Sequence[float] | None = None

    def __post_init__(self) -> None:
        positions = [self._as_xy(p, i) for i, p in enumerate(self.positions_xy)]
        if not positions:
            raise ValueError("positions_xy must contain at least one (x, y) sample")
        self.positions_xy = positions

        flags = [bool(f) for f in self.upright_flags]
        if len(flags) != len(positions):
            raise ValueError(
                f"upright_flags length ({len(flags)}) must match positions_xy "
                f"length ({len(positions)})"
            )
        self.upright_flags = flags

        if isinstance(self.dt, bool) or not isinstance(self.dt, (int, float)):
            raise ValueError(f"dt: expected a number, got {type(self.dt).__name__}")
        self.dt = float(self.dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt: must be a finite value > 0, got {self.dt!r}")

    @staticmethod
    def _as_xy(value: object, index: int) -> tuple[float, float]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError(
                f"positions_xy[{index}]: expected a 2-element (x, y) sequence, "
                f"got {type(value).__name__}"
            )
        if len(value) != 2:
            raise ValueError(
                f"positions_xy[{index}]: expected exactly 2 elements, got {len(value)}"
            )
        x, y = value
        for name, comp in (("x", x), ("y", y)):
            if isinstance(comp, bool) or not isinstance(comp, (int, float)):
                raise ValueError(
                    f"positions_xy[{index}].{name}: expected a number, "
                    f"got {type(comp).__name__}"
                )
            if not math.isfinite(float(comp)):
                raise ValueError(f"positions_xy[{index}].{name}: must be finite")
        return (float(x), float(y))

    @property
    def num_steps(self) -> int:
        """Number of recorded timesteps (samples) in the episode."""
        return len(self.upright_flags)

    @property
    def episode_length_s(self) -> float:
        """Total episode duration in seconds (``num_steps * dt``)."""
        return self.num_steps * self.dt


# --------------------------------------------------------------------------- #
# Per-episode result
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeMetrics:
    """Goal-reaching measurements for a single episode.

    These are the per-episode quantities the aggregate :class:`EvalMetrics`
    (and, in Task 11.3, the staged gates) are built from. ``success`` here is the
    exact Req 9.2 predicate -- the robot reached within the Success_Radius of the
    Goal without first falling -- and ``time_to_goal_s`` is present iff
    ``success`` is ``True`` (Req 9.3).
    """

    distance_to_goal_m: float          # Req 9.1
    success: bool                      # Req 9.2 (within radius, no fall)
    fell: bool                         # Req 9.5
    time_to_goal_s: float | None       # Req 9.3 (present iff success)
    path_efficiency: float             # Req 9.4 (in (0, 1])
    upright_time_s: float              # Req 9.5 (non-negative, <= episode length)
    net_progress_m: float              # net displacement toward Goal (for Req 17.1 gate)
    # optional secondary quality metrics (Req 9.6); None when inputs absent
    avg_forward_speed_mps: float | None = None
    energy_efficiency: float | None = None
    gait_smoothness: float | None = None
    symmetry_score: float | None = None


# --------------------------------------------------------------------------- #
# Geometry / signal helpers (pure)
# --------------------------------------------------------------------------- #
def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_length(points: Sequence[tuple[float, float]]) -> float:
    """Total polyline length through ``points`` (0.0 for a single point)."""
    total = 0.0
    for prev, cur in zip(points, points[1:]):
        total += _distance(prev, cur)
    return total


def _first_reach_index(
    positions: Sequence[tuple[float, float]],
    goal_xy: tuple[float, float],
    radius: float,
) -> int | None:
    """Index of the first sample within ``radius`` of the Goal, else ``None``."""
    for i, p in enumerate(positions):
        if _distance(p, goal_xy) <= radius:
            return i
    return None


def _path_efficiency(straight_line_m: float, path_length_m: float) -> float:
    """Path efficiency in ``(0, 1]`` (Req 9.4).

    Defined as straight-line A->Goal distance divided by the actual path length
    traveled. For an episode that reaches the Goal the denominator is at least
    the numerator (a straight run is the shortest path), so the ratio is in
    ``(0, 1]`` naturally; for episodes that wander or barely move the raw ratio
    is clamped into ``(0, 1]`` so the documented invariant always holds.

    Degenerate cases:
      * start already at the Goal (straight-line 0) and no movement -> ``1.0``;
      * no movement but not at the Goal -> the clamp floor (``_EPS``), reflecting
        zero useful progress without returning exactly 0.
    """
    if path_length_m <= 0.0:
        return 1.0 if straight_line_m <= 0.0 else _EPS
    if straight_line_m <= 0.0:
        # Started at the Goal but still moved: not "efficient" travel toward it.
        return _EPS
    ratio = straight_line_m / path_length_m
    if ratio > 1.0:
        return 1.0
    if ratio < _EPS:
        return _EPS
    return ratio


def _upright_time_s(upright_flags: Sequence[bool], dt: float) -> float:
    """Seconds the robot spent upright (Req 9.5)."""
    return sum(1 for f in upright_flags if f) * dt


def _net_progress_m(
    positions: Sequence[tuple[float, float]], goal_xy: tuple[float, float]
) -> float:
    """Net displacement *toward* the Goal: start->Goal minus end->Goal distance.

    Positive when the robot ended closer to the Goal than it started. Used by the
    Task 11.3 ``makes_progress`` gate (Req 17.1); surfaced here because it is a
    pure function of the trajectory and the Goal.
    """
    start_dist = _distance(positions[0], goal_xy)
    end_dist = _distance(positions[-1], goal_xy)
    return start_dist - end_dist


def _avg_forward_speed_mps(path_length_m: float, episode_length_s: float) -> float | None:
    """Average speed along the path = distance traveled / elapsed time (Req 9.6)."""
    if episode_length_s <= 0.0:
        return None
    return path_length_m / episode_length_s


def _energy_efficiency(
    rewards: Sequence[float] | None, torques: Sequence[Sequence[float]] | None
) -> float | None:
    """Reward per squared torque (Req 9.6).

    Numerator is total episode reward; denominator is the sum of squared joint
    torques over all steps and joints. ``None`` when either signal is absent or
    no torque was applied (an undefined ratio).
    """
    if rewards is None or torques is None:
        return None
    total_reward = float(sum(rewards))
    sq_torque = 0.0
    for step in torques:
        for t in step:
            sq_torque += float(t) * float(t)
    if sq_torque <= 0.0:
        return None
    return total_reward / sq_torque


def _gait_smoothness(
    positions: Sequence[tuple[float, float]], dt: float
) -> float | None:
    """Gait smoothness in ``(0, 1]`` from path acceleration (Req 9.6).

    Smoothness is the inverse of mean acceleration magnitude along the path:
    ``1 / (1 + mean|a|)``. A perfectly constant-velocity path scores ``1.0``;
    jerky, high-acceleration motion scores lower. ``None`` when there are too few
    samples to estimate acceleration (need at least 3 positions).
    """
    if len(positions) < 3:
        return None
    velocities = [
        ((cur[0] - prev[0]) / dt, (cur[1] - prev[1]) / dt)
        for prev, cur in zip(positions, positions[1:])
    ]
    accel_mags = [
        math.hypot((v2[0] - v1[0]) / dt, (v2[1] - v1[1]) / dt)
        for v1, v2 in zip(velocities, velocities[1:])
    ]
    if not accel_mags:
        return None
    mean_accel = sum(accel_mags) / len(accel_mags)
    return 1.0 / (1.0 + mean_accel)


def _symmetry_score(
    left: Sequence[float] | None, right: Sequence[float] | None
) -> float | None:
    """Left/right gait symmetry in ``[0, 1]`` (Req 9.6).

    Compares the summed left and right foot signals: ``1 - |L - R| / (L + R)``,
    so balanced loading scores ``1.0`` and fully one-sided gait scores ``0.0``.
    ``None`` when either signal is absent or both sum to zero (undefined).
    """
    if left is None or right is None:
        return None
    left_sum = float(sum(abs(float(v)) for v in left))
    right_sum = float(sum(abs(float(v)) for v in right))
    denom = left_sum + right_sum
    if denom <= 0.0:
        return None
    return 1.0 - abs(left_sum - right_sum) / denom


# --------------------------------------------------------------------------- #
# Per-episode computation
# --------------------------------------------------------------------------- #
def compute_episode_metrics(
    trajectory: EpisodeTrajectory,
    goal: Goal,
    *,
    include_secondary: bool = False,
) -> EpisodeMetrics:
    """Compute the goal-reaching metrics for a single episode (Req 9.1-9.6).

    Args:
        trajectory: The recorded :class:`EpisodeTrajectory`.
        goal: The :class:`~src.data_models.Goal` (point B + Success_Radius).
        include_secondary: When ``True``, also compute the optional secondary
            quality metrics (Req 9.6) from the trajectory's torque/reward/foot
            signals; each is ``None`` if its required signal is absent.

    Returns:
        An :class:`EpisodeMetrics`. ``success`` is ``True`` iff the robot reached
        within ``goal.success_radius_m`` while still upright (Req 9.2), and
        ``time_to_goal_s`` is the time of that first arrival, present iff
        ``success`` (Req 9.3).
    """
    positions = trajectory.positions_xy
    goal_xy = goal.position_xy
    radius = goal.success_radius_m
    dt = trajectory.dt

    # Req 9.1: distance to Goal at episode end.
    distance_to_goal_m = _distance(positions[-1], goal_xy)

    # Req 9.2/9.3: arrival within radius while upright, and its timestamp.
    reach_index = _first_reach_index(positions, goal_xy, radius)
    upright_until_reach = (
        reach_index is not None
        and all(trajectory.upright_flags[: reach_index + 1])
    )
    success = bool(reach_index is not None and upright_until_reach)
    time_to_goal_s = float(reach_index * dt) if success else None

    # Req 9.5: a fall occurred if any step is non-upright.
    fell = not all(trajectory.upright_flags)

    # Req 9.4: path efficiency over the traveled polyline.
    straight_line_m = _distance(positions[0], goal_xy)
    path_length_m = _path_length(positions)
    path_efficiency = _path_efficiency(straight_line_m, path_length_m)

    # Req 9.5: upright time.
    upright_time_s = _upright_time_s(trajectory.upright_flags, dt)

    net_progress_m = _net_progress_m(positions, goal_xy)

    avg_forward_speed_mps = energy_efficiency = None
    gait_smoothness = symmetry_score = None
    if include_secondary:
        avg_forward_speed_mps = _avg_forward_speed_mps(
            path_length_m, trajectory.episode_length_s
        )
        energy_efficiency = _energy_efficiency(trajectory.rewards, trajectory.torques)
        gait_smoothness = _gait_smoothness(positions, dt)
        symmetry_score = _symmetry_score(
            trajectory.left_contacts, trajectory.right_contacts
        )

    return EpisodeMetrics(
        distance_to_goal_m=distance_to_goal_m,
        success=success,
        fell=fell,
        time_to_goal_s=time_to_goal_s,
        path_efficiency=path_efficiency,
        upright_time_s=upright_time_s,
        net_progress_m=net_progress_m,
        avg_forward_speed_mps=avg_forward_speed_mps,
        energy_efficiency=energy_efficiency,
        gait_smoothness=gait_smoothness,
        symmetry_score=symmetry_score,
    )


# --------------------------------------------------------------------------- #
# Staged goal gates (Task 11.3 -- Req 17)
# --------------------------------------------------------------------------- #
class _ConfigLike(Protocol):
    """Structural view of the gate-relevant Config fields (Req 17.4).

    Declared as a protocol so :class:`GateThresholds` can be built from the real
    :class:`src.config.Config` without :mod:`src.eval.evaluator` importing it,
    keeping the evaluator free of any import cycle.
    """

    min_progress_distance_m: float
    time_to_goal_threshold_s: float
    path_efficiency_threshold: float


@dataclass(frozen=True)
class GateThresholds:
    """The Config-sourced thresholds the staged goal gates are evaluated against.

    All three values originate from the Config (Req 17.4); the Goal supplies the
    Success_Radius used by the ``reaches_goal`` / ``efficient_goal`` predicates.

    Attributes:
        min_progress_distance_m: Minimum net displacement toward the Goal for the
            ``makes_progress`` gate (Req 17.1).
        time_to_goal_threshold_s: Maximum time-to-goal for the ``efficient_goal``
            gate (Req 17.3).
        path_efficiency_threshold: Minimum path efficiency for the
            ``efficient_goal`` gate (Req 17.3).
    """

    min_progress_distance_m: float
    time_to_goal_threshold_s: float
    path_efficiency_threshold: float

    @classmethod
    def from_config(cls, config: _ConfigLike) -> "GateThresholds":
        """Build the thresholds from a Config (Req 17.4).

        Reads ``min_progress_distance_m``, ``time_to_goal_threshold_s``, and
        ``path_efficiency_threshold`` straight off the validated Config so the
        gates use the same operator-supplied bounds as the rest of the run.
        """
        return cls(
            min_progress_distance_m=float(config.min_progress_distance_m),
            time_to_goal_threshold_s=float(config.time_to_goal_threshold_s),
            path_efficiency_threshold=float(config.path_efficiency_threshold),
        )


def _episode_gate_flags(
    episode: EpisodeMetrics, thresholds: GateThresholds
) -> tuple[bool, bool, bool]:
    """Per-episode ``(makes_progress, reaches_goal, efficient_goal)`` flags.

    The three flags are computed so they are **nested** for the episode -- a
    higher flag implies the lower ones -- which is what makes the aggregate gates
    staged-monotonic (Req 17):

      * ``reaches_goal`` is exactly the Req 9.2 success predicate (within the
        Success_Radius without falling) already resolved on the episode
        (Req 17.2).
      * ``efficient_goal`` requires ``reaches_goal`` AND a time-to-goal at or
        under the threshold AND a path efficiency at or above the threshold
        (Req 17.3).
      * ``makes_progress`` is the raw net-progress predicate (Req 17.1) OR-ed with
        ``reaches_goal`` so that an episode which reached the Goal always counts
        as having made progress, even when it started closer than the
        minimum-progress distance. This is the staging guarantee: reaching the
        Goal is a strictly higher capability than merely making progress.
    """
    reaches_goal = episode.success
    efficient_goal = (
        reaches_goal
        and episode.time_to_goal_s is not None
        and episode.time_to_goal_s <= thresholds.time_to_goal_threshold_s
        and episode.path_efficiency >= thresholds.path_efficiency_threshold
    )
    makes_progress = (
        episode.net_progress_m > thresholds.min_progress_distance_m or reaches_goal
    )
    return (makes_progress, reaches_goal, efficient_goal)


def compute_capability_gates(
    episodes: Sequence[EpisodeMetrics], thresholds: GateThresholds
) -> CapabilityGates:
    """Compute the staged goal gates over evaluated episodes (Req 17).

    Each gate is met when **any** evaluated episode meets it -- the same
    "achieved at least once" aggregation the :class:`EvalMetrics` ``success`` field
    uses (so ``reaches_goal`` aligns with ``success``). Because the per-episode
    flags are nested (:func:`_episode_gate_flags`), OR-aggregating them preserves
    the staged-monotonic relationship at the aggregate level:
    ``efficient_goal => reaches_goal => makes_progress`` (Req 17.1-17.3).

    Args:
        episodes: The per-episode metrics for the evaluated episodes. Must be
            non-empty.
        thresholds: The Config-sourced :class:`GateThresholds` (Req 17.4).

    Returns:
        The aggregate :class:`CapabilityGates`.

    Raises:
        ValueError: If ``episodes`` is empty.
    """
    if not episodes:
        raise ValueError("compute_capability_gates requires at least one episode")

    makes_progress = reaches_goal = efficient_goal = False
    for episode in episodes:
        mp, rg, eg = _episode_gate_flags(episode, thresholds)
        makes_progress = makes_progress or mp
        reaches_goal = reaches_goal or rg
        efficient_goal = efficient_goal or eg

    return CapabilityGates(
        makes_progress=makes_progress,
        reaches_goal=reaches_goal,
        efficient_goal=efficient_goal,
    )


# --------------------------------------------------------------------------- #
# Aggregate computation
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def compute_eval_metrics(
    trajectories: Sequence[EpisodeTrajectory],
    goal: Goal,
    *,
    include_secondary: bool = False,
    gates: CapabilityGates | None = None,
    gate_thresholds: GateThresholds | None = None,
) -> EvalMetrics:
    """Aggregate per-episode metrics into an :class:`EvalMetrics` (Req 9, 17).

    Runs :func:`compute_episode_metrics` over every episode and combines them per
    the module's aggregation contract: rates are the mean of the per-episode 0/1
    indicators (Req 9.2, 9.5); distance, path efficiency, and upright time are
    per-episode means; time-to-goal is the mean over successful episodes (or
    ``None``); and ``success`` is ``True`` iff any episode succeeded.

    The staged goal gates (Req 17) are resolved as follows, in priority order:

      1. an explicit ``gates`` value is used as-is (caller already computed them);
      2. otherwise, when ``gate_thresholds`` is supplied, the gates are computed
         from the per-episode metrics via :func:`compute_capability_gates` using
         those Config-sourced thresholds (Req 17.1-17.4);
      3. otherwise a documented all-false placeholder is used so the returned
         :class:`EvalMetrics` is still complete and serializable.

    Args:
        trajectories: One or more recorded episodes. Must be non-empty.
        goal: The :class:`~src.data_models.Goal` evaluated against.
        include_secondary: Enable the secondary quality metrics (Req 9.6).
        gates: A precomputed :class:`CapabilityGates`. When given it takes
            precedence over ``gate_thresholds``.
        gate_thresholds: The Config-sourced :class:`GateThresholds` used to compute
            the staged goal gates (Req 17.4) when ``gates`` is not supplied.

    Returns:
        A fully-populated :class:`EvalMetrics` (Req 9), ready for ``to_json``
        (Req 9.7).

    Raises:
        ValueError: If ``trajectories`` is empty.
    """
    if not trajectories:
        raise ValueError("compute_eval_metrics requires at least one episode")

    episodes = [
        compute_episode_metrics(t, goal, include_secondary=include_secondary)
        for t in trajectories
    ]

    success_rate = _mean([1.0 if e.success else 0.0 for e in episodes])
    fall_rate = _mean([1.0 if e.fell else 0.0 for e in episodes])
    distance_to_goal_m = _mean([e.distance_to_goal_m for e in episodes])
    path_efficiency = _mean([e.path_efficiency for e in episodes])
    upright_time_s = _mean([e.upright_time_s for e in episodes])

    # time-to-goal: mean over successful episodes only; None when none succeeded.
    successful_times = [
        e.time_to_goal_s for e in episodes if e.success and e.time_to_goal_s is not None
    ]
    time_to_goal_s = _mean(successful_times) if successful_times else None

    # Aggregate success: at least one episode reached the Goal without falling,
    # which keeps "time_to_goal present iff success" exact.
    success = success_rate > 0.0

    avg_forward_speed_mps = energy_efficiency = None
    gait_smoothness = symmetry_score = None
    if include_secondary:
        avg_forward_speed_mps = _mean_optional([e.avg_forward_speed_mps for e in episodes])
        energy_efficiency = _mean_optional([e.energy_efficiency for e in episodes])
        gait_smoothness = _mean_optional([e.gait_smoothness for e in episodes])
        symmetry_score = _mean_optional([e.symmetry_score for e in episodes])

    if gates is None:
        if gate_thresholds is not None:
            # Staged goal gates from Config-sourced thresholds (Task 11.3, Req 17).
            gates = compute_capability_gates(episodes, gate_thresholds)
        else:
            # No explicit gates and no thresholds: keep a documented all-false
            # placeholder so EvalMetrics stays complete and serializable.
            gates = CapabilityGates(
                makes_progress=False, reaches_goal=False, efficient_goal=False
            )

    return EvalMetrics(
        distance_to_goal_m=distance_to_goal_m,
        success=success,
        success_rate=success_rate,
        time_to_goal_s=time_to_goal_s,
        path_efficiency=path_efficiency,
        upright_time_s=upright_time_s,
        fall_rate=fall_rate,
        gates=gates,
        avg_forward_speed_mps=avg_forward_speed_mps,
        energy_efficiency=energy_efficiency,
        gait_smoothness=gait_smoothness,
        symmetry_score=symmetry_score,
    )


# --------------------------------------------------------------------------- #
# Best/worst demo-video recording (Task 11.4 -- Req 10)
# --------------------------------------------------------------------------- #
# The selection metric the Orchestrator compares policies on also orders episodes
# for best/worst demo selection (design.md -> Evaluator: "selecting best/worst by
# the selection metric"). It names a numeric field of EvalMetrics / EpisodeMetrics.
SELECTION_METRIC_DEFAULT = "success_rate"

# A few valid Config ``selection_metric`` names are *aggregate-only* fields of
# EvalMetrics (they have no per-episode counterpart of the same name). For
# per-episode best/worst selection they map onto the equivalent EpisodeMetrics
# field, so demo selection can order episodes by the same operator-chosen metric.
SELECTION_METRIC_EPISODE_ALIAS = {
    "success_rate": "success",
    "fall_rate": "fell",
}


def episode_selection_metric(selection_metric: str) -> str:
    """Map an EvalMetrics selection metric to its per-episode EpisodeMetrics field.

    Aggregate-only fields (``success_rate`` -> ``success``, ``fall_rate`` ->
    ``fell``) are translated to the per-episode field of the same meaning so
    :func:`episode_score` can order episodes; every other metric name is returned
    unchanged.
    """
    return SELECTION_METRIC_EPISODE_ALIAS.get(selection_metric, selection_metric)

# For most metrics a *higher* score is a better episode (success_rate,
# path_efficiency, upright_time_s, ...). For these a *lower* value is better, so
# the score is negated before argmax/argmin so that "best == max score" always
# holds (Req 10.1, Property 30).
LOWER_IS_BETTER_METRICS = frozenset(
    {
        "distance_to_goal_m",  # closer to the Goal at episode end is better
        "time_to_goal_s",      # arriving sooner is better
        "fall_rate",           # falling less is better
        "fell",                # (per-episode) not falling is better
    }
)


def episode_score(episode: EpisodeMetrics, selection_metric: str) -> float:
    """Score one episode by the configured selection metric (Req 10.1).

    The score is oriented so a **larger** score is always a better episode, so
    best = argmax and worst = argmin regardless of the metric's natural
    direction. Metrics in :data:`LOWER_IS_BETTER_METRICS` are negated; a metric
    whose per-episode value is ``None`` (e.g. ``time_to_goal_s`` for an episode
    that never reached the Goal) sorts to the worst end:

      * for a higher-is-better metric, ``None`` scores ``-inf``;
      * for a lower-is-better metric, ``None`` scores ``-inf`` too (after the
        sign flip), i.e. a missing arrival time is the worst possible outcome.

    Args:
        episode: The per-episode metrics to score.
        selection_metric: The name of the :class:`EpisodeMetrics` field to score
            on (must be one the episode carries).

    Returns:
        A finite float, or ``-inf`` when the metric value is unavailable.

    Raises:
        ValueError: If ``selection_metric`` is not a scoreable episode field.
    """
    if not hasattr(episode, selection_metric):
        raise ValueError(
            f"selection_metric {selection_metric!r} is not a scoreable episode "
            f"field; expected one of "
            f"{sorted(f.name for f in fields(EpisodeMetrics))}"
        )
    raw = getattr(episode, selection_metric)
    if raw is None:
        # Missing value -> worst possible episode under either orientation.
        return float("-inf")
    if isinstance(raw, bool):
        value = 1.0 if raw else 0.0
    else:
        value = float(raw)
    return -value if selection_metric in LOWER_IS_BETTER_METRICS else value


@dataclass(frozen=True)
class BestWorstSelection:
    """Indices + scores of the best and worst episodes in a rollout (Req 10.1).

    Attributes:
        best_index / worst_index: Positions into the scored episode sequence of
            the highest- and lowest-scoring episode.
        best_score / worst_score: Their oriented scores (higher == better, see
            :func:`episode_score`). ``best_score >= worst_score`` always holds.
        selection_metric: The metric the selection was made on.
    """

    best_index: int
    worst_index: int
    best_score: float
    worst_score: float
    selection_metric: str


def select_best_worst(
    episodes: Sequence[EpisodeMetrics],
    selection_metric: str = SELECTION_METRIC_DEFAULT,
) -> BestWorstSelection:
    """Pick the best (max score) and worst (min score) episode (Req 10.1).

    Pure argmax/argmin over :func:`episode_score`, so this is fully unit-testable
    with synthetic :class:`EpisodeMetrics` and no Isaac Sim present (Property 30).
    Ties resolve to the **first** occurrence (lowest index) for both ends, which
    keeps the choice deterministic. When a single episode is supplied it is both
    the best and the worst.

    Args:
        episodes: The per-episode metrics for the rolled-out episodes. Must be
            non-empty.
        selection_metric: The metric to order episodes by.

    Returns:
        A :class:`BestWorstSelection`.

    Raises:
        ValueError: If ``episodes`` is empty.
    """
    if not episodes:
        raise ValueError("select_best_worst requires at least one episode")

    scores = [episode_score(e, selection_metric) for e in episodes]

    best_index = 0
    worst_index = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best_index]:
            best_index = i
        if scores[i] < scores[worst_index]:
            worst_index = i

    return BestWorstSelection(
        best_index=best_index,
        worst_index=worst_index,
        best_score=scores[best_index],
        worst_score=scores[worst_index],
        selection_metric=selection_metric,
    )


# --------------------------------------------------------------------------- #
# Injected rollout + recorder contracts (guarded; Isaac Sim only)
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRollout:
    """One rolled-out evaluation episode: its trajectory plus replay handle.

    The :class:`PolicyRollout` produces these by running the trained policy in
    the live (small env count) environment. ``trajectory`` feeds the pure metric
    + selection logic; ``replay`` is an opaque, simulator-specific handle the
    :class:`DemoVideoRecorder` consumes to re-render the episode through the
    external cameras (e.g. recorded actions, an env-index + seed, or a cached
    frame buffer). Selection never touches ``replay``, so best/worst selection
    stays simulator-free and testable.
    """

    trajectory: EpisodeTrajectory
    replay: Any = None


@runtime_checkable
class PolicyRollout(Protocol):
    """Runs a trained policy at a small env count to produce evaluation episodes.

    The production implementation drives the goal-conditioned Isaac Lab env with
    the external cameras attached at a small env count (~50, Req 10.4); tests
    inject a fake returning canned :class:`EpisodeRollout`s. Because this is an
    injected dependency, :mod:`src.eval.evaluator` never imports Isaac Sim.
    """

    def rollout(
        self, checkpoint: Any, goal: Goal, num_episodes: int
    ) -> Sequence[EpisodeRollout]:
        ...


@runtime_checkable
class DemoVideoRecorder(Protocol):
    """Renders one rolled-out episode to RGB video files via the demo cameras.

    Implementations (Isaac Sim-backed in production) MUST render through the
    supplied external, world-frame :class:`~src.sensors.camera_cfg.CameraSpec`
    cameras (Req 10.3) at a small env count (Req 10.4). The producer passes the
    chase + side specs from :class:`~src.sensors.camera_cfg.CameraConfig`, so the
    "external world-frame cameras only" contract is enforced at the call site.
    """

    def record(
        self,
        *,
        rollout: EpisodeRollout,
        cameras: Sequence[CameraSpec],
        label: str,
        output_dir: str,
    ) -> Sequence[str]:
        ...


@dataclass
class DemoVideo:
    """A recorded demo video for one selected (best or worst) episode (Req 10.1).

    Attributes:
        label: ``"best"`` or ``"worst"``.
        episode_index: Index of the selected episode within the rollout.
        score: The episode's oriented selection score (higher == better).
        camera_names: The cameras the video was rendered through (chase, side).
        video_paths: Local paths of the rendered RGB video files, one per camera.
    """

    label: str
    episode_index: int
    score: float
    camera_names: tuple[str, ...]
    video_paths: tuple[str, ...]


@dataclass
class DemoVideoResult:
    """Outcome of a best/worst demo-video recording cycle (Req 10.1).

    ``best`` / ``worst`` are ``None`` when recording did not run (no rollout or no
    recorder injected, or no episodes produced), so the Evaluator can proceed
    without demo video on the controller/dev host. ``selection`` carries the
    best/worst indices + scores when a selection was made.
    """

    best: DemoVideo | None = None
    worst: DemoVideo | None = None
    selection: BestWorstSelection | None = None


class DemoVideoProducer:
    """Selects and records best/worst evaluation demo video (Task 11.4, Req 10).

    Coordinates three concerns, only the first of which runs on the controller
    host:

      1. **Selection (pure).** Given per-episode metrics, pick best/worst by the
         configured selection metric via :func:`select_best_worst` (Req 10.1).
      2. **Rollout (injected, Isaac Sim).** Run the trained policy at a small env
         count to produce :class:`EpisodeRollout`s (Req 10.4).
      3. **Recording (injected, Isaac Sim).** Render the selected episodes to RGB
         video through the external world-frame chase + side cameras (Req 10.3).

    Both the rollout and the recorder are **injected and guarded**: when either is
    absent the producer is a safe no-op returning an empty
    :class:`DemoVideoResult`, exactly like :class:`src.capture.TrainingCaptureProducer`
    when no renderer is available. This keeps demo recording strictly additive and
    lets best/worst selection be unit-tested with fakes and no GPU.

    Parameters:
        cameras: The :class:`~src.sensors.camera_cfg.CameraConfig` supplying the
            external world-frame chase + side cameras (Req 10.2, 10.3).
        selection_metric: The metric episodes are ordered by (default
            ``success_rate``); should match the run's Config ``selection_metric``.
        num_episodes: How many episodes to roll out for selection (a small count;
            Req 10.4). Must be >= 1.
        include_secondary: Whether per-episode metric computation should populate
            the secondary quality metrics (only relevant when the selection metric
            is a secondary one).
        rollout: Injected :class:`PolicyRollout`; ``None`` -> no-op producer.
        recorder: Injected :class:`DemoVideoRecorder`; ``None`` -> no-op producer.
        output_dir: Local directory the recorder writes video files under.
    """

    def __init__(
        self,
        cameras: CameraConfig,
        *,
        selection_metric: str = SELECTION_METRIC_DEFAULT,
        num_episodes: int = 8,
        include_secondary: bool = False,
        rollout: PolicyRollout | None = None,
        recorder: DemoVideoRecorder | None = None,
        output_dir: str = "runs/demo_videos",
    ) -> None:
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
        self._cameras = cameras
        self._selection_metric = selection_metric
        self._num_episodes = int(num_episodes)
        self._include_secondary = bool(include_secondary)
        self._rollout = rollout
        self._recorder = recorder
        self._output_dir = output_dir

    @classmethod
    def from_config(
        cls,
        config: Any,
        cameras: CameraConfig,
        *,
        rollout: PolicyRollout | None = None,
        recorder: DemoVideoRecorder | None = None,
        output_dir: str = "runs/demo_videos",
        num_episodes: int = 8,
    ) -> "DemoVideoProducer":
        """Build a producer using the run Config's ``selection_metric`` (Req 10.1).

        Reads ``selection_metric`` off the validated :class:`src.config.Config` so
        best/worst demo selection uses the same metric the Orchestrator tracks the
        Best_Policy on; falls back to the default when absent.
        """
        selection_metric = getattr(config, "selection_metric", SELECTION_METRIC_DEFAULT)
        return cls(
            cameras,
            selection_metric=selection_metric,
            num_episodes=num_episodes,
            rollout=rollout,
            recorder=recorder,
            output_dir=output_dir,
        )

    def record_best_worst(
        self, checkpoint: Any, goal: Goal
    ) -> DemoVideoResult:
        """Roll out, select, and record the best/worst demo videos (Req 10).

        Runs the injected policy rollout at the configured small env count, scores
        every episode by the selection metric, picks best/worst
        (:func:`select_best_worst`), and renders each through the external
        chase + side cameras (Req 10.1, 10.3, 10.4). A no-op returning an empty
        :class:`DemoVideoResult` when no rollout or recorder is injected, or when
        the rollout yields no episodes -- so an evaluation run on a host without
        Isaac Sim simply produces no demo video instead of failing.

        Args:
            checkpoint: The trained-policy checkpoint reference to roll out.
            goal: The :class:`~src.data_models.Goal` evaluated against.

        Returns:
            A :class:`DemoVideoResult` with the recorded best/worst videos and the
            selection that produced them.
        """
        if self._rollout is None or self._recorder is None:
            return DemoVideoResult()

        rollouts = list(self._rollout.rollout(checkpoint, goal, self._num_episodes))
        if not rollouts:
            return DemoVideoResult()

        episodes = [
            compute_episode_metrics(
                r.trajectory, goal, include_secondary=self._include_secondary
            )
            for r in rollouts
        ]
        selection = select_best_worst(
            episodes, episode_selection_metric(self._selection_metric)
        )

        best = self._record_one(rollouts, selection.best_index, "best", selection.best_score)
        # Avoid re-rendering the same episode when best == worst (single episode
        # or a tie that resolved to one index): reuse the best video as the worst.
        if selection.worst_index == selection.best_index:
            worst = (
                DemoVideo(
                    label="worst",
                    episode_index=best.episode_index,
                    score=selection.worst_score,
                    camera_names=best.camera_names,
                    video_paths=best.video_paths,
                )
                if best is not None
                else None
            )
        else:
            worst = self._record_one(
                rollouts, selection.worst_index, "worst", selection.worst_score
            )

        return DemoVideoResult(best=best, worst=worst, selection=selection)

    def _record_one(
        self,
        rollouts: Sequence[EpisodeRollout],
        episode_index: int,
        label: str,
        score: float,
    ) -> DemoVideo | None:
        """Render one selected episode through the external cameras (Req 10.3)."""
        cameras = self._cameras.cameras()  # (chase, side); both /World, RGB
        paths = self._recorder.record(
            rollout=rollouts[episode_index],
            cameras=cameras,
            label=label,
            output_dir=self._output_dir,
        )
        if not paths:
            return None
        return DemoVideo(
            label=label,
            episode_index=episode_index,
            score=score,
            camera_names=tuple(c.name for c in cameras),
            video_paths=tuple(str(p) for p in paths),
        )


# --------------------------------------------------------------------------- #
# Live Isaac-Sim-backed rollout + recorder, and the top-level Evaluator
# (Task: live policy-rollout backend; Req 9, 10, 17)
# --------------------------------------------------------------------------- #
#
# Everything above this point is pure host-side code (metric computation,
# best/worst selection, the injected-dependency producer). The classes below are
# the *live*, simulator-backed implementations of the :class:`PolicyRollout` and
# :class:`DemoVideoRecorder` Protocols plus the top-level :class:`Evaluator` the
# Orchestrator calls.
#
# Import discipline (mirrors :func:`src.train.ppo_runner.build_isaaclab_trainer`
# and :mod:`src.envs.goal_env`):
#   * NO ``isaaclab`` / ``gymnasium`` / ``rsl_rl`` / ``torch`` import lives at
#     module scope. Every such import is lazy and INSIDE a method, guarded by a
#     try/except that raises a descriptive RuntimeError on the controller/dev
#     host. So importing :mod:`src.eval.evaluator` never requires Isaac Sim, and
#     the pure path (with injected fakes) stays unit-testable on a CPU box.
#   * The live code paths are ``# pragma: no cover`` — they only run inside the
#     Isaac Sim runtime, which CI does not have.
#
# rsl_rl / Isaac Lab API surface targeted (CONFIRM on the GPU box; versions drift):
#   * Env construction: ``gymnasium.make(task_id, cfg=env_cfg, render_mode=...)``
#     then ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper(env)`` (older trees expose this
#     at ``omni.isaac.lab_tasks.utils.wrappers.rsl_rl``). The goal-conditioned cfg
#     is produced via :func:`src.envs.goal_env.attach_goal_conditioning` against
#     the stock ``Isaac-Velocity-Flat-H1-v0`` cfg at a SMALL env count.
#   * Policy load: ``rsl_rl.runners.OnPolicyRunner(env, train_cfg, log_dir, device)``
#     then ``runner.load(checkpoint_path)`` and ``policy = runner.get_inference_policy(device)``.
#     Duck-typed: we accept an already-callable policy, a ``get_inference_policy``,
#     an ``act_inference``/``act`` method, or a TorchScript ``.pt`` loaded via
#     ``torch.jit.load`` — whichever the checkpoint/runner exposes.
#   * Per-step state read: the same articulation surface :mod:`src.envs.goal_env`
#     reads — ``env.unwrapped.scene["robot"].data`` with ``root_pos_w`` (num_envs,3)
#     and ``projected_gravity_b`` (num_envs,3); upright is decided from the z of the
#     projected gravity (or a quaternion-derived tilt fallback).
#   * Cameras: :meth:`src.sensors.camera_cfg.CameraSpec.to_isaaclab_cfg` builds the
#     ``isaaclab.sensors.CameraCfg``; the recorder reads ``camera.data.output["rgb"]``
#     each frame and muxes to file via ``imageio`` (preferred) / ``cv2`` fallback.
#
# These assumptions are intentionally duck-typed so minor rsl_rl/Isaac Lab version
# differences do not break the rollout; anything missing raises a clear error.

# A small parallel env count for evaluation so the external world-frame cameras
# can render (Req 10.4: "small env count ~50"). The live env is built at
# ``min(num_episodes, EVAL_ENV_COUNT_DEFAULT)`` so one batched rollout pass
# typically yields every requested episode at once.
EVAL_ENV_COUNT_DEFAULT = 50

# Upright decision threshold on the robot-frame projected-gravity z-component.
# When the base is perfectly upright the gravity vector projected into the base
# frame points straight down, so ``projected_gravity_b[z] ≈ -1``. As the torso
# tilts the z-component rises toward 0; once it exceeds this bound the robot is
# considered fallen. (``cos(~65°) ≈ -0.42``; -0.5 ≈ 60° of tilt.)
UPRIGHT_PROJECTED_GRAVITY_Z_MAX = -0.5

# The secondary quality-metric field names (Req 9.6); used only to document the
# include_secondary toggle and to decide whether the rollout records torque /
# reward / foot-contact signals.
SECONDARY_METRICS = (
    "avg_forward_speed_mps",
    "energy_efficiency",
    "gait_smoothness",
    "symmetry_score",
)


@dataclass(frozen=True)
class EvalConfig:
    """Config-derived parameters the :class:`Evaluator` needs (Req 9, 10, 17).

    Pure data (no Isaac Lab / torch), built from the validated
    :class:`src.config.Config` via :meth:`from_config`, so an Evaluator can be
    constructed and its pure path unit-tested on the controller host.

    Attributes:
        selection_metric: The metric best/worst demo selection is ordered by; the
            same field the Orchestrator tracks the Best_Policy on (Req 10.1, 19.1).
        gate_thresholds: The Config-sourced staged-gate thresholds (Req 17.4).
        success_radius_m: The Goal arrival threshold (Req 9.2); kept here so the
            Evaluator can validate/echo it, though the authoritative value travels
            on the :class:`~src.data_models.Goal` passed to :meth:`Evaluator.evaluate`.
        fall_threshold_s: Maximum continuous airborne/tilted time before the robot
            is treated as fallen (Req 13.1); forwarded to the live rollout's upright
            bookkeeping.
        include_secondary: Whether to compute the secondary quality metrics (Req 9.6).
        eval_env_count: Small parallel env count for the live eval env (Req 10.4).
        demo_episodes: How many episodes the demo-video producer rolls out for
            best/worst selection (a small count; Req 10.4).
        video_output_dir: Local directory the demo recorder writes video under.
    """

    selection_metric: str = SELECTION_METRIC_DEFAULT
    gate_thresholds: GateThresholds | None = None
    success_radius_m: float = 0.5
    fall_threshold_s: float = 3.0
    include_secondary: bool = False
    eval_env_count: int = EVAL_ENV_COUNT_DEFAULT
    demo_episodes: int = 8
    video_output_dir: str = "runs-mujoco/demo_videos"
    env_name: str = "H1JoystickGaitTracking"

    @classmethod
    def from_config(cls, config: Any) -> "EvalConfig":
        """Build an :class:`EvalConfig` from a validated run :class:`~src.config.Config`.

        Reads ``selection_metric``, the staged-gate thresholds (via
        :meth:`GateThresholds.from_config`), ``success_radius_m``,
        ``fall_threshold_s``, and the Isaac Lab task id (``env_id``) straight off
        the Config so evaluation uses the same operator-supplied bounds and the
        same registered env name as training (Req 17.4, 18.2, 8.1).
        """
        return cls(
            selection_metric=getattr(config, "selection_metric", SELECTION_METRIC_DEFAULT),
            gate_thresholds=GateThresholds.from_config(config),
            success_radius_m=float(getattr(config, "success_radius_m", 0.5)),
            fall_threshold_s=float(getattr(config, "fall_threshold_s", 3.0)),
            include_secondary=bool(getattr(config, "include_secondary", False)),
            eval_env_count=int(getattr(config, "eval_env_count", EVAL_ENV_COUNT_DEFAULT)),
            demo_episodes=int(getattr(config, "evaluation_episodes", 8)),
            video_output_dir=str(getattr(config, "video_output_dir", "runs-mujoco/demo_videos")),
            env_name=str(getattr(config, "env_name", "H1JoystickGaitTracking")),
        )


# NOTE: The live Isaac Lab rollout/recorder classes were removed in the
# MuJoCo port. The MJX rollout + headless demo recorder live in
# src/eval/mjx_rollout.py and are built by build_mjx_policy_rollout /
# build_mjx_demo_recorder below.


def build_mjx_policy_rollout(
    goal: Goal, eval_config: EvalConfig
) -> PolicyRollout:  # pragma: no cover - requires the GPU runtime
    """Construct a live MJX :class:`PolicyRollout` (lazy GPU-stack check).

    Raises a descriptive :class:`RuntimeError` where the MJX/Brax stack is not
    importable, mirroring :func:`src.train.mjx_trainer.build_mjx_trainer`; inject
    a fake :class:`PolicyRollout` into the :class:`Evaluator` to run the pure
    path off-GPU. The concrete MJX rollout lives in ``src/eval/mjx_rollout.py``.
    """
    try:
        import jax  # noqa: F401, PLC0415
        import mujoco_playground  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "JAX / MuJoCo MJX are not importable in this environment. Inject a "
            "PolicyRollout into Evaluator(rollout=...) to evaluate without the "
            "GPU runtime."
        ) from exc

    from .mjx_rollout import MjxPolicyRollout  # noqa: PLC0415

    return MjxPolicyRollout(goal, eval_config)


def build_mjx_demo_recorder(
    goal: Goal, eval_config: EvalConfig
) -> DemoVideoRecorder:  # pragma: no cover - requires the GPU runtime
    """Construct a live MJX demo recorder (lazy GPU-stack check).

    Mirrors :func:`build_mjx_policy_rollout`: raises off-GPU so the controller
    host injects a fake recorder. The concrete recorder lives in
    ``src/eval/mjx_rollout.py`` and uses MuJoCo's headless ``Renderer`` (EGL,
    verified on the B200).
    """
    try:
        import mujoco  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not importable in this environment. Inject a "
            "DemoVideoRecorder into Evaluator(recorder=...) to evaluate without "
            "the GPU runtime."
        ) from exc

    from .mjx_rollout import MjxDemoVideoRecorder  # noqa: PLC0415

    return MjxDemoVideoRecorder(goal, eval_config)


# --------------------------------------------------------------------------- #
# Top-level Evaluator (the surface the Orchestrator calls)
# --------------------------------------------------------------------------- #
class Evaluator:
    """Goal-reaching policy evaluation + best/worst demo recording (Req 9, 10, 17).

    This is the surface the Orchestrator drives:
    ``evaluator.evaluate(checkpoint, goal, num_episodes) -> EvalMetrics``
    (design.md -> Evaluator). It composes the already-tested pure layer with the
    injected (or lazily-built, Isaac-Sim-backed) rollout + recorder:

      1. roll out ``num_episodes`` via the :class:`PolicyRollout` backend;
      2. compute per-episode metrics and the aggregate :class:`EvalMetrics` via
         :func:`compute_eval_metrics`, with the staged goal gates resolved from
         the Config-sourced :class:`GateThresholds` (Req 17);
      3. record best/worst demo video via :class:`DemoVideoProducer` through the
         external world-frame cameras (Req 10);
      4. attach the :class:`DemoVideoResult` onto the returned ``EvalMetrics`` as a
         ``demo_video`` attribute (the Orchestrator's per-iteration persistence
         reads ``metrics.demo_video``), and return the metrics.

    Dependency injection (the testability seam):
      * ``rollout`` / ``recorder`` are injected. Tests pass fakes (no Isaac Sim).
      * When neither is injected they are built lazily via
        :func:`build_isaaclab_policy_rollout` / :func:`build_isaaclab_demo_recorder`
        at ``evaluate`` time, which raises a clear :class:`RuntimeError` on a host
        without the Isaac Sim runtime — mirroring
        :func:`src.train.ppo_runner.build_isaaclab_trainer`.

    Importing this class never imports Isaac Sim; only calling ``evaluate``
    without an injected rollout does (on the GPU box).

    Parameters:
        eval_config: The :class:`EvalConfig` (selection metric, gate thresholds,
            success radius, fall threshold, include_secondary, env/episode counts,
            video output dir). Build it from a run Config via
            :meth:`EvalConfig.from_config`.
        cameras: The :class:`~src.sensors.camera_cfg.CameraConfig` supplying the
            external world-frame chase + side cameras (Req 10.2, 10.3).
        rollout: Injected :class:`PolicyRollout`; built lazily when ``None``.
        recorder: Injected :class:`DemoVideoRecorder`; built lazily when ``None``.
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        cameras: CameraConfig,
        *,
        rollout: PolicyRollout | None = None,
        recorder: DemoVideoRecorder | None = None,
    ) -> None:
        self._config = eval_config
        self._cameras = cameras
        self._rollout = rollout
        self._recorder = recorder

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        cameras: CameraConfig | None = None,
        rollout: PolicyRollout | None = None,
        recorder: DemoVideoRecorder | None = None,
    ) -> "Evaluator":
        """Build an :class:`Evaluator` from a validated run :class:`~src.config.Config`.

        Derives the :class:`EvalConfig` (selection metric + gate thresholds +
        success radius + fall threshold) and the :class:`CameraConfig` (video
        resolution/fps) straight off the Config, so the Evaluator uses the same
        operator-supplied settings as training and the Orchestrator. Inject
        ``rollout`` / ``recorder`` for host-side tests.
        """
        eval_config = EvalConfig.from_config(config)
        if cameras is None:
            cameras = CameraConfig.from_config(config)
        return cls(eval_config, cameras, rollout=rollout, recorder=recorder)

    @property
    def selection_metric(self) -> str:
        """The metric best/worst demo selection (and the Orchestrator) order by."""
        return self._config.selection_metric

    def evaluate(
        self, checkpoint: Any, goal: Goal, num_episodes: int
    ) -> EvalMetrics:
        """Evaluate ``checkpoint`` against ``goal`` over ``num_episodes`` (Req 9, 10, 17).

        Rolls out the policy, computes the aggregate :class:`EvalMetrics` with
        staged goal gates, records best/worst demo video, attaches the
        :class:`DemoVideoResult` as ``metrics.demo_video``, and returns the metrics.

        Args:
            checkpoint: The trained-policy checkpoint reference to evaluate.
            goal: The :class:`~src.data_models.Goal` (point B + Success_Radius).
            num_episodes: Number of evaluation episodes (>= 1).

        Returns:
            The aggregate :class:`EvalMetrics` with a ``demo_video``
            :class:`DemoVideoResult` attached.

        Raises:
            ValueError: If ``num_episodes`` < 1.
            RuntimeError: If no rollout backend is injected and Isaac Lab is not
                importable (controller/dev host) — mirroring
                :func:`src.train.ppo_runner.build_isaaclab_trainer`.
        """
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

        rollout_backend = self._resolve_rollout(goal)

        # (1) Roll out the policy to produce evaluated episodes.
        rollouts = list(rollout_backend.rollout(checkpoint, goal, num_episodes))  # type: ignore[union-attr]
        if not rollouts:
            raise RuntimeError(
                "policy rollout produced no episodes; cannot compute EvalMetrics"
            )
        trajectories = [r.trajectory for r in rollouts]

        # (2) Aggregate metrics + staged goal gates (Req 9, 17).
        metrics = compute_eval_metrics(
            trajectories,
            goal,
            include_secondary=self._config.include_secondary,
            gate_thresholds=self._config.gate_thresholds,
        )

        # (3) Best/worst demo video through the external cameras (Req 10). This is
        # strictly additive: a no-op (empty DemoVideoResult) when no recorder is
        # available, so evaluation never fails for lack of video.
        demo_result = self._record_demo_video(checkpoint, goal, rollout_backend)

        # (4) Surface the demo-video result for the Orchestrator's persistence,
        # which reads ``metrics.demo_video`` (src/orchestrator.py).
        setattr(metrics, "demo_video", demo_result)
        return metrics

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_rollout(self, goal: Goal) -> PolicyRollout:
        """Return the rollout backend: injected > lazily-built Isaac Lab."""
        if self._rollout is not None:
            return self._rollout
        return build_mjx_policy_rollout(goal, self._config)

    def _resolve_recorder(self, goal: Goal) -> DemoVideoRecorder | None:
        """Return the demo recorder: injected, else lazily-built, else ``None``.

        Unlike the rollout, a missing recorder is non-fatal: best/worst recording
        is additive (Req 10), so when no recorder is injected and Isaac Lab is not
        importable the Evaluator records no video rather than failing the run.
        """
        if self._recorder is not None:
            return self._recorder
        try:
            return build_mjx_demo_recorder(goal, self._config)
        except RuntimeError:
            return None

    def _record_demo_video(
        self, checkpoint: Any, goal: Goal, rollout_backend: PolicyRollout
    ) -> DemoVideoResult:
        """Run the :class:`DemoVideoProducer` to record best/worst demo video.

        Reuses the same rollout backend resolved for metric computation (injected
        fake in tests, or the lazily-built live rollout on the GPU box) so demo
        selection runs against the same policy without re-resolving Isaac Lab.
        """
        recorder = self._resolve_recorder(goal)
        producer = DemoVideoProducer(
            self._cameras,
            selection_metric=self._config.selection_metric,
            num_episodes=self._config.demo_episodes,
            include_secondary=self._config.include_secondary,
            rollout=rollout_backend,
            recorder=recorder,
            output_dir=self._config.video_output_dir,
        )
        return producer.record_best_worst(checkpoint, goal)
