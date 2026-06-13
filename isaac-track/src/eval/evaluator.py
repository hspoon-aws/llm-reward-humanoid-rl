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
    "IsaacLabPolicyRollout",
    "IsaacLabDemoVideoRecorder",
    "build_isaaclab_policy_rollout",
    "build_isaaclab_demo_recorder",
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
        self, checkpoint: Any, goal: Goal, *, rollouts: Any = None
    ) -> DemoVideoResult:
        """Roll out, select, and record the best/worst demo videos (Req 10).

        Scores every episode by the selection metric, picks best/worst
        (:func:`select_best_worst`), and renders each through the external
        chase + side cameras (Req 10.1, 10.3, 10.4). A no-op returning an empty
        :class:`DemoVideoResult` when no rollout or recorder is injected, or when
        the rollout yields no episodes -- so an evaluation run on a host without
        Isaac Sim simply produces no demo video instead of failing.

        ``rollouts`` lets the caller pass the episodes it ALREADY rolled out for
        metric computation, so best/worst selection does not trigger a SECOND
        in-process env rollout. This matters on Isaac Lab: a second manager-based
        env in one process stalls, so the Evaluator computes metrics once and
        feeds those rollouts here. When ``rollouts`` is omitted the producer rolls
        out itself (the standalone/test path).

        Args:
            checkpoint: The trained-policy checkpoint reference to roll out.
            goal: The :class:`~src.data_models.Goal` evaluated against.
            rollouts: Optional pre-computed episode rollouts to reuse.

        Returns:
            A :class:`DemoVideoResult` with the recorded best/worst videos and the
            selection that produced them.
        """
        if self._recorder is None:
            return DemoVideoResult()
        if rollouts is not None:
            rollouts = list(rollouts)
        elif self._rollout is not None:
            rollouts = list(self._rollout.rollout(checkpoint, goal, self._num_episodes))
        else:
            return DemoVideoResult()
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
    video_output_dir: str = "runs/demo_videos"
    env_id: str = "Isaac-Velocity-Flat-H1-v0"
    record_demo_video: bool = True

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
            video_output_dir=str(getattr(config, "video_output_dir", "runs/demo_videos")),
            env_id=str(getattr(config, "env_id", "Isaac-Velocity-Flat-H1-v0")),
            record_demo_video=bool(getattr(config, "record_demo_video", True)),
        )


# --------------------------------------------------------------------------- #
# Live policy rollout (Isaac Sim only; guarded + duck-typed)
# --------------------------------------------------------------------------- #
def _default_rsl_rl_eval_cfg() -> dict:  # pragma: no cover - used only on GPU
    """A default rsl_rl OnPolicyRunner cfg for loading a checkpoint for inference.

    Mirrors the network architecture in
    :meth:`src.train.ppo_runner._IsaacLabTrainer._build_train_cfg` (ActorCritic
    with [256, 256, 256] actor/critic MLPs, ELU) so a plain rsl_rl ``.pt`` save —
    which stores weights but no cfg — reloads cleanly for the eval rollout. Only
    the policy architecture must match the trained checkpoint; the PPO algorithm
    block is present to satisfy the runner constructor but is unused at inference.
    """
    return {
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "empirical_normalization": False,
        "seed": 0,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [256, 256, 256],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "save_interval": 1,
        "experiment_name": "h1_goal_reaching",
        "run_name": "",
        "logger": "tensorboard",
    }


class IsaacLabPolicyRollout:  # pragma: no cover - requires the Isaac Sim runtime
    """Live :class:`PolicyRollout`: rolls the trained policy out in Isaac Lab.

    Builds the goal-conditioned H1 eval env at a SMALL parallel env count
    (Req 10.4), loads the trained-policy ``checkpoint``, and steps the env for one
    episode per env recording, per timestep, the base ``(x, y)`` position and an
    upright flag (and, when ``include_secondary``, joint torques / rewards / foot
    contacts for the secondary metrics). Each env's recorded series becomes one
    :class:`EpisodeTrajectory`; the returned :class:`EpisodeRollout` carries an
    opaque ``replay`` handle (the env index + the goal) the
    :class:`IsaacLabDemoVideoRecorder` re-renders from.

    All ``isaaclab`` / ``gymnasium`` / ``rsl_rl`` / ``torch`` imports are lazy and
    inside methods, so importing :mod:`src.eval.evaluator` never needs Isaac Sim;
    everything here is ``# pragma: no cover`` because it only runs on the GPU box.
    Version differences across rsl_rl / Isaac Lab releases are tolerated by
    duck-typing the env wrapper, the policy handle, and the per-step state read.

    Parameters:
        goal: The :class:`~src.data_models.Goal` to broadcast into the eval env.
        task_id: The stock manager-based task the goal-conditioning is layered on
            (default ``Isaac-Velocity-Flat-H1-v0``).
        eval_env_count: Small parallel env count (Req 10.4).
        fall_threshold_s: Continuous-tilt budget before the robot is "fallen".
        include_secondary: Record torque/reward/foot signals for Req 9.6 metrics.
        device: torch device string (default ``"cuda:0"`` within the eval process,
            whose ``CUDA_VISIBLE_DEVICES`` the Evaluator/Orchestrator restricts).
        max_episode_steps: Hard cap on steps per episode (safety bound).
    """

    def __init__(
        self,
        goal: Goal,
        *,
        task_id: str = "Isaac-Velocity-Flat-H1-v0",
        eval_env_count: int = EVAL_ENV_COUNT_DEFAULT,
        fall_threshold_s: float = 3.0,
        include_secondary: bool = False,
        device: str = "cuda:0",
        max_episode_steps: int = 1000,
        shared_env: Any = None,
        shared_policy: Any = None,
    ) -> None:
        self._goal = goal
        self._task_id = task_id
        self._eval_env_count = int(eval_env_count)
        self._fall_threshold_s = float(fall_threshold_s)
        self._include_secondary = bool(include_secondary)
        self._device = device
        self._max_episode_steps = int(max_episode_steps)
        # Single-process train+eval reuse (Isaac Sim allows one SimulationApp per
        # process): when the trainer's already-live env (and optionally its
        # inference policy) are injected, the rollout reuses them instead of
        # building a SECOND env via gym.make — which would relaunch a second
        # SimulationApp and stall on the RTX rendering-kit reinit. The rollout
        # then does NOT own the env lifecycle, so it does not close it.
        self._shared_env = shared_env
        self._shared_policy = shared_policy

    # -- env construction ------------------------------------------------- #
    def _build_env(self, goal: Goal, num_envs: int) -> Any:
        """Build the goal-conditioned eval env wrapped for rsl_rl inference.

        Lazy-imports ``gymnasium`` + Isaac Lab, parses the stock env cfg at the
        small env count, layers goal-conditioning onto it
        (:func:`src.envs.goal_env.attach_goal_conditioning`), constructs the env,
        publishes the per-env :class:`~src.envs.goal_env.GoalBuffer` at
        :data:`~src.envs.goal_env.GOAL_BUFFER_ENV_ATTR`, and returns the
        rsl_rl-wrapped vec env. Raises a descriptive RuntimeError off-GPU.
        """
        try:
            import gymnasium as gym  # noqa: PLC0415
            import isaaclab_tasks  # noqa: F401, PLC0415  (registers the tasks)
            from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Lab / gymnasium are not importable in this environment; "
                "IsaacLabPolicyRollout only runs inside the Isaac Sim runtime. "
                "Inject a fake PolicyRollout for host-side tests."
            ) from exc

        from ..envs.goal_env import (  # noqa: PLC0415
            GOAL_BUFFER_ENV_ATTR,
            attach_goal_conditioning,
            make_goal_buffer,
        )

        env_cfg = parse_env_cfg(self._task_id, num_envs=num_envs)
        # Reframe the stock velocity task as goal-reaching (Req 6.2, 8.1).
        attach_goal_conditioning(env_cfg, goal, num_envs)

        env = gym.make(self._task_id, cfg=env_cfg, render_mode=None)

        # Broadcast the per-env Goal buffer onto the live env so the goal-obs term
        # and the generated reward can read it (mirrors the training wiring).
        buffer = make_goal_buffer(goal, num_envs, device=self._device)
        target = getattr(env, "unwrapped", env)
        setattr(target, GOAL_BUFFER_ENV_ATTR, buffer)

        wrapped = self._wrap_for_rsl_rl(env)
        return wrapped

    @staticmethod
    def _wrap_for_rsl_rl(env: Any) -> Any:
        """Wrap a gym env for rsl_rl inference, tolerating import-path drift."""
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

            return RslRlVecEnvWrapper(env)
        except ImportError:
            pass
        try:
            # Older Isaac Lab trees expose the wrapper under omni.isaac.lab_tasks.
            from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (  # noqa: PLC0415
                RslRlVecEnvWrapper,
            )

            return RslRlVecEnvWrapper(env)
        except ImportError:
            # No known rsl_rl wrapper: fall back to the raw env and let the
            # duck-typed step path drive it directly.
            return env

    # -- policy loading --------------------------------------------------- #
    def _load_policy(self, checkpoint: Any, env: Any) -> Any:
        """Return a callable ``policy(obs) -> actions`` from ``checkpoint``.

        Duck-typed across rsl_rl versions and checkpoint shapes:
          1. an already-callable policy object is used as-is;
          2. an object exposing ``get_inference_policy`` / ``act_inference`` /
             ``act`` is adapted to ``policy(obs)``;
          3. otherwise ``checkpoint`` is treated as a path: a TorchScript module
             (``torch.jit.load``) when it ends in ``.pt``/``.jit``, else an
             rsl_rl ``OnPolicyRunner`` is built and ``.load()``-ed, and its
             inference policy returned.
        """
        path = self._checkpoint_path(checkpoint)

        # (1) already a policy callable.
        if path is None and callable(checkpoint):
            return checkpoint
        # (2) object exposing a policy accessor.
        for accessor in ("get_inference_policy", "act_inference", "act"):
            fn = getattr(checkpoint, accessor, None)
            if callable(fn):
                if accessor == "get_inference_policy":
                    return fn(self._device)
                return fn

        # (3) load from a checkpoint path.
        import torch  # noqa: PLC0415

        if path is not None and path.endswith((".pt", ".jit")):
            try:
                module = torch.jit.load(path, map_location=self._device)
                module.eval()
                return module
            except Exception:  # noqa: BLE001 - not TorchScript; fall through to runner
                pass

        from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

        train_cfg = self._infer_train_cfg(checkpoint)
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=self._device)
        if path is not None:
            runner.load(path)
        return runner.get_inference_policy(self._device)

    @staticmethod
    def _checkpoint_path(checkpoint: Any) -> str | None:
        """Extract a filesystem path from a checkpoint ref (duck-typed)."""
        if isinstance(checkpoint, str):
            return checkpoint
        for attr in ("path", "checkpoint_path", "uri"):
            value = getattr(checkpoint, attr, None)
            if isinstance(value, str):
                return value
        return None

    @staticmethod
    def _infer_train_cfg(checkpoint: Any) -> Any:
        """Retrieve (or synthesize) the rsl_rl train cfg for OnPolicyRunner.

        Prefers a cfg attached to the checkpoint ref; otherwise falls back to a
        default PPO cfg that MATCHES the trainer's ``_build_train_cfg`` (same
        ActorCritic MLP shape / PPO block), so a plain rsl_rl ``.pt`` save (which
        carries weights but no cfg) can be reloaded for inference without an
        explicitly attached cfg. The eval run only needs the inference policy, so
        the algorithm hyperparameters are irrelevant beyond matching the network
        architecture the checkpoint was trained with.
        """
        for attr in ("train_cfg", "agent_cfg", "rsl_rl_cfg"):
            value = getattr(checkpoint, attr, None)
            if value is not None:
                return value
        return _default_rsl_rl_eval_cfg()

    # -- rollout ---------------------------------------------------------- #
    def rollout(
        self, checkpoint: Any, goal: Goal, num_episodes: int
    ) -> Sequence[EpisodeRollout]:
        """Roll out ``num_episodes`` and return one :class:`EpisodeRollout` each.

        Builds the eval env at ``min(num_episodes, eval_env_count)`` envs (so the
        external cameras can render; Req 10.4), loads the policy, then steps until
        every env's episode has ended (or the safety step cap is hit), recording
        per-step base ``(x, y)`` and an upright flag (plus secondary signals when
        enabled). Each env yields one :class:`EpisodeTrajectory`.
        """
        import torch  # noqa: PLC0415

        goal = goal or self._goal
        # Reuse the trainer's live env when injected (single-process train+eval);
        # otherwise build a dedicated eval env (separate-process / standalone).
        reusing_env = self._shared_env is not None
        if reusing_env:
            env = self._shared_env
            num_envs = self._env_num_envs(env)
        else:
            num_envs = max(1, min(int(num_episodes), self._eval_env_count))
            env = self._build_env(goal, num_envs)
        policy = self._shared_policy if self._shared_policy is not None else self._load_policy(checkpoint, env)

        dt = self._resolve_dt(env)
        recorders = [_EpisodeRecorder(self._include_secondary) for _ in range(num_envs)]
        done_mask = [False] * num_envs

        # The reset MUST run inside inference_mode: when reusing the trainer's
        # live env, its state buffers (e.g. root_link_pose_w) were created as
        # "inference tensors" during RSL-RL rollout collection, and reset()
        # writes to them in place — which torch only permits inside inference
        # mode ("Inplace update to inference tensor outside InferenceMode is not
        # allowed"). Stepping below is inference-only too.
        with torch.inference_mode():
            obs = self._reset(env)
            for _ in range(self._max_episode_steps):
                actions = self._policy_step(policy, obs)
                obs, reward, dones, _ = self._step(env, actions)

                positions = self._read_base_xy(env)
                upright = self._read_upright(env)
                torques = self._read_torques(env) if self._include_secondary else None
                contacts = self._read_foot_contacts(env) if self._include_secondary else None

                for i in range(num_envs):
                    if done_mask[i]:
                        continue
                    recorders[i].append(
                        xy=(float(positions[i][0]), float(positions[i][1])),
                        upright=bool(upright[i]),
                        reward=(float(reward[i]) if reward is not None else None),
                        torque=(torques[i] if torques is not None else None),
                        contact=(contacts[i] if contacts is not None else None),
                    )
                    if bool(dones[i]):
                        done_mask[i] = True

                if all(done_mask):
                    break

        # Only close an env this rollout OWNS. A shared (trainer-owned) env is
        # closed by its owner so we never tear down the one SimulationApp early.
        if not reusing_env:
            self._close(env)

        rollouts: list[EpisodeRollout] = []
        for i, rec in enumerate(recorders):
            trajectory = rec.to_trajectory(dt)
            rollouts.append(
                EpisodeRollout(
                    trajectory=trajectory,
                    replay={"env_index": i, "goal": goal, "checkpoint": checkpoint},
                )
            )
        return rollouts

    # -- duck-typed env adapters ----------------------------------------- #
    @staticmethod
    def _env_num_envs(env: Any) -> int:
        """Number of parallel envs on a (possibly wrapped) env (duck-typed)."""
        target = getattr(env, "unwrapped", env)
        for obj in (env, target):
            n = getattr(obj, "num_envs", None)
            if isinstance(n, int) and n > 0:
                return n
        return 1

    @staticmethod
    def _reset(env: Any) -> Any:
        out = env.reset()
        # gymnasium returns (obs, info); rsl_rl wrappers often return obs only.
        if isinstance(out, tuple):
            return out[0]
        return out

    @staticmethod
    def _step(env: Any, actions: Any) -> tuple[Any, Any, Any, Any]:
        out = env.step(actions)
        if len(out) == 5:  # gymnasium: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = out
            dones = [bool(a) or bool(b) for a, b in zip(_as_list(terminated), _as_list(truncated))]
            return obs, reward, dones, info
        obs, reward, dones, info = out  # rsl_rl: obs, reward, dones, info
        return obs, reward, _as_list(dones), info

    @staticmethod
    def _policy_step(policy: Any, obs: Any) -> Any:
        # rsl_rl inference policies take the obs tensor directly; gymnasium-style
        # policies may take (obs) too. Tolerate a dict obs by extracting "policy".
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        return policy(obs)

    def _resolve_dt(self, env: Any) -> float:
        target = getattr(env, "unwrapped", env)
        for attr in ("step_dt", "physics_dt"):
            value = getattr(target, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        sim = getattr(target, "sim", None)
        if sim is not None:
            dt = getattr(sim, "dt", None)
            if isinstance(dt, (int, float)) and dt > 0:
                return float(dt)
        return 1.0 / 50.0  # sensible default control dt

    @staticmethod
    def _robot_data(env: Any) -> Any:
        target = getattr(env, "unwrapped", env)
        scene = target.scene
        robot = scene["robot"] if hasattr(scene, "__getitem__") else getattr(scene, "robot")
        return robot.data

    @classmethod
    def _read_base_xy(cls, env: Any) -> Any:
        """Per-env base ``(x, y)`` in the ENV-LOCAL frame.

        Isaac Lab lays out parallel envs on a grid, so ``root_pos_w`` is the
        ABSOLUTE world position which includes env ``i``'s grid-origin offset
        (``scene.env_origins[i]``, spaced metres apart). The Goal, however, is
        env-local (e.g. (2, 0) relative to each env's own origin). Comparing a
        world position against an env-local goal is a frame mismatch — it makes
        ``distance_to_goal_m`` read as the env-grid offset (tens of metres) and
        the success/upright metrics meaningless. Subtract the env origin so the
        returned position is relative to the env's own start, matching the Goal.
        """
        data = cls._robot_data(env)
        pos = data.root_pos_w
        origins = cls._env_origins(env)
        if origins is not None:
            return [
                (float(pos[i][0]) - float(origins[i][0]),
                 float(pos[i][1]) - float(origins[i][1]))
                for i in range(len(pos))
            ]
        return [(float(pos[i][0]), float(pos[i][1])) for i in range(len(pos))]

    @classmethod
    def _env_origins(cls, env: Any) -> Any:
        """Per-env grid origin ``(num_envs, 3)`` from the scene, or ``None``.

        Isaac Lab's ``InteractiveScene`` exposes ``env_origins``; absence (older
        versions / a fake env) falls back to no localization (treats world ==
        local), which is correct for a single-env scene at the world origin.
        """
        target = getattr(env, "unwrapped", env)
        scene = getattr(target, "scene", None)
        return getattr(scene, "env_origins", None) if scene is not None else None

    @classmethod
    def _read_upright(cls, env: Any) -> list[bool]:
        """Upright flag per env from the robot-frame projected gravity z.

        ``projected_gravity_b`` points straight down (z ≈ -1) when upright; once
        the torso tilts past :data:`UPRIGHT_PROJECTED_GRAVITY_Z_MAX` the env is
        treated as fallen. Falls back to a quaternion-tilt test when projected
        gravity is unavailable on a given Isaac Lab version.
        """
        data = cls._robot_data(env)
        proj = getattr(data, "projected_gravity_b", None)
        if proj is not None:
            return [float(proj[i][2]) <= UPRIGHT_PROJECTED_GRAVITY_Z_MAX for i in range(len(proj))]
        quat = data.root_quat_w  # (w, x, y, z)
        flags: list[bool] = []
        for i in range(len(quat)):
            w, x, y, z = (float(quat[i][0]), float(quat[i][1]), float(quat[i][2]), float(quat[i][3]))
            # cos(tilt) = 1 - 2(x^2 + y^2): the world-z component of the base-z axis.
            cos_tilt = 1.0 - 2.0 * (x * x + y * y)
            flags.append(cos_tilt >= -UPRIGHT_PROJECTED_GRAVITY_Z_MAX)
        return flags

    @classmethod
    def _read_torques(cls, env: Any) -> Any:
        data = cls._robot_data(env)
        torque = getattr(data, "applied_torque", None)
        if torque is None:
            return None
        return [[float(v) for v in torque[i]] for i in range(len(torque))]

    @classmethod
    def _read_foot_contacts(cls, env: Any) -> Any:
        """Per-env ``(left, right)`` foot contact magnitudes, or ``None``.

        Reads the contact-sensor net forces when a ``contact_forces`` sensor is
        present in the scene; returns ``None`` (so symmetry is simply omitted)
        when the eval scene carries no contact sensor.
        """
        target = getattr(env, "unwrapped", env)
        scene = getattr(target, "scene", None)
        if scene is None:
            return None
        try:
            sensor = scene["contact_forces"] if hasattr(scene, "__getitem__") else None
        except (KeyError, TypeError):
            return None
        if sensor is None:
            return None
        forces = getattr(sensor.data, "net_forces_w", None)
        if forces is None:
            return None
        # forces: (num_envs, num_bodies, 3); use the first two tracked bodies as
        # left/right feet (the contact sensor is configured over the foot bodies).
        out: list[tuple[float, float]] = []
        for i in range(len(forces)):
            bodies = forces[i]
            left = float(sum(float(c) ** 2 for c in bodies[0]) ** 0.5) if len(bodies) > 0 else 0.0
            right = float(sum(float(c) ** 2 for c in bodies[1]) ** 0.5) if len(bodies) > 1 else 0.0
            out.append((left, right))
        return out

    @staticmethod
    def _close(env: Any) -> None:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - closing must never mask results
                pass


class _EpisodeRecorder:  # pragma: no cover - driven only by the live rollout
    """Accumulates per-step samples for one env into an :class:`EpisodeTrajectory`."""

    def __init__(self, include_secondary: bool) -> None:
        self._include_secondary = include_secondary
        self.positions: list[tuple[float, float]] = []
        self.upright: list[bool] = []
        self.rewards: list[float] = []
        self.torques: list[list[float]] = []
        self.left: list[float] = []
        self.right: list[float] = []

    def append(
        self,
        *,
        xy: tuple[float, float],
        upright: bool,
        reward: float | None,
        torque: list[float] | None,
        contact: tuple[float, float] | None,
    ) -> None:
        self.positions.append(xy)
        self.upright.append(upright)
        if self._include_secondary:
            if reward is not None:
                self.rewards.append(reward)
            if torque is not None:
                self.torques.append(torque)
            if contact is not None:
                self.left.append(contact[0])
                self.right.append(contact[1])

    def to_trajectory(self, dt: float) -> EpisodeTrajectory:
        # A live episode always has >= 1 sample; guard the degenerate empty case
        # (env ended before the first record) with a single start sample.
        positions = self.positions or [(0.0, 0.0)]
        upright = self.upright or [True]
        torques = self.torques if (self._include_secondary and self.torques) else None
        rewards = self.rewards if (self._include_secondary and self.rewards) else None
        left = self.left if (self._include_secondary and self.left) else None
        right = self.right if (self._include_secondary and self.right) else None
        return EpisodeTrajectory(
            positions_xy=positions,
            upright_flags=upright,
            dt=dt,
            torques=torques,
            rewards=rewards,
            left_contacts=left,
            right_contacts=right,
        )


def _as_list(value: Any) -> list:
    """Coerce a tensor/array/scalar into a Python list (no eager torch import)."""
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        result = tolist()
        return result if isinstance(result, list) else [result]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# --------------------------------------------------------------------------- #
# Live demo-video recorder (Isaac Sim only; guarded)
# --------------------------------------------------------------------------- #
class IsaacLabDemoVideoRecorder:  # pragma: no cover - requires the Isaac Sim runtime
    """Live :class:`DemoVideoRecorder`: re-renders one episode to RGB video files.

    Spawns the external, world-frame chase + side cameras
    (:meth:`src.sensors.camera_cfg.CameraSpec.to_isaaclab_cfg`, Req 10.3) at a
    small env count (Req 10.4), replays the selected episode (driven from the
    :class:`EpisodeRollout`'s ``replay`` handle), and writes one RGB video file
    per camera under ``output_dir`` — returning their paths.

    The episode replay re-runs the policy deterministically for the selected env
    (the ``replay`` carries the goal + checkpoint), reading each camera's
    ``data.output["rgb"]`` frame and muxing to file. ``imageio`` is used when
    present (ffmpeg mp4), with an OpenCV fallback; if neither is importable the
    frames are written as a PNG sequence so recording never hard-fails.

    All heavy imports are lazy and inside methods; the class is ``# pragma: no
    cover`` because it only runs on the GPU box.

    Parameters:
        rollout_factory: Callable ``(cameras) -> IsaacLabPolicyRollout``-like
            object used to build the camera-instrumented replay env. Injected so
            the recorder reuses the same env-build/policy-load path as the live
            rollout. When ``None`` the recorder builds a default rollout.
        fps: Frames per second for the muxed video (defaults to each camera's fps).
        device: torch device string for the render env.
    """

    def __init__(
        self,
        *,
        goal: Goal | None = None,
        task_id: str = "Isaac-Velocity-Flat-H1-v0",
        eval_env_count: int = EVAL_ENV_COUNT_DEFAULT,
        device: str = "cuda:0",
        max_episode_steps: int = 1000,
    ) -> None:
        self._goal = goal
        self._task_id = task_id
        self._eval_env_count = int(eval_env_count)
        self._device = device
        self._max_episode_steps = int(max_episode_steps)

    def record(
        self,
        *,
        rollout: EpisodeRollout,
        cameras: Sequence[CameraSpec],
        label: str,
        output_dir: str,
    ) -> Sequence[str]:
        """Render the selected episode through ``cameras`` and return file paths.

        Builds a camera-instrumented replay env, replays the selected env's
        episode capturing each camera's RGB frames, and muxes one video file per
        camera into ``output_dir``. Returns the written paths (one per camera);
        an empty sequence when no frames could be captured so the producer treats
        it as "no video" rather than failing.
        """
        import os  # noqa: PLC0415

        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "torch / Isaac Lab are not importable; IsaacLabDemoVideoRecorder "
                "only runs inside the Isaac Sim runtime."
            ) from exc

        os.makedirs(output_dir, exist_ok=True)
        goal = self._resolve_goal(rollout)
        env_index = int(self._replay_field(rollout, "env_index", 0))
        checkpoint = self._replay_field(rollout, "checkpoint", None)

        env, camera_handles = self._build_camera_env(goal, cameras)
        policy = self._load_policy(checkpoint, env)

        # frame buffers, one list of HxWx3 uint8 arrays per camera.
        frames: list[list[Any]] = [[] for _ in cameras]
        obs = self._reset(env)
        with torch.inference_mode():
            for _ in range(self._max_episode_steps):
                actions = self._policy_step(policy, obs)
                obs, _, dones, _ = self._step(env, actions)
                for cam_i, handle in enumerate(camera_handles):
                    rgb = self._read_rgb(handle, env_index)
                    if rgb is not None:
                        frames[cam_i].append(rgb)
                done_list = _as_list(dones)
                if env_index < len(done_list) and bool(done_list[env_index]):
                    break

        self._close(env)

        paths: list[str] = []
        for cam_i, spec in enumerate(cameras):
            if not frames[cam_i]:
                continue
            path = self._mux(frames[cam_i], spec, label, output_dir)
            if path:
                paths.append(path)
        return paths

    def record_with_trajectory(
        self,
        *,
        goal: Goal,
        checkpoint: Any,
        cameras: Sequence[CameraSpec],
        label: str,
        output_dir: str,
        env_index: int = 0,
    ) -> tuple[list[str], "EpisodeTrajectory | None"]:
        """One single-env pass: capture BOTH video frames AND the trajectory.

        This is the per-iteration in-loop path. Building a camera env and doing a
        SEPARATE metrics rollout would be two manager-based envs in one process
        (which stalls), so a single rollout records the per-step camera RGB (for
        the demo video) and the per-step base position + upright flag (for the
        :class:`EpisodeTrajectory` the Evaluator scores). Returns the muxed video
        paths and the trajectory (``None`` if the rollout produced no samples).
        """
        import os  # noqa: PLC0415
        import torch  # noqa: PLC0415

        os.makedirs(output_dir, exist_ok=True)
        env, camera_handles = self._build_camera_env(goal, cameras)
        policy = self._load_policy(checkpoint, env)
        dt = self._resolve_dt(env)  # delegated from IsaacLabPolicyRollout

        frames: list[list[Any]] = [[] for _ in cameras]
        positions: list[tuple[float, float]] = []
        upright_flags: list[bool] = []

        obs = self._reset(env)
        with torch.inference_mode():
            for _ in range(self._max_episode_steps):
                actions = self._policy_step(policy, obs)
                obs, _, dones, _ = self._step(env, actions)
                # Per-step state for metrics (env_index only).
                try:
                    xy = self._read_base_xy(env)
                    up = self._read_upright(env)
                    positions.append((float(xy[env_index][0]), float(xy[env_index][1])))
                    upright_flags.append(bool(up[env_index]))
                except Exception:  # noqa: BLE001 - never let state-read break recording
                    pass
                # Per-step camera RGB for the demo video.
                for cam_i, handle in enumerate(camera_handles):
                    rgb = self._read_rgb(handle, env_index)
                    if rgb is not None:
                        frames[cam_i].append(rgb)
                done_list = _as_list(dones)
                if env_index < len(done_list) and bool(done_list[env_index]):
                    break

        self._close(env)

        paths: list[str] = []
        for cam_i, spec in enumerate(cameras):
            if not frames[cam_i]:
                continue
            path = self._mux(frames[cam_i], spec, label, output_dir)
            if path:
                paths.append(path)

        trajectory = None
        if positions:
            trajectory = EpisodeTrajectory(
                positions_xy=positions,
                upright_flags=upright_flags,
                dt=dt,
            )
        return paths, trajectory

    # -- replay handle helpers ------------------------------------------- #
    def _resolve_goal(self, rollout: EpisodeRollout) -> Goal:
        goal = self._replay_field(rollout, "goal", None)
        if isinstance(goal, Goal):
            return goal
        if self._goal is not None:
            return self._goal
        raise RuntimeError(
            "No Goal available to replay; the EpisodeRollout.replay carried none "
            "and the recorder was constructed without a goal."
        )

    @staticmethod
    def _replay_field(rollout: EpisodeRollout, key: str, default: Any) -> Any:
        replay = getattr(rollout, "replay", None)
        if isinstance(replay, dict):
            return replay.get(key, default)
        return getattr(replay, key, default)

    # -- camera env construction ----------------------------------------- #
    def _build_camera_env(
        self, goal: Goal, cameras: Sequence[CameraSpec]
    ) -> tuple[Any, list[Any]]:
        """Build the eval env with the external cameras attached to the scene."""
        try:
            import gymnasium as gym  # noqa: PLC0415
            import isaaclab_tasks  # noqa: F401, PLC0415
            from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Lab / gymnasium are not importable; IsaacLabDemoVideoRecorder "
                "only runs inside the Isaac Sim runtime."
            ) from exc

        from ..envs.goal_env import (  # noqa: PLC0415
            GOAL_BUFFER_ENV_ATTR,
            attach_goal_conditioning,
            make_goal_buffer,
        )

        # Render a SINGLE env. The recorder only captures one env's episode
        # (``env_index``, default 0), and attaching RTX cameras to a multi-env
        # scene triggers a CUDA device-side assert during camera sensor reset
        # (physx GpuRigidBodyView) on a single-GPU host — multi-env tiled camera
        # rendering is the trigger. One env is correct and sufficient for the
        # demo video and sidesteps the assert.
        num_envs = 1
        env_cfg = parse_env_cfg(self._task_id, num_envs=num_envs)
        attach_goal_conditioning(env_cfg, goal, num_envs)

        # Attach external world-frame cameras onto the scene cfg (Req 10.3).
        scene = getattr(env_cfg, "scene", None)
        for spec in cameras:
            if scene is not None:
                setattr(scene, spec.name, spec.to_isaaclab_cfg())

        env = gym.make(self._task_id, cfg=env_cfg, render_mode=None)
        target = getattr(env, "unwrapped", env)
        setattr(target, GOAL_BUFFER_ENV_ATTR, make_goal_buffer(goal, num_envs, device=self._device))

        # Resolve the live camera sensor handles from the instantiated scene.
        live_scene = getattr(target, "scene", None)
        handles: list[Any] = []
        for spec in cameras:
            handle = None
            if live_scene is not None:
                try:
                    handle = live_scene[spec.name] if hasattr(live_scene, "__getitem__") else None
                except (KeyError, TypeError):
                    handle = getattr(live_scene, spec.name, None)
            handles.append(handle)

        # Wrap for rsl_rl so the OnPolicyRunner can read observations
        # (``env.get_observations()`` exists only on the RSL-RL vec-env wrapper,
        # not the raw OrderEnforcing gym env). Camera handles were already
        # resolved off the unwrapped scene above, and the recorder's step/read
        # helpers go through ``.unwrapped``, so wrapping here is safe.
        wrapped = IsaacLabPolicyRollout._wrap_for_rsl_rl(env)
        return wrapped, handles

    @staticmethod
    def _read_rgb(camera_handle: Any, env_index: int) -> Any:
        """Read one RGB frame (HxWx3 uint8) for ``env_index`` from a camera handle."""
        if camera_handle is None:
            return None
        data = getattr(camera_handle, "data", None)
        if data is None:
            return None
        output = getattr(data, "output", None)
        if output is None:
            return None
        rgb = output["rgb"] if hasattr(output, "__getitem__") else getattr(output, "rgb", None)
        if rgb is None:
            return None
        frame = rgb[env_index] if len(rgb) > env_index else rgb[0]
        # Drop alpha if present and move to CPU/numpy uint8.
        cpu = getattr(frame, "cpu", None)
        if callable(cpu):
            frame = cpu()
        numpy_fn = getattr(frame, "numpy", None)
        if callable(numpy_fn):
            frame = numpy_fn()
        if hasattr(frame, "shape") and len(frame.shape) == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        return frame

    def _mux(
        self, frames: Sequence[Any], spec: CameraSpec, label: str, output_dir: str
    ) -> str | None:
        """Write ``frames`` to a video file for camera ``spec``; return the path."""
        import os  # noqa: PLC0415

        base = os.path.join(output_dir, f"{label}_{spec.name}")
        # Preferred: imageio + ffmpeg -> mp4.
        try:
            import imageio.v2 as imageio  # noqa: PLC0415

            path = f"{base}.mp4"
            writer = imageio.get_writer(path, fps=spec.fps)
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()
            return path
        except ImportError:
            pass
        # Fallback: OpenCV VideoWriter -> mp4.
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            path = f"{base}.mp4"
            height, width = frames[0].shape[0], frames[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, spec.fps, (width, height))
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return path
        except ImportError:
            pass
        # Last resort: a PNG sequence so recording never hard-fails.
        try:
            import imageio.v2 as imageio  # noqa: PLC0415

            first = f"{base}_0000.png"
            for idx, frame in enumerate(frames):
                imageio.imwrite(f"{base}_{idx:04d}.png", frame)
            return first
        except Exception:  # noqa: BLE001
            return None

    # -- shared duck-typed adapters (delegate to the rollout's logic) ----- #
    _reset = staticmethod(IsaacLabPolicyRollout._reset)
    _step = staticmethod(IsaacLabPolicyRollout._step)
    _policy_step = staticmethod(IsaacLabPolicyRollout._policy_step)
    _close = staticmethod(IsaacLabPolicyRollout._close)
    # State readers for the single-pass record_with_trajectory metrics capture.
    _read_base_xy = IsaacLabPolicyRollout.__dict__["_read_base_xy"]
    _read_upright = IsaacLabPolicyRollout.__dict__["_read_upright"]
    _robot_data = IsaacLabPolicyRollout.__dict__["_robot_data"]
    _resolve_dt = IsaacLabPolicyRollout._resolve_dt

    def _load_policy(self, checkpoint: Any, env: Any) -> Any:
        # Reuse the rollout's duck-typed policy loader.
        loader = IsaacLabPolicyRollout(
            self._goal or Goal(position_xy=(0.0, 0.0), success_radius_m=1.0),
            task_id=self._task_id,
            eval_env_count=self._eval_env_count,
            device=self._device,
            max_episode_steps=self._max_episode_steps,
        )
        return loader._load_policy(checkpoint, env)


# --------------------------------------------------------------------------- #
# Lazy factories (mirror build_isaaclab_trainer): built only on the GPU box
# --------------------------------------------------------------------------- #
def build_isaaclab_policy_rollout(
    goal: Goal, eval_config: EvalConfig, *, shared_env: Any = None, shared_policy: Any = None
) -> PolicyRollout:  # pragma: no cover - requires the Isaac Sim runtime
    """Construct a live :class:`IsaacLabPolicyRollout` (lazy Isaac Sim check).

    Raises a descriptive :class:`RuntimeError` on a host where Isaac Lab is not
    importable, exactly like :func:`src.train.ppo_runner.build_isaaclab_trainer`;
    inject a fake :class:`PolicyRollout` into the :class:`Evaluator` to run the
    pure path without the Isaac Sim runtime.

    ``shared_env`` / ``shared_policy`` (the trainer's live env + inference
    policy) are forwarded so the rollout reuses them rather than building a
    second Isaac Lab env — required because Isaac Sim allows only one
    ``SimulationApp`` per process.
    """
    try:
        import isaaclab  # noqa: F401, PLC0415  (presence check)
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Lab is not importable in this environment. Inject a "
            "PolicyRollout into Evaluator(rollout=...) to evaluate without the "
            "Isaac Sim runtime."
        ) from exc

    return IsaacLabPolicyRollout(
        goal,
        task_id=eval_config.env_id,
        eval_env_count=eval_config.eval_env_count,
        fall_threshold_s=eval_config.fall_threshold_s,
        include_secondary=eval_config.include_secondary,
        shared_env=shared_env,
        shared_policy=shared_policy,
    )


class SubprocessDemoRecorder:
    """Records demo video by spawning a FRESH process per call (the robust path).

    Isaac Lab cannot build a second manager-based RL env in a process that
    already created one — the in-loop recorder stalls during the second env's
    scene/sensor setup even with a shared SimulationApp and the trainer's env
    closed. This recorder sidesteps that by running ``scripts/record_demo.py`` in
    a **fresh subprocess** (one SimulationApp, one env — the configuration the
    standalone validation proved works), then collecting the written video paths.

    Conforms to the :class:`DemoVideoRecorder` protocol, so it drops into the
    :class:`DemoVideoProducer` unchanged. The goal + checkpoint are read from the
    :class:`EpisodeRollout`'s ``replay`` handle (set by the rollout backend), and
    the camera resolution/fps from the supplied :class:`CameraSpec`s.

    Cost: ~1-2 min of Isaac Sim init per call. Acceptable for a per-iteration
    demo; the recording never blocks or stalls training (it is a separate
    process the parent waits on with a timeout, fail-soft).
    """

    def __init__(
        self,
        *,
        goal: Goal | None = None,
        task_id: str = "Isaac-Velocity-Flat-H1-v0",
        eval_env_count: int = 1,
        max_episode_steps: int = 400,
        timeout_s: float = 900.0,
        isaaclab_sh: str = "/workspace/isaaclab/isaaclab.sh",
        script_path: str = "scripts/record_demo.py",
        runner: Any = None,
    ) -> None:
        self._goal_fallback = goal
        self._task_id = task_id
        # Single env: multi-env + RTX cameras triggers a CUDA device assert.
        self._eval_env_count = 1
        self._max_episode_steps = int(max_episode_steps)
        self._timeout_s = float(timeout_s)
        self._isaaclab_sh = isaaclab_sh
        self._script_path = script_path
        # Injected callable(cmd, timeout) -> (returncode, stdout) for tests; the
        # default runs the real subprocess.
        self._runner = runner or self._run_subprocess

    def record(
        self,
        *,
        rollout: "EpisodeRollout",
        cameras: Sequence[CameraSpec],
        label: str,
        output_dir: str,
    ) -> Sequence[str]:
        """Spawn record_demo.py for this episode; return the written paths.

        Fail-soft: any non-zero exit, timeout, or unparseable output yields an
        empty list so demo recording stays strictly additive (Req 10) and never
        breaks the loop.
        """
        import os  # noqa: PLC0415

        replay = getattr(rollout, "replay", None)
        if isinstance(replay, dict):
            goal = replay.get("goal")
            checkpoint = replay.get("checkpoint")
        else:
            goal = getattr(replay, "goal", None)
            checkpoint = getattr(replay, "checkpoint", None)
        if goal is None:
            goal = self._goal_fallback
        if goal is None or checkpoint is None:
            return []
        ckpt_path = checkpoint if isinstance(checkpoint, str) else getattr(checkpoint, "path", None)
        if not ckpt_path:
            return []

        os.makedirs(output_dir, exist_ok=True)
        gx, gy = goal.position_xy
        spec0 = cameras[0] if cameras else None
        cmd = [
            self._isaaclab_sh, "-p", self._script_path,
            "--checkpoint", str(ckpt_path),
            "--output-dir", output_dir,
            "--label", label,
            "--num-envs", "1",
            "--max-steps", str(self._max_episode_steps),
            "--task", self._task_id,
            "--goal-x", repr(float(gx)),
            "--goal-y", repr(float(gy)),
            "--success-radius", repr(float(goal.success_radius_m)),
        ]
        if spec0 is not None:
            cmd += [
                "--width", str(spec0.width),
                "--height", str(spec0.height),
                "--fps", repr(float(spec0.fps)),
            ]

        try:
            rc, out = self._runner(cmd, self._timeout_s)
        except Exception:  # noqa: BLE001 - never let recording break the loop
            return []
        return self._parse_paths(out) if rc == 0 else []

    @staticmethod
    def _parse_paths(stdout: str) -> list[str]:
        """Extract the JSON path list from a ``RECORD_DEMO: PASS [...]`` line."""
        import json  # noqa: PLC0415

        for line in reversed((stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("RECORD_DEMO: PASS"):
                payload = line[len("RECORD_DEMO: PASS"):].strip()
                try:
                    paths = json.loads(payload)
                    if isinstance(paths, list):
                        return [str(p) for p in paths]
                except ValueError:
                    return []
        return []

    @staticmethod
    def _run_subprocess(cmd: list[str], timeout_s: float):  # pragma: no cover - real subprocess
        import subprocess  # noqa: PLC0415

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def build_subprocess_demo_recorder(
    eval_config: EvalConfig
) -> DemoVideoRecorder:
    """Build a :class:`SubprocessDemoRecorder` from the eval config.

    Always available (no Isaac Lab import here — the subprocess does that), so
    the Evaluator can use it on the GPU host regardless of in-process env limits.
    """
    return SubprocessDemoRecorder(
        task_id=eval_config.env_id,
        eval_env_count=1,
    )


def build_final_video_recorder(config: Any, *, runner: Any = None):
    """Return a ``(checkpoint, goal, output_dir) -> list[str]`` end-of-run recorder.

    Spawns ``scripts/record_demo.py`` in a FRESH process (one SimulationApp, one
    env — the validated standalone path) to render the final Best_Policy after
    the loop. Used as the Orchestrator's ``final_video_recorder`` hook so the
    mandatory demo video (Req 10) is produced without the in-loop second-env
    stall. Fail-soft: returns ``[]`` on any failure (the Orchestrator also
    guards). ``runner`` is injectable for tests.
    """
    eval_config = EvalConfig.from_config(config)
    cameras = CameraConfig.from_config(config)
    rec = SubprocessDemoRecorder(task_id=eval_config.env_id, eval_env_count=1, runner=runner)

    def _record(checkpoint: Any, goal: Goal, output_dir: str) -> list[str]:
        rollout = EpisodeRollout(
            trajectory=None,
            replay={"env_index": 0, "goal": goal, "checkpoint": checkpoint},
        )
        return list(
            rec.record(
                rollout=rollout,
                cameras=cameras.cameras(),
                label="best_policy",
                output_dir=output_dir,
            )
        )

    return _record


def build_isaaclab_demo_recorder(
    goal: Goal, eval_config: EvalConfig
) -> DemoVideoRecorder:  # pragma: no cover - requires the Isaac Sim runtime
    """Construct a live :class:`IsaacLabDemoVideoRecorder` (lazy Isaac Sim check).

    Mirrors :func:`build_isaaclab_policy_rollout`: raises a descriptive
    :class:`RuntimeError` off-GPU so the controller host injects a fake recorder
    instead.
    """
    try:
        import isaaclab  # noqa: F401, PLC0415  (presence check)
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Lab is not importable in this environment. Inject a "
            "DemoVideoRecorder into Evaluator(recorder=...) to evaluate without "
            "the Isaac Sim runtime."
        ) from exc

    return IsaacLabDemoVideoRecorder(
        goal=goal,
        task_id=eval_config.env_id,
        eval_env_count=eval_config.eval_env_count,
    )


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
        self,
        checkpoint: Any,
        goal: Goal,
        num_episodes: int,
        *,
        live_env: Any = None,
        live_policy: Any = None,
    ) -> EvalMetrics:
        """Evaluate ``checkpoint`` against ``goal`` over ``num_episodes`` (Req 9, 10, 17).

        Rolls out the policy, computes the aggregate :class:`EvalMetrics` with
        staged goal gates, records best/worst demo video, attaches the
        :class:`DemoVideoResult` as ``metrics.demo_video``, and returns the metrics.

        Args:
            checkpoint: The trained-policy checkpoint reference to evaluate.
            goal: The :class:`~src.data_models.Goal` (point B + Success_Radius).
            num_episodes: Number of evaluation episodes (>= 1).
            live_env: Optional trainer-owned, already-live env to roll out against
                instead of building a new one (single ``SimulationApp`` per
                process). When given, the lazily-built live rollout reuses it.
            live_policy: Optional trainer inference policy paired with
                ``live_env`` so the rollout skips re-loading from the checkpoint.

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

        rollout_backend = self._resolve_rollout(
            goal, live_env=live_env, live_policy=live_policy
        )

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
        # available, so evaluation never fails for lack of video. Pass the
        # already-computed ``rollouts`` so best/worst selection does NOT trigger a
        # second in-process env rollout (which would stall on Isaac Lab).
        demo_result = self._record_demo_video(
            checkpoint, goal, rollout_backend, rollouts=rollouts
        )

        # (4) Surface the demo-video result for the Orchestrator's persistence,
        # which reads ``metrics.demo_video`` (src/orchestrator.py).
        setattr(metrics, "demo_video", demo_result)
        return metrics

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_rollout(
        self, goal: Goal, *, live_env: Any = None, live_policy: Any = None
    ) -> PolicyRollout:
        """Return the rollout backend: injected > lazily-built Isaac Lab.

        When ``live_env`` is supplied (the trainer's already-live env), the
        lazily-built live rollout reuses it instead of constructing a second
        Isaac Lab env, so only one ``SimulationApp`` exists per process. An
        injected rollout (tests) is always honored as-is.
        """
        if self._rollout is not None:
            return self._rollout
        return build_isaaclab_policy_rollout(
            goal, self._config, shared_env=live_env, shared_policy=live_policy
        )

    def _resolve_recorder(self, goal: Goal) -> DemoVideoRecorder | None:
        """Return the demo recorder: injected, else subprocess-based, else ``None``.

        Recording is additive (Req 10). When no recorder is injected and recording
        is enabled, use the :class:`SubprocessDemoRecorder`: it records in a FRESH
        process (one SimulationApp, one env), the only configuration that works —
        Isaac Lab stalls building a second manager-based env in-process. A missing
        recorder is non-fatal: the loop records no video rather than failing.
        """
        if self._recorder is not None:
            return self._recorder
        # Operator/smoke opt-out: skip demo recording entirely (additive; Req 10).
        if not getattr(self._config, "record_demo_video", True):
            return None
        # Subprocess recorder is always constructible (it shells out to a fresh
        # process that imports Isaac Lab); no in-process Isaac Lab import here.
        try:
            return build_subprocess_demo_recorder(self._config)
        except Exception:  # noqa: BLE001 - recording is additive; never fatal
            return None

    def _record_demo_video(
        self, checkpoint: Any, goal: Goal, rollout_backend: PolicyRollout,
        *, rollouts: Any = None,
    ) -> DemoVideoResult:
        """Run the :class:`DemoVideoProducer` to record best/worst demo video.

        The production recorder (:class:`SubprocessDemoRecorder`) records in a
        FRESH process from the checkpoint, so it neither touches nor needs the
        trainer's live env — no in-process second-env stall. ``rollouts`` (the
        episodes already rolled out for metric computation) are forwarded so
        best/worst SELECTION does not trigger a second in-process env rollout
        either. The subprocess then re-renders the selected episode.
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
        return producer.record_best_worst(checkpoint, goal, rollouts=rollouts)