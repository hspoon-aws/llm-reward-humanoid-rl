"""Goal-conditioned manager-based H1 flat env wiring (Task 8.2).

This module reframes the stock manager-based H1 flat task
(``Isaac-Velocity-Flat-H1-v0``) as a **point-to-point goal-reaching**
task, per design.md → "Introducing the Goal and Goal_Observation into the
environment". It does four things (Req 6.2, 8.1):

1. **Per-env Goal buffer.** A single :class:`Goal` (point B) is broadcast across
   all ``num_envs`` into a :class:`GoalBuffer` carried on the env, so the
   generated reward and the observation term can read it per environment
   (design.md step 2; Req 6.2).
2. **Goal_Observation term.** :func:`compute_goal_observation` computes, in the
   **robot base frame**, the 2-D vector to the Goal plus the scalar distance and
   heading error, appended to the ~69-dim proprio observation (design.md step 3;
   Req 6.2). This reuses :meth:`src.data_models.GoalObservation.from_world` for
   the per-sample transform so there is a single source of truth for the math.
3. **Straight-line A→Goal distance at reset.** :meth:`GoalBuffer.reset` records
   each env's start position (point A) and the straight-line A→Goal distance,
   used later for path efficiency (design.md step 2; Req 9.4 downstream).
4. **Zero the stock velocity-tracking reward.** :func:`disable_velocity_tracking_rewards`
   sets the stock ``track_lin_vel_*`` / ``track_ang_vel_*`` term weights to zero
   so the LLM-generated reward fully owns the objective (design.md: "stock
   velocity-tracking reward terms are zeroed"; Req 8.1).

Isaac Lab is an optional, lazy dependency
-----------------------------------------
Isaac Lab (``isaaclab`` manager-based env, ``ManagerBasedRLEnvCfg``,
``ObservationTermCfg``, the Unitree H1 flat task cfg) is only importable inside
the Isaac Sim runtime, which is absent on the controller/dev host and in CI.
Importing this module must therefore NOT require ``isaaclab``. Following the
pattern in :mod:`src.rewards.reward_executor` and :mod:`src.sensors.camera_cfg`,
every Isaac Lab touchpoint is **lazy and guarded** (imported inside the function
that needs it). The Goal-buffer bookkeeping and the robot-frame Goal_Observation
transform are pure and work with either ``torch`` tensors (when installed) or
plain Python sequences, so they stay fully unit-testable with synthetic data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.data_models import Goal, GoalObservation

__all__ = [
    "GOAL_OBS_DIM",
    "STOCK_VELOCITY_REWARD_TERMS",
    "GOAL_BUFFER_ENV_ATTR",
    "GoalBuffer",
    "compute_goal_observation",
    "disable_velocity_tracking_rewards",
    "make_goal_buffer",
    "yaw_from_quat",
    "goal_observation_term",
    "attach_goal_conditioning",
]

# The Goal_Observation packing is (vec_x, vec_y, distance, heading) in the robot
# base frame (design.md: "~4 dims"), appended to the ~69-dim proprio vector.
GOAL_OBS_DIM = 4

# The stock H1 velocity-command-tracking reward terms that this reframing zeroes
# so the generated reward owns the objective (design.md task framing; Req 8.1).
# Names match the manager-based velocity locomotion reward cfg used by
# ``Isaac-Velocity-Flat-H1-v0``.
STOCK_VELOCITY_REWARD_TERMS: tuple[str, ...] = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
)

# Attribute under which the per-env Goal buffer is published on the live env so
# the observation term and the generated reward can read it (mirrors the
# ``env.goal`` surface the Reward_Executor binds against; Req 6.2).
GOAL_BUFFER_ENV_ATTR = "goal_buffer"


# --------------------------------------------------------------------------- #
# torch is optional: detect it lazily without importing at module load.
# --------------------------------------------------------------------------- #
def _try_torch():
    """Return the ``torch`` module if importable, else ``None`` (lazy, guarded)."""
    try:  # pragma: no cover - torch presence is environment-dependent
        import torch  # noqa: PLC0415

        return torch
    except ImportError:
        return None


def _is_tensor(value: Any) -> bool:
    """True iff ``value`` is a ``torch.Tensor`` (without importing torch eagerly)."""
    torch = _try_torch()
    return torch is not None and isinstance(value, torch.Tensor)


# --------------------------------------------------------------------------- #
# Robot-frame Goal_Observation transform (pure; Req 6.2)
# --------------------------------------------------------------------------- #
def yaw_from_quat(quat: Any) -> Any:
    """Extract the yaw (rotation about +z) from a quaternion.

    Accepts the Isaac Lab quaternion convention ``(w, x, y, z)`` either as a
    ``torch.Tensor`` of shape ``(..., 4)`` (vectorized over envs) or as a plain
    4-element sequence (single sample). Returns a tensor of yaw angles or a float
    respectively.

    yaw = ``atan2(2 (w z + x y), 1 - 2 (y^2 + z^2))``.
    """
    if _is_tensor(quat):
        torch = _try_torch()
        w = quat[..., 0]
        x = quat[..., 1]
        y = quat[..., 2]
        z = quat[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    if len(quat) != 4:
        raise ValueError(f"quaternion must have 4 elements (w, x, y, z), got {len(quat)}")
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_goal_observation(robot_xy: Any, base_yaw: Any, goal_xy: Any) -> Any:
    """Compute the robot-frame Goal_Observation for a batch of envs (Req 6.2).

    For each env, returns ``(vec_x, vec_y, distance, heading)`` where
    ``(vec_x, vec_y)`` is the world-frame vector from the robot base to the Goal
    rotated into the robot base frame by ``-base_yaw``, ``distance`` is the
    frame-invariant straight-line distance, and ``heading`` is the bearing of the
    Goal relative to the robot's forward (+x base) axis.

    Two interchangeable code paths keep this unit-testable everywhere:

    * **torch path** — when ``robot_xy`` is a ``torch.Tensor`` of shape
      ``(num_envs, 2)`` (with ``base_yaw`` ``(num_envs,)`` and ``goal_xy``
      ``(num_envs, 2)``), the transform is fully vectorized and returns a
      ``(num_envs, GOAL_OBS_DIM)`` tensor.
    * **pure-Python path** — otherwise the inputs are treated as row sequences
      and the per-sample transform is delegated to
      :meth:`src.data_models.GoalObservation.from_world`, returning a list of
      ``[vec_x, vec_y, distance, heading]`` rows. This is the single source of
      truth for the math, shared with Task 8.1.
    """
    if _is_tensor(robot_xy):
        torch = _try_torch()
        dx = goal_xy[..., 0] - robot_xy[..., 0]
        dy = goal_xy[..., 1] - robot_xy[..., 1]
        cos_y = torch.cos(-base_yaw)
        sin_y = torch.sin(-base_yaw)
        vec_x = dx * cos_y - dy * sin_y
        vec_y = dx * sin_y + dy * cos_y
        distance = torch.sqrt(dx * dx + dy * dy)
        heading = torch.atan2(vec_y, vec_x)
        return torch.stack([vec_x, vec_y, distance, heading], dim=-1)

    # Pure-Python fallback: reuse the canonical per-sample transform.
    rows: list[list[float]] = []
    for i in range(len(robot_xy)):
        r = robot_xy[i]
        g = goal_xy[i]
        yaw_i = float(base_yaw[i])
        obs = GoalObservation.from_world(
            robot_xy=(float(r[0]), float(r[1])),
            base_yaw_rad=yaw_i,
            # success_radius is irrelevant to the transform; use a valid sentinel.
            goal=Goal(position_xy=(float(g[0]), float(g[1])), success_radius_m=1.0),
        )
        rows.append(list(obs.as_tuple()))
    return rows


def _distance_xy(a_xy: Any, b_xy: Any) -> Any:
    """Per-env straight-line distance between two ``(num_envs, 2)`` batches."""
    if _is_tensor(a_xy):
        torch = _try_torch()
        dx = b_xy[..., 0] - a_xy[..., 0]
        dy = b_xy[..., 1] - a_xy[..., 1]
        return torch.sqrt(dx * dx + dy * dy)
    return [
        math.hypot(float(b_xy[i][0]) - float(a_xy[i][0]), float(b_xy[i][1]) - float(a_xy[i][1]))
        for i in range(len(a_xy))
    ]


# --------------------------------------------------------------------------- #
# Per-env Goal buffer (broadcast across num_envs; Req 6.2)
# --------------------------------------------------------------------------- #
@dataclass
class GoalBuffer:
    """A per-env Goal buffer broadcast across all ``num_envs`` (design.md step 2).

    Carries the goal position for every parallel env, the arrival
    ``success_radius_m``, and — after a reset — each env's start position
    (point A) plus the straight-line A→Goal distance recorded at that reset
    (used for path efficiency; Req 9.4 downstream).

    The buffers are stored as ``torch.Tensor`` of shape ``(num_envs, 2)`` /
    ``(num_envs,)`` when torch is available, otherwise as plain lists of rows, so
    the bookkeeping is exercisable without a GPU or Isaac Lab.

    Attributes:
        goal_xy: ``(num_envs, 2)`` goal position per env (the broadcast Goal).
        success_radius_m: Scalar arrival threshold shared by all envs.
        num_envs: Number of parallel envs the Goal is broadcast to.
        start_xy: ``(num_envs, 2)`` start position (point A) recorded at reset;
            ``None`` until the first :meth:`reset`.
        straight_line_distance: ``(num_envs,)`` straight-line A→Goal distance
            recorded at reset; ``None`` until the first :meth:`reset`.
    """

    goal_xy: Any
    success_radius_m: float
    num_envs: int
    start_xy: Any = None
    straight_line_distance: Any = None

    def reset(self, robot_xy: Any, env_ids: Sequence[int] | None = None) -> Any:
        """Record start positions and straight-line A→Goal distance at reset.

        Args:
            robot_xy: The robot base ``(x, y)`` per env at reset — a
                ``(num_envs, 2)`` tensor or a list of rows. When ``env_ids`` is
                given, ``robot_xy`` supplies positions for exactly those envs (in
                order); otherwise it supplies all ``num_envs``.
            env_ids: Optional subset of env indices being reset (Isaac Lab resets
                a subset of envs each step). When ``None`` every env is reset.

        Returns:
            The straight-line A→Goal distance for the reset envs (same container
            type as the inputs), so a caller can use it directly if convenient.
        """
        if env_ids is None:
            self.start_xy = _clone_xy(robot_xy)
            self.straight_line_distance = _distance_xy(self.start_xy, self.goal_xy)
            return self.straight_line_distance

        # Subset reset: lazily allocate the full buffers, then update rows.
        if self.start_xy is None:
            self.start_xy = _zeros_like_xy(self.goal_xy, self.num_envs)
        if self.straight_line_distance is None:
            self.straight_line_distance = _zeros_scalar(self.goal_xy, self.num_envs)

        for src_row, env_id in enumerate(env_ids):
            rx = robot_xy[src_row]
            gx = self.goal_xy[env_id]
            _set_xy(self.start_xy, env_id, rx)
            dist = math.hypot(float(rx[0]) - float(gx[0]), float(rx[1]) - float(gx[1]))
            _set_scalar(self.straight_line_distance, env_id, dist)
        return self.straight_line_distance

    def observation(self, robot_xy: Any, base_yaw: Any) -> Any:
        """Robot-frame Goal_Observation for all envs (``(num_envs, GOAL_OBS_DIM)``).

        Thin wrapper over :func:`compute_goal_observation` bound to this buffer's
        broadcast goal (Req 6.2).
        """
        return compute_goal_observation(robot_xy, base_yaw, self.goal_xy)


def make_goal_buffer(goal: Goal, num_envs: int, device: Any = None) -> GoalBuffer:
    """Build a :class:`GoalBuffer` by broadcasting a single :class:`Goal`.

    The one configured Goal (point B) is broadcast to every one of ``num_envs``
    parallel environments (design.md step 2; Req 6.2).

    Args:
        goal: The configured :class:`Goal` (point B + Success_Radius).
        num_envs: Number of parallel envs to broadcast to. Must be positive.
        device: Optional torch device for the broadcast tensors. Ignored on the
            pure-Python fallback path (no torch installed).

    Returns:
        A :class:`GoalBuffer` whose ``goal_xy`` is shape ``(num_envs, 2)``.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")

    gx, gy = goal.position_xy
    torch = _try_torch()
    if torch is not None:
        goal_xy = torch.tensor([[gx, gy]], dtype=torch.float32, device=device)
        goal_xy = goal_xy.expand(num_envs, 2).contiguous()
    else:
        goal_xy = [[float(gx), float(gy)] for _ in range(num_envs)]

    return GoalBuffer(
        goal_xy=goal_xy,
        success_radius_m=float(goal.success_radius_m),
        num_envs=num_envs,
    )


# --------------------------------------------------------------------------- #
# Small container helpers (torch-or-list, no eager torch import)
# --------------------------------------------------------------------------- #
def _clone_xy(xy: Any) -> Any:
    if _is_tensor(xy):
        return xy.clone()
    return [[float(row[0]), float(row[1])] for row in xy]


def _zeros_like_xy(reference_xy: Any, num_envs: int) -> Any:
    if _is_tensor(reference_xy):
        torch = _try_torch()
        return torch.zeros((num_envs, 2), dtype=reference_xy.dtype, device=reference_xy.device)
    return [[0.0, 0.0] for _ in range(num_envs)]


def _zeros_scalar(reference_xy: Any, num_envs: int) -> Any:
    if _is_tensor(reference_xy):
        torch = _try_torch()
        return torch.zeros((num_envs,), dtype=reference_xy.dtype, device=reference_xy.device)
    return [0.0 for _ in range(num_envs)]


def _set_xy(buffer_xy: Any, index: int, row: Any) -> None:
    buffer_xy[index][0] = float(row[0]) if not _is_tensor(buffer_xy) else row[0]
    buffer_xy[index][1] = float(row[1]) if not _is_tensor(buffer_xy) else row[1]


def _set_scalar(buffer_scalar: Any, index: int, value: float) -> None:
    buffer_scalar[index] = value


# --------------------------------------------------------------------------- #
# Zero/disable the stock velocity-tracking reward terms (Req 8.1)
# --------------------------------------------------------------------------- #
def disable_velocity_tracking_rewards(env_cfg: Any) -> list[str]:
    """Zero the stock velocity-command-tracking reward terms (Req 8.1).

    The reframing hands the entire objective to the LLM-generated reward, so the
    stock ``track_lin_vel_*`` / ``track_ang_vel_*`` terms must not contribute
    (design.md: "stock velocity-tracking reward terms are zeroed"). This sets the
    weight of each present stock term to ``0.0`` on ``env_cfg.rewards``, which
    keeps the term registered (so the manager and any logging stay intact) while
    removing its contribution.

    Duck-typed and Isaac-Lab-free: it works on the real ``ManagerBasedRLEnvCfg``
    rewards group AND on a simple fake exposing the same term attributes, so it
    is unit-testable without the simulator.

    Args:
        env_cfg: A manager-based env cfg (or a fake) exposing a ``rewards`` group
            whose attributes are reward-term cfgs carrying a ``weight`` field.

    Returns:
        The names of the stock velocity-tracking terms that were found and
        zeroed (empty when none are present).
    """
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return []

    zeroed: list[str] = []
    for term_name in STOCK_VELOCITY_REWARD_TERMS:
        term = getattr(rewards, term_name, None)
        if term is None:
            continue
        if hasattr(term, "weight"):
            term.weight = 0.0
            zeroed.append(term_name)
        else:
            # Some configs disable a term by setting the attribute to None.
            setattr(rewards, term_name, None)
            zeroed.append(term_name)
    return zeroed


# --------------------------------------------------------------------------- #
# Isaac Lab observation term + env attachment (LAZY / GUARDED)
# --------------------------------------------------------------------------- #
def goal_observation_term(env: Any) -> Any:
    """Isaac Lab observation function: robot-frame Goal_Observation (Req 6.2).

    Conforms to the manager-based observation-term signature ``fn(env) ->
    torch.Tensor`` of shape ``(num_envs, GOAL_OBS_DIM)`` so it can be registered
    as an ``ObservationTermCfg`` and appended to the proprio observation group.

    Reads the robot base position/yaw off the env's articulation and the goal off
    the per-env :class:`GoalBuffer` published at :data:`GOAL_BUFFER_ENV_ATTR`.
    The actual tensor math is delegated to :func:`compute_goal_observation`, the
    same transform exercised in the torch-free tests.

    This function performs NO Isaac Lab import; it only reads duck-typed state off
    ``env``, so it can be driven by a fake env in tests and by the real env at
    runtime.
    """
    buffer = _resolve_goal_buffer(env)
    robot_xy, base_yaw = _read_base_pose(env)
    return compute_goal_observation(robot_xy, base_yaw, buffer.goal_xy)


def _resolve_goal_buffer(env: Any) -> GoalBuffer:
    """Locate the per-env :class:`GoalBuffer` on a (possibly wrapped) env."""
    buffer = getattr(env, GOAL_BUFFER_ENV_ATTR, None)
    if buffer is not None:
        return buffer
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:
        return _resolve_goal_buffer(unwrapped)

    # Lazy fallback: during gym.make the observation manager may evaluate the
    # Goal_Observation term BEFORE the trainer attaches the per-env GoalBuffer.
    # If the env (or its cfg) carries the stashed Goal from
    # attach_goal_conditioning (``_goal`` / ``_goal_num_envs``), build and attach
    # the buffer on demand so construction-time evaluation succeeds. The trainer's
    # later explicit attach simply overwrites this with the same broadcast Goal.
    target = getattr(env, "unwrapped", env) or env
    goal = getattr(target, "_goal", None) or getattr(env, "_goal", None)
    cfg = getattr(target, "cfg", None)
    if goal is None and cfg is not None:
        goal = getattr(cfg, "_goal", None)
    num_envs = (
        getattr(target, "num_envs", None)
        or getattr(target, "_goal_num_envs", None)
        or (getattr(cfg, "_goal_num_envs", None) if cfg is not None else None)
    )
    if goal is not None and num_envs:
        buffer = make_goal_buffer(goal, int(num_envs), device=getattr(target, "device", None))
        try:
            setattr(target, GOAL_BUFFER_ENV_ATTR, buffer)
        except (AttributeError, TypeError):
            pass
        return buffer

    raise AttributeError(
        f"env exposes no '{GOAL_BUFFER_ENV_ATTR}'; attach one with "
        f"attach_goal_conditioning() before reading the Goal_Observation."
    )


def _read_base_pose(env: Any) -> tuple[Any, Any]:
    """Read robot base ``(x, y)`` (ENV-LOCAL) and yaw from the env's articulation.

    Duck-typed against the manager-based H1 env: the robot articulation lives at
    ``env.scene["robot"]`` with ``data.root_pos_w`` ``(num_envs, 3)`` and
    ``data.root_quat_w`` ``(num_envs, 4)`` in ``(w, x, y, z)`` order. A fake env
    in tests exposes the same shape.

    ``root_pos_w`` is ABSOLUTE world position; Isaac Lab spaces parallel envs on
    a grid, so it includes env ``i``'s origin offset (``scene.env_origins[i]``).
    The Goal is env-local, so the Goal_Observation (vector/distance/heading to
    goal) handed to the POLICY must use the env-local position — otherwise every
    env but the one at the world origin sees a goal tens of metres away in the
    wrong direction, and the policy learns from a corrupted goal signal. Subtract
    the env origin when available (falls back to world == local for a single-env
    scene at the origin / fakes).
    """
    scene = getattr(env, "scene", None)
    if scene is None:
        raise AttributeError("env has no 'scene' to read the robot base pose from.")
    robot = scene["robot"] if _supports_getitem(scene) else getattr(scene, "robot")
    data = robot.data
    root_pos = data.root_pos_w
    root_quat = data.root_quat_w
    robot_xy = root_pos[..., 0:2]
    # Localize against the env grid origin so the goal vector is env-relative.
    origins = getattr(scene, "env_origins", None)
    if origins is not None:
        try:
            robot_xy = robot_xy - origins[..., 0:2]
        except Exception:  # noqa: BLE001 - shape/type mismatch on a fake env
            pass
    base_yaw = yaw_from_quat(root_quat)
    return robot_xy, base_yaw


def _supports_getitem(obj: Any) -> bool:
    try:
        return hasattr(obj, "__getitem__")
    except Exception:  # noqa: BLE001
        return False


def attach_goal_conditioning(
    env_cfg: Any,
    goal: Goal,
    num_envs: int,
    *,
    obs_group: str = "policy",
    obs_term_name: str = "goal_observation",
) -> list[str]:
    """Wire goal-conditioning onto a manager-based H1 flat env cfg (Req 6.2, 8.1).

    Performs the three cfg-level edits of the reframing, then returns the names of
    the stock velocity-tracking reward terms that were zeroed:

    1. Append a Goal_Observation :class:`~isaaclab.managers.ObservationTermCfg`
       (backed by :func:`goal_observation_term`) to the ``obs_group`` observation
       group, so the policy sees the robot-frame goal signal alongside the ~69-dim
       proprio vector (design.md step 3; Req 6.2).
    2. Zero the stock velocity-tracking reward terms via
       :func:`disable_velocity_tracking_rewards` (design.md; Req 8.1).

    The per-env :class:`GoalBuffer` itself is created from ``goal``/``num_envs``
    at env-construction time (see :func:`make_goal_buffer`) and published on the
    live env at :data:`GOAL_BUFFER_ENV_ATTR`; this cfg-level helper only wires the
    observation term and reward zeroing, which is all that lives on the cfg.

    Isaac Lab is imported **lazily and only here** so the module stays importable
    without ``isaaclab``. When Isaac Lab is unavailable (controller/dev host) the
    observation-term append is skipped with the import guarded; the reward-zeroing
    still runs because it is pure attribute manipulation. Callers that need the
    full wiring run inside the Isaac Sim runtime.

    Args:
        env_cfg: The manager-based env cfg to mutate in place.
        goal: The configured :class:`Goal` (recorded for buffer construction).
        num_envs: Parallel env count the Goal is broadcast to.
        obs_group: Observation group to append the Goal_Observation to.
        obs_term_name: Attribute name for the appended observation term.

    Returns:
        The names of the stock reward terms that were zeroed (Req 8.1).
    """
    # (1) Append the Goal_Observation term — needs Isaac Lab's ObservationTermCfg.
    try:  # pragma: no cover - exercised only where isaaclab is installed
        from isaaclab.managers import ObservationTermCfg  # noqa: PLC0415

        obs = getattr(env_cfg, "observations", None)
        group = getattr(obs, obs_group, None) if obs is not None else None
        if group is not None:
            setattr(group, obs_term_name, ObservationTermCfg(func=goal_observation_term))
    except ImportError:
        # Controller/dev host without Isaac Lab: the observation-term append is a
        # runtime-only concern. The reward zeroing below is pure and still runs.
        pass

    # (2) Zero the stock velocity-tracking reward (pure; always runs). Req 8.1.
    zeroed = disable_velocity_tracking_rewards(env_cfg)

    # Stash the Goal/num_envs on the cfg so the env builder can broadcast the
    # per-env GoalBuffer at construction time (design.md step 2).
    setattr(env_cfg, "_goal", goal)
    setattr(env_cfg, "_goal_num_envs", num_envs)
    return zeroed
