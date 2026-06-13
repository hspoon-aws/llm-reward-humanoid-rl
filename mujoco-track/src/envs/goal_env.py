"""Goal-conditioned MJX environment wrapper (point-A-to-point-B reframing).

MuJoCo analog of the Isaac project's ``goal_env.py``. Wraps a stock MuJoCo
Playground locomotion env (default ``H1JoystickGaitTracking``) and reframes its
gait/velocity-tracking objective as **point-to-point goal-reaching**:

  1. store a Goal (target x,y on the ground plane) — a constant broadcast over
     the vmap'd batch (Point A = each env's reset pose);
  2. append a robot-frame Goal_Observation (vector-to-goal + distance + heading)
     to the proprioceptive obs (NO image modality, Req 1.7);
  3. replace the stock gait/velocity-tracking reward with the LLM-generated
     goal reward so the generated reward fully owns the objective (Req 6);
  4. record the straight-line A->Goal distance at reset for path-efficiency.

The real H1 surface (verified on the B200 — see docs/lesson-mjx-b200-bringup.md):
  - stock obs: flat (113,) float32; we append 4 Goal dims -> (117,)
  - action: (19,) joint targets; dt = 0.02 s
  - state.data (mjx.Data): qpos(26)=free7+joints19, qvel(25)=base6+joint19,
    xpos(21,3), actuator_force(19), cfrc_ext(21,6)

DI seam: ``jax`` / ``mujoco_playground`` are imported lazily inside
``build_goal_env`` only, so importing this module on the controller host pulls
in no GPU stack (locked by tests/test_di_seam.py).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

DEFAULT_ENV_NAME = "H1JoystickGaitTracking"
# Documented fallback (better-supported in Playground, different robot):
FALLBACK_ENV_NAME = "G1JoystickFlatTerrain"

# Number of dims the Goal_Observation appends to the proprio obs:
# (vec_to_goal_x, vec_to_goal_y, distance, heading_error).
GOAL_OBS_DIMS = 4


def build_goal_env(
    env_name: str,
    goal: Any,
    *,
    reward_fn: Optional[Callable] = None,
    config_overrides: Any = None,
    fall_terminate: bool = False,
    fall_height_m: float = 0.5,
    standing_reward: float = 0.0,
) -> Any:
    """Load a Playground locomotion env and wrap it for goal-reaching.

    Args:
        env_name: Playground registry id (default ``H1JoystickGaitTracking``);
            sourced from ``training.env_name`` so H1<->G1 is a config edit.
        goal: The ``Goal`` (target position + success radius) to inject. Accepts
            anything exposing ``position_xy`` and ``success_radius_m`` (a
            ``data_models.Goal`` or ``GoalRef``).
        reward_fn: Optional JAX goal-reward callable
            ``compute_reward(data, action, goal_xy, success_radius) ->
            (reward, components)`` produced by the Reward_Executor. When ``None``
            a built-in shaped goal reward is used (so the env is runnable for
            smoke tests before a generated reward exists).
        config_overrides: Optional Playground config overrides.

    Returns:
        A ``GoalReachingEnv`` (a thin ``MjxEnv`` subclass) ready for Brax PPO.

    Raises:
        RuntimeError: When the MJX/Playground stack is not importable here.
    """
    try:  # pragma: no cover - exercised only on the B200 GPU host
        import jax  # noqa: F401
        from mujoco_playground import registry
    except ImportError as exc:  # pragma: no cover - dev host path
        raise RuntimeError(
            "mujoco_playground is not importable in this environment; "
            "goal-env construction runs only on the GPU host."
        ) from exc

    base_env = registry.load(env_name, config_overrides=config_overrides)
    goal_xy = tuple(float(v) for v in goal.position_xy)
    success_radius = float(goal.success_radius_m)
    _ensure_class_materialized()
    return GoalReachingEnv(  # pragma: no cover - requires the GPU runtime
        base_env,
        goal_xy=goal_xy,
        success_radius=success_radius,
        reward_fn=reward_fn,
        fall_terminate=fall_terminate,
        fall_height_m=fall_height_m,
        standing_reward=standing_reward,
    )


# The class body references jax/jnp; define it lazily so module import stays
# GPU-free. We build it on first call to build_goal_env via a factory closure.
class _LazyGoalEnvNamespace:
    """Placeholder so ``GoalReachingEnv`` is resolvable at module import without
    importing JAX. The real class is materialized on first ``build_goal_env``."""


def _materialize_goal_env_class():  # pragma: no cover - GPU host only
    import jax
    import jax.numpy as jnp
    from mujoco_playground._src import mjx_env

    class GoalReachingEnv(mjx_env.MjxEnv):
        """Stock locomotion env reframed as point-to-point goal-reaching.

        Delegates physics to the wrapped Playground env; overrides the reward
        (generated goal reward, stock reward zeroed) and appends the robot-frame
        Goal_Observation to the obs.
        """

        def __init__(
            self,
            base_env,
            *,
            goal_xy,
            success_radius,
            reward_fn=None,
            fall_terminate=False,
            fall_height_m=0.5,
            standing_reward=0.0,
        ):
            # Reuse the base env's config so MjxEnv internals (sim dt, etc.) match.
            self._base = base_env
            self._goal_xy = jnp.asarray(goal_xy, dtype=jnp.float32)
            self._success_radius = jnp.float32(success_radius)
            self._reward_fn = reward_fn
            # Balance-before-locomotion controls (run v4 finding: the H1 fell in
            # ~0.4 s every episode because the base gait-tracking env does NOT
            # terminate on fall, so a fallen robot kept earning goal reward and
            # learned to "fall toward the goal". When enabled, the episode ends
            # the moment the torso drops below ``fall_height_m`` (removing the
            # post-fall reward stream that trains fall-forward), and a small
            # per-step ``standing_reward`` floor positively reinforces staying up.
            # Default OFF so existing runs/behaviour are unchanged.
            self._fall_terminate = bool(fall_terminate)
            self._fall_height_m = jnp.float32(fall_height_m)
            self._standing_reward = jnp.float32(standing_reward)
            # MjxEnv stores config + model; mirror from the base env.
            self._config = base_env._config
            self._mjx_model = base_env.mjx_model
            self._mj_model = base_env.mj_model

        # ----- MjxEnv plumbing delegated to the base env ----------------- #
        @property
        def xml_path(self):
            return self._base.xml_path

        @property
        def action_size(self):
            return self._base.action_size

        @property
        def mj_model(self):
            return self._base.mj_model

        @property
        def mjx_model(self):
            return self._base.mjx_model

        # ----- Goal helpers ---------------------------------------------- #
        def _base_xy_yaw(self, data):
            """Extract base (x, y) and yaw from qpos (free joint: pos3 + quat4)."""
            x = data.qpos[0]
            y = data.qpos[1]
            qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            # yaw from quaternion (z-axis rotation)
            yaw = jnp.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            return x, y, yaw

        def _goal_obs(self, data):
            """Robot-frame (vec_x, vec_y, distance, heading) Goal_Observation."""
            x, y, yaw = self._base_xy_yaw(data)
            dx = self._goal_xy[0] - x
            dy = self._goal_xy[1] - y
            cos_y = jnp.cos(-yaw)
            sin_y = jnp.sin(-yaw)
            vec_x = dx * cos_y - dy * sin_y
            vec_y = dx * sin_y + dy * cos_y
            dist = jnp.sqrt(dx * dx + dy * dy + 1e-9)
            heading = jnp.arctan2(vec_y, vec_x)
            return jnp.array([vec_x, vec_y, dist, heading], dtype=jnp.float32)

        def _augment_obs(self, obs, data):
            goal_obs = self._goal_obs(data)
            if isinstance(obs, dict):
                # Append to the policy obs group if obs is a dict.
                out = dict(obs)
                key = "state" if "state" in out else next(iter(out))
                out[key] = jnp.concatenate([out[key], goal_obs])
                return out
            return jnp.concatenate([obs, goal_obs])

        # ----- env API ---------------------------------------------------- #
        def reset(self, rng):
            state = self._base.reset(rng)
            # Record straight-line A->Goal distance for path efficiency.
            x, y, _ = self._base_xy_yaw(state.data)
            d0 = jnp.sqrt(
                (self._goal_xy[0] - x) ** 2 + (self._goal_xy[1] - y) ** 2 + 1e-9
            )
            info = dict(state.info)
            info["goal_initial_distance"] = d0
            info["goal_xy"] = self._goal_xy
            # Seed the goal/* metric keys at reset so the metrics pytree structure
            # matches the post-step structure. Brax's training wrapper scans the
            # env step and requires identical carry (metrics) structure between
            # reset and step (else: "carry input/output must have the same pytree
            # structure"). We compute the real goal reward once on the reset data
            # so the keys + dtypes line up exactly.
            zero_action = jnp.zeros(self._base.action_size)
            _, components = self._compute_goal_reward(state.data, zero_action)
            metrics = dict(state.metrics)
            for name, value in components.items():
                metrics[f"goal/{name}"] = self._scalar_f32(value)
            state = state.replace(
                obs=self._augment_obs(state.obs, state.data),
                info=info,
                metrics=metrics,
            )
            return state

        def _scalar_f32(self, value):
            """Coerce any reward/component value to a finite float32 scalar.

            Untrusted generated rewards may return a component as a Python float,
            an int, or a non-scalar array (e.g. per-joint). Brax's lax.scan can
            only carry uniform scalar metrics, so reduce anything non-scalar to
            its mean and guard NaN/Inf. This makes the env boundary robust to
            whatever the model emits (the runtime analog of the jit-check)."""
            arr = jnp.asarray(value, dtype=jnp.float32)
            arr = jnp.mean(arr) if arr.ndim > 0 else arr
            return jnp.nan_to_num(arr)

        def step(self, state, action):
            nstate = self._base.step(state, action)
            reward, components = self._compute_goal_reward(nstate.data, action)
            metrics = dict(nstate.metrics)
            for name, value in components.items():
                metrics[f"goal/{name}"] = self._scalar_f32(value)

            # Balance-before-locomotion shaping (run v4 finding). Both are no-ops
            # at their defaults (standing_reward=0, fall_terminate=False), so the
            # legacy behaviour is byte-for-byte unchanged unless a config opts in.
            base_z = nstate.data.qpos[2]
            upright_now = base_z >= self._fall_height_m
            # Small per-step floor for being upright, added to the generated
            # reward so "stay standing" is positively reinforced even before the
            # LLM reward discovers it.
            reward = reward + self._standing_reward * upright_now.astype(jnp.float32)

            nstate = nstate.replace(
                obs=self._augment_obs(nstate.obs, nstate.data),
                reward=self._scalar_f32(reward),
                metrics=metrics,
            )

            if self._fall_terminate:
                # End the episode the instant the torso drops below the fall
                # height. This removes the post-fall reward stream that trained
                # "fall toward the goal" in run v4. ``done`` is a float32 flag in
                # the Brax/Playground convention; OR it with any existing done.
                fell = (~upright_now).astype(jnp.float32)
                done = jnp.maximum(self._scalar_f32(nstate.done), fell)
                nstate = nstate.replace(done=done)
            return nstate

        def _compute_goal_reward(self, data, action):
            if self._reward_fn is not None:
                return self._reward_fn(
                    data, action, self._goal_xy, self._success_radius
                )
            return self._builtin_goal_reward(data, action)

        def _builtin_goal_reward(self, data, action):
            """A simple shaped goal reward used when no LLM reward is injected.

            Lets the env run end-to-end for smoke tests before a generated reward
            exists. Distance-progress + arrival bonus + upright + alive - effort.
            """
            x, y, _ = self._base_xy_yaw(data)
            dist = jnp.sqrt(
                (self._goal_xy[0] - x) ** 2 + (self._goal_xy[1] - y) ** 2 + 1e-9
            )
            base_z = data.qpos[2]
            qw = data.qpos[3]
            upright = jnp.clip(2.0 * qw * qw - 1.0, -1.0, 1.0)  # cos of tilt-ish
            arrived = (dist < self._success_radius).astype(jnp.float32)
            comps = {
                "progress": jnp.nan_to_num(-0.1 * dist),
                "arrival": 5.0 * arrived,
                "upright": 0.5 * upright,
                "alive": jnp.float32(0.1),
                "effort": jnp.nan_to_num(-0.001 * jnp.sum(action * action)),
            }
            total = sum(comps.values())
            return jnp.nan_to_num(total), comps

    return GoalReachingEnv


# Public name; replaced with the real class on first build_goal_env call.
GoalReachingEnv: Any = _LazyGoalEnvNamespace


def _ensure_class_materialized() -> None:  # pragma: no cover - GPU host only
    global GoalReachingEnv
    if GoalReachingEnv is _LazyGoalEnvNamespace:
        GoalReachingEnv = _materialize_goal_env_class()
