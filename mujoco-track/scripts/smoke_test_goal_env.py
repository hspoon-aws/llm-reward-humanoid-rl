#!/usr/bin/env python3
"""Goal-env smoke test (Phase 2): build the goal-reframed H1 env, reset+step
under jit, confirm the Goal_Observation is appended and the goal reward flows.

Run inside the MJX venv on the B200:
    MUJOCO_GL=egl /data/mjxvenv/bin/python scripts/smoke_test_goal_env.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class _Goal:
    position_xy: tuple
    success_radius_m: float


def main() -> int:
    import jax

    sys.path.insert(0, "/data/mujoco")
    from src.envs.goal_env import GOAL_OBS_DIMS, build_goal_env

    goal = _Goal(position_xy=(5.0, 0.0), success_radius_m=0.5)
    env = build_goal_env("H1JoystickGaitTracking", goal)

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(0))
    obs = state.obs
    obs_arr = obs if not isinstance(obs, dict) else obs[next(iter(obs))]
    print(f"[goal-env] augmented obs shape={obs_arr.shape} (stock 113 + {GOAL_OBS_DIMS} goal)")
    print(f"[goal-env] goal_initial_distance={float(state.info['goal_initial_distance']):.3f}")

    action = jax.numpy.zeros(env.action_size)
    nstate = step(state, action)
    print(f"[goal-env] step reward={float(nstate.reward):.4f}")
    goal_terms = {k: float(v) for k, v in nstate.metrics.items() if k.startswith("goal/")}
    print(f"[goal-env] goal reward components: {goal_terms}")

    if obs_arr.shape[-1] != 113 + GOAL_OBS_DIMS:
        print(f"[FAIL] expected obs dim {113 + GOAL_OBS_DIMS}, got {obs_arr.shape[-1]}")
        return 1
    if not goal_terms:
        print("[FAIL] no goal/* reward components recorded")
        return 1

    # vmap a small batch to confirm the training execution model.
    keys = jax.random.split(jax.random.PRNGKey(1), 64)
    batch = jax.jit(jax.vmap(env.reset))(keys)
    batch = jax.jit(jax.vmap(env.step))(batch, jax.numpy.zeros((64, env.action_size)))
    batch.reward.block_until_ready()
    print(f"[goal-env] vmap(64) step ok: mean reward={float(jax.numpy.mean(batch.reward)):.4f}")

    print("=== GOAL ENV SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
