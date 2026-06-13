#!/usr/bin/env python3
"""Reward-contract smoke test (Phase 3): a generated-style JAX reward goes
through validate -> sandbox-compile -> jit-check -> inject into the goal env ->
vmap'd jitted training step. This is the highest-risk gate: it proves
LLM-style JAX reward code is jit-traceable and drives the env.

Also runs a NEGATIVE case: a non-jittable reward (value-dependent Python `if`)
must be rejected by the jit-check with an ExecutionError (Req 5.5).

Run inside the MJX venv on the B200:
    MUJOCO_GL=egl /data/mjxvenv/bin/python scripts/smoke_test_reward.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class _Goal:
    position_xy: tuple
    success_radius_m: float


# A realistic goal-reaching reward in the contracted JAX form. jit-safe:
# jnp ops only, jnp.where instead of a Python branch, jnp.nan_to_num guards.
GOOD_REWARD = '''
def compute_reward(data, action, goal_xy, success_radius):
    import jax.numpy as jnp
    x = data.qpos[0]
    y = data.qpos[1]
    base_z = data.qpos[2]
    qw = data.qpos[3]
    dx = goal_xy[0] - x
    dy = goal_xy[1] - y
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-9)
    upright = jnp.clip(2.0 * qw * qw - 1.0, -1.0, 1.0)
    arrived = jnp.where(dist < success_radius, 1.0, 0.0)
    progress = jnp.nan_to_num(-0.1 * dist)
    arrival = 5.0 * arrived
    upright_term = 0.5 * upright
    alive = jnp.float32(0.1)
    effort = jnp.nan_to_num(-0.001 * jnp.sum(action * action))
    components = {
        "progress": progress,
        "arrival": arrival,
        "upright": upright_term,
        "alive": alive,
        "effort": effort,
    }
    reward = progress + arrival + upright_term + alive + effort
    return jnp.nan_to_num(reward), components
'''

# A non-jittable reward: branches on a traced value with a Python if.
BAD_REWARD = '''
def compute_reward(data, action, goal_xy, success_radius):
    import jax.numpy as jnp
    dist = jnp.sqrt(jnp.sum((goal_xy - data.qpos[0:2]) ** 2))
    if dist < success_radius:        # value-dependent Python branch -> not traceable
        return 5.0, {"arrival": 5.0}
    return -0.1 * dist, {"progress": -0.1 * dist}
'''


def main() -> int:
    import jax

    sys.path.insert(0, "/data/mujoco")
    from src.rewards.reward_executor import RewardExecutor, SandboxConfig
    from src.exceptions import ExecutionError
    from src.envs.goal_env import build_goal_env

    ex = RewardExecutor(SandboxConfig(time_limit_s=10.0, jit_check=True))

    # --- NEGATIVE: non-jittable reward must be rejected up front (Req 5.5) ---
    print("[reward] negative case: non-jittable reward should be rejected...")
    try:
        ex.wrap(BAD_REWARD)
        print("[FAIL] non-jittable reward was NOT rejected by the jit-check")
        return 1
    except ExecutionError as exc:
        print(f"[ok] jit-check rejected non-traceable reward: {str(exc)[:90]}...")

    # --- POSITIVE: good reward wraps + jit-checks clean ---
    print("[reward] positive case: jit-safe reward should wrap...")
    wrapped = ex.wrap(GOOD_REWARD)[0]
    print("[ok] jit-safe reward wrapped + passed jit-check")

    # --- inject into the goal env and run a vmap'd jitted step ---
    goal = _Goal(position_xy=(5.0, 0.0), success_radius_m=0.5)
    env = build_goal_env("H1JoystickGaitTracking", goal, reward_fn=wrapped)

    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    batch = jax.jit(jax.vmap(env.reset))(keys)
    batch = jax.jit(jax.vmap(env.step))(batch, jax.numpy.zeros((64, env.action_size)))
    batch.reward.block_until_ready()
    mean_r = float(jax.numpy.mean(batch.reward))
    goal_terms = sorted(k for k in batch.metrics if k.startswith("goal/"))
    print(f"[reward] vmap(64) step with generated reward: mean reward={mean_r:.4f}")
    print(f"[reward] goal components present: {goal_terms}")

    if not goal_terms:
        print("[FAIL] generated reward components not recorded in env metrics")
        return 1
    print("=== REWARD CONTRACT SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
