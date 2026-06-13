#!/usr/bin/env python3
"""Probe the H1JoystickGaitTracking MJX env to capture the real obs/action shape
and state-access surface needed to write the goal wrapper (Phase 2).

Run inside the MJX venv on the B200:
    MUJOCO_GL=egl /data/mjxvenv/bin/python scripts/probe_h1_env.py
"""
from __future__ import annotations

import sys


def main() -> int:
    import jax
    from mujoco_playground import registry

    env_name = "H1JoystickGaitTracking"
    env = registry.load(env_name)
    cfg = registry.get_default_config(env_name)

    print(f"[probe] env={env_name}")
    print(f"[probe] observation_size={getattr(env, 'observation_size', '?')}")
    print(f"[probe] action_size={getattr(env, 'action_size', '?')}")
    print(f"[probe] dt={getattr(env, 'dt', '?')}")

    key = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(key)

    obs = state.obs
    # obs may be a dict (multiple obs groups) or a flat array.
    if isinstance(obs, dict):
        print("[probe] obs is a dict with keys + shapes:")
        for k, v in obs.items():
            print(f"         {k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"[probe] obs: shape={obs.shape} dtype={obs.dtype}")

    # mjx.Data handle lives on state.data — confirm the accessor surface the
    # generated reward will read.
    data = state.data
    print("[probe] state.data accessors present:")
    for attr in ("qpos", "qvel", "xpos", "actuator_force", "cfrc_ext"):
        v = getattr(data, attr, None)
        shape = getattr(v, "shape", None)
        print(f"         data.{attr}: {'present shape=' + str(shape) if v is not None else 'ABSENT'}")

    # Step once to confirm the transition contract.
    action = jax.numpy.zeros(env.action_size)
    nstate = jax.jit(env.step)(state, action)
    print(f"[probe] step ok: reward={float(nstate.reward):.4f} done={float(nstate.done)}")
    print("[probe] reward term keys (state.metrics):")
    for k in list(getattr(nstate, "metrics", {}) or {}):
        print(f"         {k}")
    print("=== H1 ENV PROBE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
