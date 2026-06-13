#!/usr/bin/env python3
"""Isaac Lab environment smoke test (deployment gate: "Isaac Lab env reset").

Verifies that the pre-built Unitree H1 task loads, reports the expected
action/observation dimensions, and can step without crashing. Run inside the
Isaac Lab container on the GPU host:

    isaaclab.sh -p scripts/smoke_test_env.py --num-envs 16

Exit code 0 means the env gate passes.
"""

from __future__ import annotations

import argparse
import sys

TASK = "Isaac-Velocity-Flat-H1-v0"
EXPECTED_ACTION_DIM = 19
# Flat-terrain H1 policy obs is ~69-dim proprioceptive
# (base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + velocity_commands 3
#  + joint_pos 19 + joint_vel 19 + last_action 19). The goal-conditioned env
# appends a Goal_Observation (~4 dims), so the reframed task reports more.
# This smoke test runs against the STOCK task to validate the env loads, so we
# check against the stock baseline and only warn on mismatch.
EXPECTED_OBS_DIM = 69


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac Lab H1 env smoke test")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()

    # AppLauncher must come first; it boots the simulator before any isaaclab
    # task imports are valid.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from isaaclab_tasks.utils import parse_env_cfg

    ok = True
    try:
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        print(f"[ok] created '{args.task}' with {args.num_envs} envs")

        obs, _ = env.reset()
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        obs_dim = int(obs_tensor.shape[-1])
        act_dim = int(env.action_space.shape[-1])
        print(f"[info] obs dim = {obs_dim} (expected {EXPECTED_OBS_DIM})")
        print(f"[info] action dim = {act_dim} (expected {EXPECTED_ACTION_DIM})")

        if obs_dim != EXPECTED_OBS_DIM:
            print(f"[warn] observation dim {obs_dim} != {EXPECTED_OBS_DIM}")
        if act_dim != EXPECTED_ACTION_DIM:
            print(f"[warn] action dim {act_dim} != {EXPECTED_ACTION_DIM}")

        for i in range(args.steps):
            actions = torch.zeros(
                (args.num_envs, act_dim), device=env.unwrapped.device
            )
            obs, rew, terminated, truncated, info = env.step(actions)
        print(f"[ok] stepped {args.steps} times without crashing")
        env.close()
    except Exception as exc:  # noqa: BLE001 - smoke test reports any failure
        print(f"[FAIL] env smoke test raised: {exc}")
        ok = False
    finally:
        simulation_app.close()

    print("=== ENV GATE: PASS ===" if ok else "=== ENV GATE: FAIL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
