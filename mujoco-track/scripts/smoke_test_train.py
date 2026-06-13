#!/usr/bin/env python3
"""Brax PPO training smoke test (Phase 4): a generated JAX reward -> goal env ->
Brax PPO trains a few steps -> checkpoint written. Tiny budget so it finishes in
a couple of minutes on the B200 (this is the hour-0-style gate, not a real run).

Run inside the MJX venv on the B200:
    MUJOCO_GL=egl /data/mjxvenv/bin/python scripts/smoke_test_train.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass


@dataclass
class _Goal:
    position_xy: tuple
    success_radius_m: float


@dataclass
class _TrainConfig:
    env_name: str = "H1JoystickGaitTracking"
    train_epochs: int = 1
    episode_length: int = 128
    checkpoint_dir: str = "/data/mujoco/runs/smoke"
    seed: int = 0


GOOD_REWARD = '''
def compute_reward(data, action, goal_xy, success_radius):
    x = data.qpos[0]; y = data.qpos[1]; qw = data.qpos[3]
    dx = goal_xy[0] - x; dy = goal_xy[1] - y
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-9)
    upright = jnp.clip(2.0 * qw * qw - 1.0, -1.0, 1.0)
    arrived = jnp.where(dist < success_radius, 1.0, 0.0)
    comps = {
        "progress": jnp.nan_to_num(-0.1 * dist),
        "arrival": 5.0 * arrived,
        "upright": 0.5 * upright,
        "alive": jnp.float32(0.1),
        "effort": jnp.nan_to_num(-0.001 * jnp.sum(action * action)),
    }
    return jnp.nan_to_num(sum(comps.values())), comps
'''


def main() -> int:
    sys.path.insert(0, "/data/mujoco")
    from src.rewards.reward_executor import RewardExecutor, SandboxConfig
    from src.train.mjx_trainer import build_mjx_trainer

    goal = _Goal(position_xy=(5.0, 0.0), success_radius_m=0.5)
    ex = RewardExecutor(SandboxConfig(time_limit_s=10.0, jit_check=True))
    wrapped = ex.wrap(GOOD_REWARD)[0]
    print("[train] reward wrapped + jit-checked")

    cfg = _TrainConfig()
    trainer = build_mjx_trainer(
        cfg,
        reward_fn=wrapped,
        num_envs=256,          # tiny for the smoke
        learning_rate=1e-3,
        goal=goal,
        device="cuda:1",
    )
    print("[train] trainer built; starting Brax PPO (tiny budget)...")
    t0 = time.time()
    metrics = trainer.run_epoch()
    dt = time.time() - t0
    print(f"[train] PPO run complete in {dt:.1f}s")
    # Print a few headline metrics if present.
    keys = [k for k in metrics if "reward" in k.lower() or "eval" in k.lower()][:6]
    for k in keys:
        print(f"         {k}={metrics[k]}")

    ckpt = trainer.save_checkpoint(f"{cfg.checkpoint_dir}/policy.pkl")
    import os
    ok = os.path.exists(ckpt) and os.path.getsize(ckpt) > 0
    print(f"[train] checkpoint written: {ckpt} ({os.path.getsize(ckpt)} bytes)")
    if not ok:
        print("[FAIL] checkpoint missing/empty")
        return 1
    print(f"[train] metrics rows logged: {len(trainer.metrics_log)}")
    print("=== TRAIN SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
