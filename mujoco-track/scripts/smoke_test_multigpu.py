#!/usr/bin/env python3
"""Multi-GPU Brax PPO sharding smoke (Phase 7 prep).

Validates that the trainer's batch-size derivation and num_envs divisibility
work across MULTIPLE JAX devices — the path that tripped the asserts in §5.3/5.4
of the lessons. Run on the FREE GPUs only (never GPU 0=vLLM or GPU 1=live run):

    CUDA_VISIBLE_DEVICES=2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 MUJOCO_GL=egl \
        /data/mjxvenv/bin/python scripts/smoke_test_multigpu.py

It asserts:
  * JAX sees the expected number of devices,
  * num_envs % device_count == 0,
  * Brax PPO trains a tiny budget across all devices without the divisibility
    asserts firing, and a checkpoint is written.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class _Goal:
    position_xy: tuple
    success_radius_m: float


@dataclass
class _TrainConfig:
    env_name: str = "H1JoystickGaitTracking"
    train_epochs: int = 2
    checkpoint_interval: int = 2
    checkpoint_dir: str = "/data/mujoco/runs/mgpu"
    episode_length: int = 128
    training_gpus: tuple = (2, 3, 4, 5)
    xla_python_client_mem_fraction: float = 0.4
    oom_fallback_envs: int = 1792
    seed: int = 0


GOOD_REWARD = '''
def compute_reward(data, action, goal_xy, success_radius):
    x = data.qpos[0]; y = data.qpos[1]; qw = data.qpos[3]
    dx = goal_xy[0] - x; dy = goal_xy[1] - y
    dist = jnp.sqrt(dx*dx + dy*dy + 1e-9)
    upright = jnp.clip(2.0*qw*qw - 1.0, -1.0, 1.0)
    arrived = jnp.where(dist < success_radius, 1.0, 0.0)
    comps = {
        "progress": jnp.nan_to_num(-0.1*dist),
        "arrival": 5.0*arrived,
        "upright": 0.5*upright,
        "alive": jnp.float32(0.1),
    }
    return jnp.nan_to_num(sum(comps.values())), comps
'''


def main() -> int:
    sys.path.insert(0, "/data/mujoco")
    import jax

    from src.rewards.reward_executor import RewardExecutor, SandboxConfig
    from src.train.mjx_trainer import build_mjx_trainer

    ndev = jax.device_count()
    print(f"[mgpu] jax device_count={ndev} devices={jax.devices()}")
    if ndev < 2:
        print("[FAIL] expected >=2 visible devices; set CUDA_VISIBLE_DEVICES=2,3,4,5")
        return 1

    # num_envs must be divisible by device count. 2048 is divisible by 4 but not
    # by 7; for a 7-GPU run use a 7-multiple (e.g. 7*256=1792 or 7*512=3584).
    num_envs = 512 * ndev
    if num_envs % ndev != 0:
        print(f"[FAIL] num_envs {num_envs} not divisible by {ndev}")
        return 1
    print(f"[mgpu] num_envs={num_envs} ({num_envs//ndev}/device)")

    ex = RewardExecutor(SandboxConfig(time_limit_s=10.0, jit_check=True))
    wrapped = ex.wrap(GOOD_REWARD)[0]
    goal = _Goal(position_xy=(5.0, 0.0), success_radius_m=0.5)

    trainer = build_mjx_trainer(
        _TrainConfig(), reward_fn=wrapped, num_envs=num_envs,
        learning_rate=1e-3, goal=goal, device="cuda:0",
    )
    print("[mgpu] trainer built; running tiny Brax PPO across all devices...")
    metrics = trainer.run_epoch()
    ckpt = trainer.save_checkpoint("/data/mujoco/runs/mgpu/policy.pkl")
    import os
    ok = os.path.exists(ckpt) and os.path.getsize(ckpt) > 0
    print(f"[mgpu] checkpoint: {ckpt} ({os.path.getsize(ckpt) if ok else 0} bytes)")
    if not ok:
        print("[FAIL] checkpoint missing")
        return 1
    print(f"[mgpu] sample metric keys: {sorted(list(metrics))[:5]}")
    print("=== MULTI-GPU SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
